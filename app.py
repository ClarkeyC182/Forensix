import streamlit as st
import pandas as pd
import time
import cv2
import json
import os
import uuid
import re
import altair as alt
from pathlib import Path
from google import genai
from google.genai import types
from fpdf import FPDF
from streamlit_gsheets import GSheetsConnection
import numpy as np
from datetime import datetime
from PIL import Image

# --- CONFIG ---
st.set_page_config(page_title="Forensix Personal Auditor", page_icon="🕵️‍♂️", layout="wide")
BASE_DIR = Path(__file__).parent.absolute()
DIRS = { "TEMP": BASE_DIR / "temp_uploads" }
for d in DIRS.values(): d.mkdir(exist_ok=True)

# CONSTANTS
REQUIRED_COLS = ["Date", "Vendor", "Item", "Amount", "Currency", "IVA", "Category", "Sub_Category", "Is_Vice", "File"]

# --- HELPER: SAFE TYPES ---
def safe_float(val):
    try:
        if val is None: return 0.0
        return float(val)
    except: return 0.0

def safe_date(val):
    try:
        s = str(val).strip().lower()
        if s in ['none', 'null', 'nan', '', 'yyyy-mm-dd', 'unknown']: return pd.Timestamp.now()
        dt = pd.to_datetime(val, errors='coerce')
        return pd.Timestamp.now() if pd.isna(dt) else dt
    except: return pd.Timestamp.now()

# --- DATABASE ENGINE ---
def load_data():
    df_template = pd.DataFrame(columns=REQUIRED_COLS)
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        df_sheet = conn.read(worksheet="Sheet1", ttl=0)
        if not df_sheet.empty:
            df_final = pd.concat([df_template, df_sheet], ignore_index=True)
        else:
            df_final = df_template

        df_final = df_final.reindex(columns=REQUIRED_COLS)
        
        # Sanitize
        df_final['Amount'] = df_final['Amount'].apply(safe_float)
        df_final['IVA'] = df_final['IVA'].apply(safe_float)
        df_final['Is_Vice'] = df_final['Is_Vice'].fillna(False).astype(bool)
        df_final['Date'] = df_final['Date'].apply(safe_date)
        
        for c in ['Vendor', 'Item', 'Currency', 'Category', 'Sub_Category', 'File']:
            df_final[c] = df_final[c].astype(str).replace('nan', '').replace('None', '')

        return df_final
    except: return df_template

def save_data(df):
    conn = st.connection("gsheets", type=GSheetsConnection)
    df_save = df.copy()
    df_save['Date'] = df_save['Date'].apply(safe_date).dt.strftime('%Y-%m-%d')
    df_save['Amount'] = df_save['Amount'].apply(safe_float)
    df_save['IVA'] = df_save['IVA'].apply(safe_float)
    df_save['Is_Vice'] = df_save['Is_Vice'].fillna(False).astype(bool)
    conn.update(worksheet="Sheet1", data=df_save)

# --- PDF ENGINE ---
def generate_pdf_safe(df, goal_name, currency_symbol):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16); pdf.cell(190, 10, "FORENSIX REPORT", 0, 1, 'C')
        pdf.set_font("Arial", size=10); pdf.cell(190, 10, f"Goal: {goal_name}", 0, 1, 'C'); pdf.ln(10)
        pdf.set_font("Arial", 'B', 12); pdf.cell(100, 10, f"Total: {currency_symbol}{df['Amount'].sum():.2f}", 1, 1)
        pdf.set_text_color(200, 0, 0); pdf.cell(100, 10, f"Leakage: {currency_symbol}{df[df['Is_Vice']==True]['Amount'].sum():.2f}", 1, 1); pdf.set_text_color(0,0,0); pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 7)
        pdf.cell(30, 8, "Date", 1); pdf.cell(40, 8, "Vendor", 1); pdf.cell(80, 8, "Item", 1); pdf.cell(30, 8, "Price", 1); pdf.ln()
        
        pdf.set_font("Arial", '', 7)
        for _, row in df.iterrows():
            safe_date = str(row['Date'])[:10]
            safe_vend = str(row['Vendor'])[:18]
            safe_item = str(row['Item'])[:35]
            safe_price = f"{row['Amount']:.2f}"
            pdf.cell(30, 6, safe_date, 1); pdf.cell(40, 6, safe_vend, 1); pdf.cell(80, 6, safe_item, 1); pdf.cell(30, 6, safe_price, 1); pdf.ln()
        return pdf.output(dest='S').encode('latin-1')
    except: return None

# --- AI ENGINE ---
def get_api_key():
    if "GEMINI_API_KEY" in st.secrets: return st.secrets["GEMINI_API_KEY"]
    return ""

def vision_slice_micro(image_path):
    """Slices huge images into overlapping chunks to maintain OCR clarity."""
    img = cv2.imread(str(image_path))
    if img is None: return []
    h, w = img.shape[:2]
    if h < 2500: return [img]
    slices, start = [], 0
    while start < h:
        end = min(start + 1500, h)
        if (end - start) < 200 and len(slices) > 0: break 
        slices.append(img[start:end, :])
        if end == h: break
        start += 1200 
    return slices

def analyze_chunk(content_bytes, mime_type, client, user_vices, home_currency):
    # HYBRID PROMPT: Handles both Context & Content
    prompt = f"""
    Role: Forensic Auditor.
    Task: Extract lines from this receipt slice.
    
    JSON: [{{ "d": "YYYY-MM-DD", "v": "Vendor", "n": "Item Name", "p": 1.00, "c": "£", "mc": "Category", "sc": "Sub", "vice": false }}]
    
    RULES:
    1. **NO TOTALS:** Ignore 'Total', 'Subtotal', 'Balance', 'Change'.
    2. **VENDOR LOGIC:** If you see a logo/header, output the Vendor Name. If this is the middle of a long list and NO header is visible, output 'CONT'.
    3. **CATEGORY (mc):** [Groceries, Alcohol, Dining, Transport, Shopping, Utilities, Services].
    4. **SUB-CATEGORY (sc):** Be specific (e.g., 'Dairy', 'Meat', 'Snacks', 'Fuel', 'Clothes').
    5. **VICES:** {user_vices}.
    6. **CURRENCY:** Default {home_currency}. 
    """
    try:
        res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, types.Part.from_bytes(data=content_bytes, mime_type=mime_type)],
            config=types.GenerateContentConfig(response_mime_type='application/json')
        )
        try: data = json.loads(res.text)
        except: 
            match = re.search(r'\[.*\]', res.text, re.DOTALL); 
            data = json.loads(match.group()) if match else []
        if isinstance(data, dict): data = data.get("items", [])
        return data
    except: return []

def process_upload(uploaded_file, api_key, user_vices, home_currency):
    unique_name = f"{uuid.uuid4()}_{uploaded_file.name}"
    temp_path = DIRS['TEMP'] / unique_name
    
    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
    
    client = genai.Client(api_key=api_key)
    raw_items = []
    
    try:
        if uploaded_file.type == "application/pdf":
            with open(temp_path, "rb") as f:
                raw_items.extend(analyze_chunk(f.read(), "application/pdf", client, user_vices, home_currency))
        else:
            slices = vision_slice_micro(temp_path)
            bar = st.progress(0)
            for i, s in enumerate(slices):
                _, buf = cv2.imencode(".jpg", s)
                raw_items.extend(analyze_chunk(buf.tobytes(), "image/jpeg", client, user_vices, home_currency))
                bar.progress((i + 1) / len(slices))
            time.sleep(0.2); bar.empty()
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)
    
    extracted = []
    last_known_vendor = "Unknown" 

    for i in raw_items:
        # 1. Total Filter
        name = str(i.get("n", "")).strip()
        name_lower = name.lower()
        if any(x in name_lower for x in ["total", "subtotal", "balance", "change", "cash", "due", "visa", "auth"]): continue
        
        price = safe_float(i.get("p"))
        
        # 2. Price Swap Fix
        if price == 0.0 and re.match(r'^\d+\.?\d*$', name):
            try:
                price = float(name)
                name = "Unidentified Item"
            except: pass

        # 3. VENDOR STITCHING
        raw_vendor = str(i.get("v", "Unknown"))
        if raw_vendor.upper() not in ["UNKNOWN", "CONT", "NONE", "NULL"]:
            last_known_vendor = raw_vendor
            
        # 4. Typo Fixer
        vendor = last_known_vendor.upper()
        if "SPAN" in vendor: vendor = "SPAR"
        if "MERCAD" in vendor: vendor = "MERCADONA"
        if "LID" in vendor: vendor = "LIDL"
        if "ALE" in vendor and "HOP" in vendor: vendor = "ALE-HOP"
        
        # 5. Smart Categories
        cat = str(i.get("mc", "Unknown"))
        sub_cat = str(i.get("sc", "General"))
        is_vice = bool(i.get("vice", False))
        
        if cat in ["Unknown", "Shopping", "None"]:
            if vendor in ["MERCADONA", "LIDL", "SPAR", "TESCO", "ALDI"]: cat = "Groceries"
            elif vendor in ["ALE-HOP", "AMAZON"]: cat = "Shopping"

        vice_keywords = ["chocolate", "candy", "betting", "tobacco", "cigar", "wine", "beer", "cerveza", "vino", "vodka", "ron", "gin"]
        if any(v in name_lower for v in vice_keywords):
            is_vice = True
            if any(k in name_lower for k in ["wine", "beer", "cerveza", "vino", "vodka", "ron"]): cat = "Alcohol"
            elif "chocolate" in name_lower or "candy" in name_lower: cat = "Groceries"

        # 6. Lemon Filter
        if price > 100 and cat in ["Groceries", "Dining", "Alcohol"]: price = price / 100.0

        extracted.append({
            "Date": i.get("d"), "Vendor": vendor, "Item": name, 
            "Amount": price, "Currency": i.get("c", home_currency), 
            "Category": cat, "Sub_Category": sub_cat, 
            "Is_Vice": is_vice, "File": uploaded_file.name
        })
    return pd.DataFrame(extracted)

# --- TAX LOGIC ---
def calculate_net(gross, residency, tax_code):
    net = 0.0
    if "UK" in residency:
        allowance = 12570
        if tax_code and str(tax_code).upper().endswith('L'):
            try: allowance = int(re.sub(r'\D', '', str(tax_code))) * 10
            except: pass
        if gross > 100000: allowance = max(0, allowance - (gross - 100000) / 2)
        taxable = max(0, gross - allowance)
        tax = 0.0
        if taxable > 0: tax += min(taxable, 37700) * 0.20
        if taxable > 37700: tax += min(taxable - 37700, 125140 - 37700) * 0.40
        if taxable > 125140: tax += (taxable - 125140) * 0.45
        ni = 0.0
        if gross > 12570: ni += min(max(0, gross - 12570), 50270 - 12570) * 0.08
        if gross > 50270: ni += (gross - 50270) * 0.02
        net = gross - tax - ni
    elif "Spain" in residency:
        ss = min(gross, 56646) * 0.0635
        base = gross - ss - 2000
        irpf, prev = 0.0, 0
        bands = [(12450, 0.19), (20200, 0.24), (35200, 0.30), (60000, 0.37), (300000, 0.45)]
        for lim, rate in bands:
            if base > prev: irpf += (min(base, lim) - prev) * rate; prev = lim
            else: break
        net = gross - ss - irpf
    else: net = gross * 0.78
    return net / 12

# --- UI START ---
api_key = get_api_key()
if not api_key: st.stop()

# 1. LOAD DATA
df = load_data()

# 2. SIDEBAR
with st.sidebar:
    st.title("👤 Profile")
    st.success("☁️ Connected")
    residency = st.selectbox("Residency", ["UK (GBP)", "Spain (EUR)"], index=0)
    home_curr = "£" if "UK" in residency else "€"
    tax_code = st.text_input("Tax Code") if "UK" in residency else ""
    col1, col2 = st.columns(2)
    freq = col1.selectbox("Freq", ["Yearly", "Monthly"])
    gross = col2.number_input("Gross", 35000.0, step=1000.0)
    
    gross_annual = gross if freq == "Yearly" else gross * 12
    net_monthly = calculate_net(gross_annual, residency, tax_code)
    st.metric("Net Monthly", f"{home_curr}{net_monthly:,.2f}")
    
    st.divider()
    goal_name = st.text_input("Goal", "Daughter's Bike")
    goal_target = st.number_input("Target", 150.0)
    user_vices = st.text_area("Vices", "alcohol, candy, betting")
    
    if st.button("⚠️ Force Clear DB"):
        conn = st.connection("gsheets", type=GSheetsConnection)
        conn.update(worksheet="Sheet1", data=pd.DataFrame(columns=REQUIRED_COLS))
        st.cache_data.clear(); st.rerun()
        
    if not df.empty:
        st.write("### 📥 Export")
        st.download_button("CSV", df.to_csv(index=False), "data.csv")
        pdf = generate_pdf_safe(df, goal_name, home_curr)
        if pdf: st.download_button("PDF", pdf, "report.pdf")

# 3. DASHBOARD
st.title(f"🎯 {goal_name}")
col1, col2, col3, col4 = st.columns(4)
spent = df['Amount'].sum()
vice = df[df['Is_Vice']]['Amount'].sum()
col1.metric("Spent", f"{home_curr}{spent:,.2f}")
col2.metric("Remaining", f"{home_curr}{net_monthly - spent:,.2f}")
col3.metric("Leakage", f"{home_curr}{vice:,.2f}")
col4.progress(min(vice/goal_target, 1.0) if goal_target > 0 else 0)

# 4. UPLOAD
uploaded = st.file_uploader("Upload", accept_multiple_files=True)
if uploaded and st.button("🔍 Audit"):
    rows = []
    progress_bar = st.progress(0)
    for idx, f in enumerate(uploaded):
        data = process_upload(f, api_key, user_vices, home_curr)
        if not data.empty: 
            data['Date'] = data['Date'].apply(safe_date)
            # Safe Mode for Date
            if not data['Date'].isna().all():
                mode_date = data['Date'].mode()[0]
                data['Date'] = data['Date'].fillna(mode_date)
            
            data['Currency'] = data['Currency'].fillna(home_curr)
            # Auto-VAT
            def get_vat(r):
                rate = 20.0 if "UK" in residency else 21.0
                if "grocery" in str(r['Category']).lower(): rate = 0.0 if "UK" in residency else 4.0
                return round(r['Amount'] - (r['Amount'] / (1 + rate/100)), 2)
            data['IVA'] = data.apply(get_vat, axis=1)
            rows.append(data)
        progress_bar.progress((idx + 1) / len(uploaded))
    
    if rows:
        combined = pd.concat(rows, ignore_index=True)
        st.session_state.review_data = combined
        st.rerun()

# 5. REVIEW
if 'review_data' in st.session_state and st.session_state.review_data is not None:
    st.divider()
    st.info("Review Data")
    if not st.session_state.review_data.empty:
        st.session_state.review_data['Date'] = pd.to_datetime(st.session_state.review_data['Date']).dt.date
        
    edited = st.data_editor(st.session_state.review_data, num_rows="dynamic", use_container_width=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
            "Category": st.column_config.SelectboxColumn("Category", options=["Groceries", "Alcohol", "Dining", "Transport", "Shopping", "Utilities", "Services"])
        }
    )
    if st.button("✅ Save"):
        final = pd.concat([df, edited], ignore_index=True)
        save_data(final)
        st.session_state.review_data = None
        st.success("Saved!")
        time.sleep(1); st.rerun()
    if st.button("❌ Discard"):
        st.session_state.review_data = None; st.rerun()

# 6. ANALYTICS SUITE (The New Code)
t1, t2 = st.tabs(["📊 Analytics", "📝 Ledger"])
with t1:
    if not df.empty:
        c1, c2 = st.columns(2)
        
        # CHART 1: Stacked Bar (Category + Sub_Category Breakdown)
        with c1:
            st.subheader("Spending Breakdown")
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Category', sort='-y'),
                y='Amount',
                color='Sub_Category',
                tooltip=['Item', 'Amount', 'Sub_Category']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
            
        # CHART 2: Vice Watch (Donut Chart)
        with c2:
            st.subheader("The 'Vice' Meter")
            base = alt.Chart(df).encode(theta=alt.Theta("Amount", stack=True))
            pie = base.mark_arc(outerRadius=120).encode(
                color=alt.Color("Is_Vice"),
                order=alt.Order("Amount", sort="descending"),
                tooltip=["Is_Vice", "Amount"]
            )
            text = base.mark_text(radius=140).encode(
                text=alt.Text("Amount", format=".1f"),
                order=alt.Order("Amount", sort="descending"),
                color=alt.value("black") 
            )
            st.altair_chart(pie + text, use_container_width=True)
            
        # CHART 3: Daily Trend Line
        st.subheader("Spending Timeline")
        line = alt.Chart(df).mark_line(point=True).encode(
            x='Date',
            y='Amount',
            color='Category',
            tooltip=['Date', 'Vendor', 'Amount']
        ).interactive()
        st.altair_chart(line, use_container_width=True)

with t2:
    if not df.empty: 
        st.dataframe(df, use_container_width=True)
        
