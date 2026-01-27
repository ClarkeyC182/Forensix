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
    """
    Parses dates with a preference for DAY-FIRST (UK/EU Standard).
    Fixes the 'April vs October' confusion.
    """
    try:
        s = str(val).strip().lower()
        if s in ['none', 'null', 'nan', '', 'yyyy-mm-dd', 'unknown']: return pd.Timestamp.now()
        
        # dayfirst=True forces 04/10 to be Oct 4th, not Apr 10th
        dt = pd.to_datetime(val, errors='coerce', dayfirst=True)
        
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

def resize_image_if_needed(image_path):
    try:
        with Image.open(image_path) as img:
            if img.width > 5000:
                ratio = 5000 / img.width
                new_height = int(img.height * ratio)
                img = img.resize((5000, new_height), Image.Resampling.LANCZOS)
                img.save(image_path, optimize=True, quality=85)
    except: pass

def analyze_full_image(content_bytes, mime_type, client, user_vices, home_currency):
    prompt = f"""
    Role: Senior OCR Specialist.
    Context: Collage of receipts. Extract EVERY line item.
    
    JSON: [{{ "d": "YYYY-MM-DD", "v": "Vendor", "n": "Item Name", "p": 1.00, "c": "£", "mc": "Category", "sc": "Sub", "vice": false }}]
    
    RULES:
    1. **NO TOTALS:** Ignore 'Total', 'Subtotal', 'Balance', 'Change'.
    2. **ITEM NAME:** Product name only. No prices in name.
    3. **CATEGORY (mc):** [Groceries, Alcohol, Dining, Transport, Shopping, Utilities, Services].
    4. **VICE CHECK:** If item is alcohol, candy, tobacco, gambling -> vice = true.
    5. **VENDOR:** Header of the specific receipt.
    6. **DATE:** Date of the specific receipt.
    7. **CURRENCY:** Default {home_currency}. 
    8. **USER VICES:** {user_vices}.
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
    
    if uploaded_file.type != "application/pdf":
        resize_image_if_needed(temp_path)
        
    client = genai.Client(api_key=api_key)
    raw_items = []
    
    try:
        mime = "application/pdf" if uploaded_file.type == "application/pdf" else "image/jpeg"
        with open(temp_path, "rb") as f:
            raw_items.extend(analyze_full_image(f.read(), mime, client, user_vices, home_currency))
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
        
        # 2. Price Swap
        if price == 0.0 and re.match(r'^\d+\.?\d*$', name):
            try:
                price = float(name)
                name = "Unidentified Item"
            except: pass

        # 3. Stitching
        raw_vendor = str(i.get("v", "Unknown"))
        if raw_vendor.upper() not in ["UNKNOWN", "CONT", "NONE", "NULL"]:
            last_known_vendor = raw_vendor
            
        vendor = last_known_vendor.upper()
        if "SPAN" in vendor: vendor = "SPAR"
        if "MERCAD" in vendor: vendor = "MERCADONA"
        if "LID" in vendor: vendor = "LIDL"
        if "ALE" in vendor and "HOP" in vendor: vendor = "ALE-HOP"
        
        # 4. Smart Categories
        cat = str(i.get("mc", "Unknown"))
        is_vice = bool(i.get("vice", False))
        
        if cat in ["Unknown", "Shopping", "None"]:
            if vendor in ["MERCADONA", "LIDL", "SPAR", "TESCO", "ALDI"]: cat = "Groceries"
            elif vendor in ["ALE-HOP", "AMAZON"]: cat = "Shopping"

        vice_keywords = ["chocolate", "candy", "betting", "tobacco", "cigar", "wine", "beer", "cerveza", "vino", "vodka", "ron", "gin"]
        if any(v in name_lower for v in vice_keywords):
            is_vice = True
            if any(k in name_lower for k in ["wine", "beer", "cerveza", "vino", "vodka", "ron"]): cat = "Alcohol"
            elif "chocolate" in name_lower or "candy" in name_lower: cat = "Groceries"

        # 5. Lemon Filter
        if price > 100 and cat in ["Groceries", "Dining", "Alcohol"]: price = price / 100.0

        extracted.append({
            "Date": i.get("d"), "Vendor": vendor, "Item": name, 
            "Amount": price, "Currency": i.get("c", home_currency), 
            "Category": cat, "Sub_Category": i.get("sc", "General"), 
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
    
    # FILTERING LOGIC
    st.divider()
    st.subheader("🔍 Filters")
    
    # Date Filter
    min_date = df['Date'].min().date() if not df.empty else datetime.now().date()
    max_date = df['Date'].max().date() if not df.empty else datetime.now().date()
    
    # If single date, add buffer to sliders
    if min_date == max_date:
        min_date = min_date - pd.Timedelta(days=1)
        
    date_range = st.slider("Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    
    # Category Filter
    all_cats = list(df['Category'].unique()) if not df.empty else []
    selected_cats = st.multiselect("Category", all_cats, default=all_cats)
    
    if st.button("⚠️ Force Clear DB"):
        conn = st.connection("gsheets", type=GSheetsConnection)
        conn.update(worksheet="Sheet1", data=pd.DataFrame(columns=REQUIRED_COLS))
        st.cache_data.clear(); st.rerun()
        
    if not df.empty:
        st.write("### 📥 Export")
        st.download_button("CSV", df.to_csv(index=False), "data.csv")
        pdf = generate_pdf_safe(df, goal_name, home_curr)
        if pdf: st.download_button("PDF", pdf, "report.pdf")

# 3. FILTER ENGINE
df_filtered = df.copy()
if not df.empty:
    mask = (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
    mask &= (df['Category'].isin(selected_cats))
    df_filtered = df.loc[mask]

# 4. DASHBOARD
st.title(f"🎯 {goal_name}")
col1, col2, col3, col4 = st.columns(4)
spent = df_filtered['Amount'].sum()
vice = df_filtered[df_filtered['Is_Vice']]['Amount'].sum()
col1.metric("Spent", f"{home_curr}{spent:,.2f}")
col2.metric("Remaining", f"{home_curr}{net_monthly - spent:,.2f}")
col3.metric("Leakage", f"{home_curr}{vice:,.2f}")
col4.progress(min(vice/goal_target, 1.0) if goal_target > 0 else 0)

# 5. UPLOAD
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

# 6. REVIEW
if 'review_data' in st.session_state and st.session_state.review_data is not None:
    st.divider()
    st.info("Review Data")
    if not st.session_state.review_data.empty:
        st.session_state.review_data['Date'] = pd.to_datetime(st.session_state.review_data['Date']).dt.date
        
    edited = st.data_editor(st.session_state.review_data, num_rows="dynamic", use_container_width=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
            "Is_Vice": st.column_config.CheckboxColumn("Vice?", default=False),
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

# 7. ANALYTICS SUITE (Interactive)
t1, t2 = st.tabs(["📊 Executive View", "📝 Data Ledger"])
with t1:
    if not df_filtered.empty:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("💸 Where is the money going?")
            chart = alt.Chart(df_filtered).mark_bar().encode(
                x=alt.X('Category', sort='-y'),
                y='Amount',
                color='Category',
                tooltip=['Category', 'Amount']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
            
        with c2:
            st.subheader("😈 The Vice Meter")
            base = alt.Chart(df_filtered).encode(theta=alt.Theta("Amount", stack=True))
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
            
        c3, c4 = st.columns(2)
        
        with c3:
            st.subheader("🏆 Top 5 Vendors")
            top_vendors = df_filtered.groupby("Vendor")['Amount'].sum().reset_index().sort_values("Amount", ascending=False).head(5)
            v_chart = alt.Chart(top_vendors).mark_bar().encode(
                x='Amount',
                y=alt.Y('Vendor', sort='-x'),
                color='Vendor'
            )
            st.altair_chart(v_chart, use_container_width=True)
            
        with c4:
            st.subheader("💎 Top 5 Expensive Items")
            top_items = df_filtered.sort_values("Amount", ascending=False).head(5)
            i_chart = alt.Chart(top_items).mark_bar().encode(
                x='Amount',
                y=alt.Y('Item', sort='-x'),
                color='Category'
            )
            st.altair_chart(i_chart, use_container_width=True)

with t2:
    if not df_filtered.empty: 
        st.dataframe(df_filtered, use_container_width=True)
        
