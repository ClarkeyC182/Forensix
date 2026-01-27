import streamlit as st
import pandas as pd
import time
import cv2
import json
import os
import re
import altair as alt
from pathlib import Path
from google import genai
from google.genai import types
from fpdf import FPDF
from streamlit_gsheets import GSheetsConnection
import numpy as np
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Forensix Personal Auditor", page_icon="🕵️‍♂️", layout="wide")
BASE_DIR = Path(__file__).parent.absolute()
DIRS = { "TEMP": BASE_DIR / "temp_uploads" }
for d in DIRS.values(): d.mkdir(exist_ok=True)

# CONSTANTS
REQUIRED_COLS = ["Date", "Vendor", "Item", "Amount", "Currency", "IVA", "Category", "Sub_Category", "Is_Vice", "File"]

# --- HELPER: SAFE DATE PARSER ---
def safe_parse_date(date_val):
    """Prevents app crash when AI returns garbage text instead of a date."""
    try:
        s_val = str(date_val).strip().lower()
        if s_val in ['none', 'null', 'nan', '', 'yyyy-mm-dd', 'unknown', 'date']:
            return pd.Timestamp.now()
        dt = pd.to_datetime(s_val, errors='coerce')
        if pd.isna(dt): return pd.Timestamp.now()
        return dt
    except: return pd.Timestamp.now()

# --- DATABASE ENGINE (TEMPLATE MERGE STRATEGY) ---
def load_data():
    # 1. Create the Perfect Master Template
    df_master = pd.DataFrame(columns=REQUIRED_COLS)
    
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        # 2. Read Sheet (No Cache)
        df_sheet = conn.read(worksheet="Sheet1", ttl=0)
        
        # 3. Merge into Master Template (Forces columns to exist)
        if not df_sheet.empty:
            df_combined = pd.concat([df_master, df_sheet], ignore_index=True)
        else:
            df_combined = df_master

        # 4. Strict Selection
        df_final = df_combined.reindex(columns=REQUIRED_COLS)
        
        # 5. Safe Cleaning
        df_final['Amount'] = pd.to_numeric(df_final['Amount'], errors='coerce').fillna(0.0)
        df_final['IVA'] = pd.to_numeric(df_final['IVA'], errors='coerce').fillna(0.0)
        df_final['Is_Vice'] = df_final['Is_Vice'].fillna(False).astype(bool)
        df_final['Date'] = df_final['Date'].apply(safe_parse_date)
        
        for c in ['Vendor', 'Item', 'Currency', 'Category', 'Sub_Category', 'File']:
            df_final[c] = df_final[c].astype(str).replace('nan', '').replace('None', '')

        return df_final
        
    except Exception:
        return df_master

def save_data(df):
    conn = st.connection("gsheets", type=GSheetsConnection)
    df_save = df.copy()
    # Format for storage
    df_save['Date'] = df_save['Date'].apply(safe_parse_date).dt.strftime('%Y-%m-%d')
    df_save['Amount'] = pd.to_numeric(df_save['Amount'], errors='coerce').fillna(0.0)
    df_save['IVA'] = pd.to_numeric(df_save['IVA'], errors='coerce').fillna(0.0)
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
        pdf.set_font("Arial", 'B', 8)
        pdf.cell(30, 8, "Date", 1); pdf.cell(40, 8, "Vendor", 1); pdf.cell(80, 8, "Item", 1); pdf.cell(30, 8, "Price", 1); pdf.ln()
        pdf.set_font("Arial", '', 8)
        for _, row in df.iterrows():
            d = str(row['Date'])[:10]
            v = str(row['Vendor'])[:15]
            i = str(row['Item'])[:35]
            p = f"{row['Amount']:.2f}"
            pdf.cell(30, 6, d, 1); pdf.cell(40, 6, v, 1); pdf.cell(80, 6, i, 1); pdf.cell(30, 6, p, 1); pdf.ln()
        return pdf.output(dest='S').encode('latin-1')
    except: return None

# --- TAX LOGIC ---
def calculate_accurate_tax(gross_annual, residency, tax_code):
    net_annual = 0.0
    if "UK" in residency:
        allowance = 12570
        if tax_code and str(tax_code).upper().endswith('L'):
            try: digits = int(re.sub(r'\D', '', str(tax_code))); allowance = digits * 10
            except: pass
        if gross_annual > 100000: allowance = max(0, allowance - (gross_annual - 100000) / 2)
        taxable = max(0, gross_annual - allowance)
        tax = 0.0
        if taxable > 0: tax += min(taxable, 37700) * 0.20
        if taxable > 37700: tax += min(taxable - 37700, 125140 - 37700) * 0.40
        if taxable > 125140: tax += (taxable - 125140) * 0.45
        ni = 0.0
        if gross_annual > 12570: ni += min(max(0, gross_annual - 12570), 50270 - 12570) * 0.08
        if gross_annual > 50270: ni += (gross_annual - 50270) * 0.02
        net_annual = gross_annual - tax - ni
    elif "Spain" in residency:
        ss_tax = min(gross_annual, 56646) * 0.0635
        base = gross_annual - ss_tax - 2000
        irpf, prev = 0.0, 0
        bands = [(12450, 0.19), (20200, 0.24), (35200, 0.30), (60000, 0.37), (300000, 0.45)]
        for limit, rate in bands:
            if base > prev: irpf += (min(base, limit) - prev) * rate; prev = limit
            else: break
        net_annual = gross_annual - ss_tax - irpf
    else: net_annual = gross_annual * 0.78
    return net_annual / 12

# --- AI ENGINE ---
def get_api_key():
    if "GEMINI_API_KEY" in st.secrets: return st.secrets["GEMINI_API_KEY"]
    return ""

def vision_slice_micro(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return []
    h, w = img.shape[:2]
    if h < 1000: return [img]
    slices, start = [], 0
    while start < h:
        end = min(start + 1000, h)
        if (end - start) < 100 and len(slices) > 0: break 
        slices.append(img[start:end, :])
        if end == h: break
        start += 800
    return slices

def analyze_chunk(content_bytes, mime_type, client, user_vices, home_currency):
    prompt = f"""
    Role: Forensic Auditor. Extract items.
    JSON Format: [{{ "d": "YYYY-MM-DD", "v": "Vendor", "n": "Item", "p": 1.00, "c": "£", "mc": "Category", "sc": "Sub", "vice": false }}]
    Rules: 
    1. IGNORE 'Total', 'Subtotal', 'Cash', 'Card'.
    2. Date: Use main date or null. NO 'YYYY-MM-DD'.
    3. Currency default {home_currency}. 
    4. Vice keywords: {user_vices}.
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
    temp_path = DIRS['TEMP'] / uploaded_file.name
    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
    client = genai.Client(api_key=api_key)
    raw_items = []
    
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

    os.remove(temp_path)
    
    extracted = []
    for i in raw_items:
        # 1. Filter out Totals
        item_name = str(i.get("n", "")).lower()
        if any(x in item_name for x in ["total", "subtotal", "balance", "change", "cash", "amount due"]):
            continue 

        # 2. Safe Price Conversion (FIXES TYPE ERROR)
        try: 
            raw_price = i.get("p", 0.0)
            if raw_price is None: raw_price = 0.0
            price = float(raw_price)
        except: 
            price = 0.0

        # 3. Lemon Filter (Decimal correction)
        cat = str(i.get("mc", "Shopping"))
        if price > 100 and cat in ["Groceries", "Dining", "Alcohol"]:
             price = price / 100.0

        extracted.append({
            "Date": i.get("d"), 
            "Vendor": i.get("v"), 
            "Item": i.get("n"), 
            "Amount": price,
            "Currency": i.get("c", home_currency), 
            "Category": cat,
            "Sub_Category": i.get("sc", "General"), 
            "Is_Vice": i.get("vice", False), 
            "File": uploaded_file.name
        })
    return pd.DataFrame(extracted)

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
    
    tax_code_input = ""
    if "UK" in residency:
        tax_code_input = st.text_input("Tax Code (Optional)", "1257L")
    
    col1, col2 = st.columns(2)
    income_freq = col1.selectbox("Freq", ["Yearly", "Monthly", "Hourly"])
    gross_income = col2.number_input("Gross", 35000.0, step=1000.0)
    
    gross_annual = gross_income
    if income_freq == "Monthly": gross_annual = gross_income * 12
    elif income_freq == "Hourly": gross_annual = gross_income * 160 * 12
    
    net_monthly = calculate_accurate_tax(gross_annual, residency, tax_code_input)
    st.metric("Net Monthly Income", f"{home_curr}{net_monthly:,.2f}")
    
    st.divider()
    goal_name = st.text_input("Goal Name", "Daughter's Bike")
    goal_target = st.number_input("Target Amount", 150.0)
    user_vices = st.text_area("Vices", "alcohol, candy, betting, tobacco")
    
    if st.button("⚠️ Force Clear DB"):
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            empty = pd.DataFrame(columns=REQUIRED_COLS)
            conn.update(worksheet="Sheet1", data=empty)
            st.cache_data.clear()
            st.rerun()
        except: st.error("Clear failed. Check permissions.")
        
    if not df.empty:
        st.write("### 📥 Export")
        st.download_button("📊 CSV", df.to_csv(index=False), "data.csv", "text/csv")
        pdf_data = generate_pdf_safe(df, goal_name, home_curr)
        if pdf_data: st.download_button("📄 PDF", pdf_data, "report.pdf", "application/pdf")

# 3. DASHBOARD
st.title(f"🎯 Project: {goal_name}")

col1, col2, col3, col4 = st.columns(4)
spent = df['Amount'].sum()
vice_spend = df[df['Is_Vice']]['Amount'].sum()
remaining = net_monthly - spent

col1.metric("Spent", f"{home_curr}{spent:,.2f}")
col2.metric("Remaining", f"{home_curr}{remaining:,.2f}")
col3.metric("Leakage", f"{home_curr}{vice_spend:,.2f}", delta="-Leakage")
progress = min((vice_spend / goal_target), 1.0) if goal_target > 0 else 0
col4.progress(progress)
col4.caption(f"Goal Progress: {progress*100:.1f}%")

# 4. UPLOAD SECTION
uploaded = st.file_uploader("Upload Receipts", accept_multiple_files=True)
if uploaded and st.button("🔍 Run Forensic Audit"):
    all_new_rows = []
    progress_bar = st.progress(0)
    
    for idx, file in enumerate(uploaded):
        new_data = process_upload(file, api_key, user_vices, home_curr)
        if not new_data.empty:
            # Auto-Fill
            new_data['Vendor'] = new_data['Vendor'].replace([None, 'null'], np.nan).ffill().bfill().fillna("Unknown")
            
            # Date Filling
            real_dates = new_data['Date'].replace([None, 'null'], np.nan).dropna()
            if not real_dates.empty:
                common_date = real_dates.mode()[0] 
                new_data['Date'] = common_date
            else:
                new_data['Date'] = pd.Timestamp.now().strftime('%Y-%m-%d')

            new_data['Currency'] = new_data['Currency'].fillna(home_curr)
            
            # Auto-VAT
            def calc_iva(row):
                rate = 20.0 if "UK" in residency else 21.0
                if "grocery" in str(row['Category']).lower(): rate = 0.0 if "UK" in residency else 4.0
                return round(row['Amount'] - (row['Amount'] / (1 + rate/100)), 2)
                
            new_data['IVA'] = new_data.apply(calc_iva, axis=1)
            all_new_rows.append(new_data)
        progress_bar.progress((idx + 1) / len(uploaded))
            
    if all_new_rows:
        combined_new = pd.concat(all_new_rows, ignore_index=True)
        combined_new['Date'] = pd.to_datetime(combined_new['Date']).dt.date
        st.session_state.review_data = combined_new
        st.rerun()

# 5. REVIEW ROOM
if 'review_data' in st.session_state and st.session_state.review_data is not None:
    st.divider()
    st.subheader("🧐 Review & Edit Results")
    st.info("Edit data below. Click Confirm to save.")
    
    if not st.session_state.review_data.empty:
        st.session_state.review_data['Date'] = pd.to_datetime(st.session_state.review_data['Date']).dt.date

    edited_df = st.data_editor(
        st.session_state.review_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
            "Is_Vice": st.column_config.CheckboxColumn("Vice?", default=False),
            "Category": st.column_config.SelectboxColumn("Category", options=["Groceries", "Dining", "Alcohol", "Transport", "Shopping", "Services"])
        }
    )
    
    col_save, col_cancel = st.columns([1, 4])
    if col_save.button("✅ Confirm & Save"):
        final_df = pd.concat([df, edited_df], ignore_index=True)
        save_data(final_df)
        st.session_state.review_data = None
        st.success("Saved successfully!")
        time.sleep(1)
        st.rerun()
        
    if col_cancel.button("❌ Discard"):
        st.session_state.review_data = None
        st.rerun()

# 6. TABS (FIXED TYPO)
tab1, tab2 = st.tabs(["📊 Analytics", "📝 Ledger"])
with tab1:
    if not df.empty:
        st.altair_chart(alt.Chart(df).mark_bar().encode(x='Category', y='Amount', color='Category'), use_container_width=True)
with tab2:
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
