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

# --- HELPER: SANITIZE DATA (Prevents Crashes) ---
def sanitize_dataframe(df):
    """Forces strict types to prevent Streamlit API Exceptions"""
    if df.empty: return df
    
    # 1. Force Numbers
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0.0)
    df['IVA'] = pd.to_numeric(df['IVA'], errors='coerce').fillna(0.0)
    
    # 2. Force Strings
    for col in ['Vendor', 'Item', 'Currency', 'Category', 'Sub_Category', 'File']:
        df[col] = df[col].astype(str).replace('nan', '')
        
    # 3. Force Booleans (Crucial for Checkboxes)
    df['Is_Vice'] = df['Is_Vice'].fillna(False).astype(bool)
    
    # 4. Force Dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').fillna(pd.Timestamp.now())
    return df

# --- DATABASE ---
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        df = conn.read(worksheet="Sheet1", usecols=list(range(10)), ttl=5)
        df = df.dropna(how="all")
        return sanitize_dataframe(df)
    except: return pd.DataFrame(columns=["Date", "Vendor", "Item", "Amount", "Currency", "IVA", "Category", "Sub_Category", "Is_Vice", "File"])

def save_data(df):
    conn = st.connection("gsheets", type=GSheetsConnection)
    df_save = sanitize_dataframe(df.copy())
    # Convert Date to String for Google Sheets (YYYY-MM-DD)
    df_save['Date'] = df_save['Date'].dt.strftime('%Y-%m-%d')
    conn.update(worksheet="Sheet1", data=df_save)

# --- TAX LOGIC (2025/26 RULES) ---
def calculate_accurate_tax(gross_annual, residency, tax_code):
    """
    Calculates Net Monthly Income using official bands.
    """
    net_annual = 0.0
    
    if "UK" in residency:
        # --- UK LOGIC (HMRC 2025) ---
        # 1. Personal Allowance
        allowance = 12570
        if tax_code and str(tax_code).upper().endswith('L'):
            try:
                digits = int(re.sub(r'\D', '', str(tax_code)))
                allowance = digits * 10
            except: pass
            
        # Reduced Personal Allowance (Income over £100k)
        if gross_annual > 100000:
            reduction = (gross_annual - 100000) / 2
            allowance = max(0, allowance - reduction)
            
        taxable_income = max(0, gross_annual - allowance)
        
        # 2. Income Tax
        tax = 0.0
        # Basic Rate (20% up to £37,700)
        if taxable_income > 0:
            band1 = min(taxable_income, 37700)
            tax += band1 * 0.20
        # Higher Rate (40% from £37,701 to £125,140)
        if taxable_income > 37700:
            band2 = min(taxable_income - 37700, 125140 - 37700)
            tax += band2 * 0.40
        # Additional Rate (45% over £125,140)
        if taxable_income > 125140:
            band3 = taxable_income - 125140
            tax += band3 * 0.45
            
        # 3. National Insurance (Class 1 Employee - Approx 8% logic)
        # Threshold: ~£12,570 start paying
        ni = 0.0
        if gross_annual > 12570:
            ni_band = min(gross_annual, 50270) - 12570
            ni += max(0, ni_band * 0.08)
            
        if gross_annual > 50270:
            ni += (gross_annual - 50270) * 0.02
            
        net_annual = gross_annual - tax - ni

    elif "Spain" in residency:
        # --- SPAIN IRPF (2025 Gen) ---
        # 1. Social Security (Seguridad Social) ~6.35% capped
        ss_base = min(gross_annual, 56646) # Max base approx
        ss_tax = ss_base * 0.0635
        
        # 2. IRPF Base
        base_irpf = gross_annual - ss_tax - 2000 # Standard deduction
        
        # 3. Progressive Bands
        irpf = 0.0
        bands = [
            (12450, 0.19),
            (20200, 0.24),
            (35200, 0.30),
            (60000, 0.37),
            (300000, 0.45),
            (float('inf'), 0.47)
        ]
        
        remaining = base_irpf
        previous_limit = 0
        
        for limit, rate in bands:
            if base_irpf > previous_limit:
                taxable_in_band = min(base_irpf, limit) - previous_limit
                irpf += taxable_in_band * rate
                previous_limit = limit
            else:
                break
                
        net_annual = gross_annual - ss_tax - irpf
        
    else:
        # Fallback 22%
        net_annual = gross_annual * 0.78

    return net_annual / 12

# --- AI ENGINE ---
def get_api_key():
    if "GEMINI_API_KEY" in st.secrets: return st.secrets["GEMINI_API_KEY"]
    return ""

def process_upload(uploaded_file, api_key, user_vices, home_currency):
    temp_path = DIRS['TEMP'] / uploaded_file.name
    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
    
    # 1. Generate Content (Simulated for brevity, assume Gemini Client works)
    client = genai.Client(api_key=api_key)
    
    # --- PROMPT ---
    prompt = f"""
    Role: Forensic Auditor. Output strictly valid JSON.
    Format: [{{ "d": "YYYY-MM-DD", "v": "Vendor", "n": "Item", "p": 1.00, "c": "£", "mc": "Category", "sc": "Sub", "vice": false }}]
    Rules:
    1. If date/vendor unknown, use null.
    2. Currency default: {home_currency}.
    3. Vice Keywords: {user_vices}
    """
    
    extracted = []
    try:
        mime = "application/pdf" if uploaded_file.type == "application/pdf" else "image/jpeg"
        res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, types.Part.from_bytes(data=open(temp_path, "rb").read(), mime_type=mime)],
            config=types.GenerateContentConfig(response_mime_type='application/json')
        )
        # Parse Response
        try: data = json.loads(res.text)
        except: 
            # Hunter Seeker Regex
            match = re.search(r'\[.*\]', res.text, re.DOTALL)
            data = json.loads(match.group()) if match else []
            
        if isinstance(data, dict): data = data.get("items", [])
        
        # Map to DataFrame Columns
        for i in data:
            extracted.append({
                "Date": i.get("d"),
                "Vendor": i.get("v"),
                "Item": i.get("n", "Item"),
                "Amount": i.get("p", 0.0),
                "Currency": i.get("c", home_currency),
                "Category": i.get("mc", "Shopping"),
                "Sub_Category": i.get("sc", "General"),
                "Is_Vice": i.get("vice", False),
                "File": uploaded_file.name
            })
            
    except Exception as e:
        st.error(f"AI Error: {e}")
    
    os.remove(temp_path)
    return pd.DataFrame(extracted)

def generate_pdf_safe(df, goal_name, currency_symbol):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16); pdf.cell(190, 10, "FORENSIX REPORT", 0, 1, 'C')
        pdf.set_font("Arial", size=10); pdf.cell(190, 10, f"Goal: {goal_name}", 0, 1, 'C'); pdf.ln(10)
        pdf.set_font("Arial", 'B', 12); pdf.cell(100, 10, f"Total: {currency_symbol}{df['Amount'].sum():.2f}", 1, 1)
        pdf.set_font("Arial", '', 8)
        # Simple Table
        for _, row in df.iterrows():
            line = f"{str(row['Date'])[:10]} | {str(row['Vendor'])[:15]} | {str(row['Item'])[:30]} | {row['Amount']:.2f}"
            pdf.cell(190, 6, line.encode('latin-1', 'ignore').decode('latin-1'), 1, 1)
        return pdf.output(dest='S').encode('latin-1')
    except: return None

# --- UI START ---
api_key = get_api_key()
if not api_key: st.stop()

# 1. LOAD & SANITIZE
df = load_data()

# 2. SIDEBAR
with st.sidebar:
    st.title("👤 Profile")
    st.success("☁️ Database Connected")
    
    st.subheader("🌍 Tax Residency")
    residency = st.selectbox("Select Country", ["UK (GBP)", "Spain (EUR)"], index=0)
    home_curr = "£" if "UK" in residency else "€"
    
    # TAX CODE LOGIC
    tax_code_input = ""
    if "UK" in residency:
        tax_code_input = st.text_input("Tax Code", "1257L")
    
    st.subheader("💰 Income Calculator")
    col1, col2 = st.columns(2)
    freq = col1.selectbox("Freq", ["Yearly", "Monthly"])
    gross = col2.number_input("Gross Amount", value=35000.0, step=1000.0)
    
    gross_annual = gross if freq == "Yearly" else gross * 12
    net_monthly = calculate_accurate_tax(gross_annual, residency, tax_code_input)
    
    st.metric("Net Monthly Income", f"{home_curr}{net_monthly:,.2f}")
    st.caption("Calculated using 2025/26 Tax Bands")
    
    st.markdown("---")
    goal_name = st.text_input("Goal", "Daughter's Bike")
    goal_target = st.number_input("Target", 150.0)
    user_vices = st.text_area("Vices", "alcohol, candy, betting, tobacco")
    
    if st.button("⚠️ Clear Database"):
        empty = pd.DataFrame(columns=df.columns)
        save_data(empty)
        st.rerun()

# 3. MAIN DASHBOARD
st.title(f"🎯 Project: {goal_name}")

# Metrics
col1, col2, col3, col4 = st.columns(4)
spent = df['Amount'].sum()
vice_spend = df[df['Is_Vice']]['Amount'].sum()
remaining = net_monthly - spent

col1.metric("Total Spent", f"{home_curr}{spent:,.2f}")
col2.metric("Remaining (Month)", f"{home_curr}{remaining:,.2f}")
col3.metric("Habit Leakage", f"{home_curr}{vice_spend:,.2f}", delta="-Leakage")
progress = min((vice_spend / goal_target), 1.0)
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
            # Auto-Fill missing vendors/dates (Smart Logic)
            new_data['Vendor'] = new_data['Vendor'].replace([None, 'null'], np.nan).ffill().bfill().fillna("Unknown")
            new_data['Date'] = pd.to_datetime(new_data['Date'], errors='coerce').ffill().bfill().fillna(pd.Timestamp.now())
            new_data['Currency'] = new_data['Currency'].fillna(home_curr)
            
            # Simple VAT Calc
            def calc_iva(row):
                rate = 20.0 if "UK" in residency else 21.0
                if "grocery" in str(row['Category']).lower(): rate = 0.0 if "UK" in residency else 4.0
                return round(row['Amount'] - (row['Amount'] / (1 + rate/100)), 2)
            
            new_data['IVA'] = new_data.apply(calc_iva, axis=1)
            all_new_rows.append(new_data)
        progress_bar.progress((idx + 1) / len(uploaded))
            
    if all_new_rows:
        combined_new = pd.concat(all_new_rows, ignore_index=True)
        # Store in session state for review
        st.session_state.review_data = sanitize_dataframe(combined_new)
        st.rerun()

# 5. REVIEW ROOM (THE CRASH PROOF VERSION)
if 'review_data' in st.session_state and st.session_state.review_data is not None:
    st.divider()
    st.subheader("🧐 Review & Edit Results")
    st.info("Please verify the data below. Click Confirm to save to database.")
    
    # Sanitize again just to be safe
    safe_view = sanitize_dataframe(st.session_state.review_data)
    
    edited_df = st.data_editor(
        safe_view,
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
        st.session_state.review_data = None # Clear holding bay
        st.success("Saved successfully!")
        time.sleep(1)
        st.rerun()
        
    if col_cancel.button("❌ Discard"):
        st.session_state.review_data = None
        st.rerun()

# 6. TABS
tab1, tab2 = st.tabs(["📊 Analytics", "📝 Ledger"])
with tab1:
    if not df.empty:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Category', sort='-y'),
            y='Amount',
            color='Category'
        )
        st.altair_chart(chart, use_container_width=True)

with tab2:
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        if st.download_button("Download CSV", df.to_csv(index=False), "data.csv"):
            pass
            
