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

# --- DATABASE REPAIR KIT ---
def repair_sheet():
    conn = st.connection("gsheets", type=GSheetsConnection)
    empty_structure = pd.DataFrame(columns=REQUIRED_COLS)
    conn.update(worksheet="Sheet1", data=empty_structure)
    st.cache_data.clear()

def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        df = conn.read(worksheet="Sheet1", ttl=0)
        if df.empty or len(df.columns) < 2: return pd.DataFrame(columns=REQUIRED_COLS)
        for col in REQUIRED_COLS:
            if col not in df.columns: df[col] = None
        return df[REQUIRED_COLS]
    except: return pd.DataFrame(columns=REQUIRED_COLS)

def save_data(df):
    conn = st.connection("gsheets", type=GSheetsConnection)
    df_save = df.copy()
    # Force Types
    df_save['Date'] = pd.to_datetime(df_save['Date'], errors='coerce').fillna(pd.Timestamp.now()).dt.strftime('%Y-%m-%d')
    df_save['Amount'] = pd.to_numeric(df_save['Amount'], errors='coerce').fillna(0.0)
    df_save['IVA'] = pd.to_numeric(df_save['IVA'], errors='coerce').fillna(0.0)
    df_save['Is_Vice'] = df_save['Is_Vice'].fillna(False).astype(bool)
    for c in ['Vendor', 'Item', 'Currency', 'Category']: df_save[c] = df_save[c].astype(str).replace('nan', '')
    conn.update(worksheet="Sheet1", data=df_save)

# --- PDF GENERATOR ---
def generate_pdf_safe(df, goal_name, currency_symbol):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16); pdf.cell(190, 10, "FORENSIX REPORT", 0, 1, 'C')
        pdf.set_font("Arial", size=10); pdf.cell(190, 10, f"Goal: {goal_name}", 0, 1, 'C'); pdf.ln(10)
        pdf.set_font("Arial", 'B', 12); pdf.cell(100, 10, f"Total: {currency_symbol}{df['Amount'].sum():.2f}", 1, 1)
        pdf.set_font("Arial", '', 8)
        pdf.ln(5)
        pdf.cell(30, 8, "Date", 1); pdf.cell(40, 8, "Vendor", 1); pdf.cell(80, 8, "Item", 1); pdf.cell(30, 8, "Price", 1); pdf.ln()
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
    # IF image is small, return as 1 slice
    if h < 1000: return [img]
    
    slices, start = [], 0
    # RESTORED: Slicing Loop for Long Receipts
    while start < h:
        end = min(start + 1000, h)
        if (end - start) < 100 and len(slices) > 0: break 
        slices.append(img[start:end, :])
        if end == h: break
        start += 800 # Overlap of 200px
    return slices

def analyze_chunk(content_bytes, mime_type, client, user_vices, home_currency):
    # Common function to call AI for any chunk (PDF or Image Slice)
    prompt = f"""
    Role: Forensic Auditor. Extract items.
    JSON Format: [{{ "d": "YYYY-MM-DD", "v": "Vendor", "n": "Item", "p": 1.00, "c": "£", "mc": "Category", "sc": "Sub", "vice": false }}]
    Rules: Currency default {home_currency}. Vice keywords: {user_vices}.
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
        # PDF: Send whole file
        with open(temp_path, "rb") as f:
            raw_items.extend(analyze_chunk(f.read(), "application/pdf", client, user_vices, home_currency))
    else:
        # IMAGE: Slice it!
        slices = vision_slice_micro(temp_path)
        bar = st.progress(0)
        for i, s in enumerate(slices):
            _, buf = cv2.imencode(".jpg", s)
            raw_items.extend(analyze_chunk(buf.tobytes(), "image/jpeg", client, user_vices, home_currency))
            bar.progress((i + 1) / len(slices))
        time.sleep(0.2); bar.empty()

    os.remove(temp_path)
    
    # Format into DataFrame
    extracted = []
    for i in raw_items:
        extracted.append({
            "Date": i.get("d"), "Vendor": i.get("v"), "Item": i.get("n"), "Amount": i.get("p", 0.0),
            "Currency": i.get("c", home_currency), "Category": i.get("mc", "Shopping"),
            "Sub_Category": i.get("sc", "General"), "Is_Vice": i.get("vice", False), "File": uploaded_file.name
        })
        
    return pd.DataFrame(extracted)

# --- UI START ---
api_key = get_api_key()
if not api_key: st.stop()

# 1. LOAD & CHECK
df_raw = load_data()
missing_cols = [c for c in REQUIRED_COLS if c not in df_raw.columns]

# 2. SCHEMA VALIDATION UI
if missing_cols or df_raw.empty:
    st.warning("⚠️ Database Schema Mismatch (Empty or Wrong Columns)")
    if st.button("🛠️ Repair Database Schema"):
        repair_sheet()
        st.success("Repaired! Reloading...")
        time.sleep(1)
        st.rerun()
    if not df_raw.empty: st.caption("Attempting to run with partial data...")

# 3. CLEAN DATAFRAME FOR UI
df = df_raw.copy()
for c in REQUIRED_COLS: 
    if c not in df.columns: df[c] = None

df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0.0)
df['IVA'] = pd.to_numeric(df['IVA'], errors='coerce').fillna(0.0)
df['Is_Vice'] = df['Is_Vice'].fillna(False).astype(bool)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce').fillna(pd.Timestamp.now())

# 4. SIDEBAR
with st.sidebar:
    st.title("👤 Profile")
    st.success("☁️ Connected")
    residency = st.selectbox("Residency", ["UK (GBP)", "Spain (EUR)"])
    home_curr = "£" if "UK" in residency else "€"
    tax_code = st.text_input("Tax Code") if "UK" in residency else ""
    
    col1, col2 = st.columns(2)
    freq = col1.selectbox("Freq", ["Yearly", "Monthly"])
    gross = col2.number_input("Gross", 35000.0, step=1000.0)
    
    gross_annual = gross if freq == "Yearly" else gross * 12
    net_monthly = calculate_accurate_tax(gross_annual, residency, tax_code)
    st.metric("Net Monthly", f"{home_curr}{net_monthly:,.2f}")
    
    st.divider()
    goal_name = st.text_input("Goal", "Daughter's Bike")
    goal_target = st.number_input("Target", 150.0)
    user_vices = st.text_area("Vices", "alcohol, candy, betting")
    
    if not df.empty:
        st.write("### 📥 Export")
        st.download_button("📊 CSV", df.to_csv(index=False), "data.csv", "text/csv")
        pdf_data = generate_pdf_safe(df, goal_name, home_curr)
        if pdf_data: st.download_button("📄 PDF", pdf_data, "report.pdf", "application/pdf")

    if st.button("⚠️ Wipe DB"): repair_sheet(); st.rerun()

# 5. DASHBOARD
st.title(f"🎯 {goal_name}")
col1, col2, col3, col4 = st.columns(4)
spent = df['Amount'].sum()
vice = df[df['Is_Vice']]['Amount'].sum()
col1.metric("Spent", f"{home_curr}{spent:,.2f}")
col2.metric("Remaining", f"{home_curr}{net_monthly - spent:,.2f}")
col3.metric("Leakage", f"{home_curr}{vice:,.2f}")
col4.progress(min(vice/goal_target, 1.0))

# 6. UPLOAD
uploaded = st.file_uploader("Upload", accept_multiple_files=True)
if uploaded and st.button("🔍 Audit"):
    rows = []
    for f in uploaded:
        data = process_upload(f, api_key, user_vices, home_curr)
        if not data.empty: rows.append(data)
    
    if rows:
        combined = pd.concat(rows, ignore_index=True)
        # Auto-Fill Logic
        combined['Vendor'] = combined['Vendor'].fillna("Unknown").ffill().bfill()
        combined['Date'] = pd.to_datetime(combined['Date']).ffill().bfill().fillna(pd.Timestamp.now())
        combined['Amount'] = pd.to_numeric(combined['Amount'], errors='coerce').fillna(0.0)
        
        # Auto-VAT
        def calc_vat(r):
            rate = 20.0 if "UK" in residency else 21.0
            if "grocery" in str(r['Category']).lower(): rate = 0.0 if "UK" in residency else 4.0
            return round(r['Amount'] - (r['Amount'] / (1 + rate/100)), 2)
        combined['IVA'] = combined.apply(calc_vat, axis=1)
        
        st.session_state.review = combined
        st.rerun()

# 7. REVIEW ROOM
if 'review' in st.session_state and st.session_state.review is not None:
    st.info("Review Data")
    edited = st.data_editor(st.session_state.review, num_rows="dynamic", use_container_width=True)
    if st.button("✅ Save"):
        final = pd.concat([df, edited], ignore_index=True)
        save_data(final)
        st.session_state.review = None
        st.rerun()

# 8. TABS
t1, t2 = st.tabs(["📊 Analytics", "📝 Ledger"])
with t1:
    if not df.empty:
        st.altair_chart(alt.Chart(df).mark_bar().encode(x='Category', y='Amount', color='Category'), use_container_width=True)
with t2:
    if not df.empty: st.dataframe(df, use_container_width=True)

    
