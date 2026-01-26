import streamlit as st
import pandas as pd
import time
import cv2
import json
import os
import shutil
import altair as alt
from pathlib import Path
from google import genai
from google.genai import types
from fpdf import FPDF

# --- CONFIG ---
st.set_page_config(page_title="Forensix Personal Auditor", page_icon="🕵️‍♂️", layout="wide")
BASE_DIR = Path(__file__).parent.absolute()
DIRS = { "TEMP": BASE_DIR / "temp_uploads", "REPORTS": BASE_DIR / "audit_reports" }
for d in DIRS.values(): d.mkdir(exist_ok=True)
DATA_FILE = "forensix_ledger_master.csv"

# --- AUTHENTICATION (SECURE) ---
def get_api_key():
    # Try to get from Streamlit Secrets (Cloud)
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    # Fallback for local testing (Optional: Set env var locally)
    return ""

# --- ENGINE ---
def vision_slice_micro(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return []
    h, w = img.shape[:2]
    if h < 1000: return [img]
    slices = []
    slice_height, overlap, start = 1000, 200, 0
    while start < h:
        end = min(start + slice_height, h)
        if (end - start) < 100 and len(slices) > 0: break 
        slices.append(img[start:end, :])
        if end == h: break
        start += (slice_height - overlap)
    return slices

def analyze_content(content_bytes, mime_type, client, user_vices):
    prompt = f"""
    Role: Forensic Auditor. Task: Extract EVERY line item.
    Rules: 1. Output strictly JSON. 2. Categorize items. 3. Flag 'is_vice' if matches [{user_vices}].
    JSON Schema: {{ "items": [ {{ "name": "...", "price": 0.00, "iva_rate": 0, "category": "...", "is_vice": false }} ] }}
    """
    try:
        res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, types.Part.from_bytes(data=content_bytes, mime_type=mime_type)],
            config=types.GenerateContentConfig(response_mime_type='application/json')
        )
        data = json.loads(res.text)
        if isinstance(data, list): return data
        if isinstance(data, dict): return data.get("items", []) or data.get("receipts", []) or []
        return []
    except Exception as e:
        print(f"AI Error: {e}")
        return []

def process_upload(uploaded_file, api_key, user_vices):
    # Save to temp
    temp_path = DIRS['TEMP'] / uploaded_file.name
    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
    
    client = genai.Client(api_key=api_key)
    extracted_items = []
    
    # PDF LOGIC (Explicitly Supported)
    if uploaded_file.type == "application/pdf":
        with open(temp_path, "rb") as f: 
            extracted_items.extend(analyze_content(f.read(), "application/pdf", client, user_vices))
    
    # IMAGE LOGIC
    else:
        slices = vision_slice_micro(temp_path)
        bar = st.progress(0)
        for i, s in enumerate(slices):
            _, buf = cv2.imencode(".jpg", s)
            extracted_items.extend(analyze_content(buf.tobytes(), "image/jpeg", client, user_vices))
            bar.progress((i + 1) / len(slices))
        time.sleep(0.2); bar.empty()
        
    os.remove(temp_path)
    return extracted_items

def generate_pdf_safe(df, goal_name):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16); pdf.cell(190, 10, "FORENSIX REPORT", 0, 1, 'C')
        pdf.set_font("Arial", size=10); pdf.cell(190, 10, f"Goal: {goal_name}", 0, 1, 'C'); pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(100, 10, f"Total: {df['Amount'].sum():.2f}", 1, 1)
        pdf.set_text_color(200, 0, 0); pdf.cell(100, 10, f"Leakage: {df[df['Is_Vice']==True]['Amount'].sum():.2f}", 1, 1); pdf.set_text_color(0,0,0)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 8); pdf.cell(100, 8, "Item", 1); pdf.cell(30, 8, "Price", 1); pdf.cell(50, 8, "Cat", 1); pdf.ln()
        pdf.set_font("Arial", '', 8)
        for _, row in df.iterrows():
            safe_name = str(row['Item']).encode('ascii', 'ignore').decode('ascii')[:40]
            safe_cat = str(row['Category']).encode('ascii', 'ignore').decode('ascii')
            pdf.cell(100, 6, safe_name, 1); pdf.cell(30, 6, f"{row['Amount']:.2f}", 1); pdf.cell(50, 6, safe_cat, 1); pdf.ln()
        return pdf.output(dest='S').encode('latin-1')
    except: return None

# --- UI ---
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    for c in ["Date", "Vendor", "Item", "Amount", "IVA", "Category", "Is_Vice", "File"]:
        if c not in df.columns: df[c] = 0 if c in ["Amount", "IVA"] else ""
else:
    df = pd.DataFrame(columns=["Date", "Vendor", "Item", "Amount", "IVA", "Category", "Is_Vice", "File"])

with st.sidebar:
    st.title("👤 Profile")
    
    # API KEY HANDLING
    api_key = get_api_key()
    if not api_key:
        st.error("⚠️ API Key Missing! Add 'GEMINI_API_KEY' to Streamlit Secrets.")
        st.stop()
    else:
        st.success("🔒 Secure Connection Active")

    goal_name = st.text_input("Goal Name", "Daughter's New Bike")
    goal_target = st.number_input("Target (€)", value=150.0, step=50.0)
    user_vices_input = st.text_area("Vices", "tobacco, alcohol, bet, lottery, mcdonalds, candy, game", height=100)
    st.markdown("---")
    
    # EXPORTS
    if not df.empty:
        st.write("## 📥 Export Data")
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("📊 Download CSV", csv_data, "data.csv", "text/csv")
        pdf_data = generate_pdf_safe(df, goal_name)
        if pdf_data: st.download_button("📄 Download PDF", pdf_data, "report.pdf", "application/pdf")
    
    st.markdown("---")
    if st.button("⚠️ Clear Data"):
        if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
        st.rerun()

st.title(f"🎯 Project: {goal_name}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Spend", f"€{df['Amount'].sum():.2f}")
col2.metric("Tax Recovered", f"€{df['IVA'].sum():.2f}")
col3.metric("Habit Leakage", f"€{df[df['Is_Vice']==True]['Amount'].sum():.2f}", delta="-Leakage")
progress = min((df[df['Is_Vice']==True]['Amount'].sum() / goal_target) * 100, 100)
col4.metric("Goal Progress", f"{progress:.1f}%")
st.progress(progress / 100)

# UPLOAD SECTION (PDF SUPPORTED)
uploaded = st.file_uploader("Upload Receipts", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'pdf'])

if uploaded and st.button("🔍 Run Forensic Audit"):
    new_rows = []
    for f in uploaded:
        with st.spinner(f"Analyzing {f.name}..."):
            items = process_upload(f, api_key, user_vices_input)
            for item in items:
                try: price = float(item.get('price', 0))
                except: price = 0.0
                iva_rate = item.get('iva_rate', 0)
                if iva_rate == 0 and "alcohol" in str(item.get('category','')).lower(): iva_rate = 21.0
                new_rows.append({
                    "Date": pd.Timestamp.now().strftime("%Y-%m-%d"), "Vendor": item.get('vendor', 'Unknown'),
                    "Item": str(item.get('name', 'Item')), "Amount": price,
                    "IVA": round(price - (price / (1 + (iva_rate / 100))), 2),
                    "Category": item.get('category', 'Shopping'), "Is_Vice": item.get('is_vice', False), "File": f.name
                })
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        st.rerun()

tab1, tab2 = st.tabs(["📊 Analytics", "📝 Ledger"])
with tab1:
    if not df.empty:
        chart = alt.Chart(df.groupby("Category")["Amount"].sum().reset_index()).mark_bar().encode(
            x=alt.X('Category', sort='-y'), y='Amount', color='Category'
        ).properties(height=350)
        st.altair_chart(chart, use_container_width=True)
with tab2:
    if not df.empty:
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        if st.button("💾 Save Ledger"):
            edited_df.to_csv(DATA_FILE, index=False)
            st.rerun()
            
