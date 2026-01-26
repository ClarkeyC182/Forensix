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
from streamlit_gsheets import GSheetsConnection
import numpy as np

# --- CONFIG ---
st.set_page_config(page_title="Forensix Personal Auditor", page_icon="🕵️‍♂️", layout="wide")
BASE_DIR = Path(__file__).parent.absolute()
DIRS = { "TEMP": BASE_DIR / "temp_uploads" }
for d in DIRS.values(): d.mkdir(exist_ok=True)

# --- DATABASE ---
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        df = conn.read(worksheet="Sheet1", usecols=list(range(9)), ttl=5)
        df = df.dropna(how="all")
        return df
    except: return pd.DataFrame(columns=["Date", "Vendor", "Item", "Amount", "IVA", "Category", "Sub_Category", "Is_Vice", "File"])

def save_data(df):
    conn = st.connection("gsheets", type=GSheetsConnection)
    conn.update(worksheet="Sheet1", data=df)

# --- AUTH ---
def get_api_key():
    if "GEMINI_API_KEY" in st.secrets: return st.secrets["GEMINI_API_KEY"]
    return ""

# --- ENGINE ---
def vision_slice_micro(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return []
    h, w = img.shape[:2]
    # Slice slightly smaller (900px) to ensure JSON never gets too big per chunk
    if h < 900: return [img]
    slices = []
    slice_height, overlap, start = 900, 150, 0
    while start < h:
        end = min(start + slice_height, h)
        if (end - start) < 100 and len(slices) > 0: break 
        slices.append(img[start:end, :])
        if end == h: break
        start += (slice_height - overlap)
    return slices

def analyze_content(content_bytes, mime_type, client, user_vices):
    # TOKEN DIET PROMPT: Uses short keys to prevent JSON cutoff
    prompt = f"""
    Role: Forensic Auditor. Extract ALL items.
    Context: Image may contain multiple receipts.
    Output JSON keys: n=name, p=price, v=vendor, mc=main_category, sc=sub_category, vice=is_vice(bool).
    
    Rules:
    1. If vendor is unknown, use null.
    2. Main Cat: [Groceries, Dining, Alcohol, Transport, Shopping, Utils, Services].
    3. Vice Keywords: {user_vices}
    
    JSON Schema: {{ "items": [ {{ "v": "Tesco", "n": "Milk", "p": 1.20, "mc": "Groceries", "sc": "Dairy", "vice": false }} ] }}
    """
    try:
        res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, types.Part.from_bytes(data=content_bytes, mime_type=mime_type)],
            config=types.GenerateContentConfig(response_mime_type='application/json')
        )
        data = json.loads(res.text)
        items = data.get("items", []) if isinstance(data, dict) else []
        
        # INFLATE KEYS (De-compression)
        clean_items = []
        for i in items:
            clean_items.append({
                "vendor": i.get("v", "Unknown"),
                "name": i.get("n", "Item"),
                "price": i.get("p", 0.0),
                "main_category": i.get("mc", "Shopping"),
                "sub_category": i.get("sc", "General"),
                "is_vice": i.get("vice", False)
            })
        return clean_items
    except: return []

def process_upload(uploaded_file, api_key, user_vices):
    temp_path = DIRS['TEMP'] / uploaded_file.name
    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
    client = genai.Client(api_key=api_key)
    
    extracted_items = []
    if uploaded_file.type == "application/pdf":
        with open(temp_path, "rb") as f: extracted_items.extend(analyze_content(f.read(), "application/pdf", client, user_vices))
    else:
        slices = vision_slice_micro(temp_path)
        bar = st.progress(0)
        for i, s in enumerate(slices):
            _, buf = cv2.imencode(".jpg", s)
            extracted_items.extend(analyze_content(buf.tobytes(), "image/jpeg", client, user_vices))
            bar.progress((i + 1) / len(slices))
        time.sleep(0.2); bar.empty()
    os.remove(temp_path)
    
    # --- ROLLING CONTEXT V2 ---
    if extracted_items:
        df_temp = pd.DataFrame(extracted_items)
        if 'vendor' in df_temp.columns:
            # Normalize "Unknown"
            df_temp['vendor'] = df_temp['vendor'].replace(["Unknown", "unknown", "null", None], np.nan)
            
            # Smart Fill (Forward and Backward)
            df_temp['vendor'] = df_temp['vendor'].ffill().bfill()
            df_temp['vendor'] = df_temp['vendor'].fillna("Unknown")
            
            return df_temp.to_dict('records')
            
    return extracted_items

def generate_pdf_safe(df, goal_name):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16); pdf.cell(190, 10, "FORENSIX REPORT", 0, 1, 'C')
        pdf.set_font("Arial", size=10); pdf.cell(190, 10, f"Goal: {goal_name}", 0, 1, 'C'); pdf.ln(10)
        pdf.set_font("Arial", 'B', 12); pdf.cell(100, 10, f"Total: {df['Amount'].sum():.2f}", 1, 1)
        pdf.set_text_color(200, 0, 0); pdf.cell(100, 10, f"Leakage: {df[df['Is_Vice']==True]['Amount'].sum():.2f}", 1, 1); pdf.set_text_color(0,0,0); pdf.ln(5)
        pdf.set_font("Arial", 'B', 7) 
        pdf.cell(60, 8, "Item", 1); pdf.cell(30, 8, "Vendor", 1); pdf.cell(20, 8, "Price", 1); pdf.cell(40, 8, "Cat", 1); pdf.cell(40, 8, "Sub", 1); pdf.ln()
        pdf.set_font("Arial", '', 7)
        for _, row in df.iterrows():
            safe_name = str(row['Item']).encode('ascii', 'ignore').decode('ascii')[:30]
            safe_vend = str(row['Vendor']).encode('ascii', 'ignore').decode('ascii')[:15]
            safe_cat = str(row['Category']).encode('ascii', 'ignore').decode('ascii')[:15]
            safe_sub = str(row['Sub_Category']).encode('ascii', 'ignore').decode('ascii')[:15]
            pdf.cell(60, 6, safe_name, 1); pdf.cell(30, 6, safe_vend, 1); pdf.cell(20, 6, f"{row['Amount']:.2f}", 1); pdf.cell(40, 6, safe_cat, 1); pdf.cell(40, 6, safe_sub, 1); pdf.ln()
        return pdf.output(dest='S').encode('latin-1')
    except: return None

# --- UI ---
api_key = get_api_key()
if not api_key: st.stop()
df = load_data()
required_cols = ["Date", "Vendor", "Item", "Amount", "IVA", "Category", "Sub_Category", "Is_Vice", "File"]
for c in required_cols:
    if c not in df.columns: df[c] = 0.0 if c in ["Amount", "IVA"] else ""

with st.sidebar:
    st.title("👤 Profile")
    st.success("☁️ Database Connected")
    goal_name = st.text_input("Goal Name", "Daughter's New Bike")
    goal_target = st.number_input("Target (€)", value=150.0, step=50.0)
    user_vices_input = st.text_area("Vices", "tobacco, alcohol, bet, lottery, mcdonalds, candy, game", height=100)
    st.markdown("---")
    if not df.empty:
        st.write("## 📥 Export")
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("📊 Download CSV", csv_data, "data.csv", "text/csv")
        pdf_data = generate_pdf_safe(df, goal_name)
        if pdf_data: st.download_button("📄 Download PDF", pdf_data, "report.pdf", "application/pdf")
    st.markdown("---")
    if st.button("⚠️ Clear Database"):
        empty_df = pd.DataFrame(columns=required_cols)
        save_data(empty_df)
        st.rerun()

st.title(f"🎯 Project: {goal_name}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Spend", f"€{df['Amount'].sum():.2f}")
col2.metric("Tax Recovered", f"€{df['IVA'].sum():.2f}")
col3.metric("Habit Leakage", f"€{df[df['Is_Vice']==True]['Amount'].sum():.2f}", delta="-Leakage")
progress = min((df[df['Is_Vice']==True]['Amount'].sum() / goal_target) * 100, 100)
col4.metric("Goal Progress", f"{progress:.1f}%")
st.progress(progress / 100)

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
                cat = item.get('main_category', 'Shopping')
                if not cat or cat == "Unknown": cat = "Shopping"
                if iva_rate == 0 and "alcohol" in str(cat).lower(): iva_rate = 21.0
                new_rows.append({
                    "Date": pd.Timestamp.now().strftime("%Y-%m-%d"), 
                    "Vendor": item.get('vendor', 'Unknown'),
                    "Item": str(item.get('name', 'Item')), "Amount": price,
                    "IVA": round(price - (price / (1 + (iva_rate / 100))), 2),
                    "Category": cat, "Sub_Category": item.get('sub_category', 'General'),
                    "Is_Vice": item.get('is_vice', False), "File": f.name
                })
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        st.success(f"Processing Complete! Found {len(new_rows)} items.")
        st.dataframe(df_new)
        updated_df = pd.concat([df, df_new], ignore_index=True)
        save_data(updated_df)
        st.toast("Saved to Cloud!", icon="☁️")
        time.sleep(2)
        st.rerun()

# [Tabs Logic Remains Same]
tab1, tab2 = st.tabs(["📊 Analytics", "📝 Ledger"])
with tab1:
    if not df.empty:
        chart = alt.Chart(df.groupby("Category")["Amount"].sum().reset_index()).mark_bar().encode(
            x=alt.X('Category', sort='-y'), y='Amount', color='Category', tooltip=['Category', 'Amount']
        ).properties(height=350, title="Spend by Main Category")
        st.altair_chart(chart, use_container_width=True)
        st.markdown("### 🔍 Drill Down")
        chart_sub = alt.Chart(df.groupby("Sub_Category")["Amount"].sum().reset_index()).mark_bar().encode(
            x=alt.X('Amount'), y=alt.Y('Sub_Category', sort='-x'), tooltip=['Sub_Category', 'Amount']
        ).properties(height=400, title="Spend by Detail")
        st.altair_chart(chart_sub, use_container_width=True)

with tab2:
    if not df.empty:
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        if st.button("💾 Sync to Cloud"):
            save_data(edited_df)
            st.success("Synced!")
            
