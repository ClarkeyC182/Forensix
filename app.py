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

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Forensix Personal Auditor", page_icon="🕵️‍♂️", layout="wide")

BASE_DIR = Path(__file__).parent.absolute()
DIRS = {
    "TEMP": BASE_DIR / "temp_uploads",
    "REPORTS": BASE_DIR / "audit_reports"
}
for d in DIRS.values(): d.mkdir(exist_ok=True)

DATA_FILE = "forensix_ledger_master.csv"

# --- 2. INTELLIGENCE ENGINE ---

def tax_inference_logic(item_name, category, ocr_tax, price):
    """Fills in missing tax rates based on category."""
    if ocr_tax > 0: return ocr_tax
    if category == "Groceries": return 10.0
    if category == "Alcohol/Tobacco": return 21.0
    if category == "Dining Out": return 10.0
    if category == "Transport": return 10.0
    return 21.0

def analyze_image_smart(img, client, user_vices):
    _, buf = cv2.imencode(".jpg", img)
    
    # We ask for a strict Schema, but we also handle it if the AI ignores it.
    prompt = f"""
    You are a Forensic Auditor. Extract EVERY line item from the receipt.
    
    1. CATEGORIZE each item into: [Groceries, Dining Out, Alcohol/Tobacco, Transport, Shopping, Utilities, Entertainment].
    2. DETECT VICES: If an item matches user keywords [{user_vices}], mark 'is_vice' as true.
    3. EXTRACT: Name, Price, and IVA.
    
    Output purely JSON.
    """
    
    try:
        res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, types.Part.from_bytes(data=buf.tobytes(), mime_type='image/jpeg')],
            config=types.GenerateContentConfig(response_mime_type='application/json')
        )
        return json.loads(res.text)
    except Exception as e:
        print(f"AI Error: {e}")
        return [] # Return empty list on failure

def vision_slice_smart(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return []
    h, w = img.shape[:2]
    slices = []
    
    # Slice only if the image is significantly taller than it is wide
    if h > 2000 and h > w * 1.5:
        slice_h, overlap = 1200, 300
        start = 0
        while start < h:
            end = min(start + slice_h, h)
            if (end - start) < 200 and len(slices) > 0: break
            slices.append(img[start:end, :])
            if end == h: break
            start += (slice_h - overlap)
    else:
        slices.append(img)
    return slices

def process_upload(uploaded_file, api_key, user_vices):
    temp_path = DIRS['TEMP'] / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    client = genai.Client(api_key=api_key)
    slices = vision_slice_smart(temp_path)
    
    all_rows = []
    bar = st.progress(0)
    
    for i, s in enumerate(slices):
        raw_data = analyze_image_smart(s, client, user_vices)
        
        # --- THE BULLETPROOF FIX ---
        # If AI returns a List: Wrap it in the structure we expect.
        if isinstance(raw_data, list):
            data_struct = {"receipts": [{"items": raw_data}]}
        elif isinstance(raw_data, dict):
            # If AI returns Dict but keys are weird, normalize them
            if "receipts" not in raw_data:
                # Maybe it returned {"items": [...]}
                if "items" in raw_data:
                    data_struct = {"receipts": [raw_data]}
                else:
                    # Unknown dict structure, skip safely
                    data_struct = {"receipts": []}
            else:
                data_struct = raw_data
        else:
            data_struct = {"receipts": []}
        # ---------------------------

        for block in data_struct.get('receipts', []):
            vendor = block.get('vendor', 'Unknown')
            for item in block.get('items', []):
                name = str(item.get('name', 'Item'))
                try: price = float(item.get('price', 0))
                except: price = 0.0
                
                ocr_tax = item.get('iva_rate', 0)
                category = item.get('category', 'Shopping')
                is_vice = item.get('is_vice', False)
                
                final_tax_rate = tax_inference_logic(name, category, ocr_tax, price)
                iva_amt = round(price - (price / (1 + (final_tax_rate / 100))), 2)
                
                all_rows.append({
                    "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                    "Vendor": vendor,
                    "Item": name,
                    "Amount": price,
                    "IVA": iva_amt,
                    "Category": category,
                    "Is_Vice": is_vice,
                    "File": uploaded_file.name
                })
        bar.progress((i + 1) / len(slices))
    
    time.sleep(0.5); bar.empty(); os.remove(temp_path)
    return all_rows

# --- 3. UI LAYOUT ---

with st.sidebar:
    st.title("👤 User Profile")
    api_key_input = st.text_input("Gemini API Key", value="AIzaSyC6GlQSSgIZ1UKDwGvNduggY5GR-nBO3mw", type="password")
    st.markdown("---")
    goal_name = st.text_input("Savings Goal Name", "Daughter's New Bike")
    goal_target = st.number_input("Target Amount (€)", value=150.0, step=50.0)
    st.markdown("---")
    st.subheader("🛑 My 'Vice' Triggers")
    st.caption("Tell the AI what to flag (comma separated).")
    user_vices_input = st.text_area("Keywords", "tobacco, alcohol, bet, lottery, mcdonalds, candy, game", height=100)

st.title(f"🎯 Project: {goal_name}")

if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    if "Category" not in df.columns: df["Category"] = "Shopping"
    if "Is_Vice" not in df.columns: df["Is_Vice"] = False
else:
    df = pd.DataFrame(columns=["Date", "Vendor", "Item", "Amount", "IVA", "Category", "Is_Vice", "File"])

col1, col2, col3, col4 = st.columns(4)
total_spend = df['Amount'].sum()
total_iva = df['IVA'].sum()
vice_spend = df[df['Is_Vice'] == True]['Amount'].sum()
progress = min((vice_spend / goal_target) * 100, 100)

col1.metric("Total Spend", f"€{total_spend:.2f}")
col2.metric("Tax Recovered", f"€{total_iva:.2f}")
col3.metric("Habit/Vice Cost", f"€{vice_spend:.2f}", delta="-Leakage")
col4.metric(f"Goal Progress", f"{progress:.1f}%")

st.progress(progress / 100)
if progress >= 100: st.balloons()

st.markdown("### 🧾 Upload Receipts")
uploaded = st.file_uploader("", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded:
    if st.button("🔍 Run Personalized Audit"):
        new_data = []
        for f in uploaded:
            with st.spinner(f"Auditing {f.name} against your profile..."):
                extracted = process_upload(f, api_key_input, user_vices_input)
                new_data.extend(extracted)
        
        if new_data:
            df_new = pd.DataFrame(new_data)
            df_new = df_new.drop_duplicates(subset=['Vendor', 'Item', 'Amount'])
            df = pd.concat([df, df_new], ignore_index=True) if not df.empty else df_new
            df.to_csv(DATA_FILE, index=False)
            st.success("Audit Complete!")
            st.rerun()

tab1, tab2 = st.tabs(["📊 Analytics", "📝 Ledger Editor"])

with tab1:
    st.subheader("Where is the money going?")
    if not df.empty:
        cat_data = df.groupby("Category")["Amount"].sum().reset_index()
        
        # Professional Altair Chart
        chart = alt.Chart(cat_data).mark_bar().encode(
            x=alt.X('Category', sort='-y'),
            y=alt.Y('Amount', title='€ Spent'),
            color=alt.Color('Category', scale=alt.Scale(scheme='tableau10')),
            tooltip=['Category', 'Amount']
        ).properties(height=350)
        
        st.altair_chart(chart, use_container_width=True)

        st.subheader("🚨 Top 'Vice' Items")
        vices = df[df['Is_Vice'] == True].groupby("Item")["Amount"].sum().reset_index().sort_values("Amount", ascending=False).head(5)
        if not vices.empty:
            st.table(vices.set_index("Item"))
        else:
            st.info("No vices detected yet. Keep it up!")

with tab2:
    st.subheader("Edit & Categorize")
    if not df.empty:
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            column_config={
                "Amount": st.column_config.NumberColumn(format="€%.2f"),
                "IVA": st.column_config.NumberColumn(format="€%.2f"),
                "Is_Vice": st.column_config.CheckboxColumn(label="Vice?"),
                "Category": st.column_config.SelectboxColumn(
                    options=["Groceries", "Dining Out", "Alcohol/Tobacco", "Transport", "Shopping", "Utilities", "Entertainment"]
                )
            },
            use_container_width=True
        )
        if st.button("💾 Save Ledger Changes"):
            edited_df.to_csv(DATA_FILE, index=False)
            st.success("Saved!")
            st.rerun()