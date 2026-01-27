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
from datetime import datetime
import re

# --- CONFIG ---
st.set_page_config(page_title="Forensix Personal Auditor", page_icon="🕵️‍♂️", layout="wide")
BASE_DIR = Path(__file__).parent.absolute()
DIRS = { "TEMP": BASE_DIR / "temp_uploads" }
for d in DIRS.values(): d.mkdir(exist_ok=True)

# --- SESSION STATE ---
if 'review_data' not in st.session_state:
    st.session_state.review_data = None
if 'last_error' not in st.session_state:
    st.session_state.last_error = None

# --- DATABASE ---
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        df = conn.read(worksheet="Sheet1", usecols=list(range(10)), ttl=5)
        df = df.dropna(how="all")
        # Robust Date Loading
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except: return pd.DataFrame(columns=["Date", "Vendor", "Item", "Amount", "Currency", "IVA", "Category", "Sub_Category", "Is_Vice", "File"])

def save_data(df):
    conn = st.connection("gsheets", type=GSheetsConnection)
    df_save = df.copy()
    # --- CRASH FIX: Force Date Conversion ---
    df_save['Date'] = pd.to_datetime(df_save['Date'], errors='coerce')
    # Handle any rows where date failed (Nat) by setting to today
    df_save['Date'] = df_save['Date'].fillna(pd.Timestamp.now())
    # Format safely
    df_save['Date'] = df_save['Date'].dt.strftime('%Y-%m-%d')
    conn.update(worksheet="Sheet1", data=df_save)

# --- TAX LOGIC ENGINE ---
def calculate_net_monthly_advanced(gross_annual, residency, tax_code=None):
    net_annual = 0
    
    if "UK" in residency:
        # UK LOGIC
        # 1. Parse Tax Code (e.g., 1257L -> 12570 allowance)
        allowance = 12570 # Default
        if tax_code and str(tax_code).upper().endswith('L'):
            try:
                digits = int(re.sub(r'\D', '', str(tax_code)))
                allowance = digits * 10
            except: pass
            
        taxable_income = max(0, gross_annual - allowance)
        
        # 2. Income Tax (Simplified Bands 2025)
        income_tax = 0
        if taxable_income <= 37700:
            income_tax = taxable_income * 0.20
        else:
            # Basic rate on first 37.7k
            income_tax = 37700 * 0.20
            # Higher rate (40%) on rest
            income_tax += (taxable_income - 37700) * 0.40
            
        # 3. National Insurance (Approx 8% over ~12.5k)
        ni = 0
        if gross_annual > 12570:
            ni = (gross_annual - 12570) * 0.08
            
        net_annual = gross_annual - income_tax - ni
        
    elif "Spain" in residency:
        # SPAIN IRPF ESTIMATOR (General Scale 2025)
        # Note: This is an approximation. Spain tax varies by region (Andalusia etc.)
        # We apply a standard IRPF curve + Social Security deduction (~6.35%)
        
        social_security = gross_annual * 0.0635
        base_imponible = gross_annual - social_security - 2000 # Standard deduction
        
        irpf = 0
        # Progressive Bands
        bands = [
            (12450, 0.19),
            (20200, 0.24),
            (35200, 0.30),
            (60000, 0.37),
            (300000, 0.45)
        ]
        
        remaining = base_imponible
        previous_limit = 0
        
        for limit, rate in bands:
            if base_imponible > previous_limit:
                taxable_in_band = min(base_imponible, limit) - previous_limit
                irpf += taxable_in_band * rate
                previous_limit = limit
            else:
                break
                
        net_annual = gross_annual - social_security - irpf

    return net_annual / 12

# --- ENGINE ---
def get_api_key():
    if "GEMINI_API_KEY" in st.secrets: return st.secrets["GEMINI_API_KEY"]
    return ""

def vision_slice_micro(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return []
    h, w = img.shape[:2]
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

def analyze_content(content_bytes, mime_type, client, user_vices, home_currency):
    prompt = f"""
    Role: Forensic Auditor. Extract items, DATE, and CURRENCY.
    Context: Image may contain multiple receipts.
    Output keys: d=date(YYYY-MM-DD), v=vendor, n=name, p=price, c=currency_symbol, mc=cat, sc=sub, vice=bool.
    Rules:
    1. Look for date. If null, use null.
    2. Look for currency (£, €, $). If missing, assume {home_currency}.
    3. Main Cat: [Groceries, Dining, Alcohol, Transport, Shopping, Utils, Services].
    4. Vice: {user_vices}
    5. Vendor: Look closely for the logo/header on THIS specific receipt.
    
    IMPORTANT: RETURN ONLY RAW JSON ARRAY. Example: [{{ ... }}]
    """
    try:
        res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, types.Part.from_bytes(data=content_bytes, mime_type=mime_type)],
            config=types.GenerateContentConfig(response_mime_type='application/json')
        )
        raw = res.text
        try: data = json.loads(raw)
        except:
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match: data = json.loads(match.group())
            else:
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match: data = json.loads(match.group())
                else: return []

        if isinstance(data, dict): items = data.get("items", []) or data.get("receipts", [])
        elif isinstance(data, list): items = data
        else: items = []

        clean_items = []
        for i in items:
            clean_items.append({
                "Date": i.get("d"),
                "Vendor": i.get("v", "Unknown"),
                "Item": i.get("n", "Item"),
                "Amount": i.get("p", 0.0),
                "Currency": i.get("c", home_currency),
                "Category": i.get("mc", "Shopping"),
                "Sub_Category": i.get("sc", "General"),
                "Is_Vice": i.get("vice", False)
            })
        return clean_items
    except Exception as e:
        st.session_state.last_error = f"API Error: {str(e)}"
        return []

def process_upload(uploaded_file, api_key, user_vices, home_currency):
    temp_path = DIRS['TEMP'] / uploaded_file.name
    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
    client = genai.Client(api_key=api_key)
    extracted_items = []
    
    if uploaded_file.type == "application/pdf":
        with open(temp_path, "rb") as f: extracted_items.extend(analyze_content(f.read(), "application/pdf", client, user_vices, home_currency))
    else:
        slices = vision_slice_micro(temp_path)
        bar = st.progress(0)
        for i, s in enumerate(slices):
            _, buf = cv2.imencode(".jpg", s)
            extracted_items.extend(analyze_content(buf.tobytes(), "image/jpeg", client, user_vices, home_currency))
            bar.progress((i + 1) / len(slices))
        time.sleep(0.2); bar.empty()
    os.remove(temp_path)
    
    if extracted_items:
        df_temp = pd.DataFrame(extracted_items)
        
        # --- FIX: REMOVED AGGRESSIVE VENDOR AUTO-FILL ---
        # We only standardise formats, we DO NOT copy "Ale-Hop" down to everything else.
        if 'Vendor' in df_temp.columns:
            df_temp['Vendor'] = df_temp['Vendor'].replace(["null", None], "Unknown")
            
        if 'Date' in df_temp.columns:
            df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
            df_temp['Date'] = df_temp['Date'].ffill().bfill().fillna(pd.Timestamp.now())
            
        if 'Currency' in df_temp.columns:
             df_temp['Currency'] = df_temp['Currency'].replace([None, ""], np.nan)
             df_temp['Currency'] = df_temp['Currency'].ffill().bfill().fillna(home_currency)
             
        for col in ["IVA", "File"]: 
            if col not in df_temp.columns: df_temp[col] = ""
            
        return df_temp.to_dict('records')
    return extracted_items

def generate_pdf_safe(df, goal_name, currency_symbol):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16); pdf.cell(190, 10, "FORENSIX REPORT", 0, 1, 'C')
        pdf.set_font("Arial", size=10); pdf.cell(190, 10, f"Goal: {goal_name}", 0, 1, 'C'); pdf.ln(10)
        pdf.set_font("Arial", 'B', 12); pdf.cell(100, 10, f"Total: {currency_symbol}{df['Amount'].sum():.2f}", 1, 1)
        pdf.set_text_color(200, 0, 0); pdf.cell(100, 10, f"Leakage: {currency_symbol}{df[df['Is_Vice']==True]['Amount'].sum():.2f}", 1, 1); pdf.set_text_color(0,0,0); pdf.ln(5)
        pdf.set_font("Arial", 'B', 7) 
        pdf.cell(60, 8, "Item", 1); pdf.cell(20, 8, "Date", 1); pdf.cell(20, 8, "Price", 1); pdf.cell(40, 8, "Cat", 1); pdf.cell(40, 8, "Sub", 1); pdf.ln()
        pdf.set_font("Arial", '', 7)
        for _, row in df.iterrows():
            safe_name = str(row['Item']).encode('ascii', 'ignore').decode('ascii')[:30]
            try: safe_date = pd.to_datetime(row['Date']).strftime('%d/%m')
            except: safe_date = ""
            safe_cat = str(row['Category']).encode('ascii', 'ignore').decode('ascii')[:15]
            safe_sub = str(row['Sub_Category']).encode('ascii', 'ignore').decode('ascii')[:15]
            pdf.cell(60, 6, safe_name, 1); pdf.cell(20, 6, safe_date, 1); pdf.cell(20, 6, f"{row['Amount']:.2f}", 1); pdf.cell(40, 6, safe_cat, 1); pdf.cell(40, 6, safe_sub, 1); pdf.ln()
        return pdf.output(dest='S').encode('latin-1')
    except: return None

# --- UI ---
api_key = get_api_key()
if not api_key: st.stop()
df = load_data()
required_cols = ["Date", "Vendor", "Item", "Amount", "Currency", "IVA", "Category", "Sub_Category", "Is_Vice", "File"]
for c in required_cols:
    if c not in df.columns: df[c] = 0.0 if c in ["Amount", "IVA"] else ""

with st.sidebar:
    st.title("👤 Profile")
    st.success("☁️ Database Connected")
    
    st.markdown("### 🌍 Residency & Tax")
    residency = st.selectbox("Tax Residency", ["UK (GBP)", "Spain (EUR)"], index=1)
    home_currency = "£" if "UK" in residency else "€"
    
    # TAX CODE LOGIC
    tax_code_input = ""
    if "UK" in residency:
        tax_code_input = st.text_input("Tax Code (Optional)", placeholder="e.g. 1257L")
    
    st.markdown("### 💰 Income")
    col_inc1, col_inc2 = st.columns(2)
    income_freq = col_inc1.selectbox("Freq", ["Yearly", "Monthly", "Hourly"])
    gross_income = col_inc2.number_input("Gross", value=35000.0, step=1000.0)
    
    gross_annual = 0.0
    if income_freq == "Yearly": gross_annual = gross_income
    elif income_freq == "Monthly": gross_annual = gross_income * 12
    elif income_freq == "Hourly": gross_annual = gross_income * 160 * 12
    
    # ADVANCED NET CALC
    net_monthly = calculate_net_monthly_advanced(gross_annual, residency, tax_code_input)
    st.metric("Est. Net Monthly", f"{home_currency}{net_monthly:,.2f}", help="Calculated using local tax bands")

    st.markdown("---")
    st.markdown("### 📅 Time Travel")
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        months = df['Date'].dt.to_period('M').unique().sort_values(ascending=False)
        month_strs = [str(m) for m in months]
        selected_month_str = st.selectbox("Select Month", month_strs) if month_strs else None
        
        if selected_month_str:
            mask = df['Date'].dt.to_period('M').astype(str) == selected_month_str
            filtered_df = df[mask]
        else:
            filtered_df = df
    else:
        filtered_df = df
        
    st.markdown("---")
    goal_name = st.text_input("Goal Name", "Daughter's New Bike")
    goal_target = st.number_input("Target", value=150.0, step=50.0)
    user_vices_input = st.text_area("Vices", "tobacco, alcohol, bet, lottery, mcdonalds, candy, game", height=100)
    
    if not df.empty:
        st.write("## 📥 Export")
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📊 CSV", csv_data, "data.csv", "text/csv")
        pdf_data = generate_pdf_safe(filtered_df, goal_name, home_currency)
        if pdf_data: st.download_button("📄 PDF", pdf_data, "report.pdf", "application/pdf")
    
    if st.button("⚠️ Clear Database"):
        empty_df = pd.DataFrame(columns=required_cols)
        save_data(empty_df)
        st.rerun()

st.title(f"🎯 Project: {goal_name}")

col1, col2, col3, col4 = st.columns(4)
month_spend = filtered_df['Amount'].sum()
month_tax = filtered_df['IVA'].sum()
month_vice = filtered_df[filtered_df['Is_Vice']==True]['Amount'].sum()
remaining = net_monthly - month_spend

col1.metric("Spent (Month)", f"{home_currency}{month_spend:.2f}", delta=f"Left: {home_currency}{remaining:,.2f}")
col2.metric("Tax Recovered", f"{home_currency}{month_tax:.2f}")
col3.metric("Habit Leakage", f"{home_currency}{month_vice:.2f}", delta="-Leakage")
progress = min((filtered_df[filtered_df['Is_Vice']==True]['Amount'].sum() / goal_target) * 100, 100)
col4.metric("Goal Progress", f"{progress:.1f}%")
st.progress(progress / 100)

uploaded = st.file_uploader("Upload Receipts", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'pdf'])
if uploaded and st.button("🔍 Run Forensic Audit"):
    st.session_state.last_error = None
    new_rows = []
    for f in uploaded:
        with st.spinner(f"Analyzing {f.name}..."):
            items = process_upload(f, api_key, user_vices_input, home_currency)
            for item in items:
                try: price = float(item.get('Amount', 0))
                except: price = 0.0
                
                cat = str(item.get('Category','')).lower()
                vat_rate = 21.0
                if "grocery" in cat: vat_rate = 4.0
                if "uk" in residency.lower():
                    if "grocery" in cat: vat_rate = 0.0
                    else: vat_rate = 20.0
                
                iva = round(price - (price / (1 + (vat_rate / 100))), 2)
                
                try: r_date = pd.to_datetime(item.get('Date')).strftime('%Y-%m-%d')
                except: r_date = pd.Timestamp.now().strftime('%Y-%m-%d')

                new_rows.append({
                    "Date": r_date,
                    "Vendor": item.get('Vendor', 'Unknown'),
                    "Item": str(item.get('Item', 'Item')), 
                    "Amount": price,
                    "Currency": item.get('Currency', home_currency),
                    "IVA": iva,
                    "Category": item.get('Category', 'Shopping'),
                    "Sub_Category": item.get('Sub_Category', 'General'),
                    "Is_Vice": item.get('Is_Vice', False), 
                    "File": f.name
                })
    
    if new_rows:
        st.session_state.review_data = pd.DataFrame(new_rows)
        st.rerun()
    elif st.session_state.last_error:
        st.error(f"❌ Analysis Failed: {st.session_state.last_error}")
    else:
        st.warning("No items found.")

if st.session_state.review_data is not None:
    st.write("### 🧐 Review & Edit Results")
    st.info("Edit below. Click Confirm to save.")
    
    edited_new_df = st.data_editor(
        st.session_state.review_data, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "Currency": st.column_config.SelectboxColumn("Currency", options=["£", "€", "$"]),
            "Category": st.column_config.SelectboxColumn("Category", options=["Groceries", "Dining", "Alcohol", "Transport", "Shopping", "Utils", "Services"]),
            "Is_Vice": st.column_config.CheckboxColumn("Vice?", default=False)
        }
    )
    
    col_save, col_discard = st.columns([1,4])
    if col_save.button("✅ Confirm & Save"):
        updated_df = pd.concat([df, edited_new_df], ignore_index=True)
        save_data(updated_df)
        st.session_state.review_data = None
        st.success(f"Saved!")
        time.sleep(1)
        st.rerun()
        
    if col_discard.button("❌ Discard"):
        st.session_state.review_data = None
        st.rerun()

# --- ANALYTICS ---
tab1, tab2, tab3 = st.tabs(["📊 Analytics", "📝 Ledger", "🤖 AI Insights"])
with tab1:
    if not filtered_df.empty:
        chart = alt.Chart(filtered_df.groupby("Category")["Amount"].sum().reset_index()).mark_bar().encode(
            x=alt.X('Category', sort='-y'), y='Amount', color='Category', tooltip=['Category', 'Amount']
        ).properties(height=350, title="Monthly Spend by Category")
        st.altair_chart(chart, use_container_width=True)

with tab2:
    if not filtered_df.empty:
        edited_df = st.data_editor(filtered_df, num_rows="dynamic", use_container_width=True)
        if st.button("💾 Update Ledger"):
            save_data(edited_df)
            st.success("Synced!")

with tab3:
    st.write("### 🧠 Forensic Insights")
    if not filtered_df.empty:
        top_cat = filtered_df.groupby("Category")["Amount"].sum().idxmax()
        top_val = filtered_df.groupby("Category")["Amount"].sum().max()
        vice_percent = (month_vice / month_spend) * 100 if month_spend > 0 else 0
        
        st.write(f"**Spending Alert:** Your biggest expense this month is **{top_cat}** ({home_currency}{top_val:.2f}).")
        if vice_percent > 10:
            st.warning(f"**Vice Alert:** {vice_percent:.1f}% of your income went to 'Habits' this month.")
        else:
            st.success(f"**Good Job:** Your vice spending is under control ({vice_percent:.1f}%).")
