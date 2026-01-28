import streamlit as st
import pandas as pd
import time
import cv2
import json
import os
import uuid
import re
import sqlite3
import hashlib
import altair as alt
from pathlib import Path
from google import genai
from google.genai import types
from datetime import datetime, timedelta
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Forensix Pro",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent.absolute()
DIRS = {
    "TEMP": BASE_DIR / "temp_uploads",
    "DB": BASE_DIR / "data"
}
for d in DIRS.values(): d.mkdir(exist_ok=True)

DB_PATH = DIRS["DB"] / "forensix.db"

# --- DATABASE MANAGER ---
class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        self._init_db()
        self._migrate_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            currency TEXT DEFAULT '£',
            gross_income REAL DEFAULT 0,
            income_freq TEXT DEFAULT 'Yearly',
            tax_residency TEXT DEFAULT 'UK',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            date TEXT,
            vendor TEXT,
            item TEXT,
            amount REAL,
            currency TEXT,
            iva REAL,
            category TEXT,
            sub_category TEXT,
            is_vice BOOLEAN,
            file_name TEXT,
            FOREIGN KEY(username) REFERENCES users(username)
        )''')
        conn.commit()
        conn.close()

    def _migrate_db(self):
        conn = self._get_conn()
        try:
            conn.execute("SELECT income_freq FROM users LIMIT 1")
        except sqlite3.OperationalError:
            cols = [
                ("income_freq", "TEXT DEFAULT 'Yearly'"),
                ("gross_income", "REAL DEFAULT 0"),
                ("tax_residency", "TEXT DEFAULT 'UK'"),
                ("currency", "TEXT DEFAULT '£'")
            ]
            for col, dtype in cols:
                try: conn.execute(f"ALTER TABLE users ADD COLUMN {col} {dtype}")
                except: pass
            conn.commit()
        finally:
            conn.close()

    def create_user(self, username, password):
        conn = self._get_conn()
        c = conn.cursor()
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        try:
            c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, pwd_hash))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def verify_user(self, username, password):
        conn = self._get_conn()
        c = conn.cursor()
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        c.execute("SELECT username FROM users WHERE username = ? AND password_hash = ?", (username, pwd_hash))
        user = c.fetchone()
        conn.close()
        return user is not None

    def update_user_profile(self, username, gross_income, residency, currency, freq):
        conn = self._get_conn()
        conn.execute("UPDATE users SET gross_income = ?, tax_residency = ?, currency = ?, income_freq = ? WHERE username = ?", 
                     (gross_income, residency, currency, freq, username))
        conn.commit()
        conn.close()

    def get_user_profile(self, username):
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else {}

    def add_transactions(self, username, df):
        conn = self._get_conn()
        df = df.copy()
        df['username'] = username
        df['is_vice'] = df['Is_Vice'].astype(int) 
        if not df['Date'].empty:
             df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        for c in ['selected', 'Select']:
            if c in df.columns: df = df.drop(columns=[c])

        df_sql = df.rename(columns={
            "Date": "date", "Vendor": "vendor", "Item": "item", 
            "Amount": "amount", "Currency": "currency", "IVA": "iva",
            "Category": "category", "Sub_Category": "sub_category", 
            "File": "file_name"
        })
        cols = ["username", "date", "vendor", "item", "amount", "currency", "iva", "category", "sub_category", "is_vice", "file_name"]
        for c in cols:
            if c not in df_sql.columns: df_sql[c] = None
            
        df_sql[cols].to_sql("transactions", conn, if_exists="append", index=False)
        conn.close()

    def bulk_update_ids(self, ids, date_val=None, cat_val=None, vend_val=None):
        if not ids: return
        conn = self._get_conn()
        ph = ','.join('?' * len(ids))
        if date_val:
            conn.execute(f"UPDATE transactions SET date = ? WHERE id IN ({ph})", [date_val.strftime('%Y-%m-%d')] + ids)
        if cat_val:
            conn.execute(f"UPDATE transactions SET category = ? WHERE id IN ({ph})", [cat_val] + ids)
        if vend_val:
            conn.execute(f"UPDATE transactions SET vendor = ? WHERE id IN ({ph})", [vend_val] + ids)
        conn.commit()
        conn.close()

    def get_user_data(self, username):
        conn = self._get_conn()
        df = pd.read_sql_query("SELECT id, date, vendor, item, amount, currency, iva, category, sub_category, is_vice, file_name FROM transactions WHERE username = ?", conn, params=(username,))
        conn.close()
        if df.empty: return pd.DataFrame(columns=["id"] + REQUIRED_COLS)
        
        df = df.rename(columns={
            "date": "Date", "vendor": "Vendor", "item": "Item", 
            "amount": "Amount", "currency": "Currency", "iva": "IVA",
            "category": "Category", "sub_category": "Sub_Category", 
            "is_vice": "Is_Vice", "file_name": "File"
        })
        df['Is_Vice'] = df['Is_Vice'].astype(bool)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df

    def clear_user_data(self, username):
        conn = self._get_conn()
        conn.execute("DELETE FROM transactions WHERE username = ?", (username,))
        conn.execute("UPDATE users SET gross_income = 0 WHERE username = ?", (username,))
        conn.commit()
        conn.close()

# --- FINANCIAL LOGIC ---
def calculate_net_monthly(gross, freq, residency):
    annual_gross = gross
    if freq == "Monthly": annual_gross = gross * 12
    elif freq == "Weekly": annual_gross = gross * 52
    elif freq == "Daily": annual_gross = gross * 260

    net_annual = 0.0
    if "UK" in residency:
        allowance = 12570
        if annual_gross > 100000: allowance = max(0, allowance - (annual_gross - 100000) / 2)
        taxable = max(0, annual_gross - allowance)
        tax = 0.0
        if taxable > 0: tax += min(taxable, 37700) * 0.20
        if taxable > 37700: tax += min(taxable - 37700, 125140 - 37700) * 0.40
        if taxable > 125140: tax += (taxable - 125140) * 0.45
        ni = 0.0
        if annual_gross > 12570: ni += min(max(0, annual_gross - 12570), 50270 - 12570) * 0.08
        if annual_gross > 50270: ni += (annual_gross - 50270) * 0.02
        net_annual = annual_gross - tax - ni
    elif "Spain" in residency:
        ss = min(annual_gross, 56646) * 0.0635
        base = annual_gross - ss - 2000
        irpf, prev = 0.0, 0
        bands = [(12450, 0.19), (20200, 0.24), (35200, 0.30), (60000, 0.37), (300000, 0.45)]
        for lim, rate in bands:
            if base > prev: 
                irpf += (min(base, lim) - prev) * rate
                prev = lim
            else: break
        net_annual = annual_gross - ss - irpf
    else:
        net_annual = annual_gross * 0.70 
        
    return net_annual / 12

# --- UTILS ---
REQUIRED_COLS = ["Date", "Vendor", "Item", "Amount", "Currency", "IVA", "Category", "Sub_Category", "Is_Vice", "File"]

def safe_float(val):
    try: return float(val) if val else 0.0
    except: return 0.0

def safe_date(val):
    try:
        s = str(val).strip().lower()
        if s in ['none', 'null', 'nan', '', 'yyyy-mm-dd', 'unknown']: return None
        dt = pd.to_datetime(val, errors='coerce', dayfirst=True)
        if pd.isna(dt): return None
        now = pd.Timestamp.now()
        if dt > now: return None 
        if dt.year < 2020: return None 
        return dt
    except: return None

def clean_json_response(text):
    try:
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```', '', text)
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match: return json.loads(match.group())
        return []
    except: return []

def smart_slice_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return []
    h, w = img.shape[:2]
    if h < 2000 and w < 2000: return [img]
    slices = []
    mid_h, mid_w = h // 2, w // 2
    slices.append(img[0:mid_h+100, 0:mid_w+100])
    slices.append(img[0:mid_h+100, mid_w-100:w])
    slices.append(img[mid_h-100:h, 0:mid_w+100])
    slices.append(img[mid_h-100:h, mid_w-100:w])
    return slices

# --- AI ENGINE ---
class AIEngine:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key) if api_key else None

    def analyze_image_bytes(self, image_bytes, mime_type, user_vices, currency):
        if not self.client: raise ValueError("No API Key")
        
        prompt = f"""
        Role: Senior Forensic Auditor.
        Task: OCR transaction lines.
        JSON: [{{ "rid": 1, "d": "DD/MM/YYYY", "v": "Vendor", "n": "Item Name", "p": 1.00, "c": "£", "mc": "Category", "sc": "SubCategory", "vice": false }}]
        RULES:
        1. EXTRACT EVERY VISIBLE ITEM. Do not summarize.
        2. GROUPING: Assign 'rid' (1, 2, 3...) to group items by physical receipt.
        3. VENDOR: If header missing, use 'CONT'.
        4. CATEGORY: [Groceries, Alcohol, Dining, Transport, Shopping, Utilities, Services].
        5. VICES: {user_vices}.
        6. CURRENCY: {currency}.
        """
        # USE FLASH 2.0 (Newest) or 1.5 PRO
        models_to_try = ["gemini-1.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"]
        for model in models_to_try:
            try:
                res = self.client.models.generate_content(
                    model=model, 
                    contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                    config=types.GenerateContentConfig(response_mime_type='application/json')
                )
                return clean_json_response(res.text)
            except Exception as e:
                if "429" in str(e): 
                    time.sleep(2)
                    continue
                continue 
        raise Exception("All AI models failed.")

    def generate_financial_advice(self, summary_text):
        prompt = f"""
        Act as a professional financial consultant.
        Review this spending data.
        Provide 3 specific, actionable tips to reduce 'Discretionary' (Vices) and Sub-Category spending.
        Be direct.
        DATA: {summary_text}
        """
        # FLASH IS BEST FOR TEXT GENERATION SPEED/RELIABILITY
        models_to_try = ["gemini-1.5-flash", "gemini-1.5-pro"]
        
        for model in models_to_try:
            try:
                res = self.client.models.generate_content(model=model, contents=prompt)
                return res.text
            except Exception as e:
                print(f"Advice Error {model}: {e}")
                continue
        
        # LOCAL FALLBACK IF API FAILS
        return f"""
        **AI Connectivity Issue.**
        Here is a rule-based analysis:
        1. You have spent significantly on discretionary items. Review the 'Vice Hunter' chart.
        2. Check your top Sub-Category. Is it a necessity?
        3. Ensure all receipts are uploaded to get a full picture.
        """

# --- APP LOGIC ---
db = DatabaseManager(DB_PATH)

def main():
    if 'user' not in st.session_state: st.session_state.user = None
    if not st.session_state.user: login_screen()
    else: dashboard_screen()

def login_screen():
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.markdown("<h1 style='text-align: center;'>🔐 Forensix Pro</h1>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Login", "Join"])
        with tab1:
            with st.form("login"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Sign In", use_container_width=True):
                    if db.verify_user(u, p):
                        st.session_state.user = u
                        st.rerun()
                    else: st.error("Invalid")
        with tab2:
            with st.form("signup"):
                u = st.text_input("New Username")
                p = st.text_input("New Password", type="password")
                if st.form_submit_button("Create Account", use_container_width=True):
                    if db.create_user(u, p):
                        st.session_state.user = u
                        st.rerun()
                    else: st.error("Taken")

def dashboard_screen():
    # RELOAD PROFILE
    profile = db.get_user_profile(st.session_state.user)
    home_curr = profile.get('currency', '£')
    user_residency = profile.get('tax_residency', 'UK')
    user_gross = profile.get('gross_income', 0.0)
    user_freq = profile.get('income_freq', 'Yearly')
    
    with st.sidebar:
        st.write(f"👤 **{st.session_state.user}**")
        if st.button("Log Out"):
            st.session_state.user = None
            st.rerun()
        
        st.divider()
        st.subheader("💰 Financial Profile")
        
        with st.form("profile_form", clear_on_submit=False):
            res_options = ["UK (GBP)", "Spain (EUR)", "USA (USD)", "Australia (AUD)", "Canada (CAD)", "Japan (JPY)", "Europe (EUR)"]
            def_idx = 0
            for i, r in enumerate(res_options):
                if user_residency in r: def_idx = i; break
            
            new_residency = st.selectbox("Residency", res_options, index=def_idx)
            
            c_p1, c_p2 = st.columns(2)
            new_freq = c_p1.selectbox("Freq", ["Yearly", "Monthly", "Weekly", "Daily"], index=["Yearly", "Monthly", "Weekly", "Daily"].index(user_freq))
            new_gross = c_p2.number_input("Gross", value=float(user_gross), step=100.0)
            
            if st.form_submit_button("💾 Save Profile", use_container_width=True):
                curr_map = {"GBP": "£", "EUR": "€", "USD": "$", "AUD": "A$", "CAD": "C$", "JPY": "¥"}
                curr_code = new_residency.split("(")[1].replace(")", "")
                symbol = curr_map.get(curr_code, curr_code)
                db.update_user_profile(st.session_state.user, new_gross, new_residency, symbol, new_freq)
                st.success("Saved!")
                time.sleep(0.5); st.rerun()
        
        st.divider()
        # VICES INPUT RESTORED
        vices_input = st.text_area("Vices (Keywords)", "alcohol, candy, betting", help="Separate by comma. The AI looks for these words.")
        
        if st.button("🗑️ Reset All Data"):
            db.clear_user_data(st.session_state.user)
            st.success("All data (including income) cleared.")
            time.sleep(1); st.rerun()

    st.title("📊 Forensic Dashboard")
    
    # RELOAD DATA
    df = db.get_user_data(st.session_state.user)
    
    net_monthly = calculate_net_monthly(user_gross, user_freq, user_residency)
    total_spend = df['Amount'].sum() if not df.empty else 0.0
    vice_spend = df[df['Is_Vice']]['Amount'].sum() if not df.empty else 0.0
    remaining = net_monthly - total_spend
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Monthly Income", f"{home_curr}{net_monthly:,.2f}")
    c2.metric("Total Spend", f"{home_curr}{total_spend:,.2f}")
    c3.metric("Remaining", f"{home_curr}{remaining:,.2f}", delta="Surplus" if remaining > 0 else "Deficit")
    c4.metric("Vice Leakage", f"{home_curr}{vice_spend:,.2f}", delta="Waste", delta_color="inverse")
    
    df_chart = pd.DataFrame()
    if not df.empty:
        today = datetime.now()
        start_date = df['Date'].min() if df['Date'].notna().any() else today
        end_date = df['Date'].max() if df['Date'].notna().any() else today
        
        with st.expander("🔎 Filter & Timeline Control", expanded=True):
            cf1, cf2 = st.columns([3, 1])
            with cf1:
                date_range = st.slider("Time Range", 
                    min_value=df['Date'].min().date() if df['Date'].notna().any() else today.date(), 
                    max_value=df['Date'].max().date() if df['Date'].notna().any() else today.date(),
                    value=(start_date.date(), end_date.date())
                )
            with cf2:
                st.write("") 
                if st.button("Reset to All Time", use_container_width=True):
                    date_range = (df['Date'].min().date(), df['Date'].max().date())
        
        mask = (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
        df_chart = df.loc[mask].copy()

    tab_dash, tab_upload, tab_ledger = st.tabs(["📈 Analytics", "📤 Upload", "📝 Ledger"])
    
    with tab_dash:
        if not df_chart.empty:
            col_advice, col_btn = st.columns([4, 1])
            with col_btn:
                if st.button("💡 Get AI Insight"):
                    with st.spinner("Analyzing..."):
                        summary = df_chart.groupby('Category')['Amount'].sum().to_string()
                        sub_summary = df_chart.groupby('Sub_Category')['Amount'].sum().sort_values(ascending=False).head(5).to_string()
                        vice_sum = df_chart[df_chart['Is_Vice']]['Amount'].sum()
                        context = f"Income: {net_monthly}\nTotal: {df_chart['Amount'].sum()}\nVices: {vice_sum}\nBreakdown:\n{summary}\nItems:\n{sub_summary}"
                        
                        api_key = st.secrets["GEMINI_API_KEY"]
                        st.session_state.ai_advice = AIEngine(api_key).generate_financial_advice(context)
            with col_advice:
                if 'ai_advice' in st.session_state: st.info(st.session_state.ai_advice)

            st.divider()
            
            # --- CHARTS ---
            domain = ["Groceries", "Alcohol", "Dining", "Transport", "Shopping", "Utilities", "Services", "Unknown"]
            range_ = ["#2ecc71", "#9b59b6", "#e67e22", "#3498db", "#f1c40f", "#95a5a6", "#34495e", "#bdc3c7"]
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Category Breakdown")
                # Remove Nulls for cleaner chart
                clean_chart_df = df_chart[df_chart['Category'].notna() & (df_chart['Category'] != '')]
                chart = alt.Chart(clean_chart_df).mark_bar().encode(
                    x=alt.X('Category', sort='-y'), y='Amount', color=alt.Color('Category', scale=alt.Scale(domain=domain, range=range_))
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            with c2:
                st.subheader("Top Sub-Categories")
                df_deep = df_chart.groupby('Sub_Category')['Amount'].sum().reset_index().sort_values('Amount', ascending=False).head(10)
                deep_chart = alt.Chart(df_deep).mark_bar().encode(
                    y=alt.Y('Sub_Category', sort='-x'), x='Amount', color=alt.value('#3498db'), tooltip=['Sub_Category', 'Amount']
                ).interactive()
                st.altair_chart(deep_chart, use_container_width=True)

            st.subheader("Vice Hunter")
            vice_df = df_chart[df_chart['Is_Vice'] == True]
            if not vice_df.empty:
                base = alt.Chart(vice_df).encode(theta=alt.Theta("Amount", stack=True))
                pie = base.mark_arc(outerRadius=100).encode(
                    color=alt.Color("Sub_Category"), 
                    tooltip=["Sub_Category", "Amount"]
                )
                st.altair_chart(pie, use_container_width=True)
            else:
                st.success("No Vices Found! (Or none scanned yet)")

    with tab_upload:
        uploaded = st.file_uploader("Upload Receipts", accept_multiple_files=True)
        if uploaded and st.button("🔍 Run Forensic Audit", type="primary"):
            api_key = st.secrets["GEMINI_API_KEY"]
            ai = AIEngine(api_key)
            all_rows = []
            status = st.status("Forensic Audit Running...", expanded=True)
            prog = st.progress(0)
            
            for i, f in enumerate(uploaded):
                tpath = DIRS["TEMP"] / f"{uuid.uuid4()}_{f.name}"
                try:
                    status.write(f"**File {i+1}:** `{f.name}`")
                    with open(tpath, "wb") as file: file.write(f.getbuffer())
                    
                    items = []
                    if f.type != "application/pdf":
                        status.write("  ↳ Quad Scan (High Res)...")
                        slices = smart_slice_image(tpath)
                        for idx, s in enumerate(slices):
                            _, buf = cv2.imencode(".jpg", s)
                            try: items.extend(ai.analyze_image_bytes(buf.tobytes(), "image/jpeg", vices_input, home_curr))
                            except: continue
                    else:
                        with open(tpath, "rb") as pdf_file:
                            items.extend(ai.analyze_image_bytes(pdf_file.read(), "application/pdf", vices_input, home_curr))
                    
                    if not items: continue

                    receipt_groups = {}
                    for item in items:
                        rid = item.get('rid', 0)
                        if rid not in receipt_groups: receipt_groups[rid] = []
                        receipt_groups[rid].append(item)
                    
                    for rid, group_items in receipt_groups.items():
                        group_date = None
                        for item in group_items:
                            d = safe_date(item.get("d"))
                            if d: group_date = d; break
                        
                        for item in group_items:
                            name = str(item.get("n", "Item"))
                            if any(x in name.lower() for x in ["total", "subtotal", "balance"]): continue
                            row_date = safe_date(item.get("d"))
                            if not row_date: row_date = group_date
                            price = safe_float(item.get("p"))
                            cat = item.get("mc", "Shopping")
                            if price > 100 and cat in ["Groceries", "Dining"]: price /= 100.0
                            is_vice = bool(item.get("vice", False))
                            
                            all_rows.append({
                                "Date": row_date, "Vendor": item.get("v", "Unknown"), "Item": name, "Amount": price, "Currency": home_curr, "IVA": 0.0, "Category": cat, "Sub_Category": item.get("sc", "General"), "Is_Vice": is_vice, "File": f.name
                            })
                except Exception as e:
                    status.error(f"Error: {e}")
                finally:
                    if os.path.exists(tpath): os.remove(tpath)
                    prog.progress((i+1)/len(uploaded))
            
            status.update(label="Complete!", state="complete", expanded=False)
            if all_rows:
                db.add_transactions(st.session_state.user, pd.DataFrame(all_rows))
                st.success("Audit Saved!")
                time.sleep(1); st.rerun()

    with tab_ledger:
        if not df.empty:
            # --- SUPERCHARGED FILTER BAR (EXCEL STYLE) ---
            with st.container(border=True):
                st.markdown("#### 🌪️ Excel Filters")
                col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                
                # 1. TEXT SEARCH (Global)
                search_term = col_f1.text_input("Search Text", placeholder="Search anything...")
                
                # 2. VENDOR (Dropdown)
                all_vendors = ["All"] + sorted(list(df['Vendor'].dropna().unique()))
                sel_vend = col_f2.multiselect("Vendor", all_vendors, default=[])
                
                # 3. CATEGORY (Dropdown)
                all_cats = ["All"] + sorted(list(df['Category'].dropna().unique()))
                sel_cat = col_f3.multiselect("Category", all_cats, default=[])
                
                # 4. SORT BY
                sort_col = col_f4.selectbox("Sort By", ["Date (Newest)", "Date (Oldest)", "Amount (High)", "Amount (Low)", "Vendor (A-Z)"])
            
            # APPLY FILTERS
            df_view = df.copy()
            if search_term:
                df_view = df_view[df_view.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)]
            if sel_vend and "All" not in sel_vend:
                df_view = df_view[df_view['Vendor'].isin(sel_vend)]
            if sel_cat and "All" not in sel_cat:
                df_view = df_view[df_view['Category'].isin(sel_cat)]
                
            # APPLY SORT
            if "Newest" in sort_col: df_view = df_view.sort_values("Date", ascending=False)
            elif "Oldest" in sort_col: df_view = df_view.sort_values("Date", ascending=True)
            elif "High" in sort_col: df_view = df_view.sort_values("Amount", ascending=False)
            elif "Low" in sort_col: df_view = df_view.sort_values("Amount", ascending=True)
            elif "Vendor" in sort_col: df_view = df_view.sort_values("Vendor", ascending=True)
            
            # Select Column
            df_view.insert(0, "Select", False)
            
            # RENDER GRID
            edited_df = st.data_editor(
                df_view, 
                num_rows="dynamic", 
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Select": st.column_config.CheckboxColumn("✅", width="small"),
                    "id": None,
                    "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
                }
            )
            
            # BULK ACTIONS
            selected_ids = edited_df[edited_df['Select'] == True]['id'].astype(str).tolist()
            if selected_ids:
                st.success(f"⚡ {len(selected_ids)} rows selected")
                b1, b2, b3 = st.columns(3)
                with b1:
                    bd = st.date_input("Bulk Date", value=None)
                    if st.button("Apply Date"):
                        db.bulk_update_ids(selected_ids, date_val=bd)
                        st.rerun()
                with b2:
                    bc = st.selectbox("Bulk Category", ["Groceries", "Alcohol", "Dining", "Transport", "Shopping"])
                    if st.button("Apply Category"):
                        db.bulk_update_ids(selected_ids, cat_val=bc)
                        st.rerun()
                with b3:
                    if st.button("🗑️ Delete Rows"):
                        conn = db._get_conn()
                        ph = ','.join('?' * len(selected_ids))
                        conn.execute(f"DELETE FROM transactions WHERE id IN ({ph})", selected_ids)
                        conn.commit(); conn.close()
                        st.rerun()

            if st.button("💾 Save Grid Changes"):
                conn = db._get_conn()
                for i, row in edited_df.iterrows():
                    d_val = row['Date'].strftime('%Y-%m-%d') if pd.notnull(row['Date']) else None
                    conn.execute("""
                        UPDATE transactions 
                        SET date=?, vendor=?, item=?, amount=?, category=?, sub_category=?, is_vice=?
                        WHERE id=?
                    """, (d_val, row['Vendor'], row['Item'], row['Amount'], row['Category'], row['Sub_Category'], int(row['Is_Vice']), row['id']))
                conn.commit(); conn.close()
                st.success("Saved!")
                time.sleep(0.5); st.rerun()

if __name__ == "__main__":
    main()
