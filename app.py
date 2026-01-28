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
from datetime import datetime
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(page_title="Forensix Pro", page_icon="🕵️‍♂️", layout="wide")
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

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
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

    def add_transactions(self, username, df):
        conn = self._get_conn()
        df = df.copy()
        df['username'] = username
        df['is_vice'] = df['Is_Vice'].astype(int) 
        if not df['Date'].empty:
             df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        df_sql = df.rename(columns={
            "Date": "date", "Vendor": "vendor", "Item": "item", 
            "Amount": "amount", "Currency": "currency", "IVA": "iva",
            "Category": "category", "Sub_Category": "sub_category", 
            "File": "file_name"
        })
        cols = ["username", "date", "vendor", "item", "amount", "currency", "iva", "category", "sub_category", "is_vice", "file_name"]
        df_sql[cols].to_sql("transactions", conn, if_exists="append", index=False)
        conn.close()

    def update_transaction_field(self, username, field, new_value, filter_col=None, filter_val=None):
        """Surgical Bulk Update"""
        conn = self._get_conn()
        query = f"UPDATE transactions SET {field} = ? WHERE username = ?"
        params = [new_value, username]
        
        if filter_col and filter_val:
            query += f" AND {filter_col} = ?"
            params.append(filter_val)
            
        conn.execute(query, params)
        conn.commit()
        conn.close()

    def get_user_data(self, username):
        conn = self._get_conn()
        df = pd.read_sql_query("SELECT * FROM transactions WHERE username = ?", conn, params=(username,))
        conn.close()
        if df.empty: return pd.DataFrame(columns=REQUIRED_COLS)
        
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
        conn.commit()
        conn.close()

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
        
        # IMPROVED PROMPT: ASKS FOR RECEIPT GROUPING ID
        prompt = f"""
        Role: Senior Forensic Auditor.
        Task: OCR transaction lines.
        JSON: [{{ "rid": 1, "d": "DD/MM/YYYY", "v": "Vendor", "n": "Item Name", "p": 1.00, "c": "£", "mc": "Category", "sc": "SubCategory", "vice": false }}]
        RULES:
        1. **GROUPING:** Assign a 'rid' (Receipt ID: 1, 2, 3) to items. Items from the same physical receipt MUST have the same 'rid'.
        2. SCAN ALL visible items.
        3. VENDOR: If header missing, use 'CONT'.
        4. CATEGORY: [Groceries, Alcohol, Dining, Transport, Shopping, Utilities, Services].
        5. VICES: {user_vices}.
        6. CURRENCY: {currency}.
        """
        
        models_to_try = ["gemini-1.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"]
        for model in models_to_try:
            try:
                res = self.client.models.generate_content(
                    model=model, 
                    contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                    config=types.GenerateContentConfig(response_mime_type='application/json')
                )
                return clean_json_response(res.text)
            except: continue
        raise Exception("All AI models failed.")

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
    df = db.get_user_data(st.session_state.user)
    
    with st.sidebar:
        st.write(f"👤 **{st.session_state.user}**")
        if st.button("Log Out"):
            st.session_state.user = None
            st.rerun()
        
        st.divider()
        
        # --- SMART FIXER (BULK EDIT) ---
        st.subheader("🛠️ Smart Fixer")
        
        # 1. Vendor Cleaner
        vendors = list(df['Vendor'].unique()) if not df.empty else []
        target_v = st.selectbox("Fix items from Vendor:", ["(Select Vendor)"] + vendors)
        
        if target_v != "(Select Vendor)":
            # Action A: Rename Vendor
            new_name = st.text_input("Rename Vendor to:", value=target_v)
            if new_name != target_v and st.button("Apply Rename"):
                db.update_transaction_field(st.session_state.user, "vendor", new_name, "vendor", target_v)
                st.success("Renamed!")
                time.sleep(0.5); st.rerun()
            
            # Action B: Bulk Date Set
            new_date = st.date_input("Set Date for this Vendor:", value=None)
            if st.button(f"Apply Date to all {target_v}"):
                db.update_transaction_field(st.session_state.user, "date", new_date.strftime('%Y-%m-%d'), "vendor", target_v)
                st.success("Dates Updated!")
                time.sleep(0.5); st.rerun()

        st.divider()
        residency = st.selectbox("Residency", ["UK (GBP)", "Spain (EUR)"])
        home_curr = "£" if "UK" in residency else "€"
        vices = st.text_area("Vices", "alcohol, candy, betting")
        if st.button("🗑️ Reset Data"):
            db.clear_user_data(st.session_state.user)
            st.rerun()

    st.title("📊 Forensic Dashboard")
    
    # METRICS
    if not df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Spend", f"{home_curr}{df['Amount'].sum():.2f}")
        c2.metric("Vice Leakage", f"{home_curr}{df[df['Is_Vice']]['Amount'].sum():.2f}")
        missing = df['Date'].isna().sum()
        c3.metric("Action Needed", f"{missing} Missing Dates", delta="Fix in Sidebar" if missing > 0 else "Clean", delta_color="inverse")
    
    tab_dash, tab_upload, tab_ledger = st.tabs(["📈 Analytics", "📤 Upload", "📝 Ledger"])
    
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
                        for s in slices:
                            _, buf = cv2.imencode(".jpg", s)
                            items.extend(ai.analyze_image_bytes(buf.tobytes(), "image/jpeg", vices, home_curr))
                    else:
                        with open(tpath, "rb") as pdf_file:
                            items.extend(ai.analyze_image_bytes(pdf_file.read(), "application/pdf", vices, home_curr))
                    
                    # SMART GROUPING LOGIC
                    # 1. Group items by 'rid' (Receipt ID) returned by AI
                    # 2. Find date for that group
                    # 3. Apply to siblings
                    
                    # Organize by RID
                    receipt_groups = {}
                    for item in items:
                        rid = item.get('rid', 0)
                        if rid not in receipt_groups: receipt_groups[rid] = []
                        receipt_groups[rid].append(item)
                        
                    # Process Groups
                    for rid, group_items in receipt_groups.items():
                        # Find group date
                        group_date = None
                        for item in group_items:
                            d = safe_date(item.get("d"))
                            if d: 
                                group_date = d
                                break
                        
                        # Apply
                        for item in group_items:
                            name = str(item.get("n", "Item"))
                            if any(x in name.lower() for x in ["total", "subtotal", "balance"]): continue
                            
                            row_date = safe_date(item.get("d"))
                            if not row_date: row_date = group_date
                            
                            price = safe_float(item.get("p"))
                            cat = item.get("mc", "Shopping")
                            if price > 100 and cat in ["Groceries", "Dining"]: price /= 100.0
                            
                            all_rows.append({
                                "Date": row_date,
                                "Vendor": item.get("v", "Unknown"),
                                "Item": name,
                                "Amount": price,
                                "Currency": home_curr,
                                "IVA": 0.0,
                                "Category": cat,
                                "Sub_Category": item.get("sc", "General"),
                                "Is_Vice": bool(item.get("vice", False)),
                                "File": f.name
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

    with tab_dash:
        if not df.empty:
            domain = ["Groceries", "Alcohol", "Dining", "Transport", "Shopping", "Utilities", "Services", "Unknown"]
            range_ = ["#2ecc71", "#9b59b6", "#e67e22", "#3498db", "#f1c40f", "#95a5a6", "#34495e", "#bdc3c7"]
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("💸 Categories")
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('Category', sort='-y'), y='Amount', 
                    color=alt.Color('Category', scale=alt.Scale(domain=domain, range=range_)),
                    tooltip=['Category', 'Amount']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            
            with c2:
                # SUB-CATEGORY DEEP DIVE (RESTORED)
                st.subheader("🔬 Top Items (Deep Dive)")
                df_deep = df.groupby('Sub_Category')['Amount'].sum().reset_index().sort_values('Amount', ascending=False).head(10)
                deep_chart = alt.Chart(df_deep).mark_bar().encode(
                    y=alt.Y('Sub_Category', sort='-x'),
                    x='Amount',
                    color=alt.value('#3498db'),
                    tooltip=['Sub_Category', 'Amount']
                )
                st.altair_chart(deep_chart, use_container_width=True)

            st.subheader("📈 Spending Timeline")
            df_valid = df.dropna(subset=['Date'])
            if not df_valid.empty:
                line = alt.Chart(df_valid).mark_line(point=True).encode(
                    x='Date', y='sum(Amount)', color='Category', tooltip=['Date', 'sum(Amount)']
                ).interactive()
                st.altair_chart(line, use_container_width=True)

    with tab_ledger:
        if not df.empty:
            st.info("💡 Use 'Smart Fixer' in sidebar to bulk-edit Dates or Rename Vendors.")
            edited_df = st.data_editor(
                df, 
                num_rows="dynamic", 
                use_container_width=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=False),
                    "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
                    "Category": st.column_config.SelectboxColumn("Category", options=["Groceries", "Alcohol", "Dining", "Transport", "Shopping", "Utilities", "Services"])
                }
            )
            if st.button("💾 Save Changes"):
                db.clear_user_data(st.session_state.user)
                db.add_transactions(st.session_state.user, edited_df)
                st.success("Saved!")
                st.rerun()

if __name__ == "__main__":
    main()

