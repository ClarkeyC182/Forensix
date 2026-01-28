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
from fpdf import FPDF
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

# --- CORE: DATABASE MANAGER ---
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
        df['username'] = username
        df['is_vice'] = df['Is_Vice'].astype(int) 
        df_sql = df.rename(columns={
            "Date": "date", "Vendor": "vendor", "Item": "item", 
            "Amount": "amount", "Currency": "currency", "IVA": "iva",
            "Category": "category", "Sub_Category": "sub_category", 
            "File": "file_name"
        })
        cols = ["username", "date", "vendor", "item", "amount", "currency", "iva", "category", "sub_category", "is_vice", "file_name"]
        df_sql[cols].to_sql("transactions", conn, if_exists="append", index=False)
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
        df['Date'] = pd.to_datetime(df['Date'])
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
        if s in ['none', 'null', 'nan', '', 'yyyy-mm-dd', 'unknown']: return pd.Timestamp.now()
        dt = pd.to_datetime(val, errors='coerce', dayfirst=True)
        if pd.isna(dt): return pd.Timestamp.now()
        now = pd.Timestamp.now()
        if dt > now: return now 
        if dt.year < 2020: return now
        return dt
    except: return pd.Timestamp.now()

def resize_image_force(image_path):
    try:
        with Image.open(image_path) as img:
            if img.width > 3500 or img.height > 3500:
                img.thumbnail((3500, 3500), Image.Resampling.LANCZOS)
                img.save(image_path, optimize=True, quality=80)
    except: pass

def clean_json_response(text):
    try:
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```', '', text)
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match: return json.loads(match.group())
        return []
    except: return []

# --- AI ENGINE ---
class AIEngine:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key) if api_key else None

    def analyze_image(self, image_path, mime_type, user_vices, currency):
        if not self.client: raise ValueError("No API Key")
        with open(image_path, "rb") as f: content = f.read()

        prompt = f"""
        Role: Senior Forensic Auditor.
        Task: OCR and categorize transaction lines.
        JSON STRUCTURE: [{{ "d": "DD/MM/YYYY", "v": "Vendor", "n": "Item Name", "p": 1.00, "c": "£", "mc": "Category", "sc": "SubCategory", "vice": false }}]
        RULES:
        1. SCAN ALL. No Totals.
        2. VENDOR: Header or 'CONT'.
        3. CATEGORY: [Groceries, Alcohol, Dining, Transport, Shopping, Utilities, Services].
        4. VICES: {user_vices}.
        5. CURRENCY: {currency}.
        """
        
        # SMART FALLBACK SYSTEM
        # It tries models in order. If one 404s/fails, it tries the next.
        models_to_try = ["gemini-1.5-pro-latest", "gemini-1.5-flash", "gemini-2.0-flash-exp"]
        
        last_error = None
        for model_name in models_to_try:
            try:
                res = self.client.models.generate_content(
                    model=model_name, 
                    contents=[prompt, types.Part.from_bytes(data=content, mime_type=mime_type)],
                    config=types.GenerateContentConfig(response_mime_type='application/json')
                )
                return clean_json_response(res.text)
            except Exception as e:
                last_error = e
                print(f"Model {model_name} failed: {e}")
                continue # Try next model
        
        # If all fail
        raise last_error

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
            with st.form("login_form", clear_on_submit=True):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Sign In", use_container_width=True):
                    if db.verify_user(username, password):
                        st.session_state.user = username
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("signup_form", clear_on_submit=True):
                new_user = st.text_input("New Username")
                new_pass = st.text_input("New Password", type="password")
                if st.form_submit_button("Create Account", use_container_width=True):
                    if db.create_user(new_user, new_pass):
                        # AUTO-LOGIN LOGIC
                        st.session_state.user = new_user
                        st.success("Account created! Logging in...")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Username taken.")

def dashboard_screen():
    with st.sidebar:
        st.write(f"👤 **{st.session_state.user}**")
        if st.button("Log Out"):
            st.session_state.user = None
            st.rerun()
        st.divider()
        st.subheader("⚙️ Settings")
        residency = st.selectbox("Residency", ["UK (GBP)", "Spain (EUR)"])
        home_curr = "£" if "UK" in residency else "€"
        vices = st.text_area("Vices", "alcohol, candy, betting")
        
        st.divider()
        if st.button("🗑️ Clear My Data"):
            db.clear_user_data(st.session_state.user)
            st.rerun()

    st.title("📊 Forensic Dashboard")
    df = db.get_user_data(st.session_state.user)
    
    if not df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Spend", f"{home_curr}{df['Amount'].sum():.2f}")
        c2.metric("Vice Leakage", f"{home_curr}{df[df['Is_Vice']]['Amount'].sum():.2f}")
        c3.metric("Transactions", len(df))
    
    tab_dash, tab_upload, tab_ledger = st.tabs(["📈 Analytics", "📤 Upload Receipts", "📝 Data Ledger"])
    
    with tab_upload:
        uploaded = st.file_uploader("Upload Receipts (Image/PDF)", accept_multiple_files=True)
        if uploaded and st.button("🔍 Run Forensic Audit", type="primary"):
            api_key = st.secrets["GEMINI_API_KEY"]
            ai = AIEngine(api_key)
            all_rows = []
            
            status_box = st.status("Initializing Forensic Engine...", expanded=True)
            progress_bar = st.progress(0)
            
            for i, f in enumerate(uploaded):
                tpath = DIRS["TEMP"] / f"{uuid.uuid4()}_{f.name}"
                try:
                    status_box.write(f"**Processing {i+1}/{len(uploaded)}:** `{f.name}`")
                    with open(tpath, "wb") as file: file.write(f.getbuffer())
                    
                    if f.type != "application/pdf": resize_image_force(tpath)
                    
                    status_box.write(f"  ↳ Scanning... (Auto-trying Models)")
                    mime = "application/pdf" if f.type == "application/pdf" else "image/jpeg"
                    
                    items = ai.analyze_image(tpath, mime, vices, home_curr)
                    
                    if not items:
                        status_box.warning(f"  ⚠️ No items found in {f.name}.")
                        continue
                        
                    status_box.write(f"  ✅ Extracted {len(items)} items.")
                    
                    for item in items:
                        name = str(item.get("n", "Item"))
                        blacklist = ["total", "subtotal", "balance", "change", "cash", "due", "visa", "auth", "item", "desc"]
                        if any(x in name.lower() for x in blacklist): continue
                        
                        price = safe_float(item.get("p"))
                        cat = item.get("mc", "Shopping")
                        if price > 100 and cat in ["Groceries", "Dining"]: price /= 100.0
                        
                        all_rows.append({
                            "Date": safe_date(item.get("d")),
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
                    status_box.error(f"❌ Error on {f.name}: {str(e)}")
                
                finally:
                    if os.path.exists(tpath): os.remove(tpath)
                    progress_bar.progress((i + 1) / len(uploaded))
            
            status_box.update(label="Audit Complete!", state="complete", expanded=False)
            
            if all_rows:
                new_df = pd.DataFrame(all_rows)
                db.add_transactions(st.session_state.user, new_df)
                st.success(f"Successfully audited {len(new_df)} transactions.")
                time.sleep(1)
                st.rerun()

    with tab_dash:
        if not df.empty:
            domain = ["Groceries", "Alcohol", "Dining", "Transport", "Shopping", "Utilities", "Services", "Unknown"]
            range_ = ["#2ecc71", "#9b59b6", "#e67e22", "#3498db", "#f1c40f", "#95a5a6", "#34495e", "#bdc3c7"]
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Spending by Category")
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('Category', sort='-y'),
                    y='Amount',
                    color=alt.Color('Category', scale=alt.Scale(domain=domain, range=range_)),
                    tooltip=['Category', 'Amount']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            
            with c2:
                st.subheader("Timeline")
                line = alt.Chart(df).mark_line(point=True).encode(
                    x='Date',
                    y='sum(Amount)',
                    color='Category',
                    tooltip=['Date', 'sum(Amount)']
                ).interactive()
                st.altair_chart(line, use_container_width=True)

    with tab_ledger:
        if not df.empty:
            edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
            if st.button("💾 Save Changes"):
                db.clear_user_data(st.session_state.user)
                db.add_transactions(st.session_state.user, edited_df)
                st.success("Saved!")
                st.rerun()

if __name__ == "__main__":
    main()
    
