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
        dt = pd.to_datetime(val, errors='coerce', dayfirst=True)
        now = pd.Timestamp.now()
        if pd.isna(dt): return now
        if dt > now: return now
        if dt.year < 2020: return now
        return dt
    except: return pd.Timestamp.now()

def resize_image_force(image_path):
    try:
        with Image.open(image_path) as img:
            # 3500 is a safe limit for 1.5 Pro to avoid payload errors
            if img.width > 3500 or img.height > 3500:
                img.thumbnail((3500, 3500), Image.Resampling.LANCZOS)
                img.save(image_path, optimize=True, quality=80)
    except: pass

def clean_json_response(text):
    """
    Cleans AI output to ensure valid JSON.
    Removes markdown backticks and hunts for the [...] array.
    """
    try:
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```', '', text)
        # Find the first '[' and last ']'
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
    except:
        return []

# --- AI ENGINE ---
class AIEngine:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key) if api_key else None

    def analyze_image(self, image_path, mime_type, user_vices, currency):
        if not self.client: 
            raise ValueError("No API Key provided")
        
        with open(image_path, "rb") as f: content = f.read()

        prompt = f"""
        Role: Senior Forensic Auditor.
        Task: OCR and categorize transaction lines from this image.
        
        JSON STRUCTURE: [{{ "d": "DD/MM/YYYY", "v": "Vendor", "n": "Item Name", "p": 1.00, "c": "£", "mc": "Category", "sc": "SubCategory", "vice": false }}]
        
        RULES:
        1. **SCAN ALL:** Do not stop. Extract every visible line item.
        2. **NO TOTALS:** Ignore lines that are totals, subtotals, or card info.
        3. **VENDOR:** Identify the vendor. If it's a collage, identify the vendor for *each* item based on its receipt header.
        4. **CATEGORY:** [Groceries, Alcohol, Dining, Transport, Shopping, Utilities, Services].
        5. **VICES:** Check for: {user_vices}.
        6. **CURRENCY:** Default {currency}.
        """
        
        try:
            # SWITCHED TO PRO MODEL (Smarter, handles dense text better)
            res = self.client.models.generate_content(
                model="gemini-1.5-pro", 
                contents=[prompt, types.Part.from_bytes(data=content, mime_type=mime_type)],
                config=types.GenerateContentConfig(response_mime_type='application/json')
            )
            return clean_json_response(res.text)
        except Exception as e:
            # Re-raise the error so the UI can show it
            raise e 

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
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Sign In", use_container_width=True):
                    if db.verify_user(username, password):
                        st.session_state.user = username
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("signup_form"):
                new_user = st.text_input("New Username")
                new_pass = st.text_input("New Password", type="password")
                if st.form_submit_button("Create Account", use_container_width=True):
                    if db.create_user(new_user, new_pass):
                        st.success("Created! Please Login.")
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
    
    # METRICS
    if not df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Spend", f"{home_curr}{df['Amount'].sum():.2f}")
        c2.metric("Vice Leakage", f"{home_curr}{df[df['Is_Vice']]['Amount'].sum():.2f}")
        c3.metric("Transactions", len(df))
    
    # TABS
    tab_dash, tab_upload, tab_ledger = st.tabs(["📈 Analytics", "📤 Upload Receipts", "📝 Data Ledger"])
    
    with tab_upload:
        uploaded = st.file_uploader("Upload Receipts (Image/PDF)", accept_multiple_files=True)
        if uploaded and st.button("🔍 Run Forensic Audit", type="primary"):
            
            # --- API KEY CHECK ---
            if "GEMINI_API_KEY" not in st.secrets:
                st.error("🚨 Missing API Key. Please configure .streamlit/secrets.toml")
                st.stop()
                
            api_key = st.secrets["GEMINI_API_KEY"]
            ai = AIEngine(api_key)
            all_rows = []
            
            status_box = st.status("Initializing Forensic Engine...", expanded=True)
            progress_bar = st.progress(0)
            
            for i, f in enumerate(uploaded):
                tpath = DIRS["TEMP"] / f"{uuid.uuid4()}_{f.name}"
                try:
                    status_box.write(f"**Processing File {i+1}/{len(uploaded)}:** `{f.name}`")
                    
                    with open(tpath, "wb") as file: file.write(f.getbuffer())
                    
                    if f.type != "application/pdf":
                        status_box.write(f"  ↳ Optimizing image resolution...")
                        resize_image_force(tpath)
                    
                    status_box.write(f"  ↳ Sending to Gemini 1.5 Pro (High-Intelligence Mode)...")
                    mime = "application/pdf" if f.type == "application/pdf" else "image/jpeg"
                    
                    # CALL AI WITH ERROR CATCHING
                    items = ai.analyze_image(tpath, mime, vices, home_curr)
                    
                    if not items:
                        status_box.warning(f"  ⚠️ No items found in {f.name}. Moving to next.")
                        continue
                        
                    status_box.write(f"  ✅ Found {len(items)} items. Cleaning data...")
                    
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
                    status_box.error(f"❌ Critical Error on {f.name}: {str(e)}")
                
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
            else:
                st.warning("Audit finished but no transactions were saved. Check the status log above for errors.")

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
    
