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
        df = df.copy()
        df['username'] = username
        df['is_vice'] = df['Is_Vice'].astype(int) 
        
        # Save dates as string YYYY-MM-DD or NULL
        # This preserves "None" so we know it's missing
        df['Date'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None)

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
        
        # Load dates, keep NaT for missing ones so we can flag them
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
    """
    Returns a Timestamp if valid, or None (NaT) if invalid/missing.
    No more guessing 'Today'.
    """
    try:
        s = str(val).strip().lower()
        if s in ['none', 'null', 'nan', '', 'yyyy-mm-dd', 'unknown']: return None
        
        dt = pd.to_datetime(val, errors='coerce', dayfirst=True)
        if pd.isna(dt): return None
        
        now = pd.Timestamp.now()
        if dt > now: return None # Future date = Invalid
        if dt.year < 2020: return None # Too old = Invalid
        
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

# --- VISION ENGINE (QUAD-SCAN) ---
def smart_slice_image(image_path):
    """
    For massive images, split into 4 overlapping quadrants.
    This keeps the text HUGE so the AI can actually read it.
    """
    img = cv2.imread(str(image_path))
    if img is None: return []
    h, w = img.shape[:2]
    
    # If small enough, just return original
    if h < 2000 and w < 2000: return [img]
    
    slices = []
    # Midpoints
    mid_h, mid_w = h // 2, w // 2
    
    # Add overlapping buffer (100px) so we don't cut text in half
    # Top-Left
    slices.append(img[0:mid_h+100, 0:mid_w+100])
    # Top-Right
    slices.append(img[0:mid_h+100, mid_w-100:w])
    # Bottom-Left
    slices.append(img[mid_h-100:h, 0:mid_w+100])
    # Bottom-Right
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
        Task: OCR transaction lines from this receipt section.
        JSON: [{{ "d": "DD/MM/YYYY", "v": "Vendor", "n": "Item Name", "p": 1.00, "c": "£", "mc": "Category", "sc": "SubCategory", "vice": false }}]
        RULES:
        1. SCAN ALL visible items.
        2. IGNORE totals/subtotals.
        3. VENDOR: If header missing, use 'CONT'.
        4. CATEGORY: [Groceries, Alcohol, Dining, Transport, Shopping, Utilities, Services].
        5. VICES: {user_vices}.
        6. CURRENCY: {currency}.
        """
        
        models_to_try = ["gemini-1.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"]
        
        last_error = None
        for model in models_to_try:
            try:
                res = self.client.models.generate_content(
                    model=model, 
                    contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                    config=types.GenerateContentConfig(response_mime_type='application/json')
                )
                return clean_json_response(res.text)
            except Exception as e:
                last_error = e
                continue
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
    
    # METRICS ROW
    if not df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Spend", f"{home_curr}{df['Amount'].sum():.2f}")
        c2.metric("Vice Leakage", f"{home_curr}{df[df['Is_Vice']]['Amount'].sum():.2f}")
        
        # Missing Date Alert
        missing_dates = df['Date'].isna().sum()
        c3.metric("Action Needed", f"{missing_dates} Items", delta="Missing Dates" if missing_dates > 0 else "All Good", delta_color="inverse")
    
    tab_dash, tab_upload, tab_ledger = st.tabs(["📈 Executive Analytics", "📤 Upload Receipts", "📝 Data Ledger"])
    
    with tab_upload:
        uploaded = st.file_uploader("Upload Receipts", accept_multiple_files=True)
        if uploaded and st.button("🔍 Run Forensic Audit", type="primary"):
            
            if "GEMINI_API_KEY" not in st.secrets:
                st.error("🚨 Missing API Key")
                st.stop()
                
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
                    
                    items = []
                    # QUAD-SCAN LOGIC
                    if f.type != "application/pdf":
                        # Split image into tiles for high-res scanning
                        status_box.write(f"  ↳ High-Res Quad Scan (Splitting Image)...")
                        slices = smart_slice_image(tpath)
                        for idx, s in enumerate(slices):
                            _, buf = cv2.imencode(".jpg", s)
                            status_box.write(f"    - Scanning Quadrant {idx+1}/{len(slices)}...")
                            chunk_items = ai.analyze_image_bytes(buf.tobytes(), "image/jpeg", vices, home_curr)
                            items.extend(chunk_items)
                    else:
                        # PDF (Send whole)
                        status_box.write(f"  ↳ Analyzing PDF...")
                        with open(tpath, "rb") as pdf_file:
                            items = ai.analyze_image_bytes(pdf_file.read(), "application/pdf", vices, home_curr)
                    
                    status_box.write(f"  ✅ Extracted {len(items)} items.")
                    
                    # Deduplicate logic could go here, but for now simple append
                    for item in items:
                        name = str(item.get("n", "Item"))
                        blacklist = ["total", "subtotal", "balance", "change", "cash", "visa", "auth", "item", "desc"]
                        if any(x in name.lower() for x in blacklist): continue
                        
                        price = safe_float(item.get("p"))
                        cat = item.get("mc", "Shopping")
                        if price > 100 and cat in ["Groceries", "Dining"]: price /= 100.0
                        
                        all_rows.append({
                            "Date": safe_date(item.get("d")), # Returns None if missing
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
            # Filter Logic
            min_dt = df['Date'].min().date() if df['Date'].notna().any() else datetime.now().date()
            max_dt = df['Date'].max().date() if df['Date'].notna().any() else datetime.now().date()
            if min_dt >= max_dt: min_dt = max_dt - pd.Timedelta(days=1)
            
            c_filter1, c_filter2 = st.columns(2)
            date_range = c_filter1.slider("Filter Date", min_value=min_dt, max_value=max_dt, value=(min_dt, max_dt))
            all_cats = list(df['Category'].unique())
            selected_cats = c_filter2.multiselect("Filter Category", all_cats, default=all_cats)
            
            # Apply Filters
            mask = (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
            # Include rows with missing dates in the filter? Usually no, or provide toggle.
            # For now, let's include valid dates only in timeline, but show all in totals
            
            df_filtered = df.loc[mask] if selected_cats else df
            if selected_cats: df_filtered = df_filtered[df_filtered['Category'].isin(selected_cats)]

            # CHARTS
            domain = ["Groceries", "Alcohol", "Dining", "Transport", "Shopping", "Utilities", "Services", "Unknown"]
            range_ = ["#2ecc71", "#9b59b6", "#e67e22", "#3498db", "#f1c40f", "#95a5a6", "#34495e", "#bdc3c7"]
            
            c1, c2 = st.columns(2)
            with c1:
                if len(selected_cats) == 1:
                    st.subheader(f"🔬 Breakdown: {selected_cats[0]}")
                    chart = alt.Chart(df_filtered).mark_bar().encode(
                        x=alt.X('Sub_Category', sort='-y'), y='Amount', color=alt.Color('Sub_Category', legend=None), tooltip=['Item', 'Amount']
                    ).interactive()
                else:
                    st.subheader("💸 Spending by Category")
                    chart = alt.Chart(df_filtered).mark_bar().encode(
                        x=alt.X('Category', sort='-y', axis=alt.Axis(labelAngle=-45)), 
                        y='Amount', 
                        color=alt.Color('Category', scale=alt.Scale(domain=domain, range=range_)),
                        tooltip=['Category', 'Amount']
                    ).interactive()
                st.altair_chart(chart, use_container_width=True)
            
            with c2:
                st.subheader("😈 Vice Meter")
                base = alt.Chart(df_filtered).encode(theta=alt.Theta("Amount", stack=True))
                pie = base.mark_arc(outerRadius=120).encode(
                    color=alt.Color("Is_Vice", scale=alt.Scale(domain=[True, False], range=["#e74c3c", "#ecf0f1"])),
                    order=alt.Order("Amount", sort="descending"),
                    tooltip=["Is_Vice", "Amount"]
                )
                text = base.mark_text(radius=140).encode(text=alt.Text("Amount", format=".1f"), order=alt.Order("Amount", sort="descending"), color=alt.value("black"))
                st.altair_chart(pie + text, use_container_width=True)
            
            st.subheader("📈 Financial Heartbeat")
            # Only chart rows with valid dates
            df_time = df_filtered.dropna(subset=['Date'])
            if not df_time.empty:
                base_line = alt.Chart(df_time).encode(x='Date')
                area = base_line.mark_area(opacity=0.3).encode(y='sum(Amount)', color=alt.Color('Category', scale=alt.Scale(domain=domain, range=range_)))
                line = base_line.mark_line().encode(y='sum(Amount)', color=alt.Color('Category', scale=alt.Scale(domain=domain, range=range_)), tooltip=['Date', 'Category', 'sum(Amount)'])
                st.altair_chart((area + line).interactive(), use_container_width=True)

    with tab_ledger:
        if not df.empty:
            # Show "Missing Dates" at top
            st.info("💡 Pro Tip: Items with missing dates (NaT) will not appear in the timeline. Edit them below.")
            
            edited_df = st.data_editor(
                df, 
                num_rows="dynamic", 
                use_container_width=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=False),
                    "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
                    "Is_Vice": st.column_config.CheckboxColumn("Vice?", default=False),
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
    
