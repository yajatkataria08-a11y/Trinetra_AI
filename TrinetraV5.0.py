import os
import json
import faiss
import torch
import tempfile
import librosa
import numpy as np
import streamlit as st
import re
import time
import threading
import pandas as pd
import plotly.express as px
import sqlite3
import hashlib
import logging
import smtplib
import secrets
import string
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from PIL import Image
from deep_translator import GoogleTranslator
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, ClapModel
from werkzeug.security import generate_password_hash, check_password_hash

# ==================== CONFIGURATION ==================== #
class Config:
    MAX_FILE_SIZE       = 100 * 1024 * 1024
    ALLOWED_IMAGE_EXTS  = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    ALLOWED_AUDIO_EXTS  = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    SUPPORTED_LANGUAGES = ['en', 'hi', 'ta', 'te', 'kn', 'ml', 'bn', 'mr', 'gu', 'pa']
    AUDIO_DURATION_S    = 7.0
    EMBEDDING_DIM       = 512
    AUDIO_SAMPLE_RATE   = 48_000
    AUDIO_SILENCE_RMS   = 0.001
    CONFIDENCE_HIGH     = 0.65
    CONFIDENCE_MED      = 0.45
    DUPLICATE_THRESHOLD = 0.95
    CACHE_SIZE          = 1000

CONFIG      = Config()
BASE_DIR    = "trinetra_registry"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(BASE_DIR,    exist_ok=True)

# ==================== LOGGING ==================== #
# FIX: Logging set up early so it's available everywhere
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f'logs/trinetra_{datetime.now():%Y%m%d}.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Trinetra')

# ==================== THREAD-SAFE DATABASE CONNECTION ==================== #
class DatabaseConnection:
    """
    FIX: True thread-safe SQLite connection using threading.local().
    Previously used a plain instance attribute (self.local = None) which
    was shared across all threads — not thread-local at all.
    """
    def __init__(self, db_path):
        self.db_path = db_path
        self._local  = threading.local()   # FIX: actual thread-local storage

    def get_connection(self):
        """Get a genuine per-thread database connection."""
        if not hasattr(self._local, 'conn'):
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,   # safe — each thread owns its own conn
                timeout=30.0               # FIX: longer timeout before giving up
            )
            # FIX: WAL mode allows concurrent reads + one writer without blocking
            # This is the root fix for "database is locked" on Streamlit Cloud
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=30000;")  # 30s busy wait in ms
            conn.commit()
            self._local.conn = conn
        return self._local.conn

    def execute(self, query, params=None):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor
        except sqlite3.OperationalError as e:
            conn.rollback()
            logger.error(f"DB execute error: {e}", exc_info=True)  # FIX: log instead of silent fail
            raise e

    def fetchall(self, query, params=None):
        cursor = self.execute(query, params)
        return cursor.fetchall()

    def fetchone(self, query, params=None):
        cursor = self.execute(query, params)
        return cursor.fetchone()

    def close(self):
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn


# ==================== EMAIL OTP SENDER ==================== #
class EmailOTPSender:
    """Send OTP emails for registration verification"""

    def __init__(self, smtp_server="smtp.gmail.com", smtp_port=587):
        self.smtp_server    = smtp_server
        self.smtp_port      = smtp_port
        self.sender_email   = os.getenv("SMTP_EMAIL",    "noreply@trinetra.local")
        self.sender_password = os.getenv("SMTP_PASSWORD", "password")

    def generate_otp(self, length=6):
        return ''.join(secrets.choice(string.digits) for _ in range(length))

    def send_otp_email(self, recipient_email, otp, username):
        try:
            subject   = "Trinetra Registration - OTP Verification"
            html_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px;">
                    <div style="max-width: 500px; margin: 0 auto; background-color: white; padding: 30px;
                                border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                        <h2 style="color: #e8a020; text-align: center;">TRINETRA V5.0</h2>
                        <h3 style="color: #333;">Welcome, {username}!</h3>
                        <p style="color: #666; font-size: 16px;">Your One-Time Password (OTP) for registration is:</p>
                        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 5px;
                                    text-align: center; margin: 20px 0;">
                            <h1 style="color: #e8a020; letter-spacing: 5px; margin: 0;">{otp}</h1>
                        </div>
                        <p style="color: #666;"><strong>Important:</strong> This OTP is valid for 10 minutes only.</p>
                        <p style="color: #666; font-size: 14px;">If you didn't request this, please ignore this email.</p>
                        <hr style="border: none; border-top: 1px solid #ddd; margin: 30px 0;">
                        <p style="color: #999; font-size: 12px; text-align: center;">
                            Team Human | Created with ❤️ for Bharat's Digital Future
                        </p>
                    </div>
                </body>
            </html>
            """
            message             = MIMEMultipart("alternative")
            message["Subject"]  = subject
            message["From"]     = self.sender_email
            message["To"]       = recipient_email
            message.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipient_email, message.as_string())

            logger.info(f"OTP email sent to {recipient_email}")  # FIX: log success
            return True, "OTP sent successfully"
        except Exception as e:
            logger.warning(f"Email service unavailable for {recipient_email}: {e}")  # FIX: log warning
            return False, "Email service unavailable. Using demo mode. OTP will be shown on screen."


# ==================== ENHANCED AUTHENTICATION WITH OTP ==================== #
class AuthManagerWithOTP:
    """Enhanced authentication with email OTP verification"""

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join("trinetra_registry", "users.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db         = DatabaseConnection(db_path)
        self.otp_sender = EmailOTPSender()
        self._create_tables()
        self._create_default_admin()

    def _create_tables(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username          TEXT PRIMARY KEY,
                email             TEXT UNIQUE NOT NULL,
                password_hash     TEXT NOT NULL,
                full_name         TEXT,
                role              TEXT NOT NULL,
                is_verified       INTEGER DEFAULT 0,
                created_at        TEXT NOT NULL,
                last_login        TEXT,
                registration_date TEXT
            )
        """)
        # FIX: removed plaintext `otp` column — only store the hash
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS registration_requests (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT UNIQUE NOT NULL,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name     TEXT,
                otp_hash      TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                expires_at    TEXT NOT NULL,
                is_verified   INTEGER DEFAULT 0
            )
        """)

    def _create_default_admin(self):
        try:
            admin_hash = self.hash_password("admin123")
            self.db.execute("""
                INSERT INTO users
                (username, email, password_hash, full_name, role, is_verified, created_at, registration_date)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """, ("admin", "admin@trinetra.local", admin_hash, "Administrator", "admin", 1))
        except sqlite3.IntegrityError:
            pass  # admin already exists — expected on every restart after first run
        except sqlite3.OperationalError as e:
            # FIX: "database is locked" on Streamlit Cloud cold start — safe to ignore
            # WAL mode + busy_timeout above prevents this, but belt-and-suspenders
            logger.warning(f"_create_default_admin skipped (db busy at startup): {e}")

    # FIX: Replaced bare sha256 with werkzeug PBKDF2 (salted, iterated)
    def hash_password(self, password):
        return generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

    def verify_password(self, stored_hash, password):
        return check_password_hash(stored_hash, password)

    def hash_otp(self, otp):
        return hashlib.sha256(otp.encode()).hexdigest()

    def request_registration(self, email, username, password, full_name=""):
        if not email or '@' not in email:
            return False, "Invalid email address", None
        if len(username) < 3:
            return False, "Username must be at least 3 characters", None
        if len(password) < 6:
            return False, "Password must be at least 6 characters", None

        result = self.db.fetchone(
            "SELECT username FROM registration_requests WHERE email = ? OR username = ?",
            (email, username)
        )
        if result:
            return False, "Email or username already registered", None

        result = self.db.fetchone(
            "SELECT username FROM users WHERE email = ? OR username = ?",
            (email, username)
        )
        if result:
            return False, "Email or username already in use", None

        otp      = self.otp_sender.generate_otp()
        otp_hash = self.hash_otp(otp)

        success, message = self.otp_sender.send_otp_email(email, otp, username)

        try:
            password_hash = self.hash_password(password)
            # FIX: only otp_hash stored — no plaintext otp column
            self.db.execute("""
                INSERT INTO registration_requests
                (email, username, password_hash, full_name, otp_hash, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now', '+10 minutes'))
            """, (email, username, password_hash, full_name, otp_hash))

            logger.info(f"REGISTER_REQUEST user={username} email={email}")  # FIX: log it

            # FIX: only expose raw OTP in DEBUG mode via console, never in prod UI
            demo_otp = otp if os.getenv("DEBUG_MODE", "1") == "1" else None
            return True, f"OTP sent to {email}. Valid for 10 minutes.", demo_otp
        except Exception as e:
            logger.error(f"Registration request failed for {email}: {e}", exc_info=True)
            return False, f"Registration request failed: {str(e)}", None

    def verify_otp_and_register(self, email, otp):
        otp_hash = self.hash_otp(otp)
        result   = self.db.fetchone("""
            SELECT username, password_hash, full_name, expires_at
            FROM registration_requests
            WHERE email = ? AND otp_hash = ?
        """, (email, otp_hash))

        if not result:
            logger.warning(f"OTP_FAIL email={email} reason=invalid_otp")  # FIX: log failure
            return False, "Invalid OTP"

        username, password_hash, full_name, expires_at = result

        expiry = datetime.fromisoformat(expires_at)
        if datetime.now() > expiry:
            self.db.execute("DELETE FROM registration_requests WHERE email = ?", (email,))
            logger.warning(f"OTP_FAIL email={email} reason=expired")  # FIX: log expiry
            return False, "OTP expired. Please register again."

        try:
            self.db.execute("""
                INSERT INTO users
                (username, email, password_hash, full_name, role, is_verified, created_at, registration_date)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """, (username, email, password_hash, full_name, "viewer", 1))

            self.db.execute("DELETE FROM registration_requests WHERE email = ?", (email,))
            logger.info(f"REGISTER_SUCCESS user={username} email={email}")  # FIX: log success
            return True, "Registration successful! You can now login."
        except Exception as e:
            logger.error(f"Registration DB error for {email}: {e}", exc_info=True)
            return False, f"Registration failed: {str(e)}"

    def resend_otp(self, email):
        result = self.db.fetchone(
            "SELECT username FROM registration_requests WHERE email = ?", (email,)
        )
        if not result:
            return False, "No pending registration found for this email"

        username = result[0]
        otp      = self.otp_sender.generate_otp()
        otp_hash = self.hash_otp(otp)

        self.otp_sender.send_otp_email(email, otp, username)

        try:
            # FIX: update only otp_hash — no plaintext otp column
            self.db.execute("""
                UPDATE registration_requests
                SET otp_hash = ?, created_at = datetime('now'), expires_at = datetime('now', '+10 minutes')
                WHERE email = ?
            """, (otp_hash, email))

            logger.info(f"OTP_RESEND email={email}")  # FIX: log it

            demo_otp = otp if os.getenv("DEBUG_MODE", "1") == "1" else None
            return True, "OTP resent successfully", demo_otp
        except Exception as e:
            logger.error(f"Resend OTP failed for {email}: {e}", exc_info=True)
            return False, f"Failed to resend OTP: {str(e)}", None

    def verify_user(self, username, password):
        result = self.db.fetchone("""
            SELECT username, password_hash, role, is_verified FROM users
            WHERE username = ?
        """, (username,))

        if not result:
            logger.warning(f"LOGIN_FAIL user={username} reason=not_found")  # FIX: log
            return None

        db_username, stored_hash, role, is_verified = result

        # FIX: use werkzeug check_password_hash instead of comparing raw sha256
        if not self.verify_password(stored_hash, password):
            logger.warning(f"LOGIN_FAIL user={username} reason=bad_password")  # FIX: log
            return None

        if not is_verified:
            logger.warning(f"LOGIN_FAIL user={username} reason=not_verified")  # FIX: log
            return None

        self.db.execute(
            "UPDATE users SET last_login = datetime('now') WHERE username = ?", (username,)
        )
        logger.info(f"LOGIN_SUCCESS user={username} role={role}")  # FIX: log success
        return {"username": db_username, "role": role}

    def get_all_users(self):
        return self.db.fetchall("""
            SELECT username, email, role, created_at, last_login FROM users
            WHERE is_verified = 1
        """)

    # FIX: Added missing create_user method (Admin tab called this but it didn't exist)
    def create_user(self, username, password, role, email=""):
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
        pwd_hash = self.hash_password(password)
        try:
            self.db.execute("""
                INSERT INTO users
                (username, email, password_hash, role, is_verified, created_at, registration_date)
                VALUES (?, ?, ?, ?, 1, datetime('now'), datetime('now'))
            """, (username, email or f"{username}@admin.local", pwd_hash, role))
            logger.info(f"USER_CREATED user={username} role={role} by=admin")  # FIX: log
            return True, f"User '{username}' created successfully"
        except sqlite3.IntegrityError:
            return False, "Username or email already exists"


# ==================== PAGE CONFIG ==================== #
st.set_page_config(
    page_title="Trinetra V5.0 · Bharat AI",
    layout="wide",
    page_icon="👁️",
    initial_sidebar_state="expanded",
)

# ==================== SESSION STATE ==================== #
if "theme"               not in st.session_state: st.session_state.theme = "dark"
if "search_history"      not in st.session_state: st.session_state.search_history = []
if "last_search_results" not in st.session_state: st.session_state.last_search_results = []
if "authenticated"       not in st.session_state: st.session_state.authenticated = False
if "user"                not in st.session_state: st.session_state.user = None
if "show_suggestions"    not in st.session_state: st.session_state.show_suggestions = True
if "auth_stage"          not in st.session_state: st.session_state.auth_stage = "login"
if "temp_email"          not in st.session_state: st.session_state.temp_email = ""
if "temp_username"       not in st.session_state: st.session_state.temp_username = ""
if "demo_otp"            not in st.session_state: st.session_state.demo_otp = None

is_light = st.session_state.theme == "light"

# ==================== THEME TOKENS ==================== #
DARK = dict(
    bg_base="#080c14", bg_panel="#0d1321", bg_card="#111827", bg_card_hover="#162036",
    border="#1e2d45", border_hover="#e8a020", accent="#e8a020", accent_dim="#b87818",
    accent_glow="rgba(232,160,32,0.18)", text_primary="#e8eaf0", text_muted="#6b7a99",
    text_mono="#a8bbd4", shadow_card="0 4px 24px rgba(0,0,0,0.45)",
    shadow_hover="0 8px 32px rgba(232,160,32,0.22)", grid_line="rgba(232,160,32,0.03)",
    h1_grad="linear-gradient(135deg,#e8a020 0%,#fff5d6 55%,#b87818 100%)",
    plot_bg="#080c14", plot_grid="#1e2d45", plot_text="#e8eaf0",
    icon="🌙", mode_label="Light Mode",
)
LIGHT = dict(
    bg_base="#f0f2f8", bg_panel="#ffffff", bg_card="#ffffff", bg_card_hover="#f7f8fc",
    border="#dde1ee", border_hover="#b87818", accent="#b87818", accent_dim="#8c5c10",
    accent_glow="rgba(184,120,24,0.14)", text_primary="#111827", text_muted="#6b7080",
    text_mono="#374151", shadow_card="0 2px 12px rgba(0,0,0,0.08)",
    shadow_hover="0 6px 24px rgba(184,120,24,0.18)", grid_line="rgba(184,120,24,0.04)",
    h1_grad="linear-gradient(135deg,#b87818 0%,#5c3a00 55%,#8c5c10 100%)",
    plot_bg="#f7f8fc", plot_grid="#dde1ee", plot_text="#111827",
    icon="☀️", mode_label="Dark Mode",
)
T = LIGHT if is_light else DARK

# ==================== CSS ==================== #
# FIX: wrapped with st.cache_data so it doesn't regenerate on every single rerun
@st.cache_data(ttl=None)
def _css(theme: str):
    is_light = theme == "light"
    T = LIGHT if is_light else DARK
    knob_left = "24px" if is_light else "3px"
    return (
        "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">"
        "<link href=\"https://fonts.googleapis.com/css2?family=Syne:wght@700;800"
        "&family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500"
        "&display=swap\" rel=\"stylesheet\">"
        "<style>"
        ":root { --t: 0.25s; }"
        "html,body,[data-testid=\"stAppViewContainer\"],[data-testid=\"stApp\"],.stApp {"
        "background-color:" + T["bg_base"] + " !important;"
        "color:" + T["text_primary"] + " !important;"
        "font-family:'DM Sans',sans-serif !important;"
        "transition:background-color var(--t),color var(--t); }"
        "[data-testid=\"stAppViewContainer\"]::before {"
        "content:''; position:fixed; inset:0;"
        "background-image:linear-gradient(" + T["grid_line"] + " 1px,transparent 1px),"
        "linear-gradient(90deg," + T["grid_line"] + " 1px,transparent 1px);"
        "background-size:40px 40px; pointer-events:none; z-index:0; }"
        "[data-testid=\"stMainBlockContainer\"] { padding:2rem 2.5rem; position:relative; z-index:1; }"
        "[data-testid=\"stSidebar\"] { background:" + T["bg_panel"] + " !important; border-right:1px solid " + T["border"] + " !important; }"
        "[data-testid=\"stSidebar\"] * { color:" + T["text_primary"] + " !important; }"
        "h1 { font-family:'Syne',sans-serif !important; font-weight:800 !important; font-size:2.6rem !important;"
        "background:" + T["h1_grad"] + "; -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }"
        "h2,h3,h4 { font-family:'Syne',sans-serif !important; font-weight:700 !important; color:" + T["text_primary"] + " !important; }"
        ".tagline { font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:" + T["accent"] + ";"
        "letter-spacing:0.12em; text-transform:uppercase; margin-bottom:2rem; opacity:0.85; }"
        ".stat-strip { display:flex; gap:1rem; margin-bottom:2rem; }"
        ".stat-chip { flex:1; background:" + T["bg_card"] + "; border:1px solid " + T["border"] + ";"
        "border-top:2px solid " + T["accent"] + "; border-radius:10px; padding:0.9rem 1.2rem;"
        "display:flex; flex-direction:column; gap:4px; cursor:default; transition:all var(--t); box-shadow:" + T["shadow_card"] + "; }"
        ".stat-chip:hover { background:" + T["bg_card_hover"] + "; border-color:" + T["border_hover"] + ";"
        "box-shadow:" + T["shadow_hover"] + "; transform:translateY(-3px); }"
        ".stat-label { font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:" + T["text_muted"] + "; text-transform:uppercase; letter-spacing:0.1em; }"
        ".stat-value { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:600; color:" + T["accent"] + "; }"
        "[data-testid=\"stButton\"] button { background:transparent !important; border:1px solid " + T["accent"] + " !important;"
        "color:" + T["accent"] + " !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; border-radius:6px !important; transition:all 0.2s !important; }"
        "[data-testid=\"stButton\"] button:hover { background:" + T["accent_glow"] + " !important; transform:translateY(-2px); box-shadow:0 0 16px " + T["accent_glow"] + " !important; }"
        "[data-testid=\"stButton\"] button:active { transform:translateY(0); }"
        "[data-testid=\"stTextInput\"] input { background:" + T["bg_card"] + " !important; border:1px solid " + T["border"] + " !important;"
        "border-radius:6px !important; color:" + T["text_primary"] + " !important; font-family:'DM Sans',sans-serif !important; transition:border-color var(--t),box-shadow var(--t) !important; }"
        "[data-testid=\"stTextInput\"] input:focus { border-color:" + T["accent"] + " !important; box-shadow:0 0 0 3px " + T["accent_glow"] + " !important; }"
        "[data-testid=\"stSelectbox\"] > div > div { background:" + T["bg_card"] + " !important; border:1px solid " + T["border"] + " !important; color:" + T["text_primary"] + " !important; }"
        "[data-testid=\"stSelectbox\"] > div > div:hover { border-color:" + T["accent"] + " !important; }"
        "[data-testid=\"stFileUploader\"] { background:" + T["bg_card"] + " !important; border:1px dashed " + T["accent"] + " !important; border-radius:8px !important; }"
        "[data-testid=\"stFileUploader\"]:hover { background:" + T["bg_card_hover"] + " !important; box-shadow:0 0 14px " + T["accent_glow"] + " !important; }"
        "[data-testid=\"stTabs\"] [role=\"tablist\"] { border-bottom:1px solid " + T["border"] + " !important; }"
        "[data-testid=\"stTabs\"] [role=\"tab\"] { font-family:'Syne',sans-serif !important; font-size:.85rem !important; font-weight:700 !important; color:" + T["text_muted"] + " !important; background:transparent !important; border:none !important; padding:.6rem 1.2rem !important; border-radius:6px 6px 0 0 !important; transition:color var(--t),background var(--t) !important; }"
        "[data-testid=\"stTabs\"] [role=\"tab\"]:hover { color:" + T["accent"] + " !important; background:rgba(232,160,32,0.07) !important; }"
        "[data-testid=\"stTabs\"] [role=\"tab\"][aria-selected=\"true\"] { color:" + T["accent"] + " !important; border-bottom:2px solid " + T["accent"] + " !important; }"
        "[data-testid=\"stDataFrame\"] { border:1px solid " + T["border"] + " !important; border-radius:8px !important; overflow:hidden; box-shadow:" + T["shadow_card"] + "; }"
        "[data-testid=\"stDataFrame\"]:hover { border-color:" + T["border_hover"] + " !important; }"
        "[data-testid=\"stDataFrame\"] th { background:" + T["bg_panel"] + " !important; font-family:'JetBrains Mono',monospace !important; font-size:.72rem !important; color:" + T["accent"] + " !important; text-transform:uppercase; }"
        "[data-testid=\"stDataFrame\"] td { font-family:'JetBrains Mono',monospace !important; font-size:.78rem !important; color:" + T["text_mono"] + " !important; background:" + T["bg_card"] + " !important; }"
        "[data-testid=\"stDataFrame\"] tr:hover td { background:" + T["bg_card_hover"] + " !important; }"
        "[data-testid=\"stMetric\"] { background:" + T["bg_card"] + " !important; border:1px solid " + T["border"] + " !important; border-radius:10px !important; padding:.9rem 1rem !important; box-shadow:" + T["shadow_card"] + "; cursor:default; transition:border-color var(--t),box-shadow var(--t),transform var(--t) !important; }"
        "[data-testid=\"stMetric\"]:hover { border-color:" + T["border_hover"] + " !important; box-shadow:" + T["shadow_hover"] + " !important; transform:translateY(-2px); }"
        "[data-testid=\"stMetricLabel\"] { font-family:'JetBrains Mono',monospace !important; font-size:.68rem !important; color:" + T["text_muted"] + " !important; text-transform:uppercase; }"
        "[data-testid=\"stMetricValue\"] { font-family:'JetBrains Mono',monospace !important; color:" + T["accent"] + " !important; }"
        "[data-testid=\"stExpander\"] { background:" + T["bg_card"] + " !important; border:1px solid " + T["border"] + " !important; border-radius:8px !important; }"
        "[data-testid=\"stExpander\"]:hover { border-color:" + T["border_hover"] + " !important; box-shadow:0 2px 12px " + T["accent_glow"] + " !important; }"
        "[data-testid=\"stRadio\"] label { color:" + T["text_primary"] + " !important; }"
        "[data-testid=\"stRadio\"] label:hover { color:" + T["accent"] + " !important; }"
        "[data-testid=\"stAlert\"] { background:" + T["bg_card"] + " !important; border:1px solid " + T["border"] + " !important; border-left:3px solid #00c9b1 !important; border-radius:6px !important; color:" + T["text_primary"] + " !important; }"
        ".badge { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:600; text-transform:uppercase; padding:3px 10px; border-radius:4px; margin-bottom:6px; cursor:default; transition:transform .15s,box-shadow .15s; }"
        ".badge:hover { transform:scale(1.07); box-shadow:0 2px 8px rgba(0,0,0,.2); }"
        ".badge-high   { color:#2ecc71; border:1px solid #2ecc71; background:rgba(46,204,113,0.08); }"
        ".badge-medium { color:#f39c12; border:1px solid #f39c12; background:rgba(243,156,18,0.08); }"
        ".badge-low    { color:#e74c3c; border:1px solid #e74c3c; background:rgba(231,76,60,0.08); }"
        ".score-bar-wrap { margin:6px 0 10px; }"
        ".score-bar-track { height:3px; background:" + T["border"] + "; border-radius:2px; overflow:hidden; }"
        ".score-bar-fill { height:100%; border-radius:2px; background:linear-gradient(90deg," + T["accent_dim"] + "," + T["accent"] + "); animation:barGrow 0.6s cubic-bezier(0.34,1.56,0.64,1) forwards; }"
        "@keyframes barGrow { from { width:0%; } }"
        ".result-card { background:" + T["bg_card"] + "; border:1px solid " + T["border"] + "; border-radius:10px; padding:1rem; cursor:default; transition:all var(--t); box-shadow:" + T["shadow_card"] + "; }"
        ".result-card:hover { background:" + T["bg_card_hover"] + "; border-color:" + T["border_hover"] + "; box-shadow:" + T["shadow_hover"] + "; transform:translateY(-4px) scale(1.01); }"
        ".sidebar-stat { font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:" + T["text_muted"] + "; padding:5px 0; cursor:default; transition:all 0.15s; }"
        ".sidebar-stat:hover { color:" + T["text_primary"] + " !important; transform:translateX(4px); }"
        ".sidebar-stat span { color:" + T["accent"] + "; font-weight:600; }"
        ".tog-pill { display:flex; align-items:center; gap:10px; background:" + T["bg_card"] + "; border:1px solid " + T["accent"] + "; border-radius:50px; padding:8px 14px; margin-bottom:0.75rem; cursor:pointer; user-select:none; transition:box-shadow var(--t),background var(--t); }"
        ".tog-pill:hover { box-shadow:0 0 14px " + T["accent_glow"] + "; background:" + T["bg_card_hover"] + "; }"
        ".tog-track { position:relative; width:44px; height:22px; border-radius:11px; background:" + T["border"] + "; border:1px solid " + T["border"] + "; flex-shrink:0; }"
        ".tog-knob { position:absolute; top:2px; left:" + knob_left + "; width:16px; height:16px; border-radius:50%; background:" + T["accent"] + "; box-shadow:0 1px 4px rgba(0,0,0,0.3); transition:left 0.3s cubic-bezier(0.34,1.56,0.64,1); }"
        ".tog-icon { font-size:1rem; line-height:1; }"
        ".tog-label { font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:600; color:" + T["accent"] + "; text-transform:uppercase; letter-spacing:0.08em; flex:1; }"
        ".rule { border:none; border-top:1px solid " + T["border"] + "; margin:1.5rem 0; }"
        ".login-container { max-width:420px; margin:10rem auto; padding:3rem; background:" + T["bg_card"] + "; border:1px solid " + T["border"] + "; border-radius:16px; box-shadow:" + T["shadow_card"] + "; }"
        ".login-logo { text-align:center; font-size:3.5rem; margin-bottom:1rem; }"
        ".login-title { font-family:'Syne',sans-serif; font-weight:800; font-size:2rem; text-align:center; background:" + T["h1_grad"] + "; -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:0.5rem; }"
        ".login-subtitle { font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:" + T["text_muted"] + "; text-align:center; text-transform:uppercase; letter-spacing:0.15em; margin-bottom:2.5rem; }"
        ".quality-indicator { display:inline-flex; align-items:center; gap:6px; font-family:'JetBrains Mono',monospace; font-size:0.7rem; padding:4px 10px; border-radius:4px; background:" + T["bg_card_hover"] + "; border:1px solid " + T["border"] + "; }"
        ".suggestion-chip { display:inline-block; padding:4px 12px; margin:4px; background:" + T["bg_card"] + "; border:1px solid " + T["border"] + "; border-radius:16px; font-size:0.75rem; cursor:pointer; transition:all 0.2s; }"
        ".suggestion-chip:hover { background:" + T["accent_glow"] + "; border-color:" + T["accent"] + "; transform:translateY(-2px); }"
        "@keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }"
        ".a1 { animation:fadeUp 0.4s ease both; }"
        ".a2 { animation:fadeUp 0.4s 0.08s ease both; }"
        ".a3 { animation:fadeUp 0.4s 0.16s ease both; }"
        "</style>"
    )

st.markdown(_css(st.session_state.theme), unsafe_allow_html=True)

# ==================== AUTHENTICATION ==================== #
# FIX: Wrapped in @st.cache_resource so AuthManagerWithOTP (and its DB connections)
# are created ONCE per server lifetime, not on every Streamlit rerun.
# This was the root cause of "database is locked" — bare module-level instantiation
# meant multiple threads all ran _create_default_admin simultaneously on cold start.
@st.cache_resource
def get_auth_manager():
    return AuthManagerWithOTP()

auth_manager = get_auth_manager()

def show_enhanced_login_page(auth_manager):
    st.markdown("""
    <div class="login-container">
        <div class="login-logo">👁️</div>
        <div class="login-title">TRINETRA</div>
        <div class="login-subtitle">Multimodal Neural Registry</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if st.session_state.auth_stage == "login":
                st.markdown("### Sign In")
                username = st.text_input("Username", placeholder="Enter username", key="login_user")
                password = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("Login", use_container_width=True):
                        if username and password:
                            user = auth_manager.verify_user(username, password)
                            if user:
                                st.session_state.authenticated = True
                                st.session_state.user = user
                                st.success(f"Welcome, {user['username']}!")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error("Invalid credentials or email not verified")
                        else:
                            st.warning("Please enter both username and password")

                with col_b:
                    if st.button("Register", use_container_width=True):
                        st.session_state.auth_stage = "register_step1"
                        st.rerun()

                with col_c:
                    if st.button("Guest", use_container_width=True):
                        st.session_state.authenticated = True
                        st.session_state.user = {"username": "guest", "role": "viewer"}
                        logger.info("LOGIN_GUEST")  # FIX: log guest access
                        st.rerun()

                st.markdown("---")
                st.caption("📧 Demo: admin / admin123")
                st.caption("❤️ Created by Team Human")

            elif st.session_state.auth_stage == "register_step1":
                st.markdown("### Create Account")
                email            = st.text_input("Email Address", placeholder="your.email@example.com", key="reg_email")
                username         = st.text_input("Username", placeholder="3+ characters", key="reg_user")
                password         = st.text_input("Password", type="password", placeholder="6+ characters", key="reg_pass")
                password_confirm = st.text_input("Confirm Password", type="password", key="reg_pass_conf")
                full_name        = st.text_input("Full Name (optional)", key="reg_name")

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Send OTP", use_container_width=True):
                        if not email or '@' not in email:
                            st.error("Invalid email address")
                        elif len(username) < 3:
                            st.error("Username must be at least 3 characters")
                        elif len(password) < 6:
                            st.error("Password must be at least 6 characters")
                        elif password != password_confirm:
                            st.error("Passwords don't match")
                        else:
                            success, message, otp = auth_manager.request_registration(
                                email, username, password, full_name
                            )
                            if success:
                                st.session_state.auth_stage  = "register_step2"
                                st.session_state.temp_email  = email
                                st.session_state.temp_username = username
                                st.session_state.demo_otp    = otp   # None in prod, value in DEBUG
                                st.success(message)
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(message)

                with col_b:
                    if st.button("Back to Login", use_container_width=True):
                        st.session_state.auth_stage = "login"
                        st.rerun()

            elif st.session_state.auth_stage == "register_step2":
                st.markdown("### Verify Email")
                st.info(f"📧 OTP sent to: {st.session_state.temp_email}")

                # FIX: Only show demo OTP when DEBUG_MODE=1 (set by dev), never in prod
                if st.session_state.demo_otp and os.getenv("DEBUG_MODE", "1") == "1":
                    st.warning(f"⏱️ Demo OTP: **{st.session_state.demo_otp}** (debug mode only — remove in prod)")

                otp = st.text_input("Enter 6-digit OTP", placeholder="000000", max_chars=6, key="otp_input")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("Verify OTP", use_container_width=True):
                        if len(otp) != 6:
                            st.error("OTP must be 6 digits")
                        else:
                            success, message = auth_manager.verify_otp_and_register(
                                st.session_state.temp_email, otp
                            )
                            if success:
                                st.success(message)
                                st.session_state.auth_stage = "login"
                                st.session_state.demo_otp   = None
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error(message)

                with col_b:
                    if st.button("Resend OTP", use_container_width=True):
                        result = auth_manager.resend_otp(st.session_state.temp_email)
                        # FIX: resend_otp now returns (success, message, demo_otp)
                        if len(result) == 3:
                            success, message, new_otp = result
                        else:
                            success, message = result; new_otp = None
                        if success:
                            st.session_state.demo_otp = new_otp
                            st.success(message)
                        else:
                            st.error(message)

                with col_c:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state.auth_stage = "login"
                        st.session_state.demo_otp   = None
                        st.rerun()

if not st.session_state.authenticated:
    show_enhanced_login_page(auth_manager)
    st.stop()


# ==================== METADATA DATABASE ==================== #
class MetadataDB:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(BASE_DIR, "metadata.db")
        self.db = DatabaseConnection(db_path)
        self._create_tables()

    def _create_tables(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY, modality TEXT NOT NULL, language TEXT NOT NULL,
                file_path TEXT NOT NULL, file_size INTEGER, upload_date TEXT NOT NULL,
                faiss_index INTEGER NOT NULL, tags TEXT, description TEXT, collection TEXT,
                quality_score REAL, uploaded_by TEXT,
                UNIQUE(id)
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL, user TEXT NOT NULL,
                comment TEXT NOT NULL, timestamp TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                asset_id TEXT NOT NULL, user TEXT NOT NULL,
                rating INTEGER NOT NULL, PRIMARY KEY (asset_id, user),
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            )
        """)

    def add_asset(self, asset_id, modality, language, file_path, file_size,
                  faiss_idx, tags=None, description="", collection="",
                  quality_score=None, uploaded_by="unknown"):
        try:
            self.db.execute("""
                INSERT INTO assets
                (id, modality, language, file_path, file_size, upload_date,
                 faiss_index, tags, description, collection, quality_score, uploaded_by)
                VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?)
            """, (asset_id, modality, language, file_path, file_size,
                  faiss_idx, json.dumps(tags or []), description, collection, quality_score, uploaded_by))
            logger.info(f"ASSET_ADDED id={asset_id} modality={modality} by={uploaded_by}")  # FIX: log
        except sqlite3.IntegrityError:
            pass

    def search_metadata(self, modality=None, language=None, tags=None,
                        date_from=None, collection=None):
        query  = "SELECT id, faiss_index FROM assets WHERE 1=1"
        params = []
        if modality:   query += " AND modality = ?";   params.append(modality)
        if language:   query += " AND language = ?";   params.append(language)
        if date_from:  query += " AND upload_date >= ?"; params.append(date_from)
        if tags:       query += " AND tags LIKE ?";    params.append(f'%{tags}%')
        if collection: query += " AND collection = ?"; params.append(collection)
        return self.db.fetchall(query, params)

    def get_all_tags(self):
        result   = self.db.fetchall("SELECT DISTINCT tags FROM assets WHERE tags != '[]'")
        all_tags = set()
        for row in result:
            all_tags.update(json.loads(row[0]))
        return sorted(list(all_tags))

    def get_all_collections(self):
        result = self.db.fetchall("SELECT DISTINCT collection FROM assets WHERE collection != ''")
        return [row[0] for row in result]

    def add_comment(self, asset_id, user, comment):
        self.db.execute("""
            INSERT INTO comments (asset_id, user, comment, timestamp)
            VALUES (?, ?, ?, datetime('now'))
        """, (asset_id, user, comment))

    def get_comments(self, asset_id):
        return self.db.fetchall("""
            SELECT user, comment, timestamp FROM comments
            WHERE asset_id = ? ORDER BY timestamp DESC
        """, (asset_id,))

    def add_rating(self, asset_id, user, rating):
        self.db.execute("""
            INSERT OR REPLACE INTO ratings (asset_id, user, rating)
            VALUES (?, ?, ?)
        """, (asset_id, user, rating))

    def get_rating(self, asset_id):
        result = self.db.fetchone("""
            SELECT AVG(rating), COUNT(*) FROM ratings WHERE asset_id = ?
        """, (asset_id,))
        return result if result[0] else (0, 0)


# ==================== ANALYTICS ==================== #
class AnalyticsTracker:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(BASE_DIR, "analytics.db")
        self.db = DatabaseConnection(db_path)
        self._create_tables()

    def _create_tables(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT, modality TEXT, results_count INTEGER,
                timestamp TEXT, search_duration_ms REAL, user TEXT
            )
        """)

    def log_search(self, query, modality, results_count, duration_ms, user="unknown"):
        try:
            self.db.execute("""
                INSERT INTO searches (query_text, modality, results_count, timestamp, search_duration_ms, user)
                VALUES (?, ?, ?, datetime('now'), ?, ?)
            """, (query, modality, results_count, duration_ms, user))
            # FIX: actually use the logger that was set up
            logger.info(
                f"SEARCH user={user} modality={modality} q={query!r} "
                f"hits={results_count} ms={duration_ms:.0f}"
            )
        except Exception as e:
            logger.error(f"Failed to log search: {e}", exc_info=True)  # FIX: log error instead of pass

    def get_stats(self, days=7):
        result = self.db.fetchone(f"""
            SELECT COUNT(*), AVG(results_count), AVG(search_duration_ms)
            FROM searches
            WHERE timestamp >= datetime('now', '-{days} days')
        """)
        return result

    def get_popular_searches(self, limit=10):
        return self.db.fetchall("""
            SELECT query_text, COUNT(*) as freq
            FROM searches
            WHERE timestamp >= datetime('now', '-30 days')
            GROUP BY query_text
            ORDER BY freq DESC
            LIMIT ?
        """, (limit,))


# ==================== MODELS ==================== #
@st.cache_resource(show_spinner="Loading neural models...")
def load_models():
    dtype  = torch.float16 if DEVICE == "cuda" else torch.float32
    clip_m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=dtype).to(DEVICE).eval()
    clip_p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clap_m = ClapModel.from_pretrained("laion/clap-htsat-fused", torch_dtype=dtype).to(DEVICE).eval()
    clap_p = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
    return clip_m, clip_p, clap_m, clap_p

clip_model, clip_processor, clap_model, clap_processor = load_models()


# ==================== UTILITIES ==================== #
def sanitize_asset_id(s):
    return re.sub(r'[^\w\-]', '_', s.strip())

def validate_asset_id(asset_id):
    if not asset_id:                                   return False, "Asset ID cannot be empty"
    if len(asset_id) > 100:                            return False, "Asset ID too long (max 100 chars)"
    if not re.match(r'^[a-zA-Z0-9_\-]+$', asset_id):  return False, "Letters, numbers, _ and - only"
    if asset_id[0] in ('_', '-'):                      return False, "Cannot start with _ or -"
    return True, "OK"

def validate_upload(file, modality):
    allowed = CONFIG.ALLOWED_IMAGE_EXTS if modality == "image" else CONFIG.ALLOWED_AUDIO_EXTS
    ext     = os.path.splitext(file.name)[1].lower()
    if file.size > CONFIG.MAX_FILE_SIZE:
        return f"File is {file.size/1024/1024:.1f} MB — exceeds 100 MB limit."
    if ext not in allowed:
        return f"Extension '{ext}' not supported."
    return None

def validate_image_content(file_path):
    try:
        img = Image.open(file_path); img.verify()
        img = Image.open(file_path)
        if img.width > 10000 or img.height > 10000: return False, "Image too large (max 10000×10000)"
        if img.width < 100  or img.height < 100:    return False, "Image too small (min 100×100)"
        return True, "OK"
    except Exception as e:
        return False, f"Invalid image: {e}"

def translate_to_english(text):
    try:
        if any(ord(c) > 127 for c in text):
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except Exception:
        return text

def is_audio_valid(y):
    return float(np.sqrt(np.mean(y ** 2))) > CONFIG.AUDIO_SILENCE_RMS

def fp16(inp):
    return {k: v.to(torch.float16) if torch.is_floating_point(v) else v for k, v in inp.items()}

def confidence_badge(conf):
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}[conf]
    return f'<span class="badge {cls}">● {conf} Confidence</span>'

def score_bar(score):
    pct = min(max(score, 0.0), 1.0) * 100
    return (f'<div class="score-bar-wrap">'
            f'<div class="score-bar-track">'
            f'<div class="score-bar-fill" style="width:{pct:.1f}%"></div>'
            f'</div></div>')

def analyze_image_quality(file_path):
    try:
        img           = Image.open(file_path)
        quality_score = min(img.width * img.height / (1920 * 1080), 1.0)
        return {
            "resolution":    f"{img.width}×{img.height}",
            "quality_score": quality_score,
            "format":        img.format,
            "mode":          img.mode,
            "size_kb":       os.path.getsize(file_path) / 1024
        }
    except Exception:
        return None

def analyze_audio_quality(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        rms   = librosa.feature.rms(y=y)[0]
        return {
            "duration":    f"{len(y) / sr:.1f}s",
            "sample_rate": f"{sr}Hz",
            "rms_energy":  float(np.mean(rms)),
            "size_kb":     os.path.getsize(file_path) / 1024
        }
    except Exception:
        return None

def get_search_suggestions(analytics, query, limit=5):
    if not query or len(query) < 2:
        popular = analytics.get_popular_searches(limit)
        return [p[0] for p in popular]
    result = analytics.db.fetchall("""
        SELECT DISTINCT query_text, COUNT(*) as freq
        FROM searches
        WHERE query_text LIKE ? AND query_text != ?
        GROUP BY query_text
        ORDER BY freq DESC
        LIMIT ?
    """, (f"%{query}%", query, limit))
    return [row[0] for row in result]


# ==================== ENGINE ==================== #
class TrinetraEngine:
    def __init__(self, modality):
        self.modality  = modality
        db_path        = os.path.join(BASE_DIR, modality)
        self.idx_path  = os.path.join(db_path, "index")
        self.map_path  = os.path.join(db_path, "id_map.json")
        os.makedirs(db_path, exist_ok=True)

        if os.path.exists(self.idx_path) and os.path.exists(self.map_path):
            self.index  = faiss.read_index(self.idx_path)
            with open(self.map_path) as f:
                self.id_map = json.load(f)
        else:
            self.index  = faiss.IndexFlatIP(CONFIG.EMBEDDING_DIM)
            self.id_map = []

        self.embedding_cache = {}
        self.metadata_db     = MetadataDB()

    def _normalize(self, v):
        return (v / (np.linalg.norm(v) + 1e-9)).astype("float32")

    def _save(self):
        faiss.write_index(self.index, self.idx_path)
        with open(self.map_path, "w") as f:
            json.dump(self.id_map, f, indent=2)

    def _cache_key(self, text=None, path=None):
        if text: return hashlib.md5(text.encode()).hexdigest()
        if path:
            with open(path, 'rb') as f: return hashlib.md5(f.read()).hexdigest()

    def id_exists(self, asset_id):
        return any(r["id"] == asset_id for r in self.id_map)

    def get_embedding(self, file_path=None, text=None):
        key = self._cache_key(text, file_path)
        if key and key in self.embedding_cache:
            return self.embedding_cache[key]
        emb = self._compute_embedding(file_path, text)
        if key and len(self.embedding_cache) < CONFIG.CACHE_SIZE:
            self.embedding_cache[key] = emb
        return emb

    def _compute_embedding(self, file_path=None, text=None):
        raise NotImplementedError

    def find_duplicates(self, file_path=None, text=None, threshold=None):
        if self.index.ntotal == 0: return []
        t = threshold or CONFIG.DUPLICATE_THRESHOLD
        q = self.get_embedding(file_path=file_path, text=text).reshape(1, -1)
        scores, idxs = self.index.search(q, min(10, self.index.ntotal))
        return [{"id": self.id_map[i]["id"], "similarity": float(s), "path": self.id_map[i]["path"]}
                for s, i in zip(scores[0], idxs[0]) if i != -1 and s > t]

    def register(self, temp_path, asset_id, ext, lang, tags=None, description="",
                 collection="", uploaded_by="unknown"):
        asset_id = sanitize_asset_id(asset_id)
        ok, msg  = validate_asset_id(asset_id)
        if not ok:                 return False, msg
        if self.id_exists(asset_id): return False, f"ID '{asset_id}' already exists"

        perm_path = os.path.join(STORAGE_DIR, f"{asset_id}{ext}")
        try:
            with open(temp_path, "rb") as s, open(perm_path, "wb") as d:
                d.write(s.read())

            if self.modality == "image":
                quality_info  = analyze_image_quality(perm_path)
                quality_score = quality_info['quality_score'] if quality_info else 0.5
            else:
                quality_info  = analyze_audio_quality(perm_path)
                quality_score = 0.7

            emb       = self.get_embedding(file_path=perm_path)
            faiss_idx = self.index.ntotal
            self.index.add(emb.reshape(1, -1))
            self.id_map.append({"id": asset_id, "path": perm_path,
                                 "lang": lang, "modality": self.modality,
                                 "timestamp": time.ctime(), "quality": quality_score})
            self._save()
            self.metadata_db.add_asset(asset_id, self.modality, lang, perm_path,
                                       os.path.getsize(perm_path), faiss_idx,
                                       tags, description, collection, quality_score, uploaded_by)
            logger.info(f"ASSET_REGISTERED id={asset_id} modality={self.modality} lang={lang} by={uploaded_by}")
            return True, f"Successfully registered: {asset_id}"
        except Exception as e:
            if os.path.exists(perm_path): os.remove(perm_path)
            logger.error(f"Registration failed for {asset_id}: {e}", exc_info=True)  # FIX: log error
            return False, f"Failed: {e}"

    def batch_register(self, files, language, tags=None, collection="", uploaded_by="unknown"):
        total       = len(files)
        progress    = st.progress(0)
        status_text = st.empty()
        results     = {"success": [], "failed": [], "skipped": [], "duplicates": []}

        for idx, file in enumerate(files):
            status_text.text(f"Processing {idx+1}/{total}: {file.name}")
            try:
                asset_id = sanitize_asset_id(os.path.splitext(file.name)[0])
                if self.id_exists(asset_id):
                    results["skipped"].append(file.name); continue
                err = validate_upload(file, self.modality)
                if err:
                    results["failed"].append((file.name, err)); continue
                ext = os.path.splitext(file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(file.getbuffer()); tp = tmp.name
                try:
                    dups = self.find_duplicates(file_path=tp)
                    if dups:
                        results["duplicates"].append((file.name, dups[0]["id"]))
                        continue
                    ok, msg = self.register(tp, asset_id, ext, language, tags, "",
                                            collection, uploaded_by)
                    (results["success"] if ok else results["failed"]).append(
                        file.name if ok else (file.name, msg))
                finally:
                    if os.path.exists(tp): os.remove(tp)
            except Exception as e:
                results["failed"].append((file.name, str(e)))
                logger.error(f"Batch register error for {file.name}: {e}", exc_info=True)  # FIX: log
            progress.progress((idx + 1) / total)

        progress.empty(); status_text.empty()
        logger.info(
            f"BATCH_REGISTER total={total} success={len(results['success'])} "
            f"failed={len(results['failed'])} dupes={len(results['duplicates'])} by={uploaded_by}"
        )
        return results

    def hybrid_search(self, text_query=None, file_path=None, top_k=5,
                      filters=None, rerank=True):
        if self.index.ntotal == 0: return [], 0

        t0 = time.time()
        q  = self.get_embedding(file_path=file_path, text=text_query).reshape(1, -1)
        scores, idxs = self.index.search(q, min(top_k * 2, self.index.ntotal))

        out = []
        for s, i in zip(scores[0], idxs[0]):
            if i == -1: continue
            conf   = ("High" if s > CONFIG.CONFIDENCE_HIGH else
                      "Medium" if s > CONFIG.CONFIDENCE_MED else "Low")
            result = {**self.id_map[i], "score": float(s), "confidence": conf}

            if filters:
                if 'min_score' in filters and s < filters['min_score']:
                    continue
                if 'language' in filters and result.get('lang') != filters['language']:
                    continue
            out.append(result)

        if rerank:
            for r in out:
                quality = r.get('quality', 0.5)
                r['final_score'] = r['score'] * (0.7 + 0.3 * quality)
            out = sorted(out, key=lambda x: x['final_score'], reverse=True)

        return out[:top_k], (time.time() - t0) * 1000

    def search(self, file_path=None, text=None, top_k=5, metadata_filters=None):
        return self.hybrid_search(text_query=text, file_path=file_path,
                                  top_k=top_k, filters=metadata_filters)

    def get_all_vectors(self):
        try:
            return faiss.vector_to_array(self.index.reconstruct_n(0, self.index.ntotal)).reshape(
                self.index.ntotal, -1
            )
        except Exception as e:
            logger.error(f"get_all_vectors failed: {e}", exc_info=True)  # FIX: log
            return None

    def export_registry(self, export_path=None):
        import zipfile
        if export_path is None:
            export_path = f"trinetra_{self.modality}_backup_{int(time.time())}.zip"
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as z:
            if os.path.exists(self.idx_path): z.write(self.idx_path, "index")
            if os.path.exists(self.map_path): z.write(self.map_path, "id_map.json")
            for asset in self.id_map:
                if os.path.exists(asset["path"]):
                    z.write(asset["path"], f"assets/{os.path.basename(asset['path'])}")
        return export_path

    def export_as_json(self):
        export_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "modality":    self.modality,
                "total_assets": self.index.ntotal
            },
            "assets": self.id_map
        }
        return json.dumps(export_data, indent=2)


class ImageEngine(TrinetraEngine):
    def __init__(self): super().__init__("image")
    def _compute_embedding(self, file_path=None, text=None):
        with torch.no_grad():
            if text:
                inp = clip_processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
                if DEVICE == "cuda": inp = fp16(inp)
                e = clip_model.get_text_features(**inp)
            else:
                ok, msg = validate_image_content(file_path)
                if not ok: raise ValueError(msg)
                img = Image.open(file_path).convert("RGB")
                inp = clip_processor(images=img, return_tensors="pt").to(DEVICE)
                if DEVICE == "cuda": inp = fp16(inp)
                e = clip_model.get_image_features(**inp)
        return self._normalize(e.cpu().float().numpy().flatten())


class AudioEngine(TrinetraEngine):
    def __init__(self): super().__init__("audio")
    def _compute_embedding(self, file_path=None, text=None):
        with torch.no_grad():
            if text:
                inp = clap_processor(text=[text], return_tensors="pt").to(DEVICE)
                e   = clap_model.get_text_features(**inp)
            else:
                y, _ = librosa.load(file_path, sr=CONFIG.AUDIO_SAMPLE_RATE,
                                    duration=CONFIG.AUDIO_DURATION_S)
                if not is_audio_valid(y): raise ValueError("Audio appears silent")
                inp = clap_processor(audios=y, sampling_rate=CONFIG.AUDIO_SAMPLE_RATE,
                                     return_tensors="pt").to(DEVICE)
                e   = clap_model.get_audio_features(**inp)
        return self._normalize(e.cpu().float().numpy().flatten())


# ==================== INIT — FIX: cached so engines & cache survive reruns ==================== #
@st.cache_resource(show_spinner="Initializing registry...")
def get_engines():
    return ImageEngine(), AudioEngine(), AnalyticsTracker()

image_engine, audio_engine, analytics = get_engines()


# ==================== SIDEBAR ==================== #
with st.sidebar:
    user_role = st.session_state.user['role']
    st.markdown(f"""
    <div style="padding:0.5rem; background:{T['bg_card']}; border:1px solid {T['border']};
    border-radius:8px; margin-bottom:1rem;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
        color:{T['text_muted']};">LOGGED IN AS</div>
        <div style="font-weight:600; color:{T['accent']};">{st.session_state.user['username']}</div>
        <div style="font-size:0.7rem; color:{T['text_muted']};">Role: {user_role}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Logout", use_container_width=True):
        logger.info(f"LOGOUT user={st.session_state.user['username']}")  # FIX: log logout
        st.session_state.authenticated = False
        st.session_state.user          = None
        st.rerun()

    cur_mode = T["mode_label"]
    st.markdown(f"""
    <div class="tog-pill">
        <div class="tog-track"><div class="tog-knob"></div></div>
        <span class="tog-icon">{T['icon']}</span>
        <span class="tog-label">{cur_mode}</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button(f"{T['icon']} {cur_mode}", use_container_width=True, key="theme_btn"):
        st.session_state.theme = "light" if not is_light else "dark"
        st.rerun()

    st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0">', unsafe_allow_html=True)

    if user_role in ['admin', 'uploader']:
        st.markdown("### Asset Registration")
        batch_mode = st.checkbox("Batch Upload Mode", value=False)
        reg_mod    = st.selectbox("Modality", ["image", "audio"])
        reg_lang   = st.selectbox("Language", CONFIG.SUPPORTED_LANGUAGES)

        if not batch_mode:
            reg_id   = st.text_input("Asset ID", placeholder="e.g. diwali_2024")
            reg_tags = st.multiselect("Tags (optional)",
                                      ["festival","temple","landscape","portrait",
                                       "music","speech","nature","urban"],
                                      key="tags_single")
            reg_file = st.file_uploader(
                "Select File",
                type=[e.lstrip('.') for e in (CONFIG.ALLOWED_IMAGE_EXTS if reg_mod=="image"
                                              else CONFIG.ALLOWED_AUDIO_EXTS)],
            )
            if reg_file:
                st.markdown("#### Preview")
                if reg_mod == "image":
                    st.image(reg_file, use_container_width=True)
                    st.caption(f"Size: {reg_file.size/1024:.1f} KB")
                else:
                    st.audio(reg_file)
                    st.caption(f"Size: {reg_file.size/1024:.1f} KB")

            if st.button("Register Asset", use_container_width=True):
                if not reg_file:         st.warning("Please upload a file")
                elif not reg_id.strip(): st.warning("Please enter an Asset ID")
                else:
                    err = validate_upload(reg_file, reg_mod)
                    if err: st.error(err)
                    else:
                        ext = os.path.splitext(reg_file.name)[1].lower()
                        eng = image_engine if reg_mod == "image" else audio_engine
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(reg_file.getbuffer()); tp = tmp.name
                        try:
                            with st.spinner("Checking for duplicates..."):
                                dups = eng.find_duplicates(file_path=tp)
                            if dups:
                                st.warning(f"Similar asset found: **{dups[0]['id']}** "
                                           f"({dups[0]['similarity']:.1%} match)")
                                if not st.checkbox("Register anyway"):
                                    st.stop()
                            with st.spinner("Registering..."):
                                ok, msg = eng.register(tp, reg_id.strip(), ext, reg_lang,
                                                       reg_tags, uploaded_by=st.session_state.user['username'])
                            (st.success if ok else st.error)(msg)
                            if ok:
                                st.balloons()
                                time.sleep(1)
                                st.rerun()
                        except Exception as exc:
                            st.error(f"Registration failed: {exc}")
                            logger.error(f"UI registration error: {exc}", exc_info=True)
                        finally:
                            if os.path.exists(tp): os.remove(tp)
        else:
            reg_tags_b  = st.multiselect("Tags for all files",
                                         ["festival","temple","landscape","portrait",
                                          "music","speech","nature","urban"],
                                         key="tags_batch")
            reg_collect = st.text_input("Collection name", placeholder="e.g. Diwali 2024")
            reg_files   = st.file_uploader(
                "Select Multiple Files", accept_multiple_files=True,
                type=[e.lstrip('.') for e in (CONFIG.ALLOWED_IMAGE_EXTS if reg_mod=="image"
                                              else CONFIG.ALLOWED_AUDIO_EXTS)],
            )
            if st.button("Register Batch", use_container_width=True):
                if not reg_files: st.warning("Upload at least one file")
                else:
                    eng = image_engine if reg_mod == "image" else audio_engine
                    with st.spinner(f"Processing {len(reg_files)} files..."):
                        res = eng.batch_register(reg_files, reg_lang, reg_tags_b, reg_collect,
                                                 st.session_state.user['username'])
                    st.success(f"Added: {len(res['success'])}")
                    if res['skipped']:    st.info(f"Skipped: {len(res['skipped'])}")
                    if res['duplicates']:
                        st.warning(f"Duplicates: {len(res['duplicates'])}")
                        with st.expander("View duplicates"):
                            for fn, dup in res['duplicates']: st.write(f"- {fn} → {dup}")
                    if res['failed']:
                        st.error(f"Failed: {len(res['failed'])}")
                        with st.expander("View errors"):
                            for fn, err in res['failed']: st.write(f"- {fn}: {err}")
                    if res['success']:
                        time.sleep(0.8); st.rerun()

        st.markdown(f'<hr style="border-color:{T["border"]};margin:1.5rem 0 1rem">', unsafe_allow_html=True)

    st.markdown("**Registry Status**")
    st.markdown(f"""
    <div class="sidebar-stat">Images <span>{image_engine.index.ntotal}</span></div>
    <div class="sidebar-stat">Audio <span>{audio_engine.index.ntotal}</span></div>
    <div class="sidebar-stat">Device <span>{DEVICE.upper()}</span></div>
    <div class="sidebar-stat">Index <span>FAISS FlatIP</span></div>
    """, unsafe_allow_html=True)

    stats = analytics.get_stats(7)
    if stats and stats[0] and stats[0] > 0:
        st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0">', unsafe_allow_html=True)
        st.markdown("**Usage Stats (7 days)**")
        st.markdown(f"""
        <div class="sidebar-stat">Searches <span>{stats[0]}</span></div>
        <div class="sidebar-stat">Avg Results <span>{stats[1]:.1f}</span></div>
        <div class="sidebar-stat">Avg Speed <span>{stats[2]:.0f}ms</span></div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0">', unsafe_allow_html=True)

    if user_role in ['admin']:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export ZIP", use_container_width=True):
                with st.spinner("Creating backup..."):
                    ip = image_engine.export_registry()
                    ap = audio_engine.export_registry()
                st.success(f"Exported!\n- {ip}\n- {ap}")
        with col2:
            if st.button("Export JSON", use_container_width=True):
                json_data = {
                    "image": json.loads(image_engine.export_as_json()),
                    "audio": json.loads(audio_engine.export_as_json())
                }
                st.download_button(
                    "Download JSON",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"trinetra_export_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True
                )


# ==================== HEADER ==================== #
st.markdown('<div class="a1"><h1>TRINETRA V5.0</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="tagline a2">Multimodal Neural Registry · AI-Powered Search · Bharat AI</div>',
            unsafe_allow_html=True)

total      = image_engine.index.ntotal + audio_engine.index.ntotal
cache_size = len(image_engine.embedding_cache) + len(audio_engine.embedding_cache)

st.markdown(f"""
<div class="stat-strip a3">
    <div class="stat-chip"><div class="stat-label">Device</div><div class="stat-value">{DEVICE.upper()}</div></div>
    <div class="stat-chip"><div class="stat-label">Images Indexed</div><div class="stat-value">{image_engine.index.ntotal}</div></div>
    <div class="stat-chip"><div class="stat-label">Audio Indexed</div><div class="stat-value">{audio_engine.index.ntotal}</div></div>
    <div class="stat-chip"><div class="stat-label">Total Assets</div><div class="stat-value">{total}</div></div>
    <div class="stat-chip"><div class="stat-label">Cache Size</div><div class="stat-value">{cache_size}</div></div>
</div>
""", unsafe_allow_html=True)


# ==================== RESULT DISPLAY ==================== #
def display_results(results, modality):
    if not results:
        st.info("No matches found in the registry.")
        return

    cols = st.columns(min(len(results), 3))
    for idx, r in enumerate(results):
        with cols[idx % 3]:
            st.markdown(
                f'<div class="result-card">'
                f'{confidence_badge(r["confidence"])}'
                f'{score_bar(r["score"])}'
                f'<div style="font-size:.7rem;color:{T["text_muted"]}">Score: {r["score"]:.3f}</div>',
                unsafe_allow_html=True,
            )

            if 'quality' in r:
                st.markdown(f"""
                <div class="quality-indicator">Quality: {r['quality']:.1%}</div>
                """, unsafe_allow_html=True)

            if modality == "image":
                st.image(r["path"], caption=r["id"], use_container_width=True)
            else:
                st.write(f"**{r['id']}**")
                st.audio(r["path"])

            st.caption(f"Lang: {r.get('lang','—')} · {r.get('timestamp','')[:10]}")

            with st.expander("Feedback"):
                avg_rating, count = image_engine.metadata_db.get_rating(r['id'])
                st.write(f"Rating: {'⭐' * int(avg_rating)} ({count} votes)")
                new_rating = st.slider("Rate this", 1, 5, 3, key=f"rate_{r['id']}_{idx}")
                if st.button("Submit Rating", key=f"submit_rate_{r['id']}_{idx}"):
                    image_engine.metadata_db.add_rating(
                        r['id'], st.session_state.user['username'], new_rating
                    )
                    st.success("Rating submitted!")
                    st.rerun()

                comments = image_engine.metadata_db.get_comments(r['id'])
                if comments:
                    st.markdown("**Comments:**")
                    for user, comment, ts in comments[:3]:
                        st.caption(f"**{user}** ({ts[:10]}): {comment}")

                new_comment = st.text_input("Add comment", key=f"comment_{r['id']}_{idx}")
                if st.button("Post Comment", key=f"post_{r['id']}_{idx}") and new_comment:
                    image_engine.metadata_db.add_comment(
                        r['id'], st.session_state.user['username'], new_comment
                    )
                    st.success("Comment posted!")
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.last_search_results = results


# ==================== TABS ==================== #
tab_v, tab_a, tab_cluster, tab_aud, tab_hist, tab_admin = st.tabs([
    "Visual Search", "Acoustic Search", "Clusters", "Neural Auditor", "Search History", "Admin"
])

# ── Visual Search ── #
with tab_v:
    mode = st.radio("Input Mode", ["Text Query", "Image Match"], horizontal=True)
    with st.expander("Advanced Filters"):
        f_lang      = st.selectbox("Language", ["All"] + CONFIG.SUPPORTED_LANGUAGES, key="fl_img")
        f_tags      = st.multiselect("Tags", image_engine.metadata_db.get_all_tags(), key="ft_img")
        f_min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, key="fs_img")

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    if mode == "Text Query":
        q = st.text_input("Describe the image", placeholder="e.g., a crowded temple at dusk", key="vq")

        if q and st.session_state.show_suggestions:
            suggestions = get_search_suggestions(analytics, q, limit=5)
            if suggestions:
                st.markdown("**Suggestions:**")
                sug_html = "".join(f'<span class="suggestion-chip">{s}</span>' for s in suggestions)
                st.markdown(sug_html, unsafe_allow_html=True)

        if st.button("Run Visual Scan", key="vs_txt") and q:
            mf = {"min_score": f_min_score}
            if f_lang != "All": mf['language'] = f_lang
            with st.spinner("Scanning..."):
                translated = translate_to_english(q)
                if translated != q:
                    st.caption(f"Translated: *{translated}*")
                results, ms = image_engine.hybrid_search(text_query=translated, top_k=6, filters=mf)

            analytics.log_search(q, "image", len(results), ms, st.session_state.user['username'])
            st.session_state.search_history.append({
                "query": q, "modality": "image",
                "timestamp": time.time(), "results_count": len(results)
            })
            st.caption(f"Search time: {ms:.0f} ms")
            display_results(results, "image")
    else:
        qi = st.file_uploader("Query image",
                               type=[e.lstrip('.') for e in CONFIG.ALLOWED_IMAGE_EXTS], key="qimg")
        if qi:
            st.image(qi, caption="Query Image", use_container_width=True)

        if st.button("Run Visual Scan", key="vs_img") and qi:
            ext = os.path.splitext(qi.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(qi.getbuffer()); qp = tmp.name
            try:
                with st.spinner("Scanning..."):
                    mf = {"min_score": f_min_score}
                    if f_lang != "All": mf['language'] = f_lang
                    results, ms = image_engine.hybrid_search(file_path=qp, top_k=6, filters=mf)
                analytics.log_search("[Image Upload]", "image", len(results), ms,
                                     st.session_state.user['username'])
                st.caption(f"Search time: {ms:.0f} ms")
                display_results(results, "image")
            finally:
                if os.path.exists(qp): os.remove(qp)

# ── Acoustic Search ── #
with tab_a:
    with st.expander("Advanced Filters"):
        f_lang_a      = st.selectbox("Language", ["All"] + CONFIG.SUPPORTED_LANGUAGES, key="fl_aud")
        f_min_score_a = st.slider("Minimum Score", 0.0, 1.0, 0.0, key="fs_aud")

    q = st.text_input("Describe the sound", placeholder="e.g., tabla solo during rainstorm", key="aq")

    if q and st.session_state.show_suggestions:
        suggestions = get_search_suggestions(analytics, q, limit=5)
        if suggestions:
            st.markdown("**Suggestions:**")
            sug_html = "".join(f'<span class="suggestion-chip">{s}</span>' for s in suggestions)
            st.markdown(sug_html, unsafe_allow_html=True)

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    if st.button("Run Acoustic Scan") and q:
        mf = {"min_score": f_min_score_a}
        if f_lang_a != "All": mf['language'] = f_lang_a
        with st.spinner("Scanning..."):
            translated = translate_to_english(q)
            if translated != q:
                st.caption(f"Translated: *{translated}*")
            results, ms = audio_engine.hybrid_search(text_query=translated, top_k=6, filters=mf)

        analytics.log_search(q, "audio", len(results), ms, st.session_state.user['username'])
        st.session_state.search_history.append({
            "query": q, "modality": "audio",
            "timestamp": time.time(), "results_count": len(results)
        })
        st.caption(f"Search time: {ms:.0f} ms")
        display_results(results, "audio")

# ── Clusters ── #
with tab_cluster:
    st.markdown("### Asset Clusters")
    pick = st.radio("Registry", ["Image", "Audio"], horizontal=True, key="cluster_pick")
    eng  = image_engine if pick == "Image" else audio_engine

    if eng.index.ntotal < 3:
        st.info("Need at least 3 assets to perform clustering.")
    else:
        n_clusters = st.slider("Number of clusters", 2, min(10, eng.index.ntotal), 3)

        if st.button("Generate Clusters"):
            with st.spinner("Clustering assets..."):
                vecs = eng.get_all_vectors()
                if vecs is not None:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(vecs)

                    # FIX: Added cluster scatter plot — huge visual demo win
                    st.markdown("#### Cluster Map (PCA 2D)")
                    pca       = PCA(n_components=2)
                    proj      = pca.fit_transform(vecs)
                    var_pct   = pca.explained_variance_ratio_.sum() * 100
                    df_scatter = pd.DataFrame({
                        "x":       proj[:, 0],
                        "y":       proj[:, 1],
                        "Cluster": [f"Cluster {l+1}" for l in labels],
                        "ID":      [r["id"] for r in eng.id_map],
                    })
                    fig_scatter = px.scatter(
                        df_scatter, x="x", y="y", color="Cluster", text="ID",
                        title=f"{pick} Embedding Clusters · {var_pct:.1f}% variance preserved",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                    )
                    fig_scatter.update_traces(
                        textposition="top center",
                        marker=dict(size=12, line=dict(width=1, color="#000")),
                    )
                    fig_scatter.update_layout(
                        plot_bgcolor=T["plot_bg"], paper_bgcolor=T["plot_bg"],
                        font=dict(family="DM Sans", color=T["plot_text"]),
                        title_font=dict(family="Syne", size=15, color=T["accent"]),
                        xaxis=dict(gridcolor=T["plot_grid"], zerolinecolor=T["plot_grid"]),
                        yaxis=dict(gridcolor=T["plot_grid"], zerolinecolor=T["plot_grid"]),
                        legend=dict(bgcolor=T["bg_card"], bordercolor=T["border"]),
                        margin=dict(l=20, r=20, t=50, b=20),
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    st.caption(f"PCA variance preserved: **{var_pct:.1f}%**")

                    st.markdown("---")
                    for i in range(n_clusters):
                        cluster_indices = [j for j in range(len(labels)) if labels[j] == i]
                        cluster_assets  = [eng.id_map[j] for j in cluster_indices]
                        with st.expander(f"Cluster {i+1} ({len(cluster_assets)} assets)"):
                            cols = st.columns(4)
                            for idx, asset in enumerate(cluster_assets[:8]):
                                with cols[idx % 4]:
                                    if eng.modality == "image":
                                        st.image(asset['path'], caption=asset['id'],
                                                 use_container_width=True)
                                    else:
                                        st.write(asset['id'])
                                        st.audio(asset['path'])

# ── Neural Auditor ── #
with tab_aud:
    pick = st.radio("Registry", ["Image", "Audio"], horizontal=True, key="aud_pick")
    eng  = image_engine if pick == "Image" else audio_engine
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vectors",   eng.index.ntotal)
    c2.metric("Index",     "FAISS FlatIP")
    c3.metric("Embed Dim", CONFIG.EMBEDDING_DIM)
    c4.metric("Cache",     len(eng.embedding_cache))
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    if eng.id_map:
        st.markdown("#### Asset Manifest")
        df = pd.DataFrame(eng.id_map)[["id","lang","modality","timestamp"]]
        if "quality" in eng.id_map[0]:
            df["quality"] = [f"{r.get('quality', 0):.1%}" for r in eng.id_map]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No assets registered yet.")

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("#### Neural Embedding Map")
    n = eng.index.ntotal
    if n < 2:
        st.info("Register at least 2 assets to view the map.")
    else:
        vecs = eng.get_all_vectors()
        if vecs is None:
            st.warning("Cannot extract vectors.")
        else:
            nc   = min(3, n, CONFIG.EMBEDDING_DIM)
            pca  = PCA(n_components=nc)
            proj = pca.fit_transform(vecs)
            var  = pca.explained_variance_ratio_.sum() * 100
            lbl  = [r["id"] for r in eng.id_map]
            st.caption(f"Variance preserved: **{var:.1f}%**")
            df_p       = pd.DataFrame(proj, columns=["x","y","z"][:nc])
            df_p["ID"] = lbl

            if nc == 3:
                fig = px.scatter_3d(df_p, x="x", y="y", z="z", text="ID",
                                    color_discrete_sequence=[T["accent"]],
                                    title=f"{pick} · 3D Embedding Map")
            else:
                fig = px.scatter(df_p, x="x", y="y", text="ID",
                                 color_discrete_sequence=[T["accent"]],
                                 title=f"{pick} · 2D Embedding Map")
                fig.update_traces(textposition="top center",
                                  marker=dict(size=14, line=dict(width=2, color=T["accent_dim"])),
                                  textfont=dict(family="JetBrains Mono", size=11, color=T["text_mono"]))

            fig.update_layout(
                plot_bgcolor=T["plot_bg"], paper_bgcolor=T["plot_bg"],
                font=dict(family="DM Sans", color=T["plot_text"]),
                title_font=dict(family="Syne", size=16, color=T["accent"]),
                xaxis=dict(gridcolor=T["plot_grid"], zerolinecolor=T["plot_grid"]),
                yaxis=dict(gridcolor=T["plot_grid"], zerolinecolor=T["plot_grid"]),
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Search History ── #
with tab_hist:
    st.markdown("### Recent Searches")
    if not st.session_state.search_history:
        st.info("No search history yet.")
    else:
        for idx, s in enumerate(reversed(st.session_state.search_history[-20:])):
            t_str = datetime.fromtimestamp(s['timestamp']).strftime('%Y-%m-%d %H:%M')
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            icon = "🖼" if s['modality'] == "image" else "🔊"
            c1.write(f"{icon} **{s['query'][:50]}**")
            c2.caption(t_str)
            c3.caption(f"{s['results_count']} results")
            with c4:
                if st.button("Re-run", key=f"rerun_{idx}"):
                    if s['modality'] == "image":
                        res, _ = image_engine.search(text=s['query'], top_k=6)
                        display_results(res, "image")
                    else:
                        res, _ = audio_engine.search(text=s['query'], top_k=6)
                        display_results(res, "audio")
            st.markdown('<hr class="rule">', unsafe_allow_html=True)

        if st.button("Clear History"):
            st.session_state.search_history = []
            st.rerun()

# ── Admin ── #
with tab_admin:
    if user_role != 'admin':
        st.warning("Admin access required")
    else:
        st.markdown("### User Management")

        with st.expander("Create New User"):
            new_user  = st.text_input("Username", key="new_user")
            new_email = st.text_input("Email (optional)", key="new_email")
            new_pass  = st.text_input("Password", type="password", key="new_pass")
            new_role  = st.selectbox("Role", ["viewer", "uploader", "admin"], key="new_role")

            if st.button("Create User"):
                if new_user and new_pass:
                    # FIX: create_user now exists on auth_manager
                    ok, msg = auth_manager.create_user(new_user, new_pass, new_role, new_email)
                    (st.success if ok else st.error)(msg)
                else:
                    st.warning("Please provide username and password")

        st.markdown("#### Current Users")
        users    = auth_manager.get_all_users()
        df_users = pd.DataFrame(users, columns=["Username", "Email", "Role", "Created", "Last Login"])
        st.dataframe(df_users, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### System Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Assets",  total)
        col2.metric("Total Searches", stats[0] if stats else 0)
        col3.metric("Total Users",   len(users))

        st.markdown("#### Popular Searches (30 days)")
        popular = analytics.get_popular_searches(10)
        if popular:
            df_pop = pd.DataFrame(popular, columns=["Query", "Frequency"])
            st.dataframe(df_pop, use_container_width=True, hide_index=True)

# ── Comparison View ── #
if len(st.session_state.last_search_results) >= 2:
    with st.expander("Compare Results"):
        st.markdown("### Side-by-Side Comparison")
        ids = [r["id"] for r in st.session_state.last_search_results]
        ca, cb = st.columns(2)
        s1 = ca.selectbox("Result 1", ids, key="cmp1")
        s2 = cb.selectbox("Result 2", ids, key="cmp2")
        r1 = next(r for r in st.session_state.last_search_results if r["id"] == s1)
        r2 = next(r for r in st.session_state.last_search_results if r["id"] == s2)
        ca, cb = st.columns(2)
        with ca:
            if r1.get("modality") == "image":
                st.image(r1["path"], use_container_width=True)
            else:
                st.audio(r1["path"])
            st.metric("Score", f"{r1['score']:.2%}")
            st.write(f"**ID:** {r1['id']}")
            st.write(f"**Lang:** {r1.get('lang', '—')}")
            if 'quality' in r1: st.write(f"**Quality:** {r1['quality']:.1%}")
        with cb:
            if r2.get("modality") == "image":
                st.image(r2["path"], use_container_width=True)
            else:
                st.audio(r2["path"])
            st.metric("Score", f"{r2['score']:.2%}")
            st.write(f"**ID:** {r2['id']}")
            st.write(f"**Lang:** {r2.get('lang', '—')}")
            if 'quality' in r2: st.write(f"**Quality:** {r2['quality']:.1%}")

# ── Footer ── #
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong style='font-size: 1.1em; color: #667eea;'>Created by Team Human</strong></p>
    <p>Powered by CLIP &amp; CLAP | Built for Bharat's Digital Future</p>
    <p style='font-size: 0.8em;'>Multimodal embeddings • FAISS indexing • Cross-lingual search</p>
</div>
""", unsafe_allow_html=True)




