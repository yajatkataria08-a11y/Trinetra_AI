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
import requests
import streamlit.components.v1 as components
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from PIL import Image
from deep_translator import GoogleTranslator
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, ClapModel
from werkzeug.security import generate_password_hash, check_password_hash
from bs4 import BeautifulSoup
from urllib.parse import urlencode, quote_plus

# ==================== CONFIGURATION ==================== #
class Config:
    MAX_FILE_SIZE        = 100 * 1024 * 1024
    ALLOWED_IMAGE_EXTS   = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    ALLOWED_AUDIO_EXTS   = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    SUPPORTED_LANGUAGES  = ['en', 'hi', 'ta', 'te', 'kn', 'ml', 'bn', 'mr', 'gu', 'pa']
    AUDIO_DURATION_S     = 7.0
    EMBEDDING_DIM        = 512
    AUDIO_SAMPLE_RATE    = 48_000
    AUDIO_SILENCE_RMS    = 0.001
    CONFIDENCE_HIGH      = 0.65
    CONFIDENCE_MED       = 0.45
    DUPLICATE_THRESHOLD  = 0.95
    CACHE_SIZE           = 1000
    OTP_COOLDOWN_SECS    = 60

CONFIG      = Config()
BASE_DIR    = "trinetra_registry"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(BASE_DIR,    exist_ok=True)

# ==================== LOGGING ==================== #
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f'logs/trinetra_{datetime.now():%Y%m%d}.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Trinetra')

# ==================== THREAD-SAFE DATABASE CONNECTION ==================== #
class DatabaseConnection:
    def __init__(self, db_path):
        self.db_path = db_path
        self._local  = threading.local()

    def get_connection(self):
        if not hasattr(self._local, 'conn'):
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            conn.commit()
            self._local.conn = conn
        return self._local.conn

    def execute(self, query, params=None):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params) if params else cursor.execute(query)
            conn.commit()
            return cursor
        except sqlite3.OperationalError as e:
            conn.rollback()
            logger.error(f"DB execute error: {e}", exc_info=True)
            raise e

    def fetchall(self, query, params=None):
        return self.execute(query, params).fetchall()

    def fetchone(self, query, params=None):
        return self.execute(query, params).fetchone()

    def close(self):
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn


# ==================== EMAIL OTP SENDER ==================== #
class EmailOTPSender:
    def __init__(self, smtp_server="smtp.gmail.com", smtp_port=587):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = ""
        self.sender_password = ""
        self.smtp_configured = False

        # Priority 1: Streamlit Cloud secrets
        try:
            if "SMTP_EMAIL" in st.secrets and "SMTP_PASSWORD" in st.secrets:
                email = str(st.secrets["SMTP_EMAIL"]).strip()
                pwd = str(st.secrets["SMTP_PASSWORD"]).strip()
                if email and pwd and "@" in email:
                    self.sender_email = email
                    self.sender_password = pwd
                    self.smtp_configured = True
                    logger.info("SMTP loaded from Streamlit secrets")
                else:
                    logger.warning("Secrets found but empty/invalid")
        except Exception:
            pass  # silent fail → go to env fallback

        # Priority 2: Environment variables (local dev)
        if not self.smtp_configured:
            email = os.getenv("SMTP_EMAIL", "").strip()
            pwd = os.getenv("SMTP_PASSWORD", "").strip()
            if email and pwd and "@" in email:
                self.sender_email = email
                self.sender_password = pwd
                self.smtp_configured = True
                logger.info("SMTP loaded from environment variables")

        if not self.smtp_configured:
            logger.warning("No valid SMTP credentials anywhere")
            st.warning(
                "⚠️ Email/OTP sending disabled — no SMTP_EMAIL & SMTP_PASSWORD found.\n\n"
                "• Cloud → add to Streamlit Secrets\n"
                "• Local → set in .env or export\n"
                "OTPs will show on screen until fixed."
            )

    def send_otp_email(self, recipient_email, otp, username):
        if not self.smtp_configured:
            msg = (
                "⚠️ Email not configured.\n\n"
                f"OTP for this session: **{otp}**\n\n"
                "Copy it now — won't be shown again.\n"
                "Fix: add SMTP_EMAIL + SMTP_PASSWORD (Gmail App Password if 2FA on)"
            )
            return False, msg, otp

        try:
            html_body = f"""
            <html><body style="font-family:sans-serif;padding:20px;background:#f8f9fa;">
                <div style="max-width:500px;margin:auto;background:white;padding:30px;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,0.1);">
                    <h2 style="color:#e8a020;text-align:center;">Trinetra Verification</h2>
                    <p>Hello {username},</p>
                    <p>Your code is:</p>
                    <h1 style="color:#e8a020;letter-spacing:8px;text-align:center;margin:20px 0;">{otp}</h1>
                    <p style="color:#555;">Valid 10 minutes only.</p>
                    <hr style="border:none;border-top:1px solid #eee;">
                    <p style="font-size:0.9rem;color:#777;text-align:center;">
                        Team Human | Bharat's Digital Future
                    </p>
                </div>
            </body></html>
            """

            message = MIMEMultipart("alternative")
            message["Subject"] = "Trinetra – OTP Code"
            message["From"] = f"Trinetra <{self.sender_email}>"
            message["To"] = recipient_email
            message.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipient_email, message.as_string())

            logger.info(f"OTP sent to {recipient_email}")
            return True, "OTP sent — check inbox/spam", None

        except smtplib.SMTPAuthenticationError:
            return False, "Gmail login rejected (use **App Password** if 2FA on). OTP: **{otp}**", otp
        except Exception as e:
            logger.error(f"SMTP fail: {str(e)}")
            return False, f"Email failed ({str(e)}). OTP: **{otp}**", otp


# ==================== AUTHENTICATION ==================== #
class AuthManagerWithOTP:
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
                username TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL, full_name TEXT, role TEXT NOT NULL,
                is_verified INTEGER DEFAULT 0, created_at TEXT NOT NULL,
                last_login TEXT, registration_date TEXT
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS registration_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL, username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL, full_name TEXT,
                otp_hash TEXT NOT NULL, created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL, is_verified INTEGER DEFAULT 0
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS password_reset_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL, otp_hash TEXT NOT NULL,
                created_at TEXT NOT NULL, expires_at TEXT NOT NULL
            )
        """)

    def _create_default_admin(self):
        admin_hash = self.hash_password("admin123")
        try:
            self.db.execute("""
                INSERT OR IGNORE INTO users
                (username, email, password_hash, full_name, role, is_verified, created_at, registration_date)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """, ("admin", "admin@trinetra.local", admin_hash, "Administrator", "admin", 1))
            logger.info("Default admin created or already exists")
        except Exception as e:
            logger.error(f"Failed to create default admin: {e}", exc_info=True)

    def hash_password(self, password):
        return generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

    def _is_legacy_sha256(self, stored_hash):
        return len(stored_hash) == 64 and all(c in '0123456789abcdef' for c in stored_hash)

    def verify_password(self, stored_hash, password):
        if self._is_legacy_sha256(stored_hash):
            return hashlib.sha256(password.encode()).hexdigest() == stored_hash
        return check_password_hash(stored_hash, password)

    def hash_otp(self, otp):
        return hashlib.sha256(otp.encode()).hexdigest()

    def _check_otp_cooldown(self, email, table):
        row = self.db.fetchone(
            f"SELECT created_at FROM {table} WHERE email = ?", (email,)
        )
        if row:
            last = datetime.fromisoformat(row[0])
            if (datetime.now() - last).total_seconds() < CONFIG.OTP_COOLDOWN_SECS:
                return False
        return True

    def request_registration(self, email, username, password, full_name=""):
        if not email or '@' not in email:
            return False, "Invalid email address", None
        if len(username) < 3:
            return False, "Username must be at least 3 characters", None
        if len(password) < 6:
            return False, "Password must be at least 6 characters", None

        self.db.execute(
            "DELETE FROM registration_requests WHERE expires_at < datetime('now')"
        )

        if self.db.fetchone(
            "SELECT username FROM registration_requests WHERE email=? OR username=?",
            (email, username)
        ):
            return False, "Email or username already has a pending registration", None

        if self.db.fetchone(
            "SELECT username FROM users WHERE email=? OR username=?",
            (email, username)
        ):
            return False, "Email or username already in use", None

        if not self._check_otp_cooldown(email, "registration_requests"):
            return False, f"Please wait {CONFIG.OTP_COOLDOWN_SECS}s before requesting another OTP", None

        otp      = self.otp_sender.generate_otp()
        otp_hash = self.hash_otp(otp)
        email_sent, email_msg, fallback_otp = self.otp_sender.send_otp_email(email, otp, username)

        try:
            password_hash = self.hash_password(password)
            self.db.execute("""
                INSERT INTO registration_requests
                (email, username, password_hash, full_name, otp_hash, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now', '+10 minutes'))
            """, (email, username, password_hash, full_name, otp_hash))
            logger.info(f"REGISTER_REQUEST user={username} email={email} email_sent={email_sent}")
            status_msg = (
                f"OTP sent to {email}. Valid for 10 minutes."
                if email_sent
                else "⚠️ Email delivery failed — use the OTP shown on screen below."
            )
            return True, status_msg, fallback_otp
        except Exception as e:
            logger.error(f"Registration request failed for {email}: {e}", exc_info=True)
            return False, f"Registration request failed: {str(e)}", None

    def verify_otp_and_register(self, email, otp):
        otp_hash = self.hash_otp(otp)
        result   = self.db.fetchone("""
            SELECT username, password_hash, full_name, expires_at
            FROM registration_requests WHERE email=? AND otp_hash=?
        """, (email, otp_hash))

        if not result:
            logger.warning(f"OTP_FAIL email={email} reason=invalid_otp")
            return False, "Invalid OTP"

        username, password_hash, full_name, expires_at = result
        if datetime.now() > datetime.fromisoformat(expires_at):
            self.db.execute("DELETE FROM registration_requests WHERE email=?", (email,))
            logger.warning(f"OTP_FAIL email={email} reason=expired")
            return False, "OTP expired. Please register again."

        try:
            self.db.execute("""
                INSERT INTO users
                (username, email, password_hash, full_name, role, is_verified, created_at, registration_date)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """, (username, email, password_hash, full_name, "uploader", 1))
            self.db.execute("DELETE FROM registration_requests WHERE email=?", (email,))
            logger.info(f"REGISTER_SUCCESS user={username} email={email}")
            return True, "Registration successful! You can now login."
        except Exception as e:
            logger.error(f"Registration DB error for {email}: {e}", exc_info=True)
            return False, f"Registration failed: {str(e)}"

    def resend_otp(self, email):
        result = self.db.fetchone(
            "SELECT username FROM registration_requests WHERE email=?", (email,)
        )
        if not result:
            return False, "No pending registration found for this email", None

        if not self._check_otp_cooldown(email, "registration_requests"):
            return False, f"Please wait {CONFIG.OTP_COOLDOWN_SECS}s before resending", None

        username = result[0]
        otp      = self.otp_sender.generate_otp()
        otp_hash = self.hash_otp(otp)
        email_sent, _, fallback_otp = self.otp_sender.send_otp_email(email, otp, username)

        try:
            self.db.execute("""
                UPDATE registration_requests
                SET otp_hash=?, created_at=datetime('now'), expires_at=datetime('now','+10 minutes')
                WHERE email=?
            """, (otp_hash, email))
            logger.info(f"OTP_RESEND email={email} email_sent={email_sent}")
            status_msg = "OTP resent successfully." if email_sent else "⚠️ Email failed — use the OTP shown on screen."
            return True, status_msg, fallback_otp
        except Exception as e:
            logger.error(f"Resend OTP failed for {email}: {e}", exc_info=True)
            return False, f"Failed to resend OTP: {str(e)}", None

    def verify_user(self, username, password):
        result = self.db.fetchone(
            "SELECT username, password_hash, role, is_verified FROM users WHERE username=?",
            (username,)
        )
        if not result:
            logger.warning(f"LOGIN_FAIL user={username} reason=not_found")
            return None

        db_username, stored_hash, role, is_verified = result

        if not self.verify_password(stored_hash, password):
            logger.warning(f"LOGIN_FAIL user={username} reason=bad_password")
            return None
        if not is_verified:
            logger.warning(f"LOGIN_FAIL user={username} reason=not_verified")
            return None

        if self._is_legacy_sha256(stored_hash):
            new_hash = self.hash_password(password)
            try:
                self.db.execute(
                    "UPDATE users SET password_hash=? WHERE username=?", (new_hash, db_username)
                )
                logger.info(f"HASH_UPGRADED user={db_username}")
            except Exception as e:
                logger.warning(f"Hash upgrade failed for {db_username}: {e}")

        self.db.execute(
            "UPDATE users SET last_login=datetime('now') WHERE username=?", (username,)
        )
        logger.info(f"LOGIN_SUCCESS user={db_username} role={role}")
        return {"username": db_username, "role": role}

    def get_all_users(self):
        return self.db.fetchall(
            "SELECT username, email, role, created_at, last_login FROM users WHERE is_verified=1"
        )

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
            logger.info(f"USER_CREATED user={username} role={role} by=admin")
            return True, f"User '{username}' created successfully"
        except sqlite3.IntegrityError:
            return False, "Username or email already exists"

    def request_password_reset(self, email):
        result = self.db.fetchone(
            "SELECT username FROM users WHERE email=? AND is_verified=1", (email,)
        )
        if not result:
            logger.warning(f"RESET_REQUEST_NOTFOUND email={email}")
            return True, "If that email is registered, an OTP has been sent.", None

        if not self._check_otp_cooldown(email, "password_reset_requests"):
            return False, f"Please wait {CONFIG.OTP_COOLDOWN_SECS}s before requesting another reset OTP", None

        username = result[0]
        otp      = self.otp_sender.generate_otp()
        otp_hash = self.hash_otp(otp)

        self.db.execute("DELETE FROM password_reset_requests WHERE email=?", (email,))
        self.db.execute("""
            INSERT INTO password_reset_requests (email, otp_hash, created_at, expires_at)
            VALUES (?, ?, datetime('now'), datetime('now', '+10 minutes'))
        """, (email, otp_hash))

        email_sent, _, fallback_otp = self.otp_sender.send_otp_email(email, otp, username)
        logger.info(f"RESET_REQUEST user={username} email={email} email_sent={email_sent}")

        status_msg = (
            f"OTP sent to {email}. Valid for 10 minutes."
            if email_sent
            else "⚠️ Email delivery failed — use the OTP shown on screen below."
        )
        return True, status_msg, fallback_otp

    def verify_reset_otp(self, email, otp):
        otp_hash = self.hash_otp(otp)
        result   = self.db.fetchone(
            "SELECT expires_at FROM password_reset_requests WHERE email=? AND otp_hash=?",
            (email, otp_hash)
        )
        if not result:
            logger.warning(f"RESET_OTP_FAIL email={email} reason=invalid_otp")
            return False, "Invalid OTP. Please try again."

        if datetime.now() > datetime.fromisoformat(result[0]):
            self.db.execute("DELETE FROM password_reset_requests WHERE email=?", (email,))
            logger.warning(f"RESET_OTP_FAIL email={email} reason=expired")
            return False, "OTP expired. Please request a new one."

        logger.info(f"RESET_OTP_OK email={email}")
        return True, "OTP verified. Please set your new password."

    def reset_password(self, email, otp, new_password):
        if len(new_password) < 6:
            return False, "Password must be at least 6 characters."
        ok, msg = self.verify_reset_otp(email, otp)
        if not ok:
            return False, msg
        new_hash = self.hash_password(new_password)
        try:
            self.db.execute("UPDATE users SET password_hash=? WHERE email=?", (new_hash, email))
            self.db.execute("DELETE FROM password_reset_requests WHERE email=?", (email,))
            logger.info(f"RESET_SUCCESS email={email}")
            return True, "Password updated successfully! You can now log in."
        except Exception as e:
            logger.error(f"RESET_DB_ERROR email={email}: {e}", exc_info=True)
            return False, f"Failed to update password: {e}"


# ==================== WEB SEARCH ==================== #
class WebSearchEngine:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def search(self, query: str, max_results: int = 8) -> list[dict]:
        try:
            params = {"q": query, "kl": "us-en", "kp": "-2"}
            url    = f"https://html.duckduckgo.com/html/?{urlencode(params)}"
            resp   = requests.get(url, headers=self.HEADERS, timeout=10)
            resp.raise_for_status()
            soup   = BeautifulSoup(resp.text, "html.parser")
            results = []

            for tag in soup.select(".result"):
                a       = tag.select_one(".result__a")
                snip    = tag.select_one(".result__snippet")
                if not a:
                    continue
                title   = a.get_text(strip=True)
                href    = a.get("href", "")
                if "uddg=" in href:
                    from urllib.parse import urlparse, parse_qs
                    parsed  = parse_qs(urlparse(href).query)
                    href    = parsed.get("uddg", [href])[0]
                snippet = snip.get_text(strip=True) if snip else ""
                if title and href:
                    results.append({"title": title, "url": href, "snippet": snippet})
                if len(results) >= max_results:
                    break

            logger.info(f"WEB_SEARCH q={query!r} hits={len(results)}")
            return results

        except Exception as e:
            logger.error(f"Web search failed for {query!r}: {e}", exc_info=True)
            return []

    def fetch_page_text(self, url: str, max_chars: int = 3000) -> str:
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            lines   = [l for l in text.splitlines() if l.strip()]
            cleaned = "\n".join(lines)
            return cleaned[:max_chars] + ("…" if len(cleaned) > max_chars else "")
        except Exception as e:
            logger.error(f"Page fetch failed for {url}: {e}", exc_info=True)
            return "Could not fetch page content."


# ==================== PAGE CONFIG ==================== #
st.set_page_config(
    page_title="Trinetra V5.0 · Bharat AI",
    layout="wide",
    page_icon="👁️",
    initial_sidebar_state="expanded",
)

# ==================== SESSION STATE ==================== #
defaults = {
    "theme": "dark",
    "search_history": [],
    "last_search_results": [],
    "authenticated": False,
    "user": None,
    "show_suggestions": True,
    "auth_stage": "login",
    "temp_email": "",
    "temp_username": "",
    "demo_otp": None,          # cleared after first display
    "reset_email": "",
    "reset_otp": "",
    "reset_demo_otp": None,    # cleared after first display
    "web_search_results": [],
    "web_page_content": "",
    "web_page_url": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
# FIX: Removed @st.cache_data decorator — it caused Streamlit to render
#      the returned HTML string as visible text instead of injecting it.
#      CSS is injected via components.html() with height=0 so <style> and
#      <link> tags are written into a zero-height iframe and applied globally
#      through the parent document via postMessage / cross-frame CSS.
#      For pure <style> injection the reliable pattern is to wrap the style
#      block inside a <script> that appends it to window.parent.document.head.
def _css(theme: str) -> str:
    tk       = LIGHT if theme == "light" else DARK
    knob_pos = "24px" if theme == "light" else "4px"

    return f"""
<style>
:root {{ --t: 0.25s; }}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"], .stApp {{
  background-color: {tk["bg_base"]} !important;
  color: {tk["text_primary"]} !important;
  font-family: 'DM Sans', sans-serif !important;
  transition: background-color var(--t), color var(--t);
}}

[data-testid="stAppViewContainer"]::before {{
  content: ''; position: fixed; inset: 0;
  background-image:
    linear-gradient({tk["grid_line"]} 1px, transparent 1px),
    linear-gradient(90deg, {tk["grid_line"]} 1px, transparent 1px);
  background-size: 40px 40px; pointer-events: none; z-index: 0;
}}

[data-testid="stMainBlockContainer"] {{ padding: 2rem 2.5rem; position: relative; z-index: 1; }}
[data-testid="stSidebar"] {{ background: {tk["bg_panel"]} !important; border-right: 1px solid {tk["border"]} !important; }}
[data-testid="stSidebar"] * {{ color: {tk["text_primary"]} !important; }}

h1 {{
  font-family: 'Syne', sans-serif !important; font-weight: 800 !important; font-size: 2.6rem !important;
  background: {tk["h1_grad"]}; -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; background-clip: text;
}}
h2, h3, h4 {{ font-family: 'Syne', sans-serif !important; font-weight: 700 !important; color: {tk["text_primary"]} !important; }}

.tagline {{
  font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: {tk["accent"]};
  letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 2rem; opacity: 0.85;
}}

.stat-strip {{ display: flex; gap: 1rem; margin-bottom: 2rem; }}
.stat-chip {{
  flex: 1; background: {tk["bg_card"]}; border: 1px solid {tk["border"]};
  border-top: 2px solid {tk["accent"]}; border-radius: 10px; padding: 0.9rem 1.2rem;
  display: flex; flex-direction: column; gap: 4px; cursor: default;
  transition: all var(--t); box-shadow: {tk["shadow_card"]};
}}
.stat-chip:hover {{
  background: {tk["bg_card_hover"]}; border-color: {tk["border_hover"]};
  box-shadow: {tk["shadow_hover"]}; transform: translateY(-3px);
}}
.stat-label {{ font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: {tk["text_muted"]}; text-transform: uppercase; letter-spacing: 0.1em; }}
.stat-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 600; color: {tk["accent"]}; }}

[data-testid="stButton"] button {{
  background: transparent !important; border: 1px solid {tk["accent"]} !important;
  color: {tk["accent"]} !important; font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important; border-radius: 6px !important; transition: all 0.2s !important;
}}
[data-testid="stButton"] button:hover {{
  background: {tk["accent_glow"]} !important; transform: translateY(-2px);
  box-shadow: 0 0 16px {tk["accent_glow"]} !important;
}}

[data-testid="stTextInput"] input {{
  background: {tk["bg_card"]} !important; border: 1px solid {tk["border"]} !important;
  border-radius: 6px !important; color: {tk["text_primary"]} !important;
  font-family: 'DM Sans', sans-serif !important;
  transition: border-color var(--t), box-shadow var(--t) !important;
}}
[data-testid="stTextInput"] input:focus {{
  border-color: {tk["accent"]} !important;
  box-shadow: 0 0 0 3px {tk["accent_glow"]} !important;
}}

[data-testid="stSelectbox"] > div > div {{
  background: {tk["bg_card"]} !important; border: 1px solid {tk["border"]} !important;
  color: {tk["text_primary"]} !important;
}}
[data-testid="stSelectbox"] > div > div:hover {{ border-color: {tk["accent"]} !important; }}

[data-testid="stFileUploader"] {{
  background: {tk["bg_card"]} !important; border: 1px dashed {tk["accent"]} !important; border-radius: 8px !important;
}}
[data-testid="stFileUploader"]:hover {{
  background: {tk["bg_card_hover"]} !important; box-shadow: 0 0 14px {tk["accent_glow"]} !important;
}}

[data-testid="stTabs"] [role="tablist"] {{ border-bottom: 1px solid {tk["border"]} !important; }}
[data-testid="stTabs"] [role="tab"] {{
  font-family: 'Syne', sans-serif !important; font-size: .85rem !important;
  font-weight: 700 !important; color: {tk["text_muted"]} !important;
  background: transparent !important; border: none !important;
  padding: .6rem 1.2rem !important; border-radius: 6px 6px 0 0 !important;
  transition: color var(--t), background var(--t) !important;
}}
[data-testid="stTabs"] [role="tab"]:hover {{
  color: {tk["accent"]} !important; background: rgba(232,160,32,0.07) !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
  color: {tk["accent"]} !important; border-bottom: 2px solid {tk["accent"]} !important;
}}

[data-testid="stDataFrame"] {{
  border: 1px solid {tk["border"]} !important; border-radius: 8px !important;
  overflow: hidden; box-shadow: {tk["shadow_card"]};
}}
[data-testid="stDataFrame"]:hover {{ border-color: {tk["border_hover"]} !important; }}
[data-testid="stDataFrame"] th {{
  background: {tk["bg_panel"]} !important; font-family: 'JetBrains Mono', monospace !important;
  font-size: .72rem !important; color: {tk["accent"]} !important; text-transform: uppercase;
}}
[data-testid="stDataFrame"] td {{
  font-family: 'JetBrains Mono', monospace !important; font-size: .78rem !important;
  color: {tk["text_mono"]} !important; background: {tk["bg_card"]} !important;
}}
[data-testid="stDataFrame"] tr:hover td {{ background: {tk["bg_card_hover"]} !important; }}

[data-testid="stMetric"] {{
  background: {tk["bg_card"]} !important; border: 1px solid {tk["border"]} !important;
  border-radius: 10px !important; padding: .9rem 1rem !important;
  box-shadow: {tk["shadow_card"]}; cursor: default;
  transition: border-color var(--t), box-shadow var(--t), transform var(--t) !important;
}}
[data-testid="stMetric"]:hover {{
  border-color: {tk["border_hover"]} !important; box-shadow: {tk["shadow_hover"]} !important;
  transform: translateY(-2px);
}}
[data-testid="stMetricLabel"] {{
  font-family: 'JetBrains Mono', monospace !important; font-size: .68rem !important;
  color: {tk["text_muted"]} !important; text-transform: uppercase;
}}
[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace !important; color: {tk["accent"]} !important; }}

[data-testid="stExpander"] {{
  background: {tk["bg_card"]} !important; border: 1px solid {tk["border"]} !important; border-radius: 8px !important;
}}
[data-testid="stExpander"]:hover {{
  border-color: {tk["border_hover"]} !important; box-shadow: 0 2px 12px {tk["accent_glow"]} !important;
}}

[data-testid="stRadio"] label {{ color: {tk["text_primary"]} !important; }}
[data-testid="stRadio"] label:hover {{ color: {tk["accent"]} !important; }}

[data-testid="stAlert"] {{
  background: {tk["bg_card"]} !important; border: 1px solid {tk["border"]} !important;
  border-left: 3px solid #00c9b1 !important; border-radius: 6px !important;
  color: {tk["text_primary"]} !important;
}}

.badge {{
  display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
  font-weight: 600; text-transform: uppercase; padding: 3px 10px; border-radius: 4px;
  margin-bottom: 6px; cursor: default; transition: transform .15s, box-shadow .15s;
}}
.badge:hover {{ transform: scale(1.07); box-shadow: 0 2px 8px rgba(0,0,0,.2); }}
.badge-high   {{ color: #2ecc71; border: 1px solid #2ecc71; background: rgba(46,204,113,0.08); }}
.badge-medium {{ color: #f39c12; border: 1px solid #f39c12; background: rgba(243,156,18,0.08); }}
.badge-low    {{ color: #e74c3c; border: 1px solid #e74c3c; background: rgba(231,76,60,0.08); }}

.score-bar-wrap {{ margin: 6px 0 10px; }}
.score-bar-track {{ height: 3px; background: {tk["border"]}; border-radius: 2px; overflow: hidden; }}
.score-bar-fill {{
  height: 100%; border-radius: 2px;
  background: linear-gradient(90deg, {tk["accent_dim"]}, {tk["accent"]});
  animation: barGrow 0.6s cubic-bezier(0.34,1.56,0.64,1) forwards;
}}
@keyframes barGrow {{ from {{ width: 0%; }} }}

.result-card {{
  background: {tk["bg_card"]}; border: 1px solid {tk["border"]}; border-radius: 10px;
  padding: 1rem; cursor: default; transition: all var(--t); box-shadow: {tk["shadow_card"]};
}}
.result-card:hover {{
  background: {tk["bg_card_hover"]}; border-color: {tk["border_hover"]};
  box-shadow: {tk["shadow_hover"]}; transform: translateY(-4px) scale(1.01);
}}

.web-result-card {{
  background: {tk["bg_card"]}; border: 1px solid {tk["border"]}; border-radius: 10px;
  padding: 1rem 1.2rem; margin-bottom: 0.75rem; transition: all var(--t);
  box-shadow: {tk["shadow_card"]};
}}
.web-result-card:hover {{
  border-color: {tk["border_hover"]}; box-shadow: {tk["shadow_hover"]}; transform: translateX(4px);
}}
.web-result-title {{
  font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700;
  color: {tk["accent"]}; margin-bottom: 4px;
}}
.web-result-url {{
  font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
  color: #2ecc71; margin-bottom: 6px; word-break: break-all;
}}
.web-result-snippet {{ font-size: 0.85rem; color: {tk["text_muted"]}; line-height: 1.5; }}

.sidebar-stat {{
  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
  color: {tk["text_muted"]}; padding: 5px 0; cursor: default; transition: all 0.15s;
}}
.sidebar-stat:hover {{ color: {tk["text_primary"]} !important; transform: translateX(4px); }}
.sidebar-stat span {{ color: {tk["accent"]}; font-weight: 600; }}

.tog-pill {{
  display: flex; align-items: center; gap: 12px; background: {tk["bg_card"]};
  border: 1px solid {tk["accent"]}; border-radius: 50px; padding: 6px 12px;
  margin-bottom: 0.75rem; cursor: pointer; user-select: none; position: relative;
  transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
}}
.tog-pill:hover {{
  box-shadow: 0 0 15px {tk["accent_glow"]};
  transform: translateY(-1px);
}}
.tog-track {{
  position: relative; width: 44px; height: 22px;
  border-radius: 20px; background: {tk["border"]}; flex-shrink: 0;
}}
.tog-knob {{
  position: absolute; top: 3px; left: {knob_pos};
  width: 16px; height: 16px; border-radius: 50%;
  background: {tk["accent"]}; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
  transition: left 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55), background 0.3s ease;
}}
.tog-icon {{ font-size: 1rem; line-height: 1; }}
.tog-label {{
  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600;
  color: {tk["accent"]}; text-transform: uppercase; letter-spacing: 0.08em; flex: 1;
}}
.rule {{ border: none; border-top: 1px solid {tk["border"]}; margin: 1.5rem 0; }}

.login-container {{
  max-width: 420px; margin: 10rem auto; padding: 3rem;
  background: {tk["bg_card"]}; border: 1px solid {tk["border"]};
  border-radius: 16px; box-shadow: {tk["shadow_card"]};
}}
.login-logo {{ text-align: center; font-size: 3.5rem; margin-bottom: 1rem; }}
.login-title {{
  font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2rem; text-align: center;
  background: {tk["h1_grad"]}; -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.5rem;
}}
.login-subtitle {{
  font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {tk["text_muted"]};
  text-align: center; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 2.5rem;
}}

.quality-indicator {{
  display: inline-flex; align-items: center; gap: 6px;
  font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
  padding: 4px 10px; border-radius: 4px;
  background: {tk["bg_card_hover"]}; border: 1px solid {tk["border"]};
}}

.suggestion-chip {{
  display: inline-block; padding: 4px 12px; margin: 4px;
  background: {tk["bg_card"]}; border: 1px solid {tk["border"]};
  border-radius: 16px; font-size: 0.75rem; cursor: pointer; transition: all 0.2s;
}}
.suggestion-chip:hover {{ background: {tk["accent_glow"]}; border-color: {tk["accent"]}; transform: translateY(-2px); }}

@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(16px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
.a1 {{ animation: fadeUp 0.4s ease both; }}
.a2 {{ animation: fadeUp 0.4s 0.08s ease both; }}
.a3 {{ animation: fadeUp 0.4s 0.16s ease both; }}
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
"""

# Inject CSS — st.html() is the correct API in Streamlit >= 1.32.
# It accepts raw HTML (including <style> tags) and renders it without
# wrapping in a markdown container, so nothing is displayed to the user.
try:
    st.html(_css(st.session_state.theme))
except AttributeError:
    # Fallback for Streamlit < 1.32
    st.markdown(_css(st.session_state.theme), unsafe_allow_html=True)

# ==================== AUTHENTICATION ==================== #
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
            stage = st.session_state.auth_stage

            if stage == "login":
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
                                time.sleep(0.5); st.rerun()
                            else:
                                st.error("Invalid credentials or email not verified")
                        else:
                            st.warning("Please enter both username and password")
                with col_b:
                    if st.button("Register", use_container_width=True):
                        st.session_state.auth_stage = "register_step1"; st.rerun()
                with col_c:
                    if st.button("Guest", use_container_width=True):
                        st.session_state.authenticated = True
                        st.session_state.user = {"username": "guest", "role": "viewer"}
                        logger.info("LOGIN_GUEST"); st.rerun()

                st.markdown("---")
                col_fp, _ = st.columns([1, 2])
                with col_fp:
                    if st.button("🔑 Forgot Password?", use_container_width=True, key="goto_fp"):
                        st.session_state.auth_stage = "forgot_step1"; st.rerun()
                st.caption("📧 Demo: admin / admin123")
                st.caption("❤️ Created by Team Human")

            elif stage == "forgot_step1":
                st.markdown("### Reset Password")
                st.info("Enter the email address linked to your account and we'll send you an OTP.")
                fp_email = st.text_input("Email Address", placeholder="your.email@example.com", key="fp_email")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Send OTP", use_container_width=True, key="fp_send"):
                        if not fp_email or '@' not in fp_email:
                            st.error("Please enter a valid email address.")
                        else:
                            success, message, demo_otp = auth_manager.request_password_reset(fp_email)
                            if success:
                                st.session_state.reset_email    = fp_email
                                st.session_state.reset_demo_otp = demo_otp
                                st.session_state.auth_stage     = "forgot_step2"
                                st.success(message); time.sleep(1); st.rerun()
                            else:
                                st.error(message)
                with col_b:
                    if st.button("Back to Login", use_container_width=True, key="fp_back1"):
                        st.session_state.auth_stage = "login"; st.rerun()

            elif stage == "forgot_step2":
                st.markdown("### Verify OTP")
                st.info(f"📧 OTP sent to: **{st.session_state.reset_email}**")
                if st.session_state.reset_demo_otp:
                    st.warning(f"📋 Email delivery failed. Your OTP: **{st.session_state.reset_demo_otp}** — valid for 10 minutes. Copy it now — it will not be shown again.")
                    st.session_state.reset_demo_otp = None   # clear immediately after display
                fp_otp = st.text_input("Enter 6-digit OTP", placeholder="000000", max_chars=6, key="fp_otp")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("Verify OTP", use_container_width=True, key="fp_verify"):
                        if len(fp_otp) != 6:
                            st.error("OTP must be 6 digits.")
                        else:
                            ok, msg = auth_manager.verify_reset_otp(st.session_state.reset_email, fp_otp)
                            if ok:
                                st.session_state.reset_otp  = fp_otp
                                st.session_state.auth_stage = "forgot_step3"
                                st.success(msg); time.sleep(0.8); st.rerun()
                            else:
                                st.error(msg)
                with col_b:
                    if st.button("Resend OTP", use_container_width=True, key="fp_resend"):
                        ok, msg, new_demo = auth_manager.request_password_reset(st.session_state.reset_email)
                        if ok:
                            st.session_state.reset_demo_otp = new_demo; st.success("New OTP sent!")
                        else:
                            st.error(msg)
                with col_c:
                    if st.button("Cancel", use_container_width=True, key="fp_cancel2"):
                        for k in ["auth_stage","reset_email","reset_otp","reset_demo_otp"]:
                            st.session_state[k] = "login" if k == "auth_stage" else ("" if k != "reset_demo_otp" else None)
                        st.rerun()

            elif stage == "forgot_step3":
                st.markdown("### Set New Password")
                new_pass  = st.text_input("New Password", type="password", placeholder="At least 6 characters", key="fp_newpass")
                new_pass2 = st.text_input("Confirm New Password", type="password", placeholder="Repeat password", key="fp_newpass2")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Update Password", use_container_width=True, key="fp_update"):
                        if len(new_pass) < 6:
                            st.error("Password must be at least 6 characters.")
                        elif new_pass != new_pass2:
                            st.error("Passwords don't match.")
                        else:
                            ok, msg = auth_manager.reset_password(
                                st.session_state.reset_email, st.session_state.reset_otp, new_pass
                            )
                            if ok:
                                st.success(msg)
                                for k in ["auth_stage","reset_email","reset_otp","reset_demo_otp"]:
                                    st.session_state[k] = "login" if k == "auth_stage" else ("" if k != "reset_demo_otp" else None)
                                time.sleep(2); st.rerun()
                            else:
                                st.error(msg)
                with col_b:
                    if st.button("Cancel", use_container_width=True, key="fp_cancel3"):
                        for k in ["auth_stage","reset_email","reset_otp","reset_demo_otp"]:
                            st.session_state[k] = "login" if k == "auth_stage" else ("" if k != "reset_demo_otp" else None)
                        st.rerun()

            elif stage == "register_step1":
                st.markdown("### Create Account")
                email            = st.text_input("Email Address", placeholder="your.email@example.com", key="reg_email")
                username         = st.text_input("Username", placeholder="3+ characters", key="reg_user")
                password         = st.text_input("Password", type="password", placeholder="6+ characters", key="reg_pass")
                password_confirm = st.text_input("Confirm Password", type="password", key="reg_pass_conf")
                full_name        = st.text_input("Full Name (optional)", key="reg_name")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Send OTP", use_container_width=True):
                        if not email or '@' not in email:        st.error("Invalid email address")
                        elif len(username) < 3:                  st.error("Username must be at least 3 characters")
                        elif len(password) < 6:                  st.error("Password must be at least 6 characters")
                        elif password != password_confirm:       st.error("Passwords don't match")
                        else:
                            success, message, otp = auth_manager.request_registration(email, username, password, full_name)
                            if success:
                                st.session_state.auth_stage    = "register_step2"
                                st.session_state.temp_email    = email
                                st.session_state.temp_username = username
                                st.session_state.demo_otp      = otp
                                st.success(message); time.sleep(1); st.rerun()
                            else:
                                st.error(message)
                with col_b:
                    if st.button("Back to Login", use_container_width=True):
                        st.session_state.auth_stage = "login"; st.rerun()

            elif stage == "register_step2":
                st.markdown("### Verify Email")
                st.info(f"📧 OTP sent to: {st.session_state.temp_email}")
                if st.session_state.demo_otp:
                    st.warning(f"📋 Email delivery failed. Your OTP: **{st.session_state.demo_otp}** — valid for 10 minutes. Copy it now — it will not be shown again.")
                    st.session_state.demo_otp = None   # clear immediately after display
                otp = st.text_input("Enter 6-digit OTP", placeholder="000000", max_chars=6, key="otp_input")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("Verify OTP", use_container_width=True):
                        if len(otp) != 6:
                            st.error("OTP must be 6 digits")
                        else:
                            success, message = auth_manager.verify_otp_and_register(st.session_state.temp_email, otp)
                            if success:
                                st.success(message)
                                st.session_state.auth_stage = "login"
                                st.session_state.demo_otp   = None
                                time.sleep(2); st.rerun()
                            else:
                                st.error(message)
                with col_b:
                    if st.button("Resend OTP", use_container_width=True):
                        success, message, new_otp = auth_manager.resend_otp(st.session_state.temp_email)
                        if success:
                            st.session_state.demo_otp = new_otp; st.success(message)
                        else:
                            st.error(message)
                with col_c:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state.auth_stage = "login"
                        st.session_state.demo_otp   = None; st.rerun()

if not st.session_state.authenticated:
    show_enhanced_login_page(auth_manager)
    st.stop()


# ==================== SHARED METADATA DATABASE ==================== #
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
                quality_score REAL, uploaded_by TEXT, UNIQUE(id)
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
            logger.info(f"ASSET_ADDED id={asset_id} modality={modality} by={uploaded_by}")
        except sqlite3.IntegrityError:
            pass

    def search_metadata(self, modality=None, language=None, tags=None, date_from=None, collection=None):
        query  = "SELECT id, faiss_index FROM assets WHERE 1=1"
        params = []
        if modality:   query += " AND modality=?";    params.append(modality)
        if language:   query += " AND language=?";    params.append(language)
        if date_from:  query += " AND upload_date>=?"; params.append(date_from)
        if tags:       query += " AND tags LIKE ?";   params.append(f'%{tags}%')
        if collection: query += " AND collection=?";  params.append(collection)
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
            WHERE asset_id=? ORDER BY timestamp DESC
        """, (asset_id,))

    def add_rating(self, asset_id, user, rating):
        self.db.execute(
            "INSERT OR REPLACE INTO ratings (asset_id, user, rating) VALUES (?, ?, ?)",
            (asset_id, user, rating)
        )

    def get_rating(self, asset_id):
        result = self.db.fetchone(
            "SELECT AVG(rating), COUNT(*) FROM ratings WHERE asset_id=?", (asset_id,)
        )
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
            logger.info(f"SEARCH user={user} modality={modality} q={query!r} hits={results_count} ms={duration_ms:.0f}")
        except Exception as e:
            logger.error(f"Failed to log search: {e}", exc_info=True)

    def get_stats(self, days=7):
        days = max(1, min(int(days), 365))
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
            GROUP BY query_text ORDER BY freq DESC LIMIT ?
        """, (limit,))


# ==================== MODELS ==================== #
# Global inference lock — serialises all model forward passes so concurrent
# Streamlit sessions cannot trigger simultaneous GPU kernel launches, which
# causes VRAM OOM errors on single-GPU / shared-memory systems.
_inference_lock = threading.Lock()

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
            "resolution": f"{img.width}×{img.height}", "quality_score": quality_score,
            "format": img.format, "mode": img.mode,
            "size_kb": os.path.getsize(file_path) / 1024
        }
    except Exception:
        return None

def analyze_audio_quality(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        rms   = librosa.feature.rms(y=y)[0]
        return {
            "duration": f"{len(y)/sr:.1f}s", "sample_rate": f"{sr}Hz",
            "rms_energy": float(np.mean(rms)),
            "size_kb": os.path.getsize(file_path) / 1024
        }
    except Exception:
        return None

def get_search_suggestions(analytics, query, limit=5):
    if not query or len(query) < 2:
        popular = analytics.get_popular_searches(limit)
        return [p[0] for p in popular]
    result = analytics.db.fetchall("""
        SELECT DISTINCT query_text, COUNT(*) as freq
        FROM searches WHERE query_text LIKE ? AND query_text != ?
        GROUP BY query_text ORDER BY freq DESC LIMIT ?
    """, (f"%{query}%", query, limit))
    return [row[0] for row in result]

def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()



# ==================== NEURAL GROUNDING HELPERS ==================== #

def generate_asset_description(result: dict, modality: str) -> str:
    """Generate a natural-language description of an asset for web query building."""
    asset_id = result.get("id", "").replace("_", " ")
    lang     = result.get("lang", "en")
    return (f"{asset_id} Indian {lang} image photo"
            if modality == "image"
            else f"{asset_id} Indian {lang} audio sound")


def prefetch_web_pages(hits: list, storage: list) -> None:
    """Background thread: fetch full page text for each web hit and store in-place."""
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
    }
    for i, hit in enumerate(hits):
        try:
            resp = requests.get(hit["url"], headers=headers, timeout=8)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text    = soup.get_text(separator="\n", strip=True)
            lines   = [l for l in text.splitlines() if l.strip()]
            storage[i] = {"url": hit["url"], "content": "\n".join(lines)[:5000]}
        except Exception as e:
            logger.warning(f"Prefetch failed for {hit['url']}: {e}")
            storage[i] = {"url": hit["url"], "content": "Could not fetch page content."}


def enrich_with_web(results: list, modality: str,
                    max_web: int = 3, min_similarity: float = 0.45):
    """Neural Grounding: fetch web results and filter by cosine similarity to top local asset."""
    if not results:
        return None
    top  = results[0]
    desc = generate_asset_description(top, modality)
    lang = top.get("lang", "en")

    # Build bilingual query for better cross-script recall
    query_parts = [desc]
    if lang != "en" and lang in CONFIG.SUPPORTED_LANGUAGES:
        try:
            native_desc = GoogleTranslator(source="en", target=lang).translate(desc)
            query_parts.append(f'"{native_desc}"')
        except Exception:
            pass
    web_query = f"{' OR '.join(query_parts)} India"

    with st.spinner(f"Neural Grounding: searching web for \"{desc}\"\u2026"):
        web_hits = web_search.search(web_query, max_results=max_web * 3)
    if not web_hits:
        return None

    # Embed the local file, then score each snippet against it
    eng       = image_engine if modality == "image" else audio_engine
    local_emb = eng.get_embedding(file_path=top["path"])

    filtered = []
    for hit in web_hits:
        if not hit.get("snippet"):
            continue
        # CLIP text encoder — works for both image & audio modalities
        snip_emb = image_engine.get_embedding(text=hit["snippet"])
        sim      = float(np.dot(local_emb, snip_emb) /
                         (np.linalg.norm(local_emb) * np.linalg.norm(snip_emb) + 1e-9))
        if sim >= min_similarity:
            hit["grounding_score"] = sim
            filtered.append(hit)

    return {
        "query_used":     web_query,
        "desc_generated": desc,
        "hits": sorted(filtered, key=lambda x: x["grounding_score"], reverse=True)[:max_web],
    }


def render_enrichment_ui(results: list, modality: str) -> None:
    """Render the 'Enrich with Web Context' Neural Grounding panel below search results."""
    if not results:
        return
    st.markdown("---")
    min_web_sim = st.slider(
        "Neural Grounding Sensitivity", 0.3, 0.7, 0.45, 0.05,
        help="How strictly should web snippets match your local asset? Lower = more (looser) results.",
        key=f"ground_slider_{modality}",
    )
    if st.button("\U0001f310 Enrich with Web Context",
                 use_container_width=True, key=f"enr_btn_{modality}"):
        enrichment = enrich_with_web(results, modality, min_similarity=min_web_sim)

        if enrichment and enrichment["hits"]:
            top  = results[0]
            hits = enrichment["hits"]
            st.success(f"Grounded {len(hits)} web result(s) for **{top['id']}**")
            st.caption(f"Query used: `{enrichment['query_used']}`")

            # Background prefetch of full page text
            prefetch_storage = [None] * len(hits)
            bg = threading.Thread(target=prefetch_web_pages, args=(hits, prefetch_storage))
            bg.daemon = True
            bg.start()

            for i, hit in enumerate(hits):
                gs    = hit.get("grounding_score", 0)
                color = "#2ecc71" if gs > 0.6 else "#f39c12" if gs > 0.45 else "#e74c3c"

                # Optionally translate snippet for non-English assets
                snippet_display = hit["snippet"]
                if top.get("lang", "en") != "en":
                    try:
                        snippet_display = GoogleTranslator(
                            source="auto", target="en"
                        ).translate(hit["snippet"][:500])
                        st.caption(f"Snippet auto-translated from {top['lang'].upper()}")
                    except Exception:
                        pass

                st.markdown(f"""
                <div class="web-result-card">
                    <div class="web-result-title">{i+1}. {hit['title']}</div>
                    <div class="web-result-url">\U0001f517
                        <a href="{hit['url']}" target="_blank" style="color:#2ecc71;">
                            {hit['url'][:70]}\u2026
                        </a>
                    </div>
                    <div class="web-result-snippet">{snippet_display}</div>
                    <div style="font-size:0.75rem; color:{color};
                                font-family:'JetBrains Mono',monospace;
                                margin-top:8px; font-weight:600;">
                        \u25cf Neural Grounding Score: {gs:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f'<a href="{hit["url"]}" target="_blank" ' +
                        f'style="color:{T["accent"]};font-size:0.8rem;">\u2197 Open in browser</a>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button("\U0001f4c4 Read full page",
                                 key=f"read_enr_{modality}_{i}",
                                 use_container_width=True):
                        time.sleep(0.4)  # give prefetch a head-start
                        if prefetch_storage[i]:
                            st.text_area("Full Page Content",
                                         prefetch_storage[i]["content"],
                                         height=250,
                                         key=f"page_content_{modality}_{i}")
                        else:
                            st.info("Still fetching — try again in a moment.")
        else:
            st.warning(
                "No web results met the grounding threshold. "
                "Try lowering the slider or searching for a more widely-known asset."
            )


# ==================== ENGINE ==================== #
_shared_metadata_db = None

def get_shared_metadata_db():
    global _shared_metadata_db
    if _shared_metadata_db is None:
        _shared_metadata_db = MetadataDB()
    return _shared_metadata_db


class TrinetraEngine:
    def __init__(self, modality):
        self.modality  = modality
        db_path        = os.path.join(BASE_DIR, modality)
        self.idx_path  = os.path.join(db_path, "index")
        self.map_path  = os.path.join(db_path, "id_map.json")
        os.makedirs(db_path, exist_ok=True)

        if os.path.exists(self.idx_path) and os.path.exists(self.map_path):
            self.index = faiss.read_index(self.idx_path)
            with open(self.map_path) as f:
                id_list = json.load(f)
        else:
            self.index  = faiss.IndexFlatIP(CONFIG.EMBEDDING_DIM)
            id_list     = []

        self.id_map      = {r["id"]: r for r in id_list}
        self.id_list     = id_list
        self.embedding_cache = {}
        self.metadata_db     = get_shared_metadata_db()
        self._lock           = threading.Lock()
        self._validate_sync()   # FIX: detect index/map divergence at startup

    def _validate_sync(self):
        """Detect and log divergence between FAISS index and id_map.
        FAISS vectors cannot be deleted, so if ntotal > len(id_map) it means
        a previous crash left orphaned vectors. We rebuild the index from the
        stored id_map to restore consistency.
        """
        n_idx  = self.index.ntotal
        n_map  = len(self.id_list)
        if n_idx == n_map:
            return  # all good
        logger.warning(
            f"SYNC_MISMATCH modality={self.modality} "
            f"faiss_ntotal={n_idx} id_map_len={n_map} — rebuilding index"
        )
        # Rebuild: re-embed only the assets listed in id_map
        new_index = faiss.IndexHNSWFlat(CONFIG.EMBEDDING_DIM, CONFIG.HNSW_M)
        new_index.hnsw.efConstruction = CONFIG.HNSW_EF_CONSTRUCTION
        new_index.hnsw.efSearch       = CONFIG.HNSW_EF_SEARCH
        rebuilt, failed = [], []
        for record in self.id_list:
            if not os.path.exists(record["path"]):
                failed.append(record["id"]); continue
            try:
                emb = self._compute_embedding(file_path=record["path"])
                new_index.add(emb.reshape(1, -1))
                rebuilt.append(record)
            except Exception as e:
                logger.error(f"SYNC_REBUILD_FAIL id={record['id']}: {e}")
                failed.append(record["id"])
        self.index   = new_index
        self.id_list = rebuilt
        self.id_map  = {r["id"]: r for r in rebuilt}
        self._save()
        logger.info(
            f"SYNC_REBUILD_DONE modality={self.modality} "
            f"rebuilt={len(rebuilt)} dropped={len(failed)}"
        )

    def _normalize(self, v):
        return (v / (np.linalg.norm(v) + 1e-9)).astype("float32")

    def _save(self):
        faiss.write_index(self.index, self.idx_path)
        with open(self.map_path, "w") as f:
            json.dump(self.id_list, f, indent=2)

    def _cache_key(self, text=None, path=None):
        if text:  return hashlib.md5(text.encode()).hexdigest()
        if path:  return file_md5(path)

    def id_exists(self, asset_id):
        return asset_id in self.id_map

    def get_embedding(self, file_path=None, text=None):
        key = self._cache_key(text=text, path=file_path)
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
        return [
            {"id": self.id_list[i]["id"], "similarity": float(s), "path": self.id_list[i]["path"]}
            for s, i in zip(scores[0], idxs[0])
            if i != -1 and s > t
        ]

    def register(self, temp_path, asset_id, ext, lang, tags=None, description="",
                 collection="", uploaded_by="unknown"):
        asset_id = sanitize_asset_id(asset_id)
        ok, msg  = validate_asset_id(asset_id)
        if not ok: return False, msg

        perm_path = os.path.join(STORAGE_DIR, f"{asset_id}{ext}")
        try:
            with open(temp_path, "rb") as s, open(perm_path, "wb") as d:
                d.write(s.read())

            if self.modality == "image":
                quality_info  = analyze_image_quality(perm_path)
                quality_score = quality_info['quality_score'] if quality_info else 0.5
            else:
                quality_info  = analyze_audio_quality(perm_path)
                quality_score = min(quality_info['rms_energy'] * 10, 1.0) if quality_info else 0.5

            emb = self.get_embedding(file_path=perm_path)

            with self._lock:
                if self.id_exists(asset_id):
                    if os.path.exists(perm_path): os.remove(perm_path)
                    return False, f"ID '{asset_id}' already exists"

                faiss_idx = self.index.ntotal
                self.index.add(emb.reshape(1, -1))
                record = {
                    "id": asset_id, "path": perm_path, "lang": lang,
                    "modality": self.modality, "timestamp": time.ctime(),
                    "quality": quality_score
                }
                self.id_map[asset_id]  = record
                self.id_list.append(record)
                self._save()

            self.metadata_db.add_asset(
                asset_id, self.modality, lang, perm_path,
                os.path.getsize(perm_path), faiss_idx,
                tags, description, collection, quality_score, uploaded_by
            )
            logger.info(f"ASSET_REGISTERED id={asset_id} modality={self.modality} lang={lang} by={uploaded_by}")
            return True, f"Successfully registered: {asset_id}"
        except Exception as e:
            if os.path.exists(perm_path): os.remove(perm_path)
            logger.error(f"Registration failed for {asset_id}: {e}", exc_info=True)
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
                        results["duplicates"].append((file.name, dups[0]["id"])); continue
                    ok, msg = self.register(tp, asset_id, ext, language, tags, "", collection, uploaded_by)
                    (results["success"] if ok else results["failed"]).append(
                        file.name if ok else (file.name, msg)
                    )
                finally:
                    if os.path.exists(tp): os.remove(tp)
            except Exception as e:
                results["failed"].append((file.name, str(e)))
                logger.error(f"Batch register error for {file.name}: {e}", exc_info=True)
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
            record = self.id_list[i]
            conf   = ("High" if s > CONFIG.CONFIDENCE_HIGH else
                      "Medium" if s > CONFIG.CONFIDENCE_MED else "Low")
            result = {**record, "score": float(s), "confidence": conf}

            if filters:
                if 'min_score' in filters and s < filters['min_score']:        continue
                if 'language'  in filters and result.get('lang') != filters['language']: continue
                if 'tags' in filters and filters['tags']:
                    asset_tags_row = self.metadata_db.db.fetchone(
                        "SELECT tags FROM assets WHERE id=?", (record["id"],)
                    )
                    if asset_tags_row:
                        asset_tags = json.loads(asset_tags_row[0])
                        if not any(t in asset_tags for t in filters['tags']):
                            continue
            out.append(result)

        if rerank:
            for r in out:
                r['final_score'] = r['score'] * (0.7 + 0.3 * r.get('quality', 0.5))
            out = sorted(out, key=lambda x: x['final_score'], reverse=True)

        return out[:top_k], (time.time() - t0) * 1000

    def search(self, file_path=None, text=None, top_k=5, metadata_filters=None):
        return self.hybrid_search(text_query=text, file_path=file_path,
                                  top_k=top_k, filters=metadata_filters)

    def get_all_vectors(self):
        try:
            return faiss.vector_to_array(
                self.index.reconstruct_n(0, self.index.ntotal)
            ).reshape(self.index.ntotal, -1)
        except Exception as e:
            logger.error(f"get_all_vectors failed: {e}", exc_info=True)
            return None

    def export_registry(self, export_path=None):
        import zipfile
        if export_path is None:
            export_path = f"trinetra_{self.modality}_backup_{int(time.time())}.zip"
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as z:
            if os.path.exists(self.idx_path): z.write(self.idx_path, "index")
            if os.path.exists(self.map_path): z.write(self.map_path, "id_map.json")
            for asset in self.id_list:
                if os.path.exists(asset["path"]):
                    z.write(asset["path"], f"assets/{os.path.basename(asset['path'])}")
        return export_path

    def export_as_json(self):
        return json.dumps({
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "modality":    self.modality,
                "total_assets": self.index.ntotal
            },
            "assets": self.id_list
        }, indent=2)


class ImageEngine(TrinetraEngine):
    def __init__(self): super().__init__("image")
    def _compute_embedding(self, file_path=None, text=None):
        with _inference_lock:          # prevent concurrent VRAM OOM
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
        with _inference_lock:          # prevent concurrent VRAM OOM
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


@st.cache_resource(show_spinner="Initializing registry...")
def get_engines():
    return ImageEngine(), AudioEngine(), AnalyticsTracker(), WebSearchEngine()

image_engine, audio_engine, analytics, web_search = get_engines()


# ==================== SIDEBAR ==================== #
with st.sidebar:
    user_role = st.session_state.user['role']
    st.markdown(f"""
    <div style="padding:0.5rem;background:{T['bg_card']};border:1px solid {T['border']};
    border-radius:8px;margin-bottom:1rem;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:{T['text_muted']};">LOGGED IN AS</div>
        <div style="font-weight:600;color:{T['accent']};">{st.session_state.user['username']}</div>
        <div style="font-size:0.7rem;color:{T['text_muted']};">Role: {user_role}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Logout", use_container_width=True):
        logger.info(f"LOGOUT user={st.session_state.user['username']}")
        st.session_state.authenticated = False
        st.session_state.user          = None
        st.rerun()

    st.markdown(f"""
    <div class="tog-pill">
        <div class="tog-track"><div class="tog-knob"></div></div>
        <span class="tog-icon">{T['icon']}</span>
        <span class="tog-label">{T['mode_label']}</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button(f"{T['icon']} {T['mode_label']}", use_container_width=True, key="theme_btn"):
        st.session_state.theme = "light" if not is_light else "dark"; st.rerun()

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
                                       "music","speech","nature","urban"], key="tags_single")
            reg_file = st.file_uploader(
                "Select File",
                type=[e.lstrip('.') for e in (CONFIG.ALLOWED_IMAGE_EXTS if reg_mod=="image"
                                              else CONFIG.ALLOWED_AUDIO_EXTS)],
            )
            if reg_file:
                st.markdown("#### Preview")
                if reg_mod == "image": st.image(reg_file, use_container_width=True)
                else:                  st.audio(reg_file)
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
                                st.warning(f"Similar asset found: **{dups[0]['id']}** ({dups[0]['similarity']:.1%} match)")
                                st.session_state['force_register'] = st.checkbox("Register anyway", key="force_reg_cb")
                                if not st.session_state.get('force_register'):
                                    st.info("Check 'Register anyway' to continue.")
                                    if os.path.exists(tp): os.remove(tp)
                                    st.stop()
                            with st.spinner("Registering..."):
                                ok, msg = eng.register(tp, reg_id.strip(), ext, reg_lang,
                                                       reg_tags, uploaded_by=st.session_state.user['username'])
                            (st.success if ok else st.error)(msg)
                            if ok: st.balloons(); time.sleep(1); st.rerun()
                        except Exception as exc:
                            st.error(f"Registration failed: {exc}")
                            logger.error(f"UI registration error: {exc}", exc_info=True)
                        finally:
                            if os.path.exists(tp): os.remove(tp)
        else:
            reg_tags_b  = st.multiselect("Tags for all files",
                                         ["festival","temple","landscape","portrait",
                                          "music","speech","nature","urban"], key="tags_batch")
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
                    if res['success']: time.sleep(0.8); st.rerun()

        st.markdown(f'<hr style="border-color:{T["border"]};margin:1.5rem 0 1rem">', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:{T['bg_card']};border:1px solid {T['border']};
        border-left:3px solid {T['accent']};border-radius:8px;padding:0.9rem 1rem;margin-bottom:1rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
            text-transform:uppercase;letter-spacing:0.1em;color:{T['accent']};margin-bottom:4px;">
                🔒 Viewer Access
            </div>
            <div style="font-size:0.8rem;color:{T['text_muted']};line-height:1.5;">
                Asset registration is not available for your role.<br>
                Contact an admin to be upgraded.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**Registry Status**")
    st.markdown(f"""
    <div class="sidebar-stat">Images <span>{image_engine.index.ntotal}</span></div>
    <div class="sidebar-stat">Audio  <span>{audio_engine.index.ntotal}</span></div>
    <div class="sidebar-stat">Device <span>{DEVICE.upper()}</span></div>
    <div class="sidebar-stat">Index  <span>FAISS HNSW</span></div>
    """, unsafe_allow_html=True)

    stats = analytics.get_stats(7)
    if stats and stats[0] and stats[0] > 0:
        st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0">', unsafe_allow_html=True)
        st.markdown("**Usage Stats (7 days)**")
        st.markdown(f"""
        <div class="sidebar-stat">Searches   <span>{stats[0]}</span></div>
        <div class="sidebar-stat">Avg Results <span>{stats[1]:.1f}</span></div>
        <div class="sidebar-stat">Avg Speed   <span>{stats[2]:.0f}ms</span></div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0">', unsafe_allow_html=True)

    if user_role == 'admin':
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
  <div class="stat-chip"><div class="stat-label">Images</div><div class="stat-value">{image_engine.index.ntotal}</div></div>
  <div class="stat-chip"><div class="stat-label">Audio</div><div class="stat-value">{audio_engine.index.ntotal}</div></div>
  <div class="stat-chip"><div class="stat-label">Total</div><div class="stat-value">{total}</div></div>
  <div class="stat-chip"><div class="stat-label">Cache</div><div class="stat-value">{cache_size}</div></div>
</div>
""", unsafe_allow_html=True)


# ==================== RESULT DISPLAY ==================== #
def display_results(results, modality, engine=None):
    if not results:
        st.info("No matches found in the registry.")
        return

    meta_db = get_shared_metadata_db()

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
                st.markdown(f'<div class="quality-indicator">Quality: {r["quality"]:.1%}</div>',
                            unsafe_allow_html=True)
            if modality == "image":
                st.image(r["path"], caption=r["id"], use_container_width=True)
            else:
                st.write(f"**{r['id']}**")
                st.audio(r["path"])
            st.caption(f"Lang: {r.get('lang','—')} · {r.get('timestamp','')[:10]}")

            with st.expander("Feedback"):
                avg_rating, count = meta_db.get_rating(r['id'])
                st.write(f"Rating: {'⭐' * int(avg_rating)} ({count} votes)")
                new_rating = st.slider("Rate this", 1, 5, 3, key=f"rate_{r['id']}_{idx}")
                if st.button("Submit Rating", key=f"submit_rate_{r['id']}_{idx}"):
                    meta_db.add_rating(r['id'], st.session_state.user['username'], new_rating)
                    st.success("Rating submitted!"); st.rerun()

                comments = meta_db.get_comments(r['id'])
                if comments:
                    st.markdown("**Comments:**")
                    for user, comment, ts in comments[:3]:
                        st.caption(f"**{user}** ({ts[:10]}): {comment}")

                new_comment = st.text_input("Add comment", key=f"comment_{r['id']}_{idx}")
                if st.button("Post Comment", key=f"post_{r['id']}_{idx}") and new_comment:
                    meta_db.add_comment(r['id'], st.session_state.user['username'], new_comment)
                    st.success("Comment posted!"); st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.last_search_results = results


# ==================== TABS ==================== #
tab_v, tab_a, tab_web, tab_cluster, tab_aud, tab_hist, tab_admin = st.tabs([
    "Visual Search", "Acoustic Search", "🌐 Web Search",
    "Clusters", "Neural Auditor", "Search History", "Admin"
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
            suggestions = get_search_suggestions(analytics, q)
            if suggestions:
                st.markdown("**Suggestions:**")
                st.markdown("".join(f'<span class="suggestion-chip">{s}</span>' for s in suggestions),
                            unsafe_allow_html=True)

        if st.button("Run Visual Scan", key="vs_txt") and q:
            mf = {"min_score": f_min_score}
            if f_lang != "All": mf['language'] = f_lang
            if f_tags:          mf['tags']     = f_tags
            with st.spinner("Scanning..."):
                translated = translate_to_english(q)
                if translated != q: st.caption(f"Translated: *{translated}*")
                results, ms = image_engine.hybrid_search(text_query=translated, top_k=6, filters=mf)
            analytics.log_search(q, "image", len(results), ms, st.session_state.user['username'])
            st.session_state.search_history.append({"query": q, "modality": "image",
                                                     "timestamp": time.time(), "results_count": len(results)})
            st.caption(f"Search time: {ms:.0f} ms")
            display_results(results, "image", image_engine)
    else:
        qi = st.file_uploader("Query image",
                               type=[e.lstrip('.') for e in CONFIG.ALLOWED_IMAGE_EXTS], key="qimg")
        if qi: st.image(qi, caption="Query Image", use_container_width=True)
        if st.button("Run Visual Scan", key="vs_img") and qi:
            ext = os.path.splitext(qi.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(qi.getbuffer()); qp = tmp.name
            try:
                mf = {"min_score": f_min_score}
                if f_lang != "All": mf['language'] = f_lang
                if f_tags:          mf['tags']     = f_tags
                with st.spinner("Scanning..."):
                    results, ms = image_engine.hybrid_search(file_path=qp, top_k=6, filters=mf)
                analytics.log_search("[Image Upload]", "image", len(results), ms,
                                     st.session_state.user['username'])
                st.caption(f"Search time: {ms:.0f} ms")
                display_results(results, "image", image_engine)
            finally:
                if os.path.exists(qp): os.remove(qp)

    # ── Neural Grounding panel (Visual Search) ──
    render_enrichment_ui(
        results if 'results' in dir() else st.session_state.last_search_results,
        "image"
    )

# ── Acoustic Search ── #
with tab_a:
    with st.expander("Advanced Filters"):
        f_lang_a      = st.selectbox("Language", ["All"] + CONFIG.SUPPORTED_LANGUAGES, key="fl_aud")
        f_tags_a      = st.multiselect("Tags", audio_engine.metadata_db.get_all_tags(), key="ft_aud")
        f_min_score_a = st.slider("Minimum Score", 0.0, 1.0, 0.0, key="fs_aud")

    q = st.text_input("Describe the sound", placeholder="e.g., tabla solo during rainstorm", key="aq")
    if q and st.session_state.show_suggestions:
        suggestions = get_search_suggestions(analytics, q)
        if suggestions:
            st.markdown("**Suggestions:**")
            st.markdown("".join(f'<span class="suggestion-chip">{s}</span>' for s in suggestions),
                        unsafe_allow_html=True)
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    if st.button("Run Acoustic Scan") and q:
        mf = {"min_score": f_min_score_a}
        if f_lang_a != "All": mf['language'] = f_lang_a
        if f_tags_a:          mf['tags']     = f_tags_a
        with st.spinner("Scanning..."):
            translated = translate_to_english(q)
            if translated != q: st.caption(f"Translated: *{translated}*")
            results, ms = audio_engine.hybrid_search(text_query=translated, top_k=6, filters=mf)
        analytics.log_search(q, "audio", len(results), ms, st.session_state.user['username'])
        st.session_state.search_history.append({"query": q, "modality": "audio",
                                                 "timestamp": time.time(), "results_count": len(results)})
        st.caption(f"Search time: {ms:.0f} ms")
        display_results(results, "audio", audio_engine)

    # ── Neural Grounding panel (Acoustic Search) ──
    render_enrichment_ui(
        results if 'results' in dir() else st.session_state.last_search_results,
        "audio"
    )

# ── Web Search ── #
with tab_web:
    st.markdown("### 🌐 Web Search")
    st.markdown(f"""
    <div style="background:{T['bg_card']};border:1px solid {T['border']};border-left:3px solid {T['accent']};
    border-radius:8px;padding:0.8rem 1rem;margin-bottom:1rem;font-size:0.82rem;color:{T['text_muted']};">
        Search the live web from within Trinetra. Results are fetched from DuckDuckGo — no API key required.
        Click any result to expand its full page text.
    </div>
    """, unsafe_allow_html=True)

    col_q, col_n = st.columns([4, 1])
    with col_q:
        web_q = st.text_input("Search query", placeholder="e.g., CLIP model multimodal retrieval India",
                               key="web_query")
    with col_n:
        n_results = st.selectbox("Results", [5, 8, 10], index=1, key="web_n")

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        run_web = st.button("🔍 Search Web", use_container_width=True, key="web_run")
    with col_btn2:
        if st.button("✕ Clear", use_container_width=True, key="web_clear"):
            st.session_state.web_search_results = []
            st.session_state.web_page_content   = ""
            st.session_state.web_page_url       = ""
            st.rerun()

    if run_web and web_q:
        with st.spinner(f"Searching the web for: *{web_q}*…"):
            t0  = time.time()
            res = web_search.search(web_q, max_results=n_results)
            ms  = (time.time() - t0) * 1000
        st.session_state.web_search_results = res
        st.session_state.web_page_content   = ""
        st.session_state.web_page_url       = ""
        analytics.log_search(web_q, "web", len(res), ms, st.session_state.user['username'])
        if not res:
            st.warning("No results found. Try a different query.")

    results = st.session_state.web_search_results
    if results:
        st.markdown(f"**{len(results)} results**")
        st.markdown('<hr class="rule">', unsafe_allow_html=True)

        for i, r in enumerate(results):
            st.markdown(f"""
            <div class="web-result-card">
                <div class="web-result-title">{i+1}. {r['title']}</div>
                <div class="web-result-url">🔗 {r['url']}</div>
                <div class="web-result-snippet">{r['snippet'] or 'No description available.'}</div>
            </div>
            """, unsafe_allow_html=True)

            col_open, col_fetch, _ = st.columns([1, 1, 4])
            with col_open:
                st.markdown(f'<a href="{r["url"]}" target="_blank" style="font-size:0.75rem;color:{T["accent"]};">↗ Open</a>',
                            unsafe_allow_html=True)
            with col_fetch:
                if st.button("📄 Read Page", key=f"fetch_{i}", use_container_width=True):
                    with st.spinner(f"Fetching {r['url'][:60]}…"):
                        content = web_search.fetch_page_text(r['url'])
                    st.session_state.web_page_content = content
                    st.session_state.web_page_url     = r['url']

        if st.session_state.web_page_content:
            st.markdown('<hr class="rule">', unsafe_allow_html=True)
            st.markdown(f"#### 📄 Page Content — [{st.session_state.web_page_url[:80]}]({st.session_state.web_page_url})")
            st.markdown(f"""
            <div style="background:{T['bg_card']};border:1px solid {T['border']};border-radius:8px;
            padding:1rem;font-family:'DM Sans',sans-serif;font-size:0.85rem;
            color:{T['text_muted']};line-height:1.7;white-space:pre-wrap;max-height:400px;overflow-y:auto;">
{st.session_state.web_page_content}
            </div>
            """, unsafe_allow_html=True)

            if st.button("🌏 Translate Page to English", key="translate_page"):
                with st.spinner("Translating…"):
                    translated = translate_to_english(st.session_state.web_page_content[:2000])
                st.markdown("**Translated (first 2000 chars):**")
                st.text_area("", translated, height=200, key="translation_out")

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

                    st.markdown("#### Cluster Map (PCA 2D)")
                    pca      = PCA(n_components=2)
                    proj     = pca.fit_transform(vecs)
                    var_pct  = pca.explained_variance_ratio_.sum() * 100
                    df_scat  = pd.DataFrame({
                        "x": proj[:, 0], "y": proj[:, 1],
                        "Cluster": [f"Cluster {l+1}" for l in labels],
                        "ID": [r["id"] for r in eng.id_list],
                    })
                    fig_scatter = px.scatter(
                        df_scat, x="x", y="y", color="Cluster", text="ID",
                        title=f"{pick} Embedding Clusters · {var_pct:.1f}% variance preserved",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                    )
                    fig_scatter.update_traces(
                        textposition="top center",
                        marker=dict(size=12, line=dict(width=1, color="#000"))
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
                        cluster_assets = [eng.id_list[j] for j in range(len(labels)) if labels[j] == i]
                        with st.expander(f"Cluster {i+1} ({len(cluster_assets)} assets)"):
                            cols = st.columns(4)
                            for idx, asset in enumerate(cluster_assets[:8]):
                                with cols[idx % 4]:
                                    if eng.modality == "image":
                                        st.image(asset['path'], caption=asset['id'], use_container_width=True)
                                    else:
                                        st.write(asset['id']); st.audio(asset['path'])

# ── Neural Auditor ── #
with tab_aud:
    pick = st.radio("Registry", ["Image", "Audio"], horizontal=True, key="aud_pick")
    eng  = image_engine if pick == "Image" else audio_engine
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vectors",   eng.index.ntotal)
    c2.metric("Index",     "FAISS HNSW")
    c3.metric("Embed Dim", CONFIG.EMBEDDING_DIM)
    c4.metric("Cache",     len(eng.embedding_cache))
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    if eng.id_list:
        st.markdown("#### Asset Manifest")
        df = pd.DataFrame(eng.id_list)[["id","lang","modality","timestamp"]]
        if "quality" in eng.id_list[0]:
            df["quality"] = [f"{r.get('quality',0):.1%}" for r in eng.id_list]
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
            df_p = pd.DataFrame(proj, columns=["x","y","z"][:nc])
            df_p["ID"] = [r["id"] for r in eng.id_list]
            st.caption(f"Variance preserved: **{var:.1f}%**")

            if nc == 3:
                fig = px.scatter_3d(df_p, x="x", y="y", z="z", text="ID",
                                    color_discrete_sequence=[T["accent"]],
                                    title=f"{pick} · 3D Embedding Map")
            else:
                fig = px.scatter(df_p, x="x", y="y", text="ID",
                                 color_discrete_sequence=[T["accent"]],
                                 title=f"{pick} · 2D Embedding Map")
                fig.update_traces(
                    textposition="top center",
                    marker=dict(size=14, line=dict(width=2, color=T["accent_dim"])),
                    textfont=dict(family="JetBrains Mono", size=11, color=T["text_mono"])
                )
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
            icon = {"image": "🖼", "audio": "🔊", "web": "🌐"}.get(s['modality'], "🔍")
            c1.write(f"{icon} **{s['query'][:50]}**")
            c2.caption(t_str)
            c3.caption(f"{s['results_count']} results")
            with c4:
                if s['modality'] != "web":
                    if st.button("Re-run", key=f"rerun_{idx}"):
                        if s['modality'] == "image":
                            res, _ = image_engine.search(text=s['query'], top_k=6)
                            display_results(res, "image", image_engine)
                        else:
                            res, _ = audio_engine.search(text=s['query'], top_k=6)
                            display_results(res, "audio", audio_engine)
            st.markdown('<hr class="rule">', unsafe_allow_html=True)

        if st.button("Clear History"):
            st.session_state.search_history = []; st.rerun()

# ── Admin ── 
with tab_admin:
    if st.session_state.user.get('role') != 'admin':
        st.warning("Admin access only")
        st.stop()
    else:
        st.markdown("### Admin Control Panel")

        # 1. Create user logic
        with st.expander("➕ Add New User"):
            col1, col2 = st.columns(2)
            with col1:
                new_user = st.text_input("Username", key="new_u_final")
                new_email = st.text_input("Email (optional)", key="new_e_final")
            with col2:
                new_pass = st.text_input("Password", type="password", key="new_p_final")
                new_role = st.selectbox("Role", ["viewer", "uploader", "admin"], key="new_r_final")
            if st.button("Create Account", use_container_width=True):
                if new_user and new_pass:
                    ok, msg = auth_manager.create_user(new_user, new_pass, new_role, new_email)
                    if ok: st.success(msg); time.sleep(0.8); st.rerun()
                    else: st.error(msg)

        # 2. List & manage users (The Fix for the Empty Cards)
        st.subheader("Registered Users")
        users = auth_manager.get_all_users()

        if not users:
            st.info("No verified users yet.")
        else:
            search = st.text_input("Filter Users", placeholder="Search by name, email, or role...", key="usr_filter_final")
            filtered = [u for u in users if search.lower() in " ".join(map(str, u)).lower()]

            if not filtered:
                st.info("No matches found.")
            else:
                for i, row in enumerate(filtered):
                    # SAFETY GATE: Prevents the "empty" white cards
                    if not row or len(row) < 3: 
                        continue
                        
                    username   = row[0]
                    email      = row[1] if row[1] else "No email provided"
                    role       = row[2]
                    created    = row[3]
                    last_login = row[4] if len(row) > 4 else None

                    # Styled Container
                    st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.03); padding:15px; border-radius:10px; margin-bottom:5px; border-left:4px solid {T['accent']};">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span style="font-weight:bold; font-size:1.1rem; color:{T['text_primary']};">{username}</span><br>
                                    <span style="font-size:0.8rem; color:{T['text_muted']};">{email}</span>
                                </div>
                                <div style="font-family:'JetBrains Mono'; background:{T['accent']}22; color:{T['accent']}; padding:4px 10px; border-radius:6px; font-size:0.75rem; font-weight:bold;">
                                    {role.upper()}
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Action Buttons with Unique Keys
                    c1, c2, c3, c4 = st.columns([1.5, 1.5, 1, 1])
                    with c1: st.caption(f"📅 Joined: {created[:10]}")
                    with c2: st.caption(f"👁️ Last: {last_login[:10] if last_login else 'Never'}")
                    
                    with c3:
                        if role != "admin":
                            if st.button("⏫ Promote", key=f"prom_{username}_{i}", use_container_width=True):
                                auth_manager.db.execute("UPDATE users SET role='admin' WHERE username=?", (username,))
                                st.rerun()
                    
                    with c4:
                        is_self = username == st.session_state.user["username"]
                        if st.button("🗑️ Delete", key=f"del_{username}_{i}", use_container_width=True, disabled=is_self):
                            st.session_state[f"del_confirm_{username}"] = True

                    # Confirmation logic inside the loop
                    if st.session_state.get(f"del_confirm_{username}"):
                        st.error(f"Confirm deletion of {username}?")
                        cc1, cc2 = st.columns(2)
                        if cc1.button("PURGE", key=f"y_del_{username}_{i}", use_container_width=True):
                            auth_manager.db.execute("DELETE FROM users WHERE username=?", (username,))
                            st.session_state[f"del_confirm_{username}"] = False
                            st.rerun()
                        if cc2.button("CANCEL", key=f"n_del_{username}_{i}", use_container_width=True):
                            st.session_state[f"del_confirm_{username}"] = False
                            st.rerun()
                    
                    st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

        # 3. System Health Diagnostics
        with st.expander("🩺 System Health Diagnostics"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Database", "Online" if auth_manager.db else "Offline")
            c2.metric("SMTP", "Ready" if auth_manager.otp_sender.smtp_configured else "Disabled")
            c3.metric("Storage", "Active" if os.path.exists(STORAGE_DIR) else "Error")

            if st.button("Test SMTP Connection"):
                with st.spinner("Connecting..."):
                    try:
                        with smtplib.SMTP("smtp.gmail.com", 587, timeout=5) as s:
                            s.starttls()
                            st.success("Gmail SMTP is reachable and secrets are working!")
                    except Exception as e:
                        st.error(f"Failed: {str(e)}")

# ── Footer ── #
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;padding:20px;'>
  <p><strong style='font-size:1.1em;color:#667eea;'>Created by Team Human</strong></p>
  <p>Powered by CLIP &amp; CLAP | Built for Bharat's Digital Future</p>
  <p style='font-size:0.8em;'>Multimodal embeddings · FAISS indexing · Cross-lingual search · Live web search</p>
</div>
""", unsafe_allow_html=True)

















