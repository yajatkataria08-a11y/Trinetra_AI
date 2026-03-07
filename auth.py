import os
import hashlib
import logging
import secrets
import smtplib
import sqlite3
import string

import streamlit as st
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from werkzeug.security import check_password_hash, generate_password_hash

from config import CONFIG, BASE_DIR
from database import DatabaseConnection

logger = logging.getLogger("Trinetra")


# ==================== EMAIL OTP SENDER ====================

class EmailOTPSender:

    def __init__(self, smtp_server="smtp.gmail.com", smtp_port=587):
        self.smtp_server    = smtp_server
        self.smtp_port      = smtp_port
        self.sender_email   = ""
        self.sender_password = ""
        self.smtp_configured = False

        # Priority 1: Streamlit Cloud secrets
        try:
            if "SMTP_EMAIL" in st.secrets and "SMTP_PASSWORD" in st.secrets:
                email = str(st.secrets["SMTP_EMAIL"]).strip()
                pwd   = str(st.secrets["SMTP_PASSWORD"]).strip()
                if email and pwd and "@" in email:
                    self.sender_email    = email
                    self.sender_password = pwd
                    self.smtp_configured = True
                    logger.info("SMTP loaded from Streamlit secrets")
        except Exception:
            pass

        # Priority 2: Environment variables
        if not self.smtp_configured:
            email = os.getenv("SMTP_EMAIL", "").strip()
            pwd   = os.getenv("SMTP_PASSWORD", "").strip()
            if email and pwd and "@" in email:
                self.sender_email    = email
                self.sender_password = pwd
                self.smtp_configured = True
                logger.info("SMTP loaded from environment variables")

        if not self.smtp_configured:
            logger.warning("No valid SMTP credentials found")
            st.warning(
                "⚠️ Email/OTP sending disabled — no SMTP_EMAIL & SMTP_PASSWORD found.\n\n"
                "• Cloud → add to Streamlit Secrets\n"
                "• Local → set in .env or export\n"
                "OTPs will show on screen until fixed."
            )

    def generate_otp(self, length=6) -> str:
        return "".join(secrets.choice(string.digits) for _ in range(length))

    def send_otp_email(self, recipient_email: str, otp: str, username: str):
        if not self.smtp_configured:
            msg = (
                "⚠️ Email not configured.\n\n"
                f"OTP for this session: **{otp}**\n\n"
                "Copy it now — won't be shown again."
            )
            return False, msg, otp

        try:
            html_body = f"""
            <html><body style="font-family:sans-serif;padding:20px;background:#f8f9fa;">
                <div style="max-width:500px;margin:auto;background:white;padding:30px;
                            border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,0.1);">
                    <h2 style="color:#e8a020;text-align:center;">Trinetra Verification</h2>
                    <p>Hello {username},</p>
                    <p>Your code is:</p>
                    <h1 style="color:#e8a020;letter-spacing:8px;text-align:center;
                               margin:20px 0;">{otp}</h1>
                    <p style="color:#555;">Valid 10 minutes only.</p>
                    <hr style="border:none;border-top:1px solid #eee;">
                    <p style="font-size:0.9rem;color:#777;text-align:center;">
                        Team Human | Bharat's Digital Future
                    </p>
                </div>
            </body></html>
            """
            message             = MIMEMultipart("alternative")
            message["Subject"]  = "Trinetra – OTP Code"
            message["From"]     = f"Trinetra <{self.sender_email}>"
            message["To"]       = recipient_email
            message.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipient_email, message.as_string())

            logger.info(f"OTP sent to {recipient_email}")
            return True, "OTP sent — check inbox/spam", None

        except smtplib.SMTPAuthenticationError:
            return False, f"Gmail login rejected (use App Password if 2FA on). OTP: **{otp}**", otp
        except Exception as e:
            logger.error(f"SMTP fail: {e}")
            return False, f"Email failed ({e}). OTP: **{otp}**", otp


# ==================== AUTH MANAGER ====================

class AuthManagerWithOTP:

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(BASE_DIR, "users.db")
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
                (username, email, password_hash, full_name, role, is_verified,
                 created_at, registration_date)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """, ("admin", "admin@trinetra.local", admin_hash, "Administrator", "admin", 1))
        except Exception as e:
            logger.error(f"Failed to create default admin: {e}", exc_info=True)

    # ── Password helpers ──

    def hash_password(self, password: str) -> str:
        return generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)

    def _is_legacy_sha256(self, stored_hash: str) -> bool:
        return (len(stored_hash) == 64 and
                all(c in "0123456789abcdef" for c in stored_hash))

    def verify_password(self, stored_hash: str, password: str) -> bool:
        if self._is_legacy_sha256(stored_hash):
            return hashlib.sha256(password.encode()).hexdigest() == stored_hash
        return check_password_hash(stored_hash, password)

    def hash_otp(self, otp: str) -> str:
        return hashlib.sha256(otp.encode()).hexdigest()

    # ── OTP cooldown ──

    def _check_otp_cooldown(self, email: str, table: str) -> bool:
        row = self.db.fetchone(f"SELECT created_at FROM {table} WHERE email = ?", (email,))
        if row:
            last = datetime.fromisoformat(row[0])
            if (datetime.now() - last).total_seconds() < CONFIG.OTP_COOLDOWN_SECS:
                return False
        return True

    # ── Registration ──

    def request_registration(self, email, username, password, full_name=""):
        if not email or "@" not in email:
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
            (email, username),
        ):
            return False, "Email or username already has a pending registration", None
        if self.db.fetchone(
            "SELECT username FROM users WHERE email=? OR username=?", (email, username)
        ):
            return False, "Email or username already in use", None
        if not self._check_otp_cooldown(email, "registration_requests"):
            return False, f"Please wait {CONFIG.OTP_COOLDOWN_SECS}s before requesting another OTP", None

        otp        = self.otp_sender.generate_otp()
        otp_hash   = self.hash_otp(otp)
        email_sent, email_msg, fallback_otp = self.otp_sender.send_otp_email(email, otp, username)

        try:
            password_hash = self.hash_password(password)
            self.db.execute("""
                INSERT INTO registration_requests
                (email, username, password_hash, full_name, otp_hash, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now', '+10 minutes'))
            """, (email, username, password_hash, full_name, otp_hash))
            status_msg = (
                f"OTP sent to {email}. Valid for 10 minutes."
                if email_sent
                else "⚠️ Email delivery failed — use the OTP shown on screen below."
            )
            return True, status_msg, fallback_otp
        except Exception as e:
            logger.error(f"Registration request failed for {email}: {e}", exc_info=True)
            return False, f"Registration request failed: {str(e)}", None

    def verify_otp_and_register(self, email: str, otp: str):
        otp_hash = self.hash_otp(otp)
        result   = self.db.fetchone("""
            SELECT username, password_hash, full_name, expires_at
            FROM registration_requests WHERE email=? AND otp_hash=?
        """, (email, otp_hash))

        if not result:
            return False, "Invalid OTP"

        username, password_hash, full_name, expires_at = result
        if datetime.now() > datetime.fromisoformat(expires_at):
            self.db.execute("DELETE FROM registration_requests WHERE email=?", (email,))
            return False, "OTP expired. Please register again."

        try:
            self.db.execute("""
                INSERT INTO users
                (username, email, password_hash, full_name, role, is_verified,
                 created_at, registration_date)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """, (username, email, password_hash, full_name, "uploader", 1))
            self.db.execute("DELETE FROM registration_requests WHERE email=?", (email,))
            logger.info(f"REGISTER_SUCCESS user={username}")
            return True, "Registration successful! You can now login."
        except Exception as e:
            logger.error(f"Registration DB error for {email}: {e}", exc_info=True)
            return False, f"Registration failed: {str(e)}"

    def resend_otp(self, email: str):
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
                SET otp_hash=?, created_at=datetime('now'),
                    expires_at=datetime('now','+10 minutes')
                WHERE email=?
            """, (otp_hash, email))
            status_msg = "OTP resent successfully." if email_sent else "⚠️ Email failed — use the OTP shown."
            return True, status_msg, fallback_otp
        except Exception as e:
            return False, f"Failed to resend OTP: {str(e)}", None

    # ── Login ──

    def verify_user(self, username: str, password: str):
        result = self.db.fetchone(
            "SELECT username, password_hash, role, is_verified FROM users WHERE username=?",
            (username,),
        )
        if not result:
            return None

        db_username, stored_hash, role, is_verified = result
        if not self.verify_password(stored_hash, password):
            return None
        if not is_verified:
            return None

        if self._is_legacy_sha256(stored_hash):
            new_hash = self.hash_password(password)
            try:
                self.db.execute(
                    "UPDATE users SET password_hash=? WHERE username=?",
                    (new_hash, db_username),
                )
            except Exception as e:
                logger.warning(f"Hash upgrade failed for {db_username}: {e}")

        self.db.execute(
            "UPDATE users SET last_login=datetime('now') WHERE username=?", (username,)
        )
        logger.info(f"LOGIN_SUCCESS user={db_username} role={role}")
        return {"username": db_username, "role": role}

    # ── User management ──

    def get_all_users(self):
        return self.db.fetchall(
            "SELECT username, email, role, created_at, last_login FROM users WHERE is_verified=1"
        )

    def create_user(self, username: str, password: str, role: str, email: str = ""):
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
            return True, f"User '{username}' created successfully"
        except sqlite3.IntegrityError:
            return False, "Username or email already exists"

    # ── Password reset ──

    def request_password_reset(self, email: str):
        result = self.db.fetchone(
            "SELECT username FROM users WHERE email=? AND is_verified=1", (email,)
        )
        if not result:
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
        status_msg = (
            f"OTP sent to {email}. Valid for 10 minutes."
            if email_sent
            else "⚠️ Email delivery failed — use the OTP shown on screen below."
        )
        return True, status_msg, fallback_otp

    def verify_reset_otp(self, email: str, otp: str):
        otp_hash = self.hash_otp(otp)
        result   = self.db.fetchone(
            "SELECT expires_at FROM password_reset_requests WHERE email=? AND otp_hash=?",
            (email, otp_hash),
        )
        if not result:
            return False, "Invalid OTP. Please try again."
        if datetime.now() > datetime.fromisoformat(result[0]):
            self.db.execute("DELETE FROM password_reset_requests WHERE email=?", (email,))
            return False, "OTP expired. Please request a new one."
        return True, "OTP verified. Please set your new password."

    def reset_password(self, email: str, otp: str, new_password: str):
        if len(new_password) < 6:
            return False, "Password must be at least 6 characters."
        ok, msg = self.verify_reset_otp(email, otp)
        if not ok:
            return False, msg
        new_hash = self.hash_password(new_password)
        try:
            self.db.execute("UPDATE users SET password_hash=? WHERE email=?", (new_hash, email))
            self.db.execute("DELETE FROM password_reset_requests WHERE email=?", (email,))
            return True, "Password updated successfully! You can now log in."
        except Exception as e:
            return False, f"Failed to update password: {e}"
