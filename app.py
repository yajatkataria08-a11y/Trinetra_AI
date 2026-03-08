"""
Trinetra V5.0 — app.py
Main entry point. All business logic lives in separate modules:
  config.py         — constants & Config class
  database.py       — DatabaseConnection, MetadataDB
  models.py         — load_models(), load_toxicity_model()
  utils.py          — helpers (validation, translate, hashing, etc.)
  auth.py           — EmailOTPSender, AuthManagerWithOTP
  analytics.py      — AnalyticsTracker
  search.py         — WebSearchEngine, LambdaSearchClient, AWSReverseSearchEngine
  engines.py        — TrinetraEngine, ImageEngine, AudioEngine
  ui_components.py  — CSS, display_results, render_enrichment_ui, render_aws_tab
"""

import json
import os
import smtplib
import tempfile
import time
import logging
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from config import CONFIG, BASE_DIR, STORAGE_DIR, DEVICE
from auth import AuthManagerWithOTP
from analytics import AnalyticsTracker
from engines import ImageEngine, AudioEngine
from search import WebSearchEngine, LambdaSearchClient, AWSReverseSearchEngine
from ui_components import (
    DARK, LIGHT, get_theme, build_css, TRINETRA_LOGO_SVG,
    display_results, render_enrichment_ui, render_aws_reverse_search_tab,
)
from utils import (
    translate_to_english, translate_to_english_with_sarvam,
    smart_query_preprocess, validate_upload,
)

# ==================== LOGGING ====================
logging.basicConfig(
    filename=f'logs/trinetra_{datetime.now():%Y%m%d}.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("Trinetra")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Trinetra V5.0 · Bharat AI",
    layout="wide",
    page_icon="👁️",
    initial_sidebar_state="expanded",
)

# ==================== SESSION STATE ====================
defaults = {
    "theme":               "dark",
    "search_history":      [],
    "last_search_results": [],
    "authenticated":       False,
    "user":                None,
    "show_suggestions":    True,
    "auth_stage":          "login",
    "temp_email":          "",
    "temp_username":       "",
    "demo_otp":            None,
    "reset_email":         "",
    "reset_otp":           "",
    "reset_demo_otp":      None,
    "web_search_results":  [],
    "web_page_content":    "",
    "web_page_url":        "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

is_light = st.session_state.theme == "light"
T        = LIGHT if is_light else DARK

st.markdown(build_css(st.session_state.theme), unsafe_allow_html=True)

# ==================== CACHED RESOURCES ====================

@st.cache_resource
def get_auth_manager():
    return AuthManagerWithOTP()

@st.cache_resource(show_spinner="Initializing registry…")
def get_engines():
    return (
        ImageEngine(),
        AudioEngine(),
        AnalyticsTracker(),
        WebSearchEngine(),
        AWSReverseSearchEngine(),
        LambdaSearchClient(),
    )

auth_manager                                          = get_auth_manager()
image_engine, audio_engine, analytics, web_search, \
    aws_search, lambda_client                         = get_engines()


# ==================== LOGIN PAGE ====================

def show_enhanced_login_page(auth_manager):
    st.markdown("""
    <div class="login-container">
        <div class="login-logo">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 80" width="96" height="64">
              <defs>
                <radialGradient id="lg" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stop-color="#e8a020" stop-opacity="0.45"/>
                  <stop offset="100%" stop-color="#e8a020" stop-opacity="0"/>
                </radialGradient>
                <linearGradient id="rg" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stop-color="#ff6a00"/>
                  <stop offset="45%" stop-color="#e8a020"/>
                  <stop offset="100%" stop-color="#4e54c8"/>
                </linearGradient>
                <linearGradient id="tg" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stop-color="#e8a020"/>
                  <stop offset="100%" stop-color="#fff5d6"/>
                </linearGradient>
                <filter id="glow2">
                  <feGaussianBlur stdDeviation="2.5" result="blur"/>
                  <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
              </defs>
              <ellipse cx="60" cy="40" rx="52" ry="34" fill="url(#lg)"/>
              <path d="M8,40 Q60,4 112,40 Q60,76 8,40Z" fill="#0d1321" stroke="url(#rg)" stroke-width="2.5" filter="url(#glow2)"/>
              <circle cx="60" cy="40" r="18" fill="#0a0f1e" stroke="#e8a020" stroke-width="1.2"/>
              <polygon points="60,24 75,50 45,50" fill="none" stroke="url(#tg)" stroke-width="1.8" filter="url(#glow2)"/>
              <circle cx="60" cy="40" r="6" fill="none" stroke="#e8a020" stroke-width="1.4"/>
              <text x="60" y="44" text-anchor="middle" fill="#e8a020" font-size="7" font-family="serif" font-weight="bold">©</text>
              <rect x="4" y="34" width="3" height="12" rx="1" fill="#e8a020" opacity="0.7"/>
              <rect x="1" y="36" width="2" height="3" rx="0.5" fill="#ff6a00" opacity="0.8"/>
              <rect x="1" y="41" width="2" height="3" rx="0.5" fill="#ff6a00" opacity="0.8"/>
              <rect x="103" y="37" width="2" height="6" rx="1" fill="#4e9ff5" opacity="0.9"/>
              <rect x="107" y="33" width="2" height="14" rx="1" fill="#4e9ff5" opacity="0.9"/>
              <rect x="111" y="36" width="2" height="8" rx="1" fill="#4e9ff5" opacity="0.7"/>
              <rect x="115" y="38" width="2" height="4" rx="1" fill="#4e9ff5" opacity="0.5"/>
            </svg>
        </div>
        <div class="login-title">TRINETRA</div>
        <div class="login-subtitle">Multimodal Neural Registry</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            stage = st.session_state.auth_stage

            # ── Login ──
            if stage == "login":
                with st.form("login_form", border=False):
                    username     = st.text_input("Username", placeholder="Enter username")
                    password     = st.text_input("Password", type="password", placeholder="Enter password")
                    submit_login = st.form_submit_button("Login", use_container_width=True)

                if submit_login:
                    if username and password:
                        user = auth_manager.verify_user(username, password)
                        if user:
                            st.session_state.authenticated = True
                            st.session_state.user          = user
                            st.success(f"Welcome, {user['username']}!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("Invalid credentials or email not verified")
                    else:
                        st.warning("Please enter both username and password")

                col_b, col_c = st.columns(2)
                with col_b:
                    if st.button("Register", use_container_width=True):
                        st.session_state.auth_stage = "register_step1"
                        st.rerun()
                with col_c:
                    if st.button("Guest", use_container_width=True):
                        st.session_state.authenticated = True
                        st.session_state.user = {"username": "guest", "role": "viewer"}
                        st.rerun()

                st.markdown("---")
                if st.button("🔑 Forgot Password?", use_container_width=True, key="goto_fp"):
                    st.session_state.auth_stage = "forgot_step1"
                    st.rerun()
                st.caption("📧 Demo: admin / admin123")
                st.caption("❤️ Created by Team Human")

            # ── Forgot password step 1 ──
            elif stage == "forgot_step1":
                st.markdown("### Reset Password")
                fp_email = st.text_input("Email Address", placeholder="your.email@example.com", key="fp_email")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Send OTP", use_container_width=True):
                        if not fp_email or "@" not in fp_email:
                            st.error("Please enter a valid email address.")
                        else:
                            ok, msg, demo_otp = auth_manager.request_password_reset(fp_email)
                            if ok:
                                st.session_state.reset_email    = fp_email
                                st.session_state.reset_demo_otp = demo_otp
                                st.session_state.auth_stage     = "forgot_step2"
                                st.success(msg)
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(msg)
                with col_b:
                    if st.button("Back to Login", use_container_width=True):
                        st.session_state.auth_stage = "login"
                        st.rerun()

            # ── Forgot password step 2 ──
            elif stage == "forgot_step2":
                st.markdown("### Verify OTP")
                st.info(f"📧 OTP sent to: **{st.session_state.reset_email}**")
                if st.session_state.reset_demo_otp:
                    st.warning(f"📋 Email delivery failed. OTP: **{st.session_state.reset_demo_otp}**")
                    st.session_state.reset_demo_otp = None
                fp_otp = st.text_input("Enter 6-digit OTP", max_chars=6, key="fp_otp")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("Verify OTP", use_container_width=True):
                        ok, msg = auth_manager.verify_reset_otp(st.session_state.reset_email, fp_otp)
                        if ok:
                            st.session_state.reset_otp  = fp_otp
                            st.session_state.auth_stage = "forgot_step3"
                            st.success(msg)
                            time.sleep(0.8)
                            st.rerun()
                        else:
                            st.error(msg)
                with col_b:
                    if st.button("Resend OTP", use_container_width=True):
                        ok, msg, new_demo = auth_manager.request_password_reset(st.session_state.reset_email)
                        if ok:
                            st.session_state.reset_demo_otp = new_demo
                            st.success("New OTP sent!")
                        else:
                            st.error(msg)
                with col_c:
                    if st.button("Cancel", use_container_width=True):
                        for k in ["auth_stage", "reset_email", "reset_otp", "reset_demo_otp"]:
                            st.session_state[k] = "login" if k == "auth_stage" else ("" if k != "reset_demo_otp" else None)
                        st.rerun()

            # ── Forgot password step 3 ──
            elif stage == "forgot_step3":
                st.markdown("### Set New Password")
                new_pass  = st.text_input("New Password", type="password", key="fp_newpass")
                new_pass2 = st.text_input("Confirm New Password", type="password", key="fp_newpass2")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Update Password", use_container_width=True):
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
                                for k in ["auth_stage", "reset_email", "reset_otp", "reset_demo_otp"]:
                                    st.session_state[k] = "login" if k == "auth_stage" else ("" if k != "reset_demo_otp" else None)
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error(msg)
                with col_b:
                    if st.button("Cancel", use_container_width=True):
                        for k in ["auth_stage", "reset_email", "reset_otp", "reset_demo_otp"]:
                            st.session_state[k] = "login" if k == "auth_stage" else ("" if k != "reset_demo_otp" else None)
                        st.rerun()

            # ── Register step 1 ──
            elif stage == "register_step1":
                st.markdown("### Create Account")
                email    = st.text_input("Email Address", key="reg_email")
                username = st.text_input("Username (3+ chars)", key="reg_user")
                password = st.text_input("Password (6+ chars)", type="password", key="reg_pass")
                password_confirm = st.text_input("Confirm Password", type="password", key="reg_pass_conf")
                full_name = st.text_input("Full Name (optional)", key="reg_name")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Send OTP", use_container_width=True):
                        if not email or "@" not in email:        st.error("Invalid email")
                        elif len(username) < 3:                  st.error("Username too short")
                        elif len(password) < 6:                  st.error("Password too short")
                        elif password != password_confirm:       st.error("Passwords don't match")
                        else:
                            ok, msg, otp = auth_manager.request_registration(email, username, password, full_name)
                            if ok:
                                st.session_state.update({
                                    "auth_stage": "register_step2",
                                    "temp_email":  email,
                                    "temp_username": username,
                                    "demo_otp":    otp,
                                })
                                st.success(msg)
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(msg)
                with col_b:
                    if st.button("Back to Login", use_container_width=True):
                        st.session_state.auth_stage = "login"
                        st.rerun()

            # ── Register step 2 ──
            elif stage == "register_step2":
                st.markdown("### Verify Email")
                st.info(f"📧 OTP sent to: {st.session_state.temp_email}")
                if st.session_state.demo_otp:
                    st.warning(f"📋 Email failed. OTP: **{st.session_state.demo_otp}** — copy now.")
                    st.session_state.demo_otp = None
                otp = st.text_input("Enter 6-digit OTP", max_chars=6, key="otp_input")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("Verify OTP", use_container_width=True):
                        ok, msg = auth_manager.verify_otp_and_register(st.session_state.temp_email, otp)
                        if ok:
                            st.success(msg)
                            st.session_state.auth_stage = "login"
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(msg)
                with col_b:
                    if st.button("Resend OTP", use_container_width=True):
                        ok, msg, new_otp = auth_manager.resend_otp(st.session_state.temp_email)
                        if ok:
                            st.session_state.demo_otp = new_otp
                            st.success(msg)
                        else:
                            st.error(msg)
                with col_c:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state.auth_stage = "login"
                        st.rerun()


if not st.session_state.authenticated:
    show_enhanced_login_page(auth_manager)
    st.stop()


# ==================== SIDEBAR ====================

with st.sidebar:
    user_role = st.session_state.user["role"]

    st.markdown(f"""
    <div style="padding:0.5rem;background:{T['bg_card']};border:1px solid {T['border']};
    border-radius:8px;margin-bottom:1rem;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
        color:{T['text_muted']};">LOGGED IN AS</div>
        <div style="font-weight:600;color:{T['accent']};">{st.session_state.user['username']}</div>
        <div style="font-size:0.7rem;color:{T['text_muted']};">Role: {user_role}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Logout", use_container_width=True):
        logger.info(f"LOGOUT user={st.session_state.user['username']}")
        st.session_state.authenticated = False
        st.session_state.user          = None
        st.rerun()

    # ── Theme toggle: button rendered first, pill overlaid via negative margin ──
    if st.button("​", use_container_width=True, key="theme_btn"):
        st.session_state.theme = "light" if not is_light else "dark"
        st.rerun()
    st.markdown(f"""
    <div class="theme-toggle-pill" style="margin-top:-2.8rem;pointer-events:none;position:relative;z-index:1;">
      <div class="tt-track"><div class="tt-knob"></div></div>
      <span class="tt-icon">{"🌙" if not is_light else "☀️"}</span>
      <span class="tt-label">{"Light Mode" if not is_light else "Dark Mode"}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<hr style="border-color:{T["border"]};margin:1rem 0">', unsafe_allow_html=True)

    # ── Registration panel (admins/uploaders only) ──
    if user_role in ["admin", "uploader"]:
        st.markdown("### Asset Registration")
        batch_mode = st.checkbox("Batch Upload Mode", value=False)
        reg_mod    = st.selectbox("Modality", ["image", "audio"])
        reg_lang   = st.selectbox("Language", CONFIG.SUPPORTED_LANGUAGES)

        if not batch_mode:
            reg_id   = st.text_input("Asset ID", placeholder="e.g. diwali_2024")
            reg_tags = st.multiselect("Tags (optional)",
                                      ["festival", "temple", "landscape", "portrait",
                                       "music", "speech", "nature", "urban"],
                                      key="tags_single")
            reg_file = st.file_uploader(
                "Select File",
                type=[e.lstrip(".") for e in (CONFIG.ALLOWED_IMAGE_EXTS if reg_mod == "image"
                                               else CONFIG.ALLOWED_AUDIO_EXTS)],
            )
            if reg_file:
                st.markdown("#### Preview")
                if reg_mod == "image": st.image(reg_file, use_container_width=True)
                else:                  st.audio(reg_file)
                st.caption(f"Size: {reg_file.size/1024:.1f} KB")

            if st.button("Register Asset", use_container_width=True):
                if not reg_file:
                    st.warning("Please upload a file")
                elif not reg_id.strip():
                    st.warning("Please enter an Asset ID")
                else:
                    err = validate_upload(reg_file, reg_mod)
                    if err:
                        st.error(err)
                    else:
                        ext = os.path.splitext(reg_file.name)[1].lower()
                        eng = image_engine if reg_mod == "image" else audio_engine
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(reg_file.getbuffer())
                            tp = tmp.name
                        try:
                            with st.spinner("Checking for duplicates…"):
                                dups = eng.find_duplicates(file_path=tp)
                            if dups:
                                st.warning(
                                    f"Similar: **{dups[0]['id']}** ({dups[0]['similarity']:.1%} match)"
                                )
                                if not st.checkbox("Register anyway", key="force_reg_cb"):
                                    os.remove(tp)
                                    st.stop()
                            with st.spinner("Registering…"):
                                ok, msg = eng.register(
                                    tp, reg_id.strip(), ext, reg_lang, reg_tags,
                                    uploaded_by=st.session_state.user["username"],
                                )
                            (st.success if ok else st.error)(msg)
                            if ok:
                                st.balloons()
                                time.sleep(1)
                                st.rerun()
                        finally:
                            if os.path.exists(tp):
                                os.remove(tp)
        else:
            reg_tags_b  = st.multiselect("Tags for all files",
                                         ["festival", "temple", "landscape", "portrait",
                                          "music", "speech", "nature", "urban"],
                                         key="tags_batch")
            reg_collect = st.text_input("Collection name", placeholder="e.g. Diwali 2024")
            reg_files   = st.file_uploader(
                "Select Multiple Files", accept_multiple_files=True,
                type=[e.lstrip(".") for e in (CONFIG.ALLOWED_IMAGE_EXTS if reg_mod == "image"
                                               else CONFIG.ALLOWED_AUDIO_EXTS)],
            )
            if st.button("Register Batch", use_container_width=True):
                if not reg_files:
                    st.warning("Upload at least one file")
                else:
                    eng = image_engine if reg_mod == "image" else audio_engine
                    with st.spinner(f"Processing {len(reg_files)} files…"):
                        res = eng.batch_register(
                            reg_files, reg_lang, reg_tags_b, reg_collect,
                            st.session_state.user["username"],
                        )
                    st.success(f"Added: {len(res['success'])}")
                    if res["skipped"]:    st.info(f"Skipped: {len(res['skipped'])}")
                    if res["duplicates"]:
                        st.warning(f"Duplicates: {len(res['duplicates'])}")
                        with st.expander("View duplicates"):
                            for fn, dup in res["duplicates"]:
                                st.write(f"- {fn} → {dup}")
                    if res["failed"]:
                        st.error(f"Failed: {len(res['failed'])}")
                        with st.expander("View errors"):
                            for fn, err in res["failed"]:
                                st.write(f"- {fn}: {err}")
    else:
        st.markdown(f"""
        <div style="background:{T['bg_card']};border:1px solid {T['border']};
        border-left:3px solid {T['accent']};border-radius:8px;padding:0.9rem 1rem;margin-bottom:1rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
            text-transform:uppercase;letter-spacing:0.1em;color:{T['accent']};margin-bottom:4px;">
                🔒 Viewer Access
            </div>
            <div style="font-size:0.8rem;color:{T['text_muted']};line-height:1.5;">
                Asset registration not available for your role.
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
        <div class="sidebar-stat">Searches    <span>{stats[0]}</span></div>
        <div class="sidebar-stat">Avg Results <span>{stats[1]:.1f}</span></div>
        <div class="sidebar-stat">Avg Speed   <span>{stats[2]:.0f}ms</span></div>
        """, unsafe_allow_html=True)

    if user_role == "admin":
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export ZIP", use_container_width=True):
                ip = image_engine.export_registry()
                ap = audio_engine.export_registry()
                st.success(f"Exported!\n- {ip}\n- {ap}")
        with col2:
            if st.button("Export JSON", use_container_width=True):
                json_data = {
                    "image": json.loads(image_engine.export_as_json()),
                    "audio": json.loads(audio_engine.export_as_json()),
                }
                st.download_button(
                    "Download JSON",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"trinetra_export_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        st.markdown("---")
        st.markdown("### 🔌 AWS Test")
        if st.button("Test AWS Services", use_container_width=True):
            ok, msg = aws_search.test_aws_connection()
            (st.success if ok else st.error)(msg)


# ==================== HEADER ====================

st.markdown(f"""
<div class="trinetra-hero a1">
  {TRINETRA_LOGO_SVG}
  <div class="trinetra-wordmark">
    <h1 style="margin:0;padding:0;">TRINETRA V5.0</h1>
  </div>
</div>
<div class="tagline a2">Multimodal Neural Registry · AI-Powered Search · Bharat AI</div>
""", unsafe_allow_html=True)

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


# ==================== TABS ====================

tab_v, tab_a, tab_web, tab_aws, tab_cluster, tab_aud, tab_hist, tab_admin = st.tabs([
    "Visual Search", "Acoustic Search", "🌐 Web Search", "🔍 AWS Reverse",
    "Clusters", "Neural Auditor", "Search History", "Admin",
])


# ── Visual Search ──────────────────────────────────────────────────────────────
with tab_v:
    use_lambda = st.checkbox(
        "⚡ Hybrid Cloud Search (Lambda + DynamoDB)",
        value=False,
        help="Serverless keyword search on labels & transcripts.",
        key="hybrid_visual",
    )
    use_sarvam_v = st.checkbox(
        "Boost Indic queries with Sarvam translation 🇮🇳",
        value=True,
        key="sarvam_visual",
        help="Auto-translates Hindi/Tamil/etc. queries to English before embedding.",
    )
    mode = st.radio("Input Mode", ["Text Query", "Image Match"], horizontal=True)

    with st.expander("Advanced Filters"):
        f_lang      = st.selectbox("Language", ["All"] + CONFIG.SUPPORTED_LANGUAGES, key="fl_img")
        f_tags      = st.multiselect("Tags", image_engine.metadata_db.get_all_tags(), key="ft_img")
        f_min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, key="fs_img")

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    if mode == "Text Query":
        q = st.text_input("Describe the image",
                          placeholder="e.g., a crowded temple at dusk", key="vq")
        if q and st.session_state.show_suggestions:
            suggestions = analytics.get_search_suggestions(q)
            if suggestions:
                st.markdown("**Suggestions:**")
                st.markdown(
                    "".join(f'<span class="suggestion-chip">{s}</span>' for s in suggestions),
                    unsafe_allow_html=True,
                )
    else:
        q = None
        img_query = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png", "webp"], key="img_query_upload")

    run_visual = st.button("Run Visual Scan", key="vs_run")

    if run_visual:
        mf = {"min_score": f_min_score}
        if f_lang != "All": mf["language"] = f_lang
        if f_tags:          mf["tags"]     = f_tags

        results, ms = [], 0

        if mode == "Image Match":
            if img_query:
                ext = os.path.splitext(img_query.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(img_query.getbuffer())
                    tp = tmp.name
                try:
                    with st.spinner("Scanning by image…"):
                        results, ms = image_engine.hybrid_search(file_path=tp, top_k=6, filters=mf)
                finally:
                    os.remove(tp)
            else:
                st.warning("Please upload a query image.")

        elif q:
            q_processed = smart_query_preprocess(q, use_sarvam=use_sarvam_v)

            if use_lambda and lambda_client.is_configured():
                with st.spinner("Querying cloud-powered index… 🚀"):
                    t0         = time.time()
                    lambda_res = lambda_client.search(q_processed, modality="image", limit=6)
                    if "error" in lambda_res:
                        st.error(f"Cloud search failed: {lambda_res['error']}")
                        results, ms = image_engine.hybrid_search(text_query=q_processed, top_k=6, filters=mf)
                    else:
                        for hit in lambda_res.get("results", []):
                            score_raw = hit.get("score", 0)
                            results.append({
                                "id":         hit.get("asset_id", "unknown"),
                                "modality":   "image",
                                "score":      score_raw / 10.0 if score_raw > 1.0 else score_raw,
                                "confidence": "High" if score_raw > 7 else "Medium" if score_raw > 4 else "Low",
                                "path":       None,
                                "lang":       "en",
                            })
                        ms = (time.time() - t0) * 1000
                        st.caption("Results from **AWS Lambda + DynamoDB**")
            else:
                with st.spinner("Scanning local registry…"):
                    results, ms = image_engine.hybrid_search(text_query=q_processed, top_k=6, filters=mf)

        # Show Sarvam translation caption if translation happened
        if "last_translated" in st.session_state:
            orig, trans = st.session_state["last_translated"]
            if orig != trans:
                st.caption(f"🇮🇳 Sarvam magic: {orig} → **{trans}**")

        if results:
            analytics.log_search(
                q or "image_match", "image", len(results), ms,
                st.session_state.user["username"],
            )
            st.session_state.search_history.append({
                "query": q or "image_match", "modality": "image",
                "timestamp": time.time(), "results_count": len(results),
            })
            st.caption(f"Search time: {ms:.0f} ms")
            display_results(results, "image", image_engine)
            render_enrichment_ui(results, "image", image_engine, audio_engine, web_search)


# ── Acoustic Search ────────────────────────────────────────────────────────────
with tab_a:
    st.markdown("### 🔊 Acoustic Search")
    use_lambda_audio = st.checkbox(
        "⚡ Hybrid Cloud Search (Lambda + DynamoDB)",
        value=False,
        key="hybrid_acoustic",
    )
    use_sarvam_a = st.checkbox(
        "Boost Indic queries with Sarvam translation 🇮🇳",
        value=True,
        key="sarvam_acoustic",
        help="Auto-translates Hindi/Tamil/etc. queries to English before embedding.",
    )

    with st.expander("Advanced Filters"):
        f_lang_a      = st.selectbox("Language", ["All"] + CONFIG.SUPPORTED_LANGUAGES, key="fl_aud")
        f_tags_a      = st.multiselect("Tags", audio_engine.metadata_db.get_all_tags(), key="ft_aud")
        f_min_score_a = st.slider("Minimum Score", 0.0, 1.0, 0.0, key="fs_aud")

    q_a = st.text_input("Describe the sound",
                        placeholder="e.g., tabla solo during rainstorm", key="aq_input")
    if q_a and st.session_state.show_suggestions:
        suggestions = analytics.get_search_suggestions(q_a)
        if suggestions:
            st.markdown("**Suggestions:**")
            st.markdown(
                "".join(f'<span class="suggestion-chip">{s}</span>' for s in suggestions),
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    if st.button("Run Acoustic Scan", key="vs_aud_btn") and q_a:
        mf = {"min_score": f_min_score_a}
        if f_lang_a != "All": mf["language"] = f_lang_a
        if f_tags_a:          mf["tags"]     = f_tags_a

        results, ms  = [], 0
        q_processed  = smart_query_preprocess(q_a, use_sarvam=use_sarvam_a)

        if use_lambda_audio and lambda_client.is_configured():
            with st.spinner("Querying cloud-powered acoustic index… 🚀"):
                t0         = time.time()
                lambda_res = lambda_client.search(q_processed, modality="audio", limit=6)
                if "error" in lambda_res:
                    st.error(f"Cloud search failed: {lambda_res['error']}")
                    results, ms = audio_engine.hybrid_search(text_query=q_processed, top_k=6, filters=mf)
                else:
                    for hit in lambda_res.get("results", []):
                        score_raw = hit.get("score", 0)
                        results.append({
                            "id":         hit.get("asset_id", "unknown"),
                            "modality":   "audio",
                            "score":      score_raw / 10.0 if score_raw > 1.0 else score_raw,
                            "confidence": "High" if score_raw > 7 else "Medium" if score_raw > 4 else "Low",
                            "path":       None,
                            "lang":       "en",
                        })
                    ms = (time.time() - t0) * 1000
                    st.caption("Results from **AWS Lambda + DynamoDB**")
        else:
            with st.spinner("Scanning local audio registry…"):
                results, ms = audio_engine.hybrid_search(text_query=q_processed, top_k=6, filters=mf)

        # Show Sarvam translation caption if translation happened
        if "last_translated" in st.session_state:
            orig, trans = st.session_state["last_translated"]
            if orig != trans:
                st.caption(f"🇮🇳 Sarvam magic: {orig} → **{trans}**")

        analytics.log_search(q_a, "audio", len(results), ms, st.session_state.user["username"])
        st.session_state.search_history.append({
            "query": q_a, "modality": "audio",
            "timestamp": time.time(), "results_count": len(results),
        })
        st.caption(f"Search time: {ms:.0f} ms")
        display_results(results, "audio", audio_engine)
        render_enrichment_ui(results, "audio", image_engine, audio_engine, web_search)


# ── Web Search ─────────────────────────────────────────────────────────────────
with tab_web:
    st.markdown("### 🌐 Web Search")
    st.markdown(f"""
    <div style="background:{T['bg_card']};border:1px solid {T['border']};
    border-left:3px solid {T['accent']};border-radius:8px;padding:0.8rem 1rem;
    margin-bottom:1rem;font-size:0.82rem;color:{T['text_muted']};">
        Live DuckDuckGo search. No API key required.
    </div>
    """, unsafe_allow_html=True)

    col_q, col_n = st.columns([4, 1])
    with col_q:
        web_q = st.text_input("Search query", placeholder="e.g., CLIP model India", key="web_query")
    with col_n:
        n_results = st.selectbox("Results", [5, 8, 10], index=1, key="web_n")

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        run_web = st.button("🔍 Search Web", use_container_width=True, key="web_run")
    with col_btn2:
        if st.button("✕ Clear", use_container_width=True, key="web_clear"):
            st.session_state.web_search_results = []
            st.session_state.web_page_content   = ""
            st.rerun()

    if run_web and web_q:
        with st.spinner(f"Searching for: *{web_q}*…"):
            t0  = time.time()
            res = web_search.search(web_q, max_results=n_results)
            ms  = (time.time() - t0) * 1000
        st.session_state.web_search_results = res
        analytics.log_search(web_q, "web", len(res), ms, st.session_state.user["username"])
        if not res:
            st.warning("No results found.")

    web_results = st.session_state.web_search_results
    if web_results:
        st.markdown(f"**{len(web_results)} results**")
        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        for i, r in enumerate(web_results):
            st.markdown(f"""
            <div class="web-result-card">
                <div class="web-result-title">{i+1}. {r['title']}</div>
                <div class="web-result-url">🔗 {r['url']}</div>
                <div class="web-result-snippet">{r['snippet'] or 'No description available.'}</div>
            </div>
            """, unsafe_allow_html=True)
            col_open, col_fetch, _ = st.columns([1, 1, 4])
            with col_open:
                st.markdown(
                    f'<a href="{r["url"]}" target="_blank" '
                    f'style="font-size:0.75rem;color:{T["accent"]};">↗ Open</a>',
                    unsafe_allow_html=True,
                )
            with col_fetch:
                if st.button("📄 Read Page", key=f"fetch_{i}", use_container_width=True):
                    with st.spinner("Fetching…"):
                        content = web_search.fetch_page_text(r["url"])
                    st.session_state.web_page_content = content
                    st.session_state.web_page_url     = r["url"]

        if st.session_state.web_page_content:
            st.markdown('<hr class="rule">', unsafe_allow_html=True)
            st.markdown(
                f"#### 📄 [{st.session_state.web_page_url[:80]}]({st.session_state.web_page_url})"
            )
            st.markdown(f"""
            <div style="background:{T['bg_card']};border:1px solid {T['border']};
            border-radius:8px;padding:1rem;font-size:0.85rem;color:{T['text_muted']};
            line-height:1.7;white-space:pre-wrap;max-height:400px;overflow-y:auto;">
{st.session_state.web_page_content}
            </div>
            """, unsafe_allow_html=True)

            if st.button("🌏 Translate Page to English", key="translate_page"):
                with st.spinner("Translating…"):
                    translated = translate_to_english(st.session_state.web_page_content[:2000])
                st.markdown("**Translated (first 2000 chars):**")
                st.text_area("", translated, height=200, key="translation_out")


# ── AWS Reverse Search ─────────────────────────────────────────────────────────
with tab_aws:
    render_aws_reverse_search_tab(T, aws_search, web_search)


# ── Clusters ───────────────────────────────────────────────────────────────────
with tab_cluster:
    st.markdown("### Asset Clusters")
    pick = st.radio("Registry", ["Image", "Audio"], horizontal=True, key="cluster_pick")
    eng  = image_engine if pick == "Image" else audio_engine

    if eng.index.ntotal < 3:
        st.info("Need at least 3 assets to cluster.")
    else:
        n_clusters = st.slider("Number of clusters", 2, min(10, eng.index.ntotal), 3)
        if st.button("Generate Clusters"):
            vecs = eng.get_all_vectors()
            if vecs is not None:
                kmeans  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels  = kmeans.fit_predict(vecs)
                pca     = PCA(n_components=2)
                proj    = pca.fit_transform(vecs)
                var_pct = pca.explained_variance_ratio_.sum() * 100
                df_scat = pd.DataFrame({
                    "x": proj[:, 0], "y": proj[:, 1],
                    "Cluster": [f"Cluster {l+1}" for l in labels],
                    "ID":      [r["id"] for r in eng.id_list],
                })
                fig = px.scatter(
                    df_scat, x="x", y="y", color="Cluster", text="ID",
                    title=f"{pick} Clusters · {var_pct:.1f}% variance",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig.update_layout(
                    plot_bgcolor=T["plot_bg"], paper_bgcolor=T["plot_bg"],
                    font=dict(family="DM Sans", color=T["plot_text"]),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"PCA variance: **{var_pct:.1f}%**")
                for i in range(n_clusters):
                    cluster_assets = [eng.id_list[j] for j in range(len(labels)) if labels[j] == i]
                    with st.expander(f"Cluster {i+1} ({len(cluster_assets)} assets)"):
                        cols = st.columns(4)
                        for idx, asset in enumerate(cluster_assets[:8]):
                            with cols[idx % 4]:
                                if eng.modality == "image":
                                    st.image(asset["path"], caption=asset["id"], use_container_width=True)
                                else:
                                    st.write(asset["id"])
                                    st.audio(asset["path"])


# ── Neural Auditor ─────────────────────────────────────────────────────────────
with tab_aud:
    pick = st.radio("Registry", ["Image", "Audio"], horizontal=True, key="aud_pick")
    eng  = image_engine if pick == "Image" else audio_engine

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vectors",   eng.index.ntotal)
    c2.metric("Index",     "FAISS HNSW")
    c3.metric("Embed Dim", CONFIG.EMBEDDING_DIM)
    c4.metric("Cache",     len(eng.embedding_cache))

    if eng.id_list:
        df = pd.DataFrame(eng.id_list)[["id", "lang", "modality", "timestamp"]]
        if "quality" in eng.id_list[0]:
            df["quality"] = [f"{r.get('quality', 0):.1%}" for r in eng.id_list]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No assets registered yet.")

    n = eng.index.ntotal
    if n >= 2:
        vecs = eng.get_all_vectors()
        if vecs is not None:
            nc   = min(3, n, CONFIG.EMBEDDING_DIM)
            pca  = PCA(n_components=nc)
            proj = pca.fit_transform(vecs)
            var  = pca.explained_variance_ratio_.sum() * 100
            df_p = pd.DataFrame(proj, columns=["x", "y", "z"][:nc])
            df_p["ID"] = [r["id"] for r in eng.id_list]
            if nc == 3:
                fig = px.scatter_3d(df_p, x="x", y="y", z="z", text="ID",
                                    color_discrete_sequence=[T["accent"]],
                                    title=f"{pick} · 3D Embedding Map")
            else:
                fig = px.scatter(df_p, x="x", y="y", text="ID",
                                 color_discrete_sequence=[T["accent"]],
                                 title=f"{pick} · 2D Embedding Map")
            fig.update_layout(
                plot_bgcolor=T["plot_bg"], paper_bgcolor=T["plot_bg"],
                font=dict(family="DM Sans", color=T["plot_text"]),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Search History ─────────────────────────────────────────────────────────────
with tab_hist:
    st.markdown("### Recent Searches")
    if not st.session_state.search_history:
        st.info("No search history yet.")
    else:
        for idx, s in enumerate(reversed(st.session_state.search_history[-20:])):
            t_str = datetime.fromtimestamp(s["timestamp"]).strftime("%Y-%m-%d %H:%M")
            icon  = {"image": "🖼", "audio": "🔊", "web": "🌐"}.get(s["modality"], "🔍")
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            c1.write(f"{icon} **{s['query'][:50]}**")
            c2.caption(t_str)
            c3.caption(f"{s['results_count']} results")
            with c4:
                if s["modality"] != "web":
                    if st.button("Re-run", key=f"rerun_{idx}"):
                        if s["modality"] == "image":
                            res, _ = image_engine.search(text=s["query"], top_k=6)
                            display_results(res, "image", image_engine)
                        else:
                            res, _ = audio_engine.search(text=s["query"], top_k=6)
                            display_results(res, "audio", audio_engine)
            st.markdown('<hr class="rule">', unsafe_allow_html=True)
        if st.button("Clear History"):
            st.session_state.search_history = []
            st.rerun()


# ── Admin ──────────────────────────────────────────────────────────────────────
with tab_admin:
    if st.session_state.user.get("role") != "admin":
        st.warning("Admin access only")
        st.stop()

    st.markdown("### Admin Control Panel")

    with st.expander("➕ Add New User"):
        col1, col2 = st.columns(2)
        with col1:
            new_user  = st.text_input("Username", key="new_u")
            new_email = st.text_input("Email (optional)", key="new_e")
        with col2:
            new_pass = st.text_input("Password", type="password", key="new_p")
            new_role = st.selectbox("Role", ["viewer", "uploader", "admin"], key="new_r")
        if st.button("Create Account", use_container_width=True):
            if new_user and new_pass:
                ok, msg = auth_manager.create_user(new_user, new_pass, new_role, new_email)
                (st.success if ok else st.error)(msg)
                if ok:
                    time.sleep(0.8)
                    st.rerun()

    st.subheader("Registered Users")
    users  = auth_manager.get_all_users()
    search = st.text_input("Filter Users", placeholder="Search…", key="usr_filter")
    filtered = [u for u in users if search.lower() in " ".join(map(str, u)).lower()]

    if not filtered:
        st.info("No matches found.")
    else:
        for i, row in enumerate(filtered):
            if not row or len(row) < 3:
                continue
            username, email, role = row[0], row[1] or "No email", row[2]
            created    = row[3]
            last_login = row[4] if len(row) > 4 else None

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);padding:15px;border-radius:10px;
            margin-bottom:5px;border-left:4px solid {T['accent']};">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-weight:bold;font-size:1.1rem;
                        color:{T['text_primary']};">{username}</span><br>
                        <span style="font-size:0.8rem;color:{T['text_muted']};">{email}</span>
                    </div>
                    <div style="font-family:'JetBrains Mono';background:{T['accent']}22;
                    color:{T['accent']};padding:4px 10px;border-radius:6px;font-size:0.75rem;">
                        {role.upper()}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns([1.5, 1.5, 1, 1])
            with c1: st.caption(f"📅 Joined: {created[:10]}")
            with c2: st.caption(f"👁️ Last: {last_login[:10] if last_login else 'Never'}")
            with c3:
                if role != "admin":
                    if st.button("⏫ Promote", key=f"prom_{username}_{i}", use_container_width=True):
                        auth_manager.db.execute(
                            "UPDATE users SET role='admin' WHERE username=?", (username,)
                        )
                        st.rerun()
            with c4:
                is_self = username == st.session_state.user["username"]
                with st.popover("🗑️ Delete", use_container_width=True, disabled=is_self):
                    st.error(f"Confirm deletion of {username}?")
                    if st.button("PURGE", key=f"y_del_{username}_{i}", use_container_width=True):
                        auth_manager.db.execute(
                            "DELETE FROM users WHERE username=?", (username,)
                        )
                        st.rerun()

    with st.expander("🩺 System Health"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Database", "Online" if auth_manager.db else "Offline")
        c2.metric("SMTP",     "Ready" if auth_manager.otp_sender.smtp_configured else "Disabled")
        c3.metric("Storage",  "Active" if os.path.exists(STORAGE_DIR) else "Error")
        if st.button("Test SMTP Connection"):
            try:
                with smtplib.SMTP("smtp.gmail.com", 587, timeout=5) as s:
                    s.starttls()
                st.success("Gmail SMTP reachable!")
            except Exception as e:
                st.error(f"Failed: {e}")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;padding:20px;'>
  <p><strong style='font-size:1.1em;color:#667eea;'>Created by Team Human</strong></p>
  <p>Powered by CLIP &amp; CLAP | Built for Bharat's Digital Future</p>
</div>
""", unsafe_allow_html=True)
