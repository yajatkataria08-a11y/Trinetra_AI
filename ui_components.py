import os
import time
import threading
import logging
import tempfile

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator

from config import CONFIG, DEVICE

logger = logging.getLogger("Trinetra")

# ==================== THEME TOKENS ====================

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


def get_theme(session_state) -> dict:
    return LIGHT if session_state.get("theme") == "light" else DARK


# ==================== LOGO SVG ====================
# Inline SVG recreation of the Trinetra eye/triangle logo
TRINETRA_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 80" width="64" height="43">
  <defs>
    <radialGradient id="eyeGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#e8a020" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="#e8a020" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="rimGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#ff6a00"/>
      <stop offset="45%" stop-color="#e8a020"/>
      <stop offset="100%" stop-color="#4e54c8"/>
    </linearGradient>
    <linearGradient id="triGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#e8a020"/>
      <stop offset="100%" stop-color="#fff5d6"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="2" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <!-- outer glow -->
  <ellipse cx="60" cy="40" rx="52" ry="34" fill="url(#eyeGlow)"/>
  <!-- eye shape -->
  <path d="M8,40 Q60,4 112,40 Q60,76 8,40Z" fill="#0d1321" stroke="url(#rimGrad)" stroke-width="2.5" filter="url(#glow)"/>
  <!-- inner iris -->
  <circle cx="60" cy="40" r="18" fill="#0a0f1e" stroke="#e8a020" stroke-width="1.2"/>
  <!-- triangle -->
  <polygon points="60,24 75,50 45,50" fill="none" stroke="url(#triGrad)" stroke-width="1.8" filter="url(#glow)"/>
  <!-- copyright dot -->
  <circle cx="60" cy="40" r="6" fill="none" stroke="#e8a020" stroke-width="1.4"/>
  <text x="60" y="44" text-anchor="middle" fill="#e8a020" font-size="7" font-family="serif" font-weight="bold">©</text>
  <!-- film strip left -->
  <rect x="4" y="34" width="3" height="12" rx="1" fill="#e8a020" opacity="0.7"/>
  <rect x="1" y="36" width="2" height="3" rx="0.5" fill="#ff6a00" opacity="0.8"/>
  <rect x="1" y="41" width="2" height="3" rx="0.5" fill="#ff6a00" opacity="0.8"/>
  <!-- audio bars right -->
  <rect x="103" y="37" width="2" height="6" rx="1" fill="#4e9ff5" opacity="0.9"/>
  <rect x="107" y="33" width="2" height="14" rx="1" fill="#4e9ff5" opacity="0.9"/>
  <rect x="111" y="36" width="2" height="8" rx="1" fill="#4e9ff5" opacity="0.7"/>
  <rect x="115" y="38" width="2" height="4" rx="1" fill="#4e9ff5" opacity="0.5"/>
</svg>
"""


# ==================== CSS ====================

def build_css(theme: str) -> str:
    tk       = LIGHT if theme == "light" else DARK
    is_dark  = theme != "light"

    return f"""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800;900&family=JetBrains+Mono:wght@400;500;600&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Reset & Root ── */
:root {{
  --t:        0.28s;
  --ease:     cubic-bezier(0.23,1,0.32,1);
  --accent:   {tk["accent"]};
  --accent-d: {tk["accent_dim"]};
  --glow:     {tk["accent_glow"]};
  --bg:       {tk["bg_base"]};
  --card:     {tk["bg_card"]};
  --border:   {tk["border"]};
  --text:     {tk["text_primary"]};
  --muted:    {tk["text_muted"]};
  --mono:     {tk["text_mono"]};
  --shadow:   {tk["shadow_card"]};
}}

#MainMenu {{ visibility: hidden !important; }}
footer {{ visibility: hidden !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}
/* Hide toolbar icons but NOT the sidebar toggle */
[data-testid="stToolbarActions"] {{ visibility: hidden !important; }}
/* Style header bar to match theme */
header[data-testid="stHeader"] {{
  background: {tk["bg_base"]} !important;
  border-bottom: 1px solid {tk["border"]} !important;
}}
/* Sidebar collapse/expand button - make it visible and amber */
[data-testid="stSidebarCollapsedControl"] {{
  visibility: visible !important;
  display: flex !important;
  background: {tk["bg_card"]} !important;
  border: 1px solid {tk["accent"]} !important;
  border-radius: 0 8px 8px 0 !important;
}}
[data-testid="stSidebarCollapsedControl"] button {{
  color: {tk["accent"]} !important;
  visibility: visible !important;
}}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"], .stApp {{
  background-color: {tk["bg_base"]} !important;
  color: {tk["text_primary"]} !important;
  font-family: 'DM Sans', sans-serif !important;
  transition: background-color var(--t) var(--ease), color var(--t) var(--ease);
}}

/* ── Grid background (dark only) ── */
{"[data-testid='stAppViewContainer']::before { content:''; position:fixed; inset:0; background-image: linear-gradient(" + tk["grid_line"] + " 1px,transparent 1px), linear-gradient(90deg," + tk["grid_line"] + " 1px,transparent 1px); background-size:44px 44px; pointer-events:none; z-index:0; }" if is_dark else ""}

[data-testid="stMainBlockContainer"] {{
  padding: 1.5rem 2.5rem;
  position: relative;
  z-index: 1;
}}
.block-container {{ padding-top: 3.5rem !important; }}

/* ── Empty container fix ── */
[data-testid="stVerticalBlock"] > div:empty,
[data-testid="stHorizontalBlock"] > div:empty,
.element-container:empty,
div[data-testid="column"]:empty {{
  display: none !important; height: 0 !important;
  min-height: 0 !important; padding: 0 !important; margin: 0 !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
  background: {tk["bg_panel"]} !important;
  border-right: 1px solid {tk["border"]} !important;
}}
[data-testid="stSidebar"] * {{ color: {tk["text_primary"]} !important; }}

/* ── Typography ── */
h1 {{
  font-family: 'Syne', sans-serif !important;
  font-weight: 900 !important;
  font-size: 2.8rem !important;
  letter-spacing: -0.02em !important;
  background: {tk["h1_grad"]};
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1 !important;
}}
h2, h3, h4 {{
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  color: {tk["text_primary"]} !important;
  letter-spacing: -0.01em;
}}

/* ── Hero header block ── */
.trinetra-hero {{
  display: flex;
  align-items: center;
  gap: 1.2rem;
  margin-bottom: 0.25rem;
}}
.trinetra-hero svg {{
  flex-shrink: 0;
  filter: drop-shadow(0 0 12px rgba(232,160,32,0.5));
  transition: filter 0.3s ease;
}}
.trinetra-hero:hover svg {{
  filter: drop-shadow(0 0 20px rgba(232,160,32,0.8));
}}
.trinetra-wordmark {{
  display: flex;
  flex-direction: column;
  gap: 0;
}}

.tagline {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: {tk["accent"]};
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-bottom: 1.8rem;
  margin-top: 0.1rem;
  opacity: 0.8;
  padding-left: 0.15rem;
}}

/* ── Stat strip ── */
.stat-strip {{
  display: flex;
  gap: 0.85rem;
  margin-bottom: 2rem;
}}
.stat-chip {{
  flex: 1;
  background: {tk["bg_card"]};
  border: 1px solid {tk["border"]};
  border-top: 2px solid {tk["accent"]};
  border-radius: 10px;
  padding: 0.85rem 1.1rem;
  display: flex;
  flex-direction: column;
  gap: 3px;
  cursor: default;
  transition: all var(--t) var(--ease);
  box-shadow: {tk["shadow_card"]};
  position: relative;
  overflow: hidden;
}}
.stat-chip::after {{
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, {tk["accent_glow"]}, transparent 60%);
  opacity: 0;
  transition: opacity var(--t);
}}
.stat-chip:hover {{ transform: translateY(-4px); box-shadow: {tk["shadow_hover"]}; border-color: {tk["border_hover"]}; }}
.stat-chip:hover::after {{ opacity: 1; }}
.stat-label {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.6rem;
  color: {tk["text_muted"]};
  text-transform: uppercase;
  letter-spacing: 0.12em;
}}
.stat-value {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.45rem;
  font-weight: 600;
  color: {tk["accent"]};
  line-height: 1;
}}

/* ── Buttons ── */
[data-testid="stButton"] button {{
  background: transparent !important;
  border: 1px solid {tk["accent"]} !important;
  color: {tk["accent"]} !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  border-radius: 7px !important;
  letter-spacing: 0.03em !important;
  transition: all 0.2s var(--ease) !important;
  position: relative;
  overflow: hidden;
}}
[data-testid="stButton"] button::before {{
  content: '';
  position: absolute;
  inset: 0;
  background: {tk["accent_glow"]};
  opacity: 0;
  transition: opacity 0.2s;
}}
[data-testid="stButton"] button:hover {{
  transform: translateY(-2px) !important;
  box-shadow: 0 0 18px {tk["accent_glow"]} !important;
  border-color: {tk["accent"]} !important;
}}
[data-testid="stButton"] button:hover::before {{ opacity: 1; }}

/* ── Form submit button (Login) ── */
[data-testid="stFormSubmitButton"] button {{
  background: transparent !important;
  border: 1px solid {tk["accent"]} !important;
  color: {tk["accent"]} !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  border-radius: 7px !important;
  letter-spacing: 0.03em !important;
  transition: all 0.2s var(--ease) !important;
}}
[data-testid="stFormSubmitButton"] button:hover {{
  background: {tk["accent"]} !important;
  color: #080c14 !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 0 18px {tk["accent_glow"]} !important;
}}

/* ── Inputs ── */
[data-testid="stTextInput"] input {{
  background: {tk["bg_card"]} !important;
  border: 1px solid {tk["border"]} !important;
  border-radius: 7px !important;
  color: {tk["text_primary"]} !important;
  font-family: 'DM Sans', sans-serif !important;
  transition: border-color var(--t), box-shadow var(--t) !important;
}}
[data-testid="stTextInput"] input:focus {{
  border-color: {tk["accent"]} !important;
  box-shadow: 0 0 0 3px {tk["accent_glow"]} !important;
  outline: none !important;
}}
[data-testid="stSelectbox"] > div > div {{
  background: {tk["bg_card"]} !important;
  border: 1px solid {tk["border"]} !important;
  color: {tk["text_primary"]} !important;
  border-radius: 7px !important;
}}
[data-testid="stSelectbox"] > div > div:hover {{ border-color: {tk["accent"]} !important; }}
[data-testid="stFileUploader"] {{
  background: {tk["bg_card"]} !important;
  border: 1px dashed {tk["accent"]} !important;
  border-radius: 8px !important;
}}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {{
  border-bottom: 1px solid {tk["border"]} !important;
  gap: 0.2rem;
}}
[data-testid="stTabs"] [role="tab"] {{
  font-family: 'Syne', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  color: {tk["text_muted"]} !important;
  background: transparent !important;
  border: none !important;
  padding: 0.55rem 1.1rem !important;
  border-radius: 7px 7px 0 0 !important;
  transition: color var(--t), background var(--t) !important;
  letter-spacing: 0.02em;
}}
[data-testid="stTabs"] [role="tab"]:hover {{
  color: {tk["accent"]} !important;
  background: {tk["accent_glow"]} !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
  color: {tk["accent"]} !important;
  border-bottom: 2px solid {tk["accent"]} !important;
}}

/* ── Data tables ── */
[data-testid="stDataFrame"] {{
  border: 1px solid {tk["border"]} !important;
  border-radius: 9px !important;
  overflow: hidden;
  box-shadow: {tk["shadow_card"]};
}}
[data-testid="stDataFrame"] th {{
  background: {tk["bg_panel"]} !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.7rem !important;
  color: {tk["accent"]} !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}}
[data-testid="stDataFrame"] td {{
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem !important;
  color: {tk["text_mono"]} !important;
  background: {tk["bg_card"]} !important;
}}

/* ── Metrics ── */
[data-testid="stMetric"] {{
  background: {tk["bg_card"]} !important;
  border: 1px solid {tk["border"]} !important;
  border-radius: 10px !important;
  padding: 0.9rem 1rem !important;
  box-shadow: {tk["shadow_card"]};
  transition: border-color var(--t), box-shadow var(--t), transform var(--t) !important;
}}
[data-testid="stMetric"]:hover {{
  border-color: {tk["border_hover"]} !important;
  box-shadow: {tk["shadow_hover"]} !important;
  transform: translateY(-2px);
}}
[data-testid="stMetricLabel"] {{
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.66rem !important;
  color: {tk["text_muted"]} !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}}
[data-testid="stMetricValue"] {{
  font-family: 'JetBrains Mono', monospace !important;
  color: {tk["accent"]} !important;
}}

/* ── Result cards ── */
.result-card {{
  background: {tk["bg_card"]};
  border: 1px solid {tk["border"]};
  border-radius: 12px;
  padding: 1rem;
  cursor: default;
  transition: all var(--t) var(--ease);
  box-shadow: {tk["shadow_card"]};
  position: relative;
  overflow: hidden;
}}
.result-card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, {tk["accent"]}, transparent);
  opacity: 0;
  transition: opacity var(--t);
}}
.result-card:hover {{
  background: {tk["bg_card_hover"]};
  border-color: {tk["border_hover"]};
  box-shadow: {tk["shadow_hover"]};
  transform: translateY(-5px) scale(1.01);
}}
.result-card:hover::before {{ opacity: 1; }}

/* ── Web result cards ── */
.web-result-card {{
  background: {tk["bg_card"]};
  border: 1px solid {tk["border"]};
  border-left: 3px solid transparent;
  border-radius: 10px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.75rem;
  transition: all var(--t) var(--ease);
  box-shadow: {tk["shadow_card"]};
}}
.web-result-card:hover {{
  border-left-color: {tk["accent"]};
  border-color: {tk["border_hover"]};
  box-shadow: {tk["shadow_hover"]};
  transform: translateX(5px);
}}
.web-result-title {{
  font-family: 'Syne', sans-serif;
  font-size: 0.98rem;
  font-weight: 700;
  color: {tk["accent"]};
  margin-bottom: 4px;
  letter-spacing: -0.01em;
}}
.web-result-url {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.63rem;
  color: #3dd68c;
  margin-bottom: 6px;
  word-break: break-all;
  opacity: 0.9;
}}
.web-result-snippet {{
  font-size: 0.84rem;
  color: {tk["text_muted"]};
  line-height: 1.6;
}}

/* ── Badges ── */
.badge {{
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
  padding: 3px 10px;
  border-radius: 5px;
  margin-bottom: 6px;
  cursor: default;
  letter-spacing: 0.08em;
  transition: transform 0.15s, box-shadow 0.15s;
}}
.badge:hover {{ transform: scale(1.06); box-shadow: 0 2px 8px rgba(0,0,0,.2); }}
.badge-high   {{ color: #3dd68c; border: 1px solid #3dd68c; background: rgba(61,214,140,0.08); }}
.badge-medium {{ color: #f5a623; border: 1px solid #f5a623; background: rgba(245,166,35,0.08); }}
.badge-low    {{ color: #f05252; border: 1px solid #f05252; background: rgba(240,82,82,0.08); }}

/* ── Score bar ── */
.score-bar-wrap {{ margin: 6px 0 10px; }}
.score-bar-track {{
  height: 3px;
  background: {tk["border"]};
  border-radius: 2px;
  overflow: hidden;
}}
.score-bar-fill {{
  height: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, {tk["accent_dim"]}, {tk["accent"]}, #fff5d6);
  animation: barGrow 0.7s cubic-bezier(0.34,1.56,0.64,1) forwards;
}}
@keyframes barGrow {{ from {{ width: 0%; }} }}

/* ── Sidebar stat ── */
.sidebar-stat {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.71rem;
  color: {tk["text_muted"]};
  padding: 5px 0;
  cursor: default;
  transition: all 0.15s;
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.sidebar-stat:hover {{
  color: {tk["text_primary"]} !important;
  padding-left: 4px;
}}
.sidebar-stat span {{
  color: {tk["accent"]};
  font-weight: 600;
}}

/* ── Theme toggle — WORKING FUNCTIONAL TOGGLE ── */
/* The actual toggle is a Streamlit button; this styles the decorative pill shown above it */
/* ── Theme toggle ── */
.theme-toggle-wrap {{
  position: relative;
  margin-bottom: 0.5rem;
}}
/* Pull the real button up to sit exactly over the pill */
.theme-toggle-wrap + div[data-testid="stButton"] {{
  margin-top: -2.85rem !important;
  position: relative;
  z-index: 2;
}}
.theme-toggle-wrap + div[data-testid="stButton"] button {{
  background: transparent !important;
  border: 1px solid {tk["accent"]} !important;
  border-radius: 50px !important;
  box-shadow: none !important;
  outline: none !important;
  color: transparent !important;
  height: 2.6rem !important;
  cursor: pointer !important;
}}
.theme-toggle-wrap + div[data-testid="stButton"] button:hover,
.theme-toggle-wrap + div[data-testid="stButton"] button:focus,
.theme-toggle-wrap + div[data-testid="stButton"] button:active {{
  background: transparent !important;
  border: 1px solid {tk["accent"]} !important;
  box-shadow: none !important;
  transform: none !important;
}}

.theme-toggle-pill {{
  margin-top: -2.8rem !important;
}}
.theme-toggle-pill + * div[data-testid="stButton"] button,
div[data-testid="stButton"]:has(~ .theme-toggle-pill) button {{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}}

.theme-toggle-pill {{
  display: flex;
  align-items: center;
  gap: 10px;
  background: {tk["bg_card"]};
  border: 1px solid {tk["border"]};
  border-radius: 50px;
  padding: 7px 14px;
  user-select: none;
  pointer-events: none;  /* clicks go to the real button underneath */
}}
.tt-track {{
  position: relative;
  width: 42px;
  height: 22px;
  border-radius: 11px;
  background: {"#e8a020" if is_dark else tk["border"]};
  transition: background 0.35s var(--ease);
  flex-shrink: 0;
  box-shadow: {"inset 0 0 8px rgba(232,160,32,0.4)" if is_dark else "none"};
}}
.tt-knob {{
  position: absolute;
  top: 3px;
  left: {"22px" if is_dark else "3px"};
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: {"#fff5d6" if is_dark else "#b87818"};
  box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  transition: left 0.35s cubic-bezier(0.68,-0.55,0.265,1.55);
}}
.tt-label {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  font-weight: 600;
  color: {tk["accent"]};
  text-transform: uppercase;
  letter-spacing: 0.1em;
  flex: 1;
}}
.tt-icon {{ font-size: 0.95rem; line-height: 1; }}

/* ── Checkboxes ── */
[data-testid="stCheckbox"] > label > div:first-child {{
  background: {tk["bg_card"]} !important;
  border: 2px solid {tk["accent"]} !important;
  border-radius: 4px !important;
}}
[data-testid="stCheckbox"] svg {{
  color: #080c14 !important;
  fill: #080c14 !important;
}}

/* ── Radio buttons ── */
[data-testid="stRadio"] label p {{
  color: {tk["text_primary"]} !important;
}}

/* ── Delete/trash buttons in admin panel ── */
[data-testid="stButton"] button[kind="secondary"] {{
  background: transparent !important;
}}

/* ── Popovers (delete confirm) ── */
[data-testid="stPopover"] > div,
[data-testid="stPopoverBody"] {{
  background: {tk["bg_card"]} !important;
  border: 1px solid {tk["border"]} !important;
  color: {tk["text_primary"]} !important;
}}
/* Popover trigger button (🗑️ Delete) */
[data-testid="stPopover"] button {{
  background: transparent !important;
  border: 1px solid {tk["border"]} !important;
  color: {tk["text_primary"]} !important;
}}
[data-testid="stPopover"] button:hover {{
  border-color: #cc3333 !important;
  color: #cc3333 !important;
}}
/* Promote button */
[data-testid="stButton"] button:has(span) {{
  background: transparent !important;
}}

/* ── Login page ── */
.login-container {{
  max-width: 420px;
  margin: 8rem auto;
  padding: 3rem;
  background: {tk["bg_card"]};
  border: 1px solid {tk["border"]};
  border-radius: 18px;
  box-shadow: {tk["shadow_card"]};
  position: relative;
  overflow: hidden;
}}
.login-container::before {{
  content: '';
  position: absolute;
  top: -1px; left: 20%; right: 20%;
  height: 2px;
  background: linear-gradient(90deg, transparent, {tk["accent"]}, transparent);
}}
.login-logo {{
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 0.75rem;
  filter: drop-shadow(0 0 16px rgba(232,160,32,0.6));
}}
.login-title {{
  font-family: 'Syne', sans-serif;
  font-weight: 900;
  font-size: 2.2rem;
  text-align: center;
  background: {tk["h1_grad"]};
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.3rem;
  letter-spacing: -0.02em;
}}
.login-subtitle {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  color: {tk["text_muted"]};
  text-align: center;
  text-transform: uppercase;
  letter-spacing: 0.2em;
  margin-bottom: 2.5rem;
}}

/* ── Misc ── */
.rule {{ border: none; border-top: 1px solid {tk["border"]}; margin: 1.4rem 0; }}

.quality-indicator {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  padding: 3px 9px;
  border-radius: 5px;
  background: {tk["bg_card_hover"]};
  border: 1px solid {tk["border"]};
  color: {tk["text_muted"]};
}}

.suggestion-chip {{
  display: inline-block;
  padding: 4px 12px;
  margin: 3px;
  background: {tk["bg_card"]};
  border: 1px solid {tk["border"]};
  border-radius: 20px;
  font-size: 0.73rem;
  cursor: pointer;
  transition: all 0.2s var(--ease);
  font-family: 'DM Sans', sans-serif;
  color: {tk["text_muted"]};
}}
.suggestion-chip:hover {{
  background: {tk["accent_glow"]};
  border-color: {tk["accent"]};
  color: {tk["accent"]};
  transform: translateY(-2px);
}}

[data-testid="stAlert"] {{
  background: {tk["bg_card"]} !important;
  border: 1px solid {tk["border"]} !important;
  border-left: 3px solid {tk["accent"]} !important;
  border-radius: 7px !important;
  color: {tk["text_primary"]} !important;
}}

/* ── Animations ── */
@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(18px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
  from {{ opacity: 0; }}
  to   {{ opacity: 1; }}
}}
.a1 {{ animation: fadeUp 0.45s var(--ease) both; }}
.a2 {{ animation: fadeUp 0.45s 0.09s var(--ease) both; }}
.a3 {{ animation: fadeUp 0.45s 0.18s var(--ease) both; }}

/* ── Mobile ── */
@media (max-width: 768px) {{
  h1 {{ font-size: 2rem !important; }}
  .stat-strip {{ flex-direction: column; gap: 0.5rem; }}
  [data-testid="stMainBlockContainer"] {{ padding: 1rem; }}
  [data-testid="stButton"] button {{ width: 100% !important; }}
  [data-testid="stButton"] button:hover {{ transform: none !important; }}
  [data-testid="stTextInput"] input {{ font-size: 16px !important; padding: 0.8rem !important; }}
  .result-card:hover, .web-result-card:hover {{ transform: none !important; }}
  * {{ animation-duration: 0.2s !important; transition-duration: 0.15s !important; }}
}}
</style>
"""


# ==================== BADGE & SCORE BAR ====================

def confidence_badge(conf: str) -> str:
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}[conf]
    return f'<span class="badge {cls}">● {conf} Confidence</span>'


def score_bar(score: float) -> str:
    pct = min(max(score, 0.0), 1.0) * 100
    return (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar-track">'
        f'<div class="score-bar-fill" style="width:{pct:.1f}%"></div>'
        f'</div></div>'
    )


# ==================== RESULT DISPLAY ====================

def display_results(results: list, modality: str, engine=None):
    from database import get_shared_metadata_db

    T = get_theme(st.session_state)

    if not results:
        st.info("No matches found in the registry.")
        return

    meta_db = get_shared_metadata_db()
    cols    = st.columns(min(len(results), 3))

    for idx, r in enumerate(results):
        with cols[idx % 3]:
            st.markdown(
                f'<div class="result-card">'
                f'{confidence_badge(r["confidence"])}'
                f'{score_bar(r["score"])}'
                f'<div style="font-size:.7rem;color:{T["text_muted"]}">Score: {r["score"]:.3f}</div>',
                unsafe_allow_html=True,
            )
            if "quality" in r:
                st.markdown(
                    f'<div class="quality-indicator">Quality: {r["quality"]:.1%}</div>',
                    unsafe_allow_html=True,
                )
            if modality == "image":
                if r.get("path") and os.path.exists(r["path"]):
                    st.image(r["path"], caption=r["id"], use_container_width=True)
                else:
                    st.write(f"**{r['id']}** *(file not found locally)*")
            else:
                st.write(f"**{r['id']}**")
                if r.get("path") and os.path.exists(r["path"]):
                    st.audio(r["path"])
            st.caption(f"Lang: {r.get('lang','—')} · {r.get('timestamp','')[:10]}")

            with st.expander("Feedback"):
                avg_rating, count = meta_db.get_rating(r["id"])
                st.write(f"Rating: {'⭐' * int(avg_rating)} ({count} votes)")
                new_rating = st.slider("Rate this", 1, 5, 3, key=f"rate_{r['id']}_{idx}")
                if st.button("Submit Rating", key=f"submit_rate_{r['id']}_{idx}"):
                    meta_db.add_rating(r["id"], st.session_state.user["username"], new_rating)
                    st.success("Rating submitted!")
                    st.rerun()

                comments = meta_db.get_comments(r["id"])
                if comments:
                    st.markdown("**Comments:**")
                    for user, comment, ts in comments[:3]:
                        st.caption(f"**{user}** ({ts[:10]}): {comment}")

                new_comment = st.text_input("Add comment", key=f"comment_{r['id']}_{idx}")
                if st.button("Post Comment", key=f"post_{r['id']}_{idx}") and new_comment:
                    meta_db.add_comment(
                        r["id"], st.session_state.user["username"], new_comment
                    )
                    st.success("Comment posted!")
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.last_search_results = results


# ==================== NEURAL GROUNDING ====================

def _generate_asset_description(result: dict, modality: str) -> str:
    asset_id = result.get("id", "").replace("_", " ")
    lang     = result.get("lang", "en")
    return (
        f"{asset_id} Indian {lang} image photo"
        if modality == "image"
        else f"{asset_id} Indian {lang} audio sound"
    )


def _prefetch_web_pages(hits: list, storage: list) -> None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
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
            storage[i] = {"url": hit["url"], "content": "Could not fetch page content."}


def render_enrichment_ui(results: list, modality: str,
                         image_engine=None, audio_engine=None,
                         web_search=None) -> None:
    """Neural Grounding panel — renders below search results."""
    if not results:
        return

    T = get_theme(st.session_state)

    st.markdown("---")
    min_web_sim = st.slider(
        "Neural Grounding Sensitivity", 0.3, 0.7, 0.45, 0.05,
        help="Lower = more (looser) web results.",
        key=f"ground_slider_{modality}",
    )

    if not st.button("🌐 Enrich with Web Context",
                     use_container_width=True, key=f"enr_btn_{modality}"):
        return

    top  = results[0]
    desc = _generate_asset_description(top, modality)
    lang = top.get("lang", "en")

    query_parts = [desc]
    if lang != "en" and lang in CONFIG.SUPPORTED_LANGUAGES:
        try:
            native = GoogleTranslator(source="en", target=lang).translate(desc)
            query_parts.append(f'"{native}"')
        except Exception:
            pass
    web_query = f"{' OR '.join(query_parts)} India"

    with st.spinner(f"Searching web for \"{desc}\"…"):
        hits_raw = web_search.search(web_query, max_results=min_web_sim * 3 or 9)

    if not hits_raw:
        st.warning("No web results found. Try lowering the sensitivity slider.")
        return

    eng       = image_engine if modality == "image" else audio_engine
    local_emb = eng.get_embedding(file_path=top["path"]) if top.get("path") else None

    filtered = []
    for hit in hits_raw:
        if not hit.get("snippet"):
            continue
        snip_emb = image_engine.get_embedding(text=hit["snippet"])
        if local_emb is not None:
            sim = float(
                np.dot(local_emb, snip_emb)
                / (np.linalg.norm(local_emb) * np.linalg.norm(snip_emb) + 1e-9)
            )
        else:
            sim = 0.5
        if sim >= min_web_sim:
            hit["grounding_score"] = sim
            filtered.append(hit)

    max_web  = 3
    filtered = sorted(filtered, key=lambda x: x["grounding_score"], reverse=True)[:max_web]

    if not filtered:
        st.warning(
            "No web results met the grounding threshold. "
            "Try lowering the slider or using a more well-known asset."
        )
        return

    st.success(f"Grounded {len(filtered)} web result(s) for **{top['id']}**")
    st.caption(f"Query used: `{web_query}`")

    prefetch_storage = [None] * len(filtered)
    bg = threading.Thread(target=_prefetch_web_pages, args=(filtered, prefetch_storage))
    bg.daemon = True
    bg.start()

    for i, hit in enumerate(filtered):
        gs    = hit.get("grounding_score", 0)
        color = "#2ecc71" if gs > 0.6 else "#f39c12" if gs > 0.45 else "#e74c3c"

        snippet_display = hit["snippet"]
        if top.get("lang", "en") != "en":
            try:
                snippet_display = GoogleTranslator(
                    source="auto", target="en"
                ).translate(hit["snippet"][:500])
            except Exception:
                pass

        st.markdown(f"""
        <div class="web-result-card">
            <div class="web-result-title">{i+1}. {hit['title']}</div>
            <div class="web-result-url">🔗
                <a href="{hit['url']}" target="_blank" style="color:#2ecc71;">
                    {hit['url'][:70]}…
                </a>
            </div>
            <div class="web-result-snippet">{snippet_display}</div>
            <div style="font-size:0.75rem;color:{color};font-family:'JetBrains Mono',monospace;
                        margin-top:8px;font-weight:600;">
                ● Neural Grounding Score: {gs:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<a href="{hit["url"]}" target="_blank" '
                f'style="color:{T["accent"]};font-size:0.8rem;">↗ Open in browser</a>',
                unsafe_allow_html=True,
            )
        with c2:
            if st.button("📄 Read full page",
                         key=f"read_enr_{modality}_{i}", use_container_width=True):
                time.sleep(0.4)
                if prefetch_storage[i]:
                    st.text_area("Full Page Content", prefetch_storage[i]["content"],
                                 height=250, key=f"page_content_{modality}_{i}")
                else:
                    st.info("Still fetching — try again in a moment.")


# ==================== AWS REVERSE SEARCH TAB ====================

def render_aws_reverse_search_tab(T: dict, aws_search, web_search) -> None:
    st.markdown("### 🔍 AWS Reverse Search")
    st.markdown(f"""
    <div style="background:{T['bg_card']};border:1px solid {T['border']};
    border-left:3px solid {T['accent']};border-radius:8px;padding:0.8rem 1rem;
    margin-bottom:1rem;font-size:0.82rem;color:{T['text_muted']};">
        Use AWS AI to analyze uploads and search the web.
        Powered by your $100 hackathon credit!
    </div>
    """, unsafe_allow_html=True)

    is_configured, config_msg = aws_search.test_aws_connection()

    if not is_configured:
        st.error(config_msg)
        st.code("""# Add to .streamlit/secrets.toml:
AWS_ACCESS_KEY_ID = "AKIA..."
AWS_SECRET_ACCESS_KEY = "wJalr..."
AWS_REGION = "us-east-1"
""", language="toml")
        return

    st.success(config_msg)
    img_tab, cost_tab = st.tabs(["🖼️ Image Analysis", "💰 Cost Tracker"])

    with img_tab:
        img_file = st.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png", "webp"], key="aws_img"
        )
        if img_file:
            st.image(img_file, caption="Uploaded", use_container_width=True)
            st.caption(f"Size: {img_file.size / 1024:.1f} KB")

        if img_file and st.button("🔍 Analyze & Search",
                                   use_container_width=True, key="aws_analyze"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(img_file.getbuffer())
                tmp_path = tmp.name
            try:
                with st.spinner("Analyzing with AWS…"):
                    results = aws_search.search_web_from_image_analysis(tmp_path, web_search)

                if "error" in results:
                    st.error(f"Error: {results['error']}")
                else:
                    st.markdown("#### 🤖 AWS Analysis")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Labels", len(results["analysis"]["labels"]))
                    col2.metric("Text (OCR)", len(results["analysis"]["text"]))
                    col3.metric("Faces", results["analysis"]["faces"])

                    if results["analysis"]["labels"]:
                        st.markdown("**🏷️ Detected:**")
                        cols = st.columns(3)
                        for i, label in enumerate(results["analysis"]["labels"][:9]):
                            with cols[i % 3]:
                                st.markdown(f"""
                                <div style="background:{T['bg_card']};border:1px solid {T['border']};
                                border-radius:6px;padding:0.5rem;margin-bottom:0.5rem;">
                                    <div style="font-weight:600;color:{T['accent']};">
                                        {label['name']}
                                    </div>
                                    <div style="font-size:0.7rem;color:{T['text_muted']};">
                                        {label['confidence']:.1f}%
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                    if results["analysis"]["text"]:
                        st.markdown("**📝 Text Found:**")
                        st.info(" · ".join(results["analysis"]["text"]))

                    st.markdown("---")
                    st.markdown("#### 🔎 Search Queries Generated")
                    for i, q in enumerate(results["search_queries"], 1):
                        st.code(f"{i}. {q}")

                    if results["web_results"]:
                        st.markdown(f"#### 🌐 Web Results ({len(results['web_results'])})")
                        for i, r in enumerate(results["web_results"], 1):
                            st.markdown(f"""
                            <div class="web-result-card">
                                <div class="web-result-title">{i}. {r['title']}</div>
                                <div class="web-result-url">🔗 {r['url']}</div>
                                <div class="web-result-snippet">{r['snippet']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(
                                f'<a href="{r["url"]}" target="_blank">↗ Open</a>',
                                unsafe_allow_html=True,
                            )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    with cost_tab:
        st.markdown("### 💰 AWS Cost Tracker")
        num_images = st.number_input("Images to process", 0, 10000, 100, 10)
        if st.button("Calculate", use_container_width=True):
            costs = aws_search.estimate_costs(num_images)
            col1, col2 = st.columns(2)
            col1.metric("Cost",      f"${costs['costs']['total']:.2f}")
            col2.metric("Remaining", f"${costs['remaining_credit']:.2f}")
            st.write(f"🖼️ Can process {costs['can_process']['images']:,} more images")
            st.progress(min(costs["costs"]["total"] / 100, 1.0))
