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
import pandas as pd
import plotly.express as px
import sqlite3
import hashlib
import logging
from datetime import datetime
from sklearn.decomposition import PCA
from PIL import Image
from deep_translator import GoogleTranslator
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, ClapModel

# ==================== LOGGING ==================== #
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f'logs/trinetra_{datetime.now():%Y%m%d}.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Trinetra')

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

# ==================== PAGE CONFIG ==================== #
st.set_page_config(
    page_title="Trinetra V4.1 · Bharat AI",
    layout="wide",
    page_icon="👁️",
    initial_sidebar_state="expanded",
)

# ==================== SESSION STATE ==================== #
if "theme"               not in st.session_state: st.session_state.theme = "dark"
if "search_history"      not in st.session_state: st.session_state.search_history = []
if "last_search_results" not in st.session_state: st.session_state.last_search_results = []

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
# CRITICAL: every literal CSS brace must be doubled {{ }} inside an f-string.
# Single braces are reserved for Python variable interpolation only.
def _css(T, is_light):
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
        "font-family:\'DM Sans\',sans-serif !important;"
        "transition:background-color var(--t),color var(--t); }"
        "[data-testid=\"stAppViewContainer\"]::before {"
        "content:\'\'; position:fixed; inset:0;"
        "background-image:linear-gradient(" + T["grid_line"] + " 1px,transparent 1px),"
        "linear-gradient(90deg," + T["grid_line"] + " 1px,transparent 1px);"
        "background-size:40px 40px; pointer-events:none; z-index:0; }"
        "[data-testid=\"stMainBlockContainer\"] { padding:2rem 2.5rem; position:relative; z-index:1; }"
        "[data-testid=\"stSidebar\"] { background:" + T["bg_panel"] + " !important; border-right:1px solid " + T["border"] + " !important; }"
        "[data-testid=\"stSidebar\"] * { color:" + T["text_primary"] + " !important; }"
        "h1 { font-family:\'Syne\',sans-serif !important; font-weight:800 !important; font-size:2.6rem !important;"
        "background:" + T["h1_grad"] + "; -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }"
        "h2,h3,h4 { font-family:\'Syne\',sans-serif !important; font-weight:700 !important; color:" + T["text_primary"] + " !important; }"
        ".tagline { font-family:\'JetBrains Mono\',monospace; font-size:0.78rem; color:" + T["accent"] + ";"
        "letter-spacing:0.12em; text-transform:uppercase; margin-bottom:2rem; opacity:0.85; }"
        ".stat-strip { display:flex; gap:1rem; margin-bottom:2rem; }"
        ".stat-chip { flex:1; background:" + T["bg_card"] + "; border:1px solid " + T["border"] + ";"
        "border-top:2px solid " + T["accent"] + "; border-radius:10px; padding:0.9rem 1.2rem;"
        "display:flex; flex-direction:column; gap:4px; cursor:default; transition:all var(--t); box-shadow:" + T["shadow_card"] + "; }"
        ".stat-chip:hover { background:" + T["bg_card_hover"] + "; border-color:" + T["border_hover"] + ";"
        "box-shadow:" + T["shadow_hover"] + "; transform:translateY(-3px); }"
        ".stat-label { font-family:\'JetBrains Mono\',monospace; font-size:0.65rem; color:" + T["text_muted"] + "; text-transform:uppercase; letter-spacing:0.1em; }"
        ".stat-value { font-family:\'JetBrains Mono\',monospace; font-size:1.5rem; font-weight:600; color:" + T["accent"] + "; }"
        "[data-testid=\"stButton\"] button { background:transparent !important; border:1px solid " + T["accent"] + " !important;"
        "color:" + T["accent"] + " !important; font-family:\'Syne\',sans-serif !important; font-weight:700 !important; border-radius:6px !important; transition:all 0.2s !important; }"
        "[data-testid=\"stButton\"] button:hover { background:" + T["accent_glow"] + " !important; transform:translateY(-2px); box-shadow:0 0 16px " + T["accent_glow"] + " !important; }"
        "[data-testid=\"stButton\"] button:active { transform:translateY(0); }"
        "[data-testid=\"stTextInput\"] input { background:" + T["bg_card"] + " !important; border:1px solid " + T["border"] + " !important;"
        "border-radius:6px !important; color:" + T["text_primary"] + " !important; font-family:\'DM Sans\',sans-serif !important; transition:border-color var(--t),box-shadow var(--t) !important; }"
        "[data-testid=\"stTextInput\"] input:focus { border-color:" + T["accent"] + " !important; box-shadow:0 0 0 3px " + T["accent_glow"] + " !important; }"
        "[data-testid=\"stSelectbox\"] > div > div { background:" + T["bg_card"] + " !important; border:1px solid " + T["border"] + " !important; color:" + T["text_primary"] + " !important; }"
        "[data-testid=\"stSelectbox\"] > div > div:hover { border-color:" + T["accent"] + " !important; }"
        "[data-testid=\"stFileUploader\"] { background:" + T["bg_card"] + " !important; border:1px dashed " + T["accent"] + " !important; border-radius:8px !important; }"
        "[data-testid=\"stFileUploader\"]:hover { background:" + T["bg_card_hover"] + " !important; box-shadow:0 0 14px " + T["accent_glow"] + " !important; }"
        "[data-testid=\"stTabs\"] [role=\"tablist\"] { border-bottom:1px solid " + T["border"] + " !important; }"
        "[data-testid=\"stTabs\"] [role=\"tab\"] { font-family:\'Syne\',sans-serif !important; font-size:.85rem !important; font-weight:700 !important; color:" + T["text_muted"] + " !important; background:transparent !important; border:none !important; padding:.6rem 1.2rem !important; border-radius:6px 6px 0 0 !important; transition:color var(--t),background var(--t) !important; }"
        "[data-testid=\"stTabs\"] [role=\"tab\"]:hover { color:" + T["accent"] + " !important; background:rgba(232,160,32,0.07) !important; }"
        "[data-testid=\"stTabs\"] [role=\"tab\"][aria-selected=\"true\"] { color:" + T["accent"] + " !important; border-bottom:2px solid " + T["accent"] + " !important; }"
        "[data-testid=\"stDataFrame\"] { border:1px solid " + T["border"] + " !important; border-radius:8px !important; overflow:hidden; box-shadow:" + T["shadow_card"] + "; }"
        "[data-testid=\"stDataFrame\"]:hover { border-color:" + T["border_hover"] + " !important; }"
        "[data-testid=\"stDataFrame\"] th { background:" + T["bg_panel"] + " !important; font-family:\'JetBrains Mono\',monospace !important; font-size:.72rem !important; color:" + T["accent"] + " !important; text-transform:uppercase; }"
        "[data-testid=\"stDataFrame\"] td { font-family:\'JetBrains Mono\',monospace !important; font-size:.78rem !important; color:" + T["text_mono"] + " !important; background:" + T["bg_card"] + " !important; }"
        "[data-testid=\"stDataFrame\"] tr:hover td { background:" + T["bg_card_hover"] + " !important; }"
        "[data-testid=\"stMetric\"] { background:" + T["bg_card"] + " !important; border:1px solid " + T["border"] + " !important; border-radius:10px !important; padding:.9rem 1rem !important; box-shadow:" + T["shadow_card"] + "; cursor:default; transition:border-color var(--t),box-shadow var(--t),transform var(--t) !important; }"
        "[data-testid=\"stMetric\"]:hover { border-color:" + T["border_hover"] + " !important; box-shadow:" + T["shadow_hover"] + " !important; transform:translateY(-2px); }"
        "[data-testid=\"stMetricLabel\"] { font-family:\'JetBrains Mono\',monospace !important; font-size:.68rem !important; color:" + T["text_muted"] + " !important; text-transform:uppercase; }"
        "[data-testid=\"stMetricValue\"] { font-family:\'JetBrains Mono\',monospace !important; color:" + T["accent"] + " !important; }"
        "[data-testid=\"stExpander\"] { background:" + T["bg_card"] + " !important; border:1px solid " + T["border"] + " !important; border-radius:8px !important; }"
        "[data-testid=\"stExpander\"]:hover { border-color:" + T["border_hover"] + " !important; box-shadow:0 2px 12px " + T["accent_glow"] + " !important; }"
        "[data-testid=\"stRadio\"] label { color:" + T["text_primary"] + " !important; }"
        "[data-testid=\"stRadio\"] label:hover { color:" + T["accent"] + " !important; }"
        "[data-testid=\"stAlert\"] { background:" + T["bg_card"] + " !important; border:1px solid " + T["border"] + " !important; border-left:3px solid #00c9b1 !important; border-radius:6px !important; color:" + T["text_primary"] + " !important; }"
        ".badge { display:inline-block; font-family:\'JetBrains Mono\',monospace; font-size:0.68rem; font-weight:600; text-transform:uppercase; padding:3px 10px; border-radius:4px; margin-bottom:6px; cursor:default; transition:transform .15s,box-shadow .15s; }"
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
        ".sidebar-stat { font-family:\'JetBrains Mono\',monospace; font-size:0.72rem; color:" + T["text_muted"] + "; padding:5px 0; cursor:default; transition:all 0.15s; }"
        ".sidebar-stat:hover { color:" + T["text_primary"] + " !important; transform:translateX(4px); }"
        ".sidebar-stat span { color:" + T["accent"] + "; font-weight:600; }"
        ".tog-pill { display:flex; align-items:center; gap:10px; background:" + T["bg_card"] + "; border:1px solid " + T["accent"] + "; border-radius:50px; padding:8px 14px; margin-bottom:0.75rem; cursor:pointer; user-select:none; transition:box-shadow var(--t),background var(--t); }"
        ".tog-pill:hover { box-shadow:0 0 14px " + T["accent_glow"] + "; background:" + T["bg_card_hover"] + "; }"
        ".tog-track { position:relative; width:44px; height:22px; border-radius:11px; background:" + T["border"] + "; border:1px solid " + T["border"] + "; flex-shrink:0; }"
        ".tog-knob { position:absolute; top:2px; left:" + knob_left + "; width:16px; height:16px; border-radius:50%; background:" + T["accent"] + "; box-shadow:0 1px 4px rgba(0,0,0,0.3); transition:left 0.3s cubic-bezier(0.34,1.56,0.64,1); }"
        ".tog-icon { font-size:1rem; line-height:1; }"
        ".tog-label { font-family:\'JetBrains Mono\',monospace; font-size:0.72rem; font-weight:600; color:" + T["accent"] + "; text-transform:uppercase; letter-spacing:0.08em; flex:1; }"
        ".rule { border:none; border-top:1px solid " + T["border"] + "; margin:1.5rem 0; }"
        "@keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }"
        ".a1 { animation:fadeUp 0.4s ease both; }"
        ".a2 { animation:fadeUp 0.4s 0.08s ease both; }"
        ".a3 { animation:fadeUp 0.4s 0.16s ease both; }"
        "</style>"
    )

st.markdown(_css(T, is_light), unsafe_allow_html=True)

# ==================== METADATA DATABASE ==================== #
class MetadataDB:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(BASE_DIR, "metadata.db")
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY, modality TEXT NOT NULL, language TEXT NOT NULL,
                file_path TEXT NOT NULL, file_size INTEGER, upload_date TEXT NOT NULL,
                faiss_index INTEGER NOT NULL, tags TEXT, description TEXT, collection TEXT,
                UNIQUE(id)
            )
        """)
        self.conn.commit()

    def add_asset(self, asset_id, modality, language, file_path, file_size,
                  faiss_idx, tags=None, description="", collection=""):
        try:
            self.conn.execute("""
                INSERT INTO assets
                (id, modality, language, file_path, file_size, upload_date,
                 faiss_index, tags, description, collection)
                VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?)
            """, (asset_id, modality, language, file_path, file_size,
                  faiss_idx, json.dumps(tags or []), description, collection))
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass

    def search_metadata(self, modality=None, language=None, tags=None,
                        date_from=None, collection=None):
        query  = "SELECT id, faiss_index FROM assets WHERE 1=1"
        params = []
        if modality:   query += " AND modality = ?";             params.append(modality)
        if language:   query += " AND language = ?";             params.append(language)
        if date_from:  query += " AND upload_date >= ?";         params.append(date_from)
        if tags:       query += " AND tags LIKE ?";              params.append(f'%{tags}%')
        if collection: query += " AND collection = ?";           params.append(collection)
        return self.conn.execute(query, params).fetchall()

    def get_all_tags(self):
        cursor  = self.conn.execute("SELECT DISTINCT tags FROM assets WHERE tags != '[]'")
        all_tags = set()
        for row in cursor:
            all_tags.update(json.loads(row[0]))
        return sorted(list(all_tags))

    def get_all_collections(self):
        cursor = self.conn.execute("SELECT DISTINCT collection FROM assets WHERE collection != ''")
        return [row[0] for row in cursor]


# ==================== ANALYTICS ==================== #
class AnalyticsTracker:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(BASE_DIR, "analytics.db")
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT, modality TEXT, results_count INTEGER,
                timestamp TEXT, search_duration_ms REAL
            )
        """)
        self.conn.commit()

    def log_search(self, query, modality, results_count, duration_ms):
        try:
            self.conn.execute("""
                INSERT INTO searches (query_text, modality, results_count, timestamp, search_duration_ms)
                VALUES (?, ?, ?, datetime('now'), ?)
            """, (query, modality, results_count, duration_ms))
            self.conn.commit()
        except Exception:
            pass

    def get_stats(self, days=7):
        cursor = self.conn.execute(f"""
            SELECT COUNT(*), AVG(results_count), AVG(search_duration_ms)
            FROM searches
            WHERE timestamp >= datetime('now', '-{days} days')
        """)
        return cursor.fetchone()


# ==================== MODELS ==================== #
@st.cache_resource(show_spinner="🔮 Loading neural models…")
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

    def register(self, temp_path, asset_id, ext, lang, tags=None, description="", collection=""):
        asset_id = sanitize_asset_id(asset_id)
        ok, msg  = validate_asset_id(asset_id)
        if not ok:            return False, msg
        if self.id_exists(asset_id): return False, f"ID '{asset_id}' already exists"

        perm_path = os.path.join(STORAGE_DIR, f"{asset_id}{ext}")
        try:
            with open(temp_path, "rb") as s, open(perm_path, "wb") as d:
                d.write(s.read())
            emb       = self.get_embedding(file_path=perm_path)
            faiss_idx = self.index.ntotal
            self.index.add(emb.reshape(1, -1))
            self.id_map.append({"id": asset_id, "path": perm_path,
                                 "lang": lang, "modality": self.modality,
                                 "timestamp": time.ctime()})
            self._save()
            self.metadata_db.add_asset(asset_id, self.modality, lang, perm_path,
                                       os.path.getsize(perm_path), faiss_idx,
                                       tags, description, collection)
            return True, f"✅ Registered: {asset_id}"
        except Exception as e:
            if os.path.exists(perm_path): os.remove(perm_path)
            return False, f"❌ Failed: {e}"

    def batch_register(self, files, language, tags=None, collection=""):
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
                    ok, msg = self.register(tp, asset_id, ext, language, tags, "", collection)
                    (results["success"] if ok else results["failed"]).append(
                        file.name if ok else (file.name, msg))
                finally:
                    if os.path.exists(tp): os.remove(tp)
            except Exception as e:
                results["failed"].append((file.name, str(e)))
            progress.progress((idx + 1) / total)

        progress.empty(); status_text.empty()
        return results

    def search(self, file_path=None, text=None, top_k=5, metadata_filters=None):
        if self.index.ntotal == 0: return [], 0
        t0 = time.time()
        q  = self.get_embedding(file_path=file_path, text=text).reshape(1, -1)
        scores, idxs = self.index.search(q, min(top_k, self.index.ntotal))
        out = []
        for s, i in zip(scores[0], idxs[0]):
            if i == -1: continue
            conf = ("High" if s > CONFIG.CONFIDENCE_HIGH else
                    "Medium" if s > CONFIG.CONFIDENCE_MED else "Low")
            out.append({**self.id_map[i], "score": float(s), "confidence": conf})
        return sorted(out, key=lambda x: x["score"], reverse=True), (time.time()-t0)*1000

    def get_all_vectors(self):
        try:
            return faiss.vector_to_array(self.index).reshape(self.index.ntotal, CONFIG.EMBEDDING_DIM)
        except Exception:
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


# ==================== INIT ==================== #
image_engine = ImageEngine()
audio_engine = AudioEngine()
analytics    = AnalyticsTracker()

# ==================== SIDEBAR ==================== #
with st.sidebar:

    # theme toggle — pill display + real button below it
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
    st.markdown("### 📥 Asset Registration")

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
        if st.button("⬆ Register Asset", use_container_width=True):
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
                        with st.spinner("Checking for duplicates…"):
                            dups = eng.find_duplicates(file_path=tp)
                        if dups:
                            st.warning(f"⚠️ Similar asset found: **{dups[0]['id']}** "
                                       f"({dups[0]['similarity']:.1%} match)")
                        with st.spinner("Registering…"):
                            ok, msg = eng.register(tp, reg_id.strip(), ext, reg_lang, reg_tags)
                        (st.success if ok else st.error)(msg)
                        if ok: st.rerun()
                    except Exception as exc:
                        st.error(f"Registration failed: {exc}")
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
        if st.button("⬆ Register Batch", use_container_width=True):
            if not reg_files: st.warning("Upload at least one file")
            else:
                eng = image_engine if reg_mod == "image" else audio_engine
                with st.spinner(f"Processing {len(reg_files)} files…"):
                    res = eng.batch_register(reg_files, reg_lang, reg_tags_b, reg_collect)
                st.success(f"✅ Added: {len(res['success'])}")
                if res['skipped']:  st.info(f"⏭ Skipped: {len(res['skipped'])}")
                if res['duplicates']:
                    st.warning(f"🔄 Duplicates: {len(res['duplicates'])}")
                    with st.expander("View duplicates"):
                        for fn, dup in res['duplicates']: st.write(f"- {fn} → {dup}")
                if res['failed']:
                    st.error(f"❌ Failed: {len(res['failed'])}")
                    with st.expander("View errors"):
                        for fn, err in res['failed']: st.write(f"- {fn}: {err}")
                if res['success']:
                    time.sleep(0.8); st.rerun()

    st.markdown(f'<hr style="border-color:{T["border"]};margin:1.5rem 0 1rem">', unsafe_allow_html=True)
    st.markdown("**Registry Status**")
    st.markdown(f"""
    <div class="sidebar-stat">🖼 Images <span>{image_engine.index.ntotal}</span></div>
    <div class="sidebar-stat">🔊 Audio <span>{audio_engine.index.ntotal}</span></div>
    <div class="sidebar-stat">💻 Device <span>{DEVICE.upper()}</span></div>
    <div class="sidebar-stat">📐 Index <span>FAISS FlatIP</span></div>
    """, unsafe_allow_html=True)

    stats = analytics.get_stats(7)
    if stats and stats[0] and stats[0] > 0:
        st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0">', unsafe_allow_html=True)
        st.markdown("**Usage Stats (7 days)**")
        st.markdown(f"""
        <div class="sidebar-stat">🔍 Searches <span>{stats[0]}</span></div>
        <div class="sidebar-stat">📊 Avg Results <span>{stats[1]:.1f}</span></div>
        <div class="sidebar-stat">⚡ Avg Speed <span>{stats[2]:.0f}ms</span></div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0">', unsafe_allow_html=True)
    if st.button("💾 Export Registry", use_container_width=True):
        with st.spinner("Creating backup…"):
            ip = image_engine.export_registry()
            ap = audio_engine.export_registry()
        st.success(f"✅ Exported!\n- {ip}\n- {ap}")

# ==================== HEADER ==================== #
st.markdown('<div class="a1"><h1>👁️ TRINETRA V4.1</h1></div>', unsafe_allow_html=True)
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
                f'<div style="font-size:.7rem;color:{T["text_muted"]}">Score: {r["score"]:.3f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if modality == "image":
                st.image(r["path"], caption=r["id"], use_container_width=True)
            else:
                st.write(f"**{r['id']}**")
                st.audio(r["path"])
            st.caption(f"Lang: {r.get('lang','—')} · {r.get('timestamp','')[:10]}")
    st.session_state.last_search_results = results

# ==================== TABS ==================== #
tab_v, tab_a, tab_aud, tab_hist = st.tabs([
    "🖼  Visual Search", "🔊  Acoustic Search", "📊  Neural Auditor", "📜  Search History"
])

# ── Visual ── #
with tab_v:
    mode = st.radio("Input Mode", ["Text Query", "Image Match"], horizontal=True)
    with st.expander("🔍 Advanced Filters"):
        f_lang = st.selectbox("Language", ["All"] + CONFIG.SUPPORTED_LANGUAGES, key="fl_img")
        f_tags = st.multiselect("Tags", image_engine.metadata_db.get_all_tags(), key="ft_img")
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    if mode == "Text Query":
        q = st.text_input("Describe the image", placeholder="e.g., a crowded temple at dusk")
        if st.button("⟳ Run Visual Scan", key="vs_txt") and q:
            mf = {}
            if f_lang != "All": mf['language'] = f_lang
            if f_tags:          mf['tags']     = f_tags[0]
            with st.spinner("Scanning…"):
                results, ms = image_engine.search(text=translate_to_english(q), top_k=6,
                                                   metadata_filters=mf or None)
            analytics.log_search(q, "image", len(results), ms)
            st.session_state.search_history.append(
                {"query": q, "modality": "image", "timestamp": time.time(), "results_count": len(results)})
            st.caption(f"⚡ {ms:.0f} ms")
            display_results(results, "image")
    else:
        qi = st.file_uploader("Query image",
                               type=[e.lstrip('.') for e in CONFIG.ALLOWED_IMAGE_EXTS], key="qimg")
        if st.button("⟳ Run Visual Scan", key="vs_img") and qi:
            ext = os.path.splitext(qi.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(qi.getbuffer()); qp = tmp.name
            try:
                with st.spinner("Scanning…"):
                    results, ms = image_engine.search(file_path=qp, top_k=6)
                analytics.log_search("[Image Upload]", "image", len(results), ms)
                st.caption(f"⚡ {ms:.0f} ms")
                display_results(results, "image")
            finally:
                if os.path.exists(qp): os.remove(qp)

# ── Acoustic ── #
with tab_a:
    with st.expander("🔍 Advanced Filters"):
        f_lang_a = st.selectbox("Language", ["All"] + CONFIG.SUPPORTED_LANGUAGES, key="fl_aud")
    q = st.text_input("Describe the sound", placeholder="e.g., tabla solo during rainstorm")
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    if st.button("⟳ Run Acoustic Scan") and q:
        mf = {}
        if f_lang_a != "All": mf['language'] = f_lang_a
        with st.spinner("Scanning…"):
            results, ms = audio_engine.search(text=translate_to_english(q), top_k=6,
                                               metadata_filters=mf or None)
        analytics.log_search(q, "audio", len(results), ms)
        st.session_state.search_history.append(
            {"query": q, "modality": "audio", "timestamp": time.time(), "results_count": len(results)})
        st.caption(f"⚡ {ms:.0f} ms")
        display_results(results, "audio")

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
        st.dataframe(pd.DataFrame(eng.id_map)[["id","lang","modality","timestamp"]],
                     use_container_width=True, hide_index=True)
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
            fig = (px.scatter_3d if nc == 3 else px.scatter)(
                df_p, **({} if nc < 3 else {}),
                x="x", y="y", **({"z": "z"} if nc == 3 else {}),
                text="ID", color_discrete_sequence=[T["accent"]],
                title=f"{pick} · {'3-D' if nc==3 else '2-D'} Embedding Map",
            )
            if nc < 3:
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
            with st.expander("Variance Breakdown"):
                st.dataframe(pd.DataFrame({
                    "Component":  [f"PC{i+1}" for i in range(nc)],
                    "Variance %": (pca.explained_variance_ratio_ * 100).round(2),
                }), hide_index=True, use_container_width=True)

# ── Search History ── #
with tab_hist:
    st.markdown("### 📜 Recent Searches")
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
            st.session_state.search_history = []; st.rerun()

# ── Comparison View ── #
if len(st.session_state.last_search_results) >= 2:
    with st.expander("🔀 Compare Results"):
        st.markdown("### Side-by-Side Comparison")
        ids = [r["id"] for r in st.session_state.last_search_results]
        ca, cb = st.columns(2)
        s1 = ca.selectbox("Result 1", ids, key="cmp1")
        s2 = cb.selectbox("Result 2", ids, key="cmp2")
        r1 = next(r for r in st.session_state.last_search_results if r["id"] == s1)
        r2 = next(r for r in st.session_state.last_search_results if r["id"] == s2)
        ca, cb = st.columns(2)
        with ca:
            (st.image if r1.get("modality") == "image" else st.audio)(r1["path"],
                **({"use_container_width": True} if r1.get("modality") == "image" else {}))
            st.metric("Score", f"{r1['score']:.2%}")
            st.write(f"**ID:** {r1['id']}  **Lang:** {r1['lang']}")
        with cb:
            (st.image if r2.get("modality") == "image" else st.audio)(r2["path"],
                **({"use_container_width": True} if r2.get("modality") == "image" else {}))
            st.metric("Score", f"{r2['score']:.2%}")
            st.write(f"**ID:** {r2['id']}  **Lang:** {r2['lang']}")