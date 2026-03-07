import os
import re
import hashlib
import logging

import numpy as np
import librosa
from PIL import Image
from deep_translator import GoogleTranslator

from config import CONFIG

logger = logging.getLogger("Trinetra")


# ==================== ASSET ID HELPERS ====================

def sanitize_asset_id(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s.strip())


def validate_asset_id(asset_id: str):
    if not asset_id:
        return False, "Asset ID cannot be empty"
    if len(asset_id) > 100:
        return False, "Asset ID too long (max 100 chars)"
    if not re.match(r"^[a-zA-Z0-9_\-]+$", asset_id):
        return False, "Letters, numbers, _ and - only"
    if asset_id[0] in ("_", "-"):
        return False, "Cannot start with _ or -"
    return True, "OK"


# ==================== FILE VALIDATION ====================

def validate_upload(file, modality: str):
    allowed = (CONFIG.ALLOWED_IMAGE_EXTS if modality == "image"
               else CONFIG.ALLOWED_AUDIO_EXTS)
    ext = os.path.splitext(file.name)[1].lower()
    if file.size > CONFIG.MAX_FILE_SIZE:
        return f"File is {file.size/1024/1024:.1f} MB — exceeds 100 MB limit."
    if ext not in allowed:
        return f"Extension '{ext}' not supported."
    return None


def validate_image_content(file_path: str):
    try:
        img = Image.open(file_path)
        img.verify()
        img = Image.open(file_path)
        if img.width > 10000 or img.height > 10000:
            return False, "Image too large (max 10000×10000)"
        if img.width < 100 or img.height < 100:
            return False, "Image too small (min 100×100)"
        return True, "OK"
    except Exception as e:
        return False, f"Invalid image: {e}"


# ==================== QUALITY ANALYSIS ====================

def analyze_image_quality(file_path: str):
    try:
        img           = Image.open(file_path)
        quality_score = min(img.width * img.height / (1920 * 1080), 1.0)
        return {
            "resolution":   f"{img.width}×{img.height}",
            "quality_score": quality_score,
            "format":       img.format,
            "mode":         img.mode,
            "size_kb":      os.path.getsize(file_path) / 1024,
        }
    except Exception:
        return None


def analyze_audio_quality(file_path: str):
    try:
        y, sr = librosa.load(file_path, sr=None)
        rms   = librosa.feature.rms(y=y)[0]
        return {
            "duration":    f"{len(y)/sr:.1f}s",
            "sample_rate": f"{sr}Hz",
            "rms_energy":  float(np.mean(rms)),
            "size_kb":     os.path.getsize(file_path) / 1024,
        }
    except Exception:
        return None


# ==================== AUDIO CHECK ====================

def is_audio_valid(y) -> bool:
    return float(np.sqrt(np.mean(y ** 2))) > CONFIG.AUDIO_SILENCE_RMS


# ==================== TRANSLATION ====================

def translate_to_english(text: str) -> str:
    """Fallback translation using Google Translate (deep_translator)."""
    try:
        if any(ord(c) > 127 for c in text):
            return GoogleTranslator(source="auto", target="en").translate(text)
        return text
    except Exception:
        return text


def translate_to_english_with_sarvam(text: str) -> str:
    """
    Translate Indic text to English using Sarvam AI.
    Falls back to GoogleTranslator if Sarvam is unavailable.
    Add your Sarvam API key to secrets.toml as SARVAM_API_KEY.
    """
    import requests
    import streamlit as st

    api_key = ""
    try:
        api_key = st.secrets.get("SARVAM_API_KEY", "")
    except Exception:
        pass

    if not api_key:
        logger.warning("SARVAM_API_KEY not set — falling back to GoogleTranslator")
        return translate_to_english(text)

    try:
        resp = requests.post(
            "https://api.sarvam.ai/translate",
            headers={"API-Subscription-Key": api_key, "Content-Type": "application/json"},
            json={
                "input":           text,
                "source_language_code": "auto",
                "target_language_code": "en-IN",
                "speaker_gender":  "Male",
                "mode":            "formal",
            },
            timeout=10,
        )
        resp.raise_for_status()
        translated = resp.json().get("translated_text", text)
        logger.info(f"SARVAM_TRANSLATE: '{text[:40]}' → '{translated[:40]}'")
        return translated
    except Exception as e:
        logger.warning(f"Sarvam translation failed: {e} — falling back to GoogleTranslator")
        return translate_to_english(text)


# ==================== SMART QUERY PREPROCESSOR ====================

def smart_query_preprocess(q: str, use_sarvam: bool = True) -> str:
    """
    Sarvam translates Indic → English before embedding/search.
    - If already ASCII (English), skips translation entirely — no latency.
    - Stores (original, translated) in st.session_state['last_translated']
      so the UI can show "🇮🇳 Sarvam magic: <orig> → <trans>" after results.
    - If use_sarvam=False or translation fails, returns q unchanged.
    """
    import streamlit as st

    if not q.strip():
        return q

    # Already pure ASCII → definitely English, skip API call
    if all(ord(c) < 128 for c in q):
        return q

    if not use_sarvam:
        return q

    try:
        translated = translate_to_english_with_sarvam(q)
        st.session_state["last_translated"] = (q, translated)
        return translated
    except Exception:
        return q


# ==================== HASHING ====================

def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ==================== TENSOR HELPERS ====================

def fp16(inp: dict) -> dict:
    return {
        k: v.to(__import__("torch").float16) if __import__("torch").is_floating_point(v) else v
        for k, v in inp.items()
    }
