# Trinetra V6.0 — System Design Document

> **Multimodal Neural Search Registry · AI for Bharat**
> Team Human

---

## 1. Overview

Trinetra is a multimodal asset registry and search engine built for India's multilingual, multi-format digital content ecosystem. It allows users to register image and audio assets, then search them semantically in any of 10 Indian languages using neural embeddings — no keywords required.

**Core problem it solves:** Most search systems are text-keyword-based and English-centric. Trinetra lets you type a query in Hindi, Tamil, or Bengali and find visually or acoustically similar content through meaning, not metadata.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        app.py (Streamlit UI)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Visual    │  │Acoustic  │  │Web Search│  │AWS Revrse│   │
│  │Search    │  │Search    │  │(DDG)     │  │(Rekogntn)│   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼──────────────┼─────────┘
        │             │             │              │
   ┌────▼────┐   ┌────▼────┐        │         ┌───▼──────────┐
   │ImageEng │   │AudioEng │        │         │AWSReverseSrch│
   │(CLIP)   │   │(CLAP)   │        │         │(Rekognition) │
   └────┬────┘   └────┬────┘        │         └──────────────┘
        │             │             │
   ┌────▼─────────────▼────┐   ┌────▼────────────┐
   │  FAISS HNSW Index     │   │ WebSearchEngine  │
   │  (512-dim embeddings) │   │ (DuckDuckGo)     │
   └────┬──────────────────┘   └─────────────────┘
        │
   ┌────▼──────────────┐   ┌──────────────────────────────┐
   │  MetadataDB       │   │  LambdaSearchClient           │
   │  (SQLite)         │   │  TrinetraIngestClient         │
   │  assets/comments/ │   │  (AWS Lambda + DynamoDB)      │
   │  ratings/tags     │   └──────────────────────────────┘
   └───────────────────┘
        │
   ┌────▼──────────────┐   ┌──────────────────────────────┐
   │  AuthManager      │   │  AnalyticsTracker             │
   │  (OTP + PBKDF2)   │   │  (SQLite — searches.db)      │
   └───────────────────┘   └──────────────────────────────┘
```

---

## 3. Module Breakdown

| File | Responsibility |
|---|---|
| `app.py` | Streamlit entry point. Renders all tabs, sidebar, login page. Orchestrates all modules. |
| `config.py` | Single source of truth for all constants: HNSW params, file limits, supported languages, thresholds. |
| `models.py` | `@st.cache_resource` loaders for CLIP and CLAP models. One GPU load per app lifetime. |
| `engines.py` | `TrinetraEngine` base class + `ImageEngine` (CLIP) + `AudioEngine` (CLAP). Registration, search, dedup, export. |
| `search.py` | `WebSearchEngine` (DuckDuckGo), `LambdaSearchClient`, `TrinetraIngestClient`, `AWSReverseSearchEngine`. |
| `database.py` | Thread-safe `DatabaseConnection` (WAL mode). `MetadataDB` with assets, comments, ratings tables. |
| `auth.py` | `EmailOTPSender` (Gmail SMTP) + `AuthManagerWithOTP` (PBKDF2 passwords, OTP flow, password reset). |
| `analytics.py` | `AnalyticsTracker` — logs every search to SQLite, exposes stats and autocomplete suggestions. |
| `utils.py` | Validation, image/audio quality analysis, Sarvam + Google translation, `smart_query_preprocess`, MD5 hashing, fp16 helper. |
| `ui_components.py` | Theme tokens (DARK/LIGHT), full CSS injection, `display_results`, `render_enrichment_ui`, `render_aws_reverse_search_tab`. |

---

## 4. Embedding Pipeline

### 4.1 Image Search (CLIP)
```
Query text / Image file
        ↓
smart_query_preprocess()     ← Sarvam AI translates Indic → English
        ↓
CLIPProcessor → CLIPModel.get_text_features() / get_image_features()
        ↓
L2-normalize → float32 numpy (512-dim)
        ↓
FAISS HNSW index.search(query_vec, top_k*2)
        ↓
Score filtering + quality reranking
        ↓
Results (id, path, score, confidence, lang)
```

### 4.2 Audio Search (CLAP)
```
Query text / Audio file
        ↓
smart_query_preprocess()     ← same Sarvam path
        ↓
librosa.load() → waveform     (audio path only)
ClapProcessor → ClapModel.get_text_features() / get_audio_features()
        ↓
L2-normalize → float32 numpy (512-dim)
        ↓
FAISS HNSW index.search() → rerank → results
```

### 4.3 Inference Safety
A single `threading.Lock` (`_inference_lock`) serialises all model forward passes. This prevents VRAM OOM crashes when multiple Streamlit sessions hit the server simultaneously on a single-GPU machine.

---

## 5. FAISS Index Design

| Parameter | Value | Rationale |
|---|---|---|
| Index type | `IndexHNSWFlat` | Approximate nearest-neighbour; no training needed; good recall at small scale |
| Embedding dim | 512 | CLIP/CLAP output dimension |
| HNSW M | 32 | Neighbours per node; good recall-speed tradeoff |
| efConstruction | 200 | Build-time graph quality |
| efSearch | 50 | Query-time beam width |

**Persistence:** Index is saved to `trinetra_registry/<modality>/index` (FAISS binary) + `id_map.json` (ordered list of asset metadata). Both are written atomically on every registration.

**Self-healing:** On startup, `_validate_sync()` checks `index.ntotal == len(id_list)`. If they diverge (e.g. crash mid-write), the index is rebuilt from scratch by re-embedding every asset file.

---

## 6. Indic Language Support

Trinetra supports 10 languages: `en, hi, ta, te, kn, ml, bn, mr, gu, pa`.

**Translation pipeline in `smart_query_preprocess()`:**
1. If query is pure ASCII → skip translation entirely (zero latency).
2. If query contains non-ASCII characters → call **Sarvam AI** (`/translate` endpoint, `source_language_code: auto`).
3. If Sarvam fails or key is missing → fall back to `deep-translator` (GoogleTranslator).
4. Translated query is stored in `st.session_state["last_translated"]` so the UI can show "🇮🇳 Sarvam magic: _orig_ → **translated**".

Limitation to disclose: CLIP/CLAP were trained on predominantly English/Western data. Indic audio content (folk instruments, regional speech) may embed with lower fidelity than English content. The Sarvam translation layer mitigates this for text queries but not for raw audio embeddings.

---

## 7. Authentication System

```
Register: email + username + password
    → OTP generated (secrets.choice, 6 digits)
    → OTP hashed (SHA-256) stored in registration_requests
    → OTP emailed (Gmail SMTP / shown on-screen as fallback)
    → User submits OTP → verified → moved to users table

Login: username + password
    → PBKDF2-SHA256 verify (werkzeug)
    → Legacy SHA-256 hashes auto-upgraded on first login
    → Session state: {username, role}

Roles: viewer (search only) | uploader (+ register assets) | admin (+ user mgmt)

Password Reset: email → OTP → new password (same OTP flow)
OTP Cooldown: 60 seconds between requests (CONFIG.OTP_COOLDOWN_SECS)
OTP Expiry: 10 minutes
```

**Known issue:** Default admin credentials (`admin / admin123`) are hardcoded in `auth.py`. Change immediately after first deploy.

---

## 8. Storage Layout

```
trinetra_registry/
├── metadata.db              ← MetadataDB (assets, comments, ratings)
├── users.db                 ← AuthManagerWithOTP
├── analytics.db             ← AnalyticsTracker
├── image/
│   ├── index                ← FAISS binary
│   └── id_map.json          ← ordered asset list
├── audio/
│   ├── index
│   └── id_map.json
└── storage/
    ├── <asset_id>.jpg       ← permanent copies of all registered files
    ├── <asset_id>.png
    └── <asset_id>.wav
logs/
└── trinetra_YYYYMMDD.log
```

---

## 9. AWS Integration

### Rekognition (Reverse Image Search)
- `detect_labels` → top 20 labels (≥70% confidence)
- `detect_text` → OCR lines (≥80% confidence)
- `detect_faces` → face count
- Labels + detected text are used to construct DuckDuckGo search queries, returning web results enriched with image understanding.
- Cost estimate: ~$0.001 per image (Rekognition standard pricing).

### Lambda + DynamoDB (Hybrid Cloud Search)
- `LambdaSearchClient` — keyword search on labels/transcripts stored in DynamoDB via a Lambda Function URL.
- `TrinetraIngestClient` — sends text assets to Lambda for ingestion; includes SHA-256 content-hash dedup check before write.
- Compressed content (`zlib+b64` encoding) is transparently decoded by `decode_dynamo_content()`.
- Both are optional — the app degrades gracefully to local FAISS search when Lambda URLs are not configured.

---

## 10. UI Design System

**Theme:** Dual dark/light with live toggle. Dark theme uses `#080c14` base with amber accent `#e8a020`. Light theme uses `#f0f2f8` base with darker amber `#b87818`.

**Fonts:**
- Headlines: `Syne` (900 weight) with gradient text fill
- Body: `DM Sans` (300–500)
- Code/labels: `JetBrains Mono`

**Layout:** Wide Streamlit layout. Grid background overlay (dark only). Empty container fix prevents phantom whitespace from Streamlit's empty div emission.

**Tabs:** Visual Search · Acoustic Search · Web Search · AWS Reverse · Clusters · Neural Auditor · Search History · Admin

---

## 11. Known Limitations & Future Work

| Issue | Status | Fix |
|---|---|---|
| Default `admin/admin123` credentials | ⚠️ | Force password change on first login |
| `batch_register()` calls `st.progress()` inside engine layer | ⚠️ | Decouple: pass a progress callback instead |
| Rekognition cost guard | ⚠️ | Add per-session call counter with configurable cap |
| CLIP/CLAP bias toward English/Western content | Known | Fine-tune on Indic datasets (future) |
| No rate limiting on search endpoints | Planned | Add per-user request throttling |
| `estimate_costs()` hardcodes $100 remaining credit | ⚠️ | Pull from AWS Budgets API or make configurable |



