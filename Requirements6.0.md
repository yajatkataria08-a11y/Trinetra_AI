# Trinetra V6.0 — Requirements & Dependency Guide

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python **3.10 or higher** required.

---

## Dependency Breakdown

### Core UI
| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.35.0 | Web app framework, session state, caching, file uploads |

### Neural Models
| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥2.1.0 | Model inference backend; auto-detects CUDA vs CPU |
| `transformers` | ≥4.40.0 | CLIP (`openai/clip-vit-base-patch32`) + CLAP (`laion/clap-htsat-fused`) |

> **GPU note:** If you have a CUDA GPU, replace `faiss-cpu` with `faiss-gpu` and install `torch` with the appropriate CUDA wheel from pytorch.org. The app auto-detects via `torch.cuda.is_available()`.

### Vector Search
| Package | Version | Purpose |
|---|---|---|
| `faiss-cpu` | ≥1.8.0 | HNSW flat index for 512-dim embeddings; swap to `faiss-gpu` for CUDA |

### Image & Audio Processing
| Package | Version | Purpose |
|---|---|---|
| `Pillow` | ≥10.0.0 | Image open/verify/convert in `utils.py` and `engines.py` |
| `librosa` | ≥0.10.1 | Audio loading, RMS energy, quality analysis |
| `soundfile` | ≥0.12.1 | Non-WAV format backend for librosa (MP3, FLAC, OGG, M4A) |

### Web Search
| Package | Version | Purpose |
|---|---|---|
| `requests` | ≥2.31.0 | DuckDuckGo scraper, Sarvam API, Lambda calls, page fetch |
| `beautifulsoup4` | ≥4.12.0 | HTML parsing for DuckDuckGo results and page text extraction |

### Translation (Indic Language Support)
| Package | Version | Purpose |
|---|---|---|
| `deep-translator` | ≥1.11.4 | `GoogleTranslator` fallback when Sarvam API is unavailable |

> Primary translation is via **Sarvam AI** (configured in `secrets.toml`). `deep-translator` is only the fallback.

### Auth & Security
| Package | Version | Purpose |
|---|---|---|
| `werkzeug` | ≥3.0.1 | `generate_password_hash` / `check_password_hash` with PBKDF2-SHA256 |

### Data & Visualisation
| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.26.0 | Embedding math, normalisation, audio RMS |
| `pandas` | ≥2.1.0 | Asset dataframe display in Neural Auditor tab |
| `plotly` | ≥5.20.0 | PCA scatter plots (2D/3D) and cluster visualisation |
| `scikit-learn` | ≥1.4.0 | `KMeans` and `PCA` in the Clusters tab |

### AWS
| Package | Version | Purpose |
|---|---|---|
| `boto3` | ≥1.34.0 | Rekognition client for reverse image search |
| `botocore` | ≥1.34.0 | AWS core — pulled in automatically by boto3 |

---

## Stdlib (No Install Needed)

`sqlite3`, `hashlib`, `threading`, `smtplib`, `json`, `os`, `re`, `tempfile`, `zipfile`, `zlib`, `base64`, `secrets`, `logging`, `time` — all Python standard library.

---

## Secrets Configuration

All credentials go in `.streamlit/secrets.toml` (never commit this file):

```toml
SARVAM_API_KEY        = "..."
SMTP_EMAIL            = "you@gmail.com"
SMTP_PASSWORD         = "xxxx xxxx xxxx xxxx"   # Gmail App Password
AWS_ACCESS_KEY_ID     = "AKIAxxxxx"
AWS_SECRET_ACCESS_KEY = "xxxxxxxx"
AWS_REGION            = "us-east-1"
LAMBDA_SEARCH_URL     = ""   # optional
LAMBDA_INGEST_URL     = ""   # optional
```

---

## Optional / Future

| Package | Purpose |
|---|---|
| `faiss-gpu` | Drop-in replacement for `faiss-cpu` on CUDA machines |
| `unitary/multilingual-toxic-xlm-roberta` | Loaded via `transformers.pipeline` in `models.py` — no extra pip install |
