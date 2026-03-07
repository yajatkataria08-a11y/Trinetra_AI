# Trinetra V5.0 — Problem Statement

**Team:** Team Human
**Members:** Yajat Kataria · Aklesh Swain
**Track:** AI for Bharat
**Version:** 5.0 (March 2026)

---

## The Problem

India produces one of the largest volumes of digital media in the world — festival photographs, regional music recordings, oral histories, archival footage, educational content across 22 scheduled languages, documentary audio from every corner of the country. The infrastructure to *find* this content, however, is almost entirely built around English keywords and Western-trained retrieval systems.

This creates three compounding failures:

**1. Language exclusion.** A journalist in Tamil Nadu searching for archival audio of a local folk performance cannot query in Tamil and expect meaningful results from any existing system. A researcher cataloguing photographs from rural Bengal cannot type a Hindi description and retrieve visually similar images. The assumption baked into every mainstream search tool is that the user writes in English — and that assumption excludes the majority of India.

**2. Keyword dependency.** Existing digital asset management systems — from enterprise DAMs to basic file explorers — rely entirely on filenames, tags, and manually written metadata. If an asset wasn't labeled correctly at upload time, it is effectively unfindable. India's cultural and documentary archives are full of assets that were never labeled at all.

**3. Modality blindness.** Sound and image carry information that words cannot fully capture. A raga, a specific temple's architecture, the ambience of a monsoon market — these are not easily described with tags. Searching for audio by describing how it *sounds*, or finding an image by describing what it *means*, requires a fundamentally different approach.

The result: India's richest digital content sits in folders, hard drives, and cold storage — unsearchable, undiscoverable, effectively dark.

---

## What Trinetra Does

Trinetra V5.0 is a **multimodal neural search registry** — a system where image and audio assets are registered once, encoded into semantic vectors, and then retrieved forever using natural language queries or example files, in any of 10 Indian languages.

### Semantic search, not keyword search

When a user types "भीड़ भरे बाज़ार में दीये जलते हुए" (lit diyas in a crowded market), Trinetra:

1. Detects that the query is in Hindi (non-ASCII characters present)
2. Routes it through **Sarvam AI** — an Indian language model — for translation to English
3. Encodes the translated meaning into a 512-dimensional vector using **OpenAI CLIP**
4. Searches the FAISS HNSW index and returns the closest matching images by semantic similarity
5. Shows the user "🇮🇳 Sarvam magic: भीड़ भरे बाज़ार... → lit diyas in a crowded market" so the translation is transparent

No tags. No filenames. No manual labeling. Just meaning.

The same pipeline applies to audio: "tabla solo during a rainstorm" retrieves acoustically similar audio using **LAION CLAP**, a cross-modal audio-language model.

### 10 Indian languages, natively

Supported: Hindi · Tamil · Telugu · Kannada · Malayalam · Bengali · Marathi · Gujarati · Punjabi · English.

The translation layer is intelligent. Pure ASCII queries skip the API call entirely — zero added latency for English users. Non-ASCII queries go to Sarvam AI, with automatic fallback to Google Translate if Sarvam is unavailable. This means the system stays functional even under partial infrastructure failure.

### Reverse image search with AWS Rekognition

Beyond text-to-image search, Trinetra can analyze an uploaded photograph using AWS Rekognition — detecting labels, reading embedded text via OCR, and counting faces — then automatically construct DuckDuckGo search queries from those labels to surface relevant web results. This turns any image into a research starting point.

### Hybrid cloud + local architecture

Trinetra can operate in two modes simultaneously:

- **Local mode:** FAISS HNSW vector index on-device. Works offline. Fast. No API cost per query.
- **Cloud mode:** AWS Lambda + DynamoDB for keyword search on labels and transcripts, with a serverless ingestion pipeline for text assets. Compressed storage (`zlib+b64`) reduces DynamoDB payload sizes for large documents.

Both modes degrade gracefully — if Lambda URLs are not configured, the system falls back to local search without any error visible to the user.

### Production-grade foundations

Trinetra is not a prototype. The codebase (~3,700 lines across 10 modules) includes:

- **Thread-safe SQLite** with WAL journaling for concurrent Streamlit sessions
- **PBKDF2-SHA256 password hashing** with automatic legacy hash upgrade on login
- **OTP-based email authentication** for registration and password reset, with 60-second cooldown and 10-minute expiry
- **Self-healing FAISS index** that rebuilds automatically if the index and ID map fall out of sync after a crash
- **Embedding cache** (LRU up to 1000 entries) to avoid re-encoding the same asset multiple times
- **Duplicate detection** before registration using cosine similarity threshold (0.95)
- **Quality scoring** for both images (pixel area vs. 1080p baseline) and audio (RMS energy)
- **Role-based access control** — viewer, uploader, admin — with a full admin panel for user management

---

## Why This Matters for Bharat

The digital divide in India is not just about connectivity — it is about whether the tools built on top of that connectivity serve India's actual users. A system that only understands English queries, only processes Western-labeled content, and only works with manually tagged metadata is a system that does not serve most of India.

Trinetra is an attempt to build the search layer that India's digital content ecosystem actually needs: one that understands the language you speak, finds assets by what they mean rather than what they're called, and works on the hardware and infrastructure that's available — not just in ideal conditions.

The immediate use case is cultural archiving and media management. The broader vision is a foundation for any Indian institution — newspapers, government archives, educational content producers, music libraries, documentary filmmakers — to make their collections genuinely searchable without requiring English literacy or manual tagging discipline.

---

## Technical Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (Python) |
| Image embeddings | OpenAI CLIP (`clip-vit-base-patch32`) |
| Audio embeddings | LAION CLAP (`clap-htsat-fused`) |
| Vector index | FAISS IndexHNSWFlat (512-dim) |
| Indic translation | Sarvam AI → Google Translate fallback |
| Reverse image search | AWS Rekognition (labels, OCR, faces) |
| Web search | DuckDuckGo HTML scraper |
| Cloud search | AWS Lambda + DynamoDB |
| Auth | PBKDF2-SHA256 + OTP via Gmail SMTP |
| Persistence | SQLite (WAL) · FAISS binary · JSON |
| Inference safety | Global threading lock (single-GPU safe) |

---

## Team

**Yajat Kataria** — Architecture, full-stack implementation, AI pipeline integration, search engine, auth system, UI, Lambda integrations, DynamoDB storage logic.

**Aklesh Swain** — AWS infrastructure pipeline setup and service configuration.

---

*Trinetra V5.0 · March 2026 · Built for Bharat's Digital Future*

