# 🧠 Trinetra — System Design Document

## 1. Overview

**Trinetra** is a multimodal asset registry and verification system that enables
cross-modal search and similarity-based identification of images and audio using
neural embeddings.

It is designed for **AI for Bharat–scale digital infrastructure**, where assets
may be queried using:
- Regional language text
- Reference images
- Reference audio samples

Trinetra converts heterogeneous inputs into a **shared semantic vector space**,
allowing reliable similarity search across modalities.

---

## 2. Design Goals

### Primary Goals
- 🔍 Multimodal search (Image ↔ Text, Audio ↔ Text)
- 🌐 Cross-lingual support for Indian languages
- ⚡ Low-latency similarity search
- 📦 Persistent and auditable asset registry
- 🧪 Interpretable similarity scoring

### Non-Goals
- Biometric identification (face or speaker recognition)
- Real-time streaming inference
- Model training or fine-tuning

---

## 3. High-Level Architecture

┌──────────────┐
│  User Input  │
│──────────────│
│ Text / Image │
│ Audio        │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Preprocessing Layer  │
│──────────────────────│
│ • Language Translation│
│ • Image Loading       │
│ • Audio Resampling    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Embedding Models     │
│──────────────────────│
│ CLIP  (Image/Text)   │
│ CLAP  (Audio/Text)   │
└──────┬───────────────┘
       │ 512-D vectors
       ▼
┌──────────────────────┐
│ Vector Normalization │
│ (Unit L2 Norm)       │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ FAISS Vector Index   │
│──────────────────────│
│ IndexFlatIP          │
│ (Cosine Similarity)  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Ranked Results       │
│ Similarity Scores    │
└──────────────────────┘

---

## 4. Core Components

### 4.1 Frontend (Streamlit)

- Asset registration and management
- Visual and acoustic search workflows
- Result visualization with similarity indicators
- Optimized for demo clarity and usability

---

### 4.2 Embedding Layer

#### Image & Text — CLIP
- Model: `openai/clip-vit-base-patch32`
- Embedding dimension: 512
- Supports:
  - Image → Image
  - Text → Image
  - Cross-lingual queries via translation

#### Audio & Text — CLAP
- Model: `laion/clap-htsat-fused`
- Embedding dimension: 512
- Supports:
  - Audio → Audio
  - Text → Audio
  - Environmental and semantic sound descriptions

---

### 4.3 Translation Layer

- Regional language → English translation
- Implemented using `deep-translator`
- Cached to reduce latency and external calls
- Graceful fallback to original text on failure

---

### 4.4 Vector Index (FAISS)

- Index type: `IndexFlatIP`
- Similarity metric: cosine similarity
- Implementation details:
  - All embeddings are L2-normalized
  - Inner product equals cosine similarity
- Persistent storage:
  - `registry.index` → FAISS index
  - `id_map.pkl` → Asset metadata mapping

---

### 4.5 Storage Layout

trinetra_registry/
├── image/
│   ├── registry.index
│   └── id_map.pkl
├── audio/
│   ├── registry.index
│   └── id_map.pkl
└── storage/
    ├── asset_001.jpg
    ├── asset_002.wav

- Asset IDs are sanitized to prevent path traversal
- Duplicate asset IDs are rejected

---

## 5. Data Flow

### Asset Registration
1. User uploads image or audio
2. File size and format validation
3. Embedding generation using CLIP or CLAP
4. Vector normalization
5. Insertion into FAISS index
6. Metadata persistence

### Search
1. User submits text or reference file
2. Text translated if required
3. Query embedding generated
4. FAISS similarity search executed
5. Top-K ranked results returned

---

## 6. Similarity Scoring

- Score range: −1.0 to 1.0
- Displayed as percentage for clarity
- Interpretation:
  - > 0.85 → Strong semantic match
  - 0.65–0.85 → Related content
  - < 0.65 → Weak similarity

Trinetra is designed as a **decision-support system**, not an automated
adjudication or identity verification tool.

---

## 7. Security & Validation

- File size limits enforced
- Whitelisted file extensions
- Asset ID sanitization
- Temporary file cleanup
- No direct use of user-supplied paths

---

## 8. Performance Characteristics

| Component | Notes |
|--------|------|
| Embedding | GPU-accelerated when available |
| Search | Exact similarity search |
| Latency | Sub-second for small registries |
| Scaling | Upgradeable to IVF / HNSW |

---

## 9. Alignment with AI for Bharat

Trinetra addresses key challenges in the Indian digital ecosystem:

- **Language diversity**: Supports regional language queries
- **Modality diversity**: Works with images and audio common in low-literacy contexts
- **Open AI stack**: Uses inspectable, non-proprietary models
- **Offline-first potential**: Can run on local or government infrastructure
- **Public-good focus**: Designed for trust, transparency, and auditability

---

## 10. Example Use Case (India-Specific)

### Public Media Verification

A district administration registers verified images and audio from official
events. Field officers can later query the system using regional language
descriptions to verify whether newly circulating media matches official records,
helping counter misinformation.

---

## 11. Scalability Considerations

Future improvements:
- Approximate nearest neighbor indexes
- Batch ingestion pipelines
- Sharded registries
- Verification thresholds
- Audit logs and access controls

---

## 12. Known Limitations

- No model fine-tuning
- Single-node deployment
- Exact search only
- CLAP performance varies for very short audio clips

---

## 13. Technology Stack

- Frontend: Streamlit
- Models: CLIP, CLAP
- Vector Index: FAISS
- ML Framework: PyTorch
- Audio Processing: Librosa
- Translation: deep-translator
- Language: Python 3

---

## 14. Why Trinetra for Bharat 🇮🇳

- Language-agnostic access
- Modality-agnostic verification
- Transparent AI pipeline
- Designed for public digital infrastructure
- Suitable for registries, archives, and governance workflows

---

## 15. Future Roadmap

- Video modality support
- On-device inference
- Confidence calibration
- API-first deployment
- Integration with digital signature systems
