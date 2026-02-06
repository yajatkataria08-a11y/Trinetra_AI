# 👁️ TRINETRA: Multimodal IP Defense System

**Trinetra** is a forensic intellectual property (IP) protection engine designed to shield digital assets through a dual-layered defense strategy: **DCT-based Forensic Watermarking** and **AI-driven Multimodal Fingerprinting**. 

The system enables creators to register original images and videos into a secured vector database and scan suspicious content to detect unauthorized use, even after modifications like cropping, brightness shifts, or compression.



---

## 🚀 Key Features

### 1. Forensic DCT Watermarking
* **Invisible Protection**: Injects a forensic signal into the frequency domain (Y-channel of YUV space) using Discrete Cosine Transform.
* **Tamper Resistance**: Designed to survive common image manipulations while remaining invisible to the human eye.

### 2. AI Fingerprinting (CLIP)
* **Neural Fingerprinting**: Utilizes OpenAI’s CLIP (ViT-B/32) to generate high-dimensional embeddings for images and video keyframes.
* **Semantic Matching**: Detects infringements based on visual content rather than just pixel-matching, making it resistant to significant edits.

### 3. Video Forensic Scanning
* **Adaptive Keyframing**: Automatically detects scene changes and extracts unique keyframes for registration to save memory.
* **Temporal Evidence Graph**: Generates a similarity timeline during scans to show exactly where and when an infringement occurs in a video file.



### 4. High-Speed Vector Retrieval
* **FAISS Powered**: Uses Facebook AI Similarity Search (FAISS) for near-instant retrieval of matching assets from the registry.
* **Local Privacy**: All vector indices and labels are stored locally in the `trinetra_registry` folder.

---

## 🛠️ Technical Architecture

* **Frontend**: Gradio (Web Dashboard)
* **Vector Engine**: FAISS (IndexFlatIP)
* **Vision Model**: CLIP (ViT-B/32)
* **Processing**: OpenCV, PyTorch, NumPy

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yajatkataria08-a11y/Trinetra_AI.git
   cd Trinetra_AI