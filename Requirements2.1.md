#  Trinetra — Runtime & Device Requirements

This document describes the **hardware, software, and environment requirements**
needed to run the Trinetra application locally or on a server.

---

## 1. Supported Operating Systems

Trinetra has been tested on:

- Linux (Ubuntu 20.04+ recommended)
- macOS (Intel and Apple Silicon)
- Windows 10 / 11 (WSL2 recommended for best performance)

---

## 2. Hardware Requirements

### Minimum (CPU-Only)

| Component | Requirement |
|--------|------------|
| CPU | 4-core 64-bit processor |
| RAM | 8 GB |
| Storage | 10 GB free disk space |
| GPU | Not required |

- Suitable for small demos and testing
- Slower embedding generation

---

### Recommended (GPU-Accelerated)

| Component | Requirement |
|--------|------------|
| CPU | 8-core 64-bit processor |
| RAM | 16 GB or more |
| Storage | 20 GB free disk space |
| GPU | NVIDIA GPU with CUDA support |
| VRAM | 8 GB or more |

- Significantly faster embeddings
- Recommended for hackathon demos

---

### GPU Compatibility

- NVIDIA GPUs with CUDA Compute Capability ≥ 7.0 recommended
- CUDA 11.7 or later
- cuDNN compatible with installed CUDA version

> Note: AMD GPUs are not currently supported for acceleration.

---

## 3. Software Requirements

### Programming Language

- Python **3.9 – 3.11**

---

### Python Libraries

Core dependencies:

- torch
- transformers
- faiss
- librosa
- numpy
- pillow
- streamlit
- deep-translator

All dependencies are listed in `requirements.txt`.

---

### System Packages

Required system-level packages:

- `ffmpeg` (audio decoding)
- `libsndfile` (audio file support)

Linux (Ubuntu):
```bash
sudo apt install ffmpeg libsndfile1
