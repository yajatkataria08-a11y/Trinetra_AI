# Trinetra – System & Dependency Requirements

  **Trinetra v1.6.2** is a multimodal IP defense system for image and video similarity detection using CLIP + FAISS.

This document defines the **runtime, hardware, and software requirements** needed to run Trinetra reliably.

---

## 1. Supported Platforms

- **Operating Systems**
  - Linux (Ubuntu 20.04+ recommended)
  - macOS 12+ (Apple Silicon supported, CPU mode only)
  - Windows 10/11 (CUDA support requires WSL or native CUDA setup)

- **Python**
  - Python **3.9 – 3.11**
  - Python 3.12 is **not recommended** (FAISS / PyTorch compatibility issues)

---

## 2. Hardware Requirements

### Minimum (CPU-only)
- CPU: 4 cores
- RAM: 8 GB
- Storage: 5 GB free
- GPU: ❌ Not required

### Recommended (GPU-accelerated)
- GPU: NVIDIA CUDA-capable (≥ 8 GB VRAM)
- CUDA: 11.8 or 12.x
- RAM: 16 GB+
- Storage: SSD strongly recommended

> Trinetra automatically switches between CPU and GPU at runtime.

---

## 3. Core Python Dependencies

### Required Packages

| Package | Version | Purpose |
|------|------|------|
| torch | ≥ 2.0 | CLIP inference |
| transformers | ≥ 4.36 | CLIP model + processor |
| faiss-cpu / faiss-gpu | ≥ 1.7 | Vector similarity search |
| numpy | ≥ 1.23 | Numerical operations |
| opencv-python | ≥ 4.8 | Image & video processing |
| pillow | ≥ 9.5 | Image I/O |
| gradio | ≥ 4.0 | Web UI |
| matplotlib | ≥ 3.7 | Video similarity plots |

### Install (CPU-only)
```bash
pip install torch transformers faiss-cpu numpy opencv-python pillow gradio matplotlib

