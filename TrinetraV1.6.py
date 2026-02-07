import os
import cv2
import faiss
import pickle
import torch
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
from transformers import (
    CLIPModel,
    CLIPImageProcessor
)
import time
import logging

# ─────────────────── CONFIG ───────────────────
logging.basicConfig(level=logging.INFO)
BASE_PATH = "trinetra_registry"
EXPORT_PATH = "watermarked_exports"

# ─────────────────── CORE ENGINE ───────────────────
class Trinetra:
    def __init__(self, base_path=BASE_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nTRINETRA STARTING ON {self.device.upper()}")

        os.makedirs(base_path, exist_ok=True)
        os.makedirs(EXPORT_PATH, exist_ok=True)

        self.base_path = base_path

        # ── CLIP (IMAGE / VIDEO) ──
        print("Loading CLIP weights...")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.clip_model.eval()

        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_fast=False
        )

        # ── CLAP (AUDIO) - DISABLED FOR SPEED ──
        # Audio logic remains commented to prevent 0% hang on CPU load
        
        self.modalities = {
            "Image": self.clip_model.config.projection_dim,
            "Video": self.clip_model.config.projection_dim
        }

        self.indices = {}
        self.labels = {}

        for mod, dim in self.modalities.items():
            idx = os.path.join(base_path, f"{mod}.index")
            lbl = os.path.join(base_path, f"{mod}.pkl")

            if os.path.exists(idx) and os.path.exists(lbl):
                self.indices[mod] = faiss.read_index(idx)
                with open(lbl, "rb") as f:
                    self.labels[mod] = pickle.load(f)
            else:
                self.indices[mod] = faiss.IndexFlatIP(dim)
                self.labels[mod] = []

        print("TRINETRA ONLINE (IMAGE/VIDEO MODE).")

    def _normalize(self, x):
        return (x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)).astype(np.float32)

    def _save(self, mod):
        faiss.write_index(self.indices[mod], os.path.join(self.base_path, f"{mod}.index"))
        with open(os.path.join(self.base_path, f"{mod}.pkl"), "wb") as f:
            pickle.dump(self.labels[mod], f)

    def _apply_watermark(self, img_np, strength=0.12):
        yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(yuv)
        dct = cv2.dct(np.float32(y))
        dct[8:12, 8:12] += strength
        y_idct = cv2.idct(dct)
        merged = cv2.merge([np.clip(y_idct, 0, 255).astype(np.uint8), u, v])
        return cv2.cvtColor(merged, cv2.COLOR_YUV2RGB)

    # ─────────────────── UPDATED EMBEDDING LOGIC ───────────────────
    def image_embedding(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")

        inputs = self.clip_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get raw features directly to avoid the pooling object error
            emb = self.clip_model.get_image_features(**inputs)

        return self._normalize(emb.cpu().numpy())

    def video_embeddings(self, path):
        cap = cv2.VideoCapture(path)
        embeddings = []
        last_gray = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
            diff = 1.0 if last_gray is None else np.mean(cv2.absdiff(gray, last_gray)) / 255.0
            if diff > 0.15:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                embeddings.append(self.image_embedding(img))
                last_gray = gray.copy()
        cap.release()
        return np.vstack(embeddings) if embeddings else None

    def register(self, label, source, modality):
        tag = f"TRINETRA::{label}"
        try:
            if modality == "Image":
                img = cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2RGB)
                wm = self._apply_watermark(img)
                out = os.path.join(EXPORT_PATH, f"SECURE_{os.path.splitext(label)[0]}.png")
                cv2.imwrite(out, cv2.cvtColor(wm, cv2.COLOR_RGB2BGR))
                emb = self.image_embedding(wm)
                self.indices[modality].add(emb)
                self.labels[modality].append(tag)
            elif modality == "Video":
                embs = self.video_embeddings(source)
                if embs is not None:
                    for i, e in enumerate(embs):
                        self.indices[modality].add(e[None])
                        self.labels[modality].append(f"{tag}::KF_{i}")
            self._save(modality)
            return f"REGISTERED: {label}"
        except Exception as e:
            return f"ERROR: {str(e)}"

    def scan(self, source, modality):
        if self.indices[modality].ntotal == 0: return "REGISTRY EMPTY", None
        threshold = {"Image": 0.88, "Video": 0.85}[modality]
        if modality == "Video":
            embs = self.video_embeddings(source)
            if embs is None: return "VIDEO FAILED", None
            D, I = self.indices[modality].search(embs, 1)
            scores = D[:, 0]
            idx = np.argmax(scores)
            plt.figure(figsize=(8, 2)); plt.plot(scores); plt.axhline(threshold, color='red', linestyle='--')
            graph = f"report_{int(time.time())}.png"; plt.savefig(graph); plt.close()
            if scores[idx] >= threshold:
                return f"INFRINGEMENT: {self.labels[modality][I[idx,0]]} ({scores[idx]:.2%})", graph
            return f"CLEAN ({scores[idx]:.2%})", graph
        
        emb = self.image_embedding(source)
        D, I = self.indices[modality].search(emb, 1)
        if D[0,0] >= threshold:
            return f"INFRINGEMENT: {self.labels[modality][I[0,0]]} ({D[0,0]:.2%})", None
        return f"CLEAN ({D[0,0]:.2%})", None

# ─────────────────── UI ───────────────────
engine = Trinetra()

def handler(files, modality, action):
    if not files: return "NO FILES", None
    out = []; graph = None
    for f in files:
        if action == "Register":
            out.append(engine.register(os.path.basename(f.name), f.name, modality))
        else:
            res, g = engine.scan(f.name, modality)
            out.append(res)
            if g: graph = g
    return "\n".join(out), graph

with gr.Blocks(title="Trinetra", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 👁️ TRINETRA v1.6.2\nMultimodal IP Defense System")
    with gr.Tabs():
        with gr.Tab("🛡️ Register"):
            r_mod = gr.Radio(["Image", "Video"], value="Image", label="Select Modality")
            r_files = gr.File(file_count="multiple", label="Upload Originals")
            r_btn = gr.Button("SHIELD ASSETS", variant="primary")
            r_out = gr.Textbox(label="Status Log", lines=8)
        with gr.Tab("🔍 Scan"):
            s_mod = gr.Radio(["Image", "Video"], label="Select Modality", value="Image")
            s_files = gr.File(file_count="multiple", label="Upload Suspicious Content")
            s_btn = gr.Button("RUN FORENSIC SCAN", variant="stop")
            s_out = gr.Textbox(label="Analysis Results", lines=6)
            s_img = gr.Image(label="Temporal Evidence (Video Only)")

    r_btn.click(handler, [r_files, r_mod, gr.State("Register")], [r_out, s_img])
    s_btn.click(handler, [s_files, s_mod, gr.State("Scan")], [s_out, s_img])

if __name__ == "__main__":
    app.launch()
