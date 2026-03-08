import os
import json
import time
import logging
import hashlib
import tempfile
import threading

import faiss
import numpy as np
import streamlit as st
import torch
import librosa
from PIL import Image

from config import CONFIG, BASE_DIR, STORAGE_DIR, DEVICE
from database import get_shared_metadata_db
from utils import (
    sanitize_asset_id, validate_asset_id, validate_upload,
    validate_image_content, analyze_image_quality, analyze_audio_quality,
    is_audio_valid, file_md5, fp16,
)
from models import load_models
from titan_embeddings import get_titan_embedder

logger = logging.getLogger("Trinetra")

# Serialise all model forward passes — prevents VRAM OOM on single-GPU systems
_inference_lock = threading.Lock()


# ==================== BASE ENGINE ====================

class TrinetraEngine:

    def __init__(self, modality: str):
        self.modality = modality
        db_path       = os.path.join(BASE_DIR, modality)
        self.idx_path = os.path.join(db_path, "index")
        self.map_path = os.path.join(db_path, "id_map.json")
        os.makedirs(db_path, exist_ok=True)

        if os.path.exists(self.idx_path) and os.path.exists(self.map_path):
            self.index = faiss.read_index(self.idx_path)
            with open(self.map_path) as f:
                id_list = json.load(f)
        else:
            self.index = faiss.IndexHNSWFlat(CONFIG.EMBEDDING_DIM, CONFIG.HNSW_M)
            self.index.hnsw.efConstruction = CONFIG.HNSW_EF_CONSTRUCTION
            self.index.hnsw.efSearch       = CONFIG.HNSW_EF_SEARCH
            id_list = []

        self.id_map          = {r["id"]: r for r in id_list}
        self.id_list         = id_list
        self.embedding_cache = {}
        self.metadata_db     = get_shared_metadata_db()
        self._lock           = threading.Lock()
        self._validate_sync()

    # ── Index integrity ──

    def _validate_sync(self):
        n_idx = self.index.ntotal
        n_map = len(self.id_list)
        if n_idx == n_map:
            return

        logger.warning(
            f"SYNC_MISMATCH modality={self.modality} "
            f"faiss_ntotal={n_idx} id_map_len={n_map} — rebuilding index"
        )
        new_index = faiss.IndexHNSWFlat(CONFIG.EMBEDDING_DIM, CONFIG.HNSW_M)
        new_index.hnsw.efConstruction = CONFIG.HNSW_EF_CONSTRUCTION
        new_index.hnsw.efSearch       = CONFIG.HNSW_EF_SEARCH
        rebuilt, failed = [], []

        for record in self.id_list:
            if not os.path.exists(record["path"]):
                failed.append(record["id"])
                continue
            try:
                emb = self._compute_embedding(file_path=record["path"])
                new_index.add(emb.reshape(1, -1))
                rebuilt.append(record)
            except Exception as e:
                logger.error(f"SYNC_REBUILD_FAIL id={record['id']}: {e}")
                failed.append(record["id"])

        self.index   = new_index
        self.id_list = rebuilt
        self.id_map  = {r["id"]: r for r in rebuilt}
        self._save()
        logger.info(
            f"SYNC_REBUILD_DONE modality={self.modality} "
            f"rebuilt={len(rebuilt)} dropped={len(failed)}"
        )

    # ── Internals ──

    def _normalize(self, v):
        return (v / (np.linalg.norm(v) + 1e-9)).astype("float32")

    def _save(self):
        faiss.write_index(self.index, self.idx_path)
        with open(self.map_path, "w") as f:
            json.dump(self.id_list, f, indent=2)

    def _cache_key(self, text=None, path=None):
        if text:  return hashlib.md5(text.encode()).hexdigest()
        if path:  return file_md5(path)

    def _compute_embedding(self, file_path=None, text=None):
        raise NotImplementedError

    # ── Public helpers ──

    def id_exists(self, asset_id: str) -> bool:
        return asset_id in self.id_map

    def get_embedding(self, file_path=None, text=None):
        key = self._cache_key(text=text, path=file_path)
        if key and key in self.embedding_cache:
            return self.embedding_cache[key]
        emb = self._compute_embedding(file_path, text)
        if key and len(self.embedding_cache) < CONFIG.CACHE_SIZE:
            self.embedding_cache[key] = emb
        return emb

    def find_duplicates(self, file_path=None, text=None, threshold=None):
        if self.index.ntotal == 0:
            return []
        t = threshold or CONFIG.DUPLICATE_THRESHOLD
        q = self.get_embedding(file_path=file_path, text=text).reshape(1, -1)
        scores, idxs = self.index.search(q, min(10, self.index.ntotal))
        return [
            {"id": self.id_list[i]["id"], "similarity": float(s),
             "path": self.id_list[i]["path"]}
            for s, i in zip(scores[0], idxs[0])
            if i != -1 and s > t
        ]

    # ── Registration ──

    def register(self, temp_path, asset_id, ext, lang, tags=None,
                 description="", collection="", uploaded_by="unknown"):
        asset_id = sanitize_asset_id(asset_id)
        ok, msg  = validate_asset_id(asset_id)
        if not ok:
            return False, msg

        perm_path = os.path.join(STORAGE_DIR, f"{asset_id}{ext}")
        try:
            with open(temp_path, "rb") as s, open(perm_path, "wb") as d:
                d.write(s.read())

            if self.modality == "image":
                quality_info  = analyze_image_quality(perm_path)
                quality_score = quality_info["quality_score"] if quality_info else 0.5
            else:
                quality_info  = analyze_audio_quality(perm_path)
                quality_score = (
                    min(quality_info["rms_energy"] * 10, 1.0) if quality_info else 0.5
                )

            emb = self.get_embedding(file_path=perm_path)

            with self._lock:
                if self.id_exists(asset_id):
                    if os.path.exists(perm_path):
                        os.remove(perm_path)
                    return False, f"ID '{asset_id}' already exists"

                faiss_idx = self.index.ntotal
                self.index.add(emb.reshape(1, -1))
                record = {
                    "id": asset_id, "path": perm_path, "lang": lang,
                    "modality": self.modality, "timestamp": time.ctime(),
                    "quality": quality_score,
                }
                self.id_map[asset_id] = record
                self.id_list.append(record)
                self._save()

            self.metadata_db.add_asset(
                asset_id, self.modality, lang, perm_path,
                os.path.getsize(perm_path), faiss_idx,
                tags, description, collection, quality_score, uploaded_by,
            )
            logger.info(
                f"ASSET_REGISTERED id={asset_id} modality={self.modality} "
                f"lang={lang} by={uploaded_by}"
            )
            return True, f"Successfully registered: {asset_id}"
        except Exception as e:
            if os.path.exists(perm_path):
                os.remove(perm_path)
            logger.error(f"Registration failed for {asset_id}: {e}", exc_info=True)
            return False, f"Failed: {e}"

    def batch_register(self, files, language, tags=None, collection="",
                       uploaded_by="unknown"):
        total       = len(files)
        progress    = st.progress(0)
        status_text = st.empty()
        results     = {"success": [], "failed": [], "skipped": [], "duplicates": []}

        for idx, file in enumerate(files):
            status_text.text(f"Processing {idx+1}/{total}: {file.name}")
            try:
                asset_id = sanitize_asset_id(os.path.splitext(file.name)[0])
                if self.id_exists(asset_id):
                    results["skipped"].append(file.name)
                    continue
                err = validate_upload(file, self.modality)
                if err:
                    results["failed"].append((file.name, err))
                    continue
                ext = os.path.splitext(file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(file.getbuffer())
                    tp = tmp.name
                try:
                    dups = self.find_duplicates(file_path=tp)
                    if dups:
                        results["duplicates"].append((file.name, dups[0]["id"]))
                        continue
                    ok, msg = self.register(
                        tp, asset_id, ext, language, tags, "", collection, uploaded_by
                    )
                    (results["success"] if ok else results["failed"]).append(
                        file.name if ok else (file.name, msg)
                    )
                finally:
                    if os.path.exists(tp):
                        os.remove(tp)
            except Exception as e:
                results["failed"].append((file.name, str(e)))
                logger.error(f"Batch register error for {file.name}: {e}", exc_info=True)
            progress.progress((idx + 1) / total)

        progress.empty()
        status_text.empty()
        return results

    # ── Search ──

    def hybrid_search(self, text_query=None, file_path=None, top_k=5,
                      filters=None, rerank=True):
        if self.index.ntotal == 0:
            return [], 0

        t0 = time.time()
        q  = self.get_embedding(file_path=file_path, text=text_query).reshape(1, -1)
        scores, idxs = self.index.search(q, min(top_k * 2, self.index.ntotal))

        out = []
        for s, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            record = self.id_list[i]
            conf   = ("High"   if s > CONFIG.CONFIDENCE_HIGH else
                      "Medium" if s > CONFIG.CONFIDENCE_MED  else "Low")
            result = {**record, "score": float(s), "confidence": conf}

            if filters:
                if "min_score" in filters and s < filters["min_score"]:
                    continue
                if "language" in filters and result.get("lang") != filters["language"]:
                    continue
                if "tags" in filters and filters["tags"]:
                    row = self.metadata_db.db.fetchone(
                        "SELECT tags FROM assets WHERE id=?", (record["id"],)
                    )
                    if row:
                        asset_tags = json.loads(row[0])
                        if not any(t in asset_tags for t in filters["tags"]):
                            continue
            out.append(result)

        if rerank:
            for r in out:
                r["final_score"] = r["score"] * (0.7 + 0.3 * r.get("quality", 0.5))
            out = sorted(out, key=lambda x: x["final_score"], reverse=True)

        return out[:top_k], (time.time() - t0) * 1000

    def search(self, file_path=None, text=None, top_k=5, metadata_filters=None):
        return self.hybrid_search(
            text_query=text, file_path=file_path, top_k=top_k, filters=metadata_filters
        )

    def get_all_vectors(self):
        try:
            return faiss.vector_to_array(
                self.index.reconstruct_n(0, self.index.ntotal)
            ).reshape(self.index.ntotal, -1)
        except Exception as e:
            logger.error(f"get_all_vectors failed: {e}", exc_info=True)
            return None

    def export_registry(self, export_path=None):
        import zipfile
        if export_path is None:
            export_path = f"trinetra_{self.modality}_backup_{int(time.time())}.zip"
        with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as z:
            if os.path.exists(self.idx_path):
                z.write(self.idx_path, "index")
            if os.path.exists(self.map_path):
                z.write(self.map_path, "id_map.json")
            for asset in self.id_list:
                if os.path.exists(asset["path"]):
                    z.write(asset["path"], f"assets/{os.path.basename(asset['path'])}")
        return export_path

    def export_as_json(self):
        from datetime import datetime
        return json.dumps({
            "metadata": {
                "export_date":  datetime.now().isoformat(),
                "modality":     self.modality,
                "total_assets": self.index.ntotal,
            },
            "assets": self.id_list,
        }, indent=2)


# ==================== IMAGE ENGINE ====================

class ImageEngine(TrinetraEngine):

    def __init__(self):
        super().__init__("image")

    def _compute_embedding(self, file_path=None, text=None):
        # ── Text query: try Titan V2 first, fall back to CLIP text encoder ──
        if text:
            titan = get_titan_embedder()
            if titan.is_available():
                vec = titan.embed(text)
                if vec is not None:
                    return self._normalize(vec)
                logger.warning("Titan embed returned None — falling back to CLIP")

        clip_model, clip_processor, _, _ = load_models()
        with _inference_lock:
            with torch.no_grad():
                if text:
                    inp = clip_processor(
                        text=[text], return_tensors="pt", padding=True
                    ).to(DEVICE)
                    if DEVICE == "cuda":
                        inp = fp16(inp)
                    e = clip_model.get_text_features(**inp)
                else:
                    ok, msg = validate_image_content(file_path)
                    if not ok:
                        raise ValueError(msg)
                    img = Image.open(file_path).convert("RGB")
                    inp = clip_processor(images=img, return_tensors="pt").to(DEVICE)
                    if DEVICE == "cuda":
                        inp = fp16(inp)
                    e = clip_model.get_image_features(**inp)
        return self._normalize(e.cpu().float().numpy().flatten())


# ==================== AUDIO ENGINE ====================

class AudioEngine(TrinetraEngine):

    def __init__(self):
        super().__init__("audio")

    def _compute_embedding(self, file_path=None, text=None):
        # ── Text query: try Titan V2 first, fall back to CLAP text encoder ──
        if text:
            titan = get_titan_embedder()
            if titan.is_available():
                vec = titan.embed(text)
                if vec is not None:
                    return self._normalize(vec)
                logger.warning("Titan embed returned None — falling back to CLAP")

        _, _, clap_model, clap_processor = load_models()
        with _inference_lock:
            with torch.no_grad():
                if text:
                    inp = clap_processor(text=[text], return_tensors="pt").to(DEVICE)
                    e   = clap_model.get_text_features(**inp)
                else:
                    y, _ = librosa.load(
                        file_path,
                        sr=CONFIG.AUDIO_SAMPLE_RATE,
                        duration=CONFIG.AUDIO_DURATION_S,
                    )
                    if not is_audio_valid(y):
                        raise ValueError("Audio appears silent")
                    inp = clap_processor(
                        audios=y,
                        sampling_rate=CONFIG.AUDIO_SAMPLE_RATE,
                        return_tensors="pt",
                    ).to(DEVICE)
                    e = clap_model.get_audio_features(**inp)
        return self._normalize(e.cpu().float().numpy().flatten())
