"""
titan_embeddings.py
────────────────────────────────────────────────────────────────
Amazon Titan Text Embeddings V2 integration for Trinetra V5.0

Replaces CLIP/CLAP text encoders on the QUERY side only.
Asset embeddings (image → CLIP, audio → CLAP) are unchanged.

Usage:
    from titan_embeddings import TitanEmbedder
    embedder = TitanEmbedder()
    if embedder.is_available():
        vec = embedder.embed(query_text)   # np.ndarray shape (512,)

Secrets required in .streamlit/secrets.toml:
    AWS_ACCESS_KEY_ID     = "AKIAxxxxx"
    AWS_SECRET_ACCESS_KEY = "xxxxxxxx"
    AWS_REGION            = "us-east-1"   # must be a Bedrock-enabled region

Cost: ~$0.00002 per 1K tokens — negligible for hackathon scale.
"""

import os
import json
import logging

import numpy as np
import streamlit as st

logger = logging.getLogger("Trinetra")

# Titan V2 output dimensions — we use 512 to match CLIP/CLAP FAISS index
TITAN_MODEL_ID  = "amazon.titan-embed-text-v2:0"
TITAN_DIM       = 512          # request 512-dim output (Titan V2 supports 256/512/1024)
TITAN_MODEL_ALT = "amazon.titan-embed-text-v2:0"  # fallback


class TitanEmbedder:
    """
    Wraps Amazon Bedrock Titan Text Embeddings V2.

    Falls back gracefully to None if AWS is not configured or
    Bedrock is unavailable — callers must check is_available() first.
    """

    def __init__(self):
        self._client   = None
        self._ready    = False
        self._err      = ""
        self._init()

    def _init(self):
        try:
            import boto3

            # Pull credentials same way as AWSReverseSearchEngine
            key, secret, region = "", "", "us-east-1"
            try:
                if "AWS_ACCESS_KEY_ID" in st.secrets:
                    key    = str(st.secrets["AWS_ACCESS_KEY_ID"]).strip()
                    secret = str(st.secrets["AWS_SECRET_ACCESS_KEY"]).strip()
                    region = str(st.secrets.get("AWS_REGION", "us-east-1")).strip()
            except Exception:
                pass

            if not key:
                key    = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
                secret = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
                region = os.getenv("AWS_REGION", "us-east-1").strip()

            if not key or not secret:
                self._err = "AWS credentials not configured"
                return

            self._client = boto3.client(
                service_name="bedrock-runtime",
                region_name=region,
                aws_access_key_id=key,
                aws_secret_access_key=secret,
            )
            self._ready = True
            logger.info(f"TitanEmbedder ready (region={region})")

        except ImportError:
            self._err = "boto3 not installed"
        except Exception as e:
            self._err = str(e)
            logger.warning(f"TitanEmbedder init failed: {e}")

    # ── Public API ────────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._ready

    def error(self) -> str:
        return self._err

    def embed(self, text: str) -> np.ndarray | None:
        """
        Returns a normalised float32 numpy array of shape (512,),
        or None on failure.
        """
        if not self._ready:
            logger.warning(f"TitanEmbedder not ready: {self._err}")
            return None

        try:
            body = json.dumps({
                "inputText":      text,
                "dimensions":     TITAN_DIM,
                "normalize":      True,       # unit-norm — matches CLIP/CLAP convention
            })
            response = self._client.invoke_model(
                modelId     = TITAN_MODEL_ID,
                body        = body,
                contentType = "application/json",
                accept      = "application/json",
            )
            result = json.loads(response["body"].read())
            vec    = np.array(result["embedding"], dtype="float32")

            # Safety: re-normalise just in case
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm

            logger.debug(f"TitanEmbedder embedded {len(text)} chars → dim={vec.shape[0]}")
            return vec

        except Exception as e:
            import streamlit as st
            st.error(f"❌ Titan embed error: {type(e).__name__}: {e}")
            logger.error(f"Titan embed failed: {type(e).__name__}: {e}", exc_info=True)
            return None

    def embed_batch(self, texts: list[str]) -> list[np.ndarray | None]:
        """Embed a list of texts. Titan V2 is single-input; we loop."""
        return [self.embed(t) for t in texts]

    def test_connection(self) -> tuple[bool, str]:
        """Smoke-test: embed a short string and check shape."""
        if not self._ready:
            return False, f"❌ Titan not ready: {self._err}"
        vec = self.embed("test")
        if vec is None:
            return False, "❌ Titan embed returned None"
        if vec.shape[0] != TITAN_DIM:
            return False, f"❌ Unexpected dim: {vec.shape[0]}"
        return True, f"✅ Titan Embeddings V2 connected! dim={TITAN_DIM}"


# ── Module-level singleton (cached by Streamlit) ──────────────────

@st.cache_resource(show_spinner=False)
def get_titan_embedder() -> TitanEmbedder:
    return TitanEmbedder()


# ── Module-level singleton (cached by Streamlit) ──────────────────

@st.cache_resource(show_spinner=False)
def get_titan_embedder() -> TitanEmbedder:
    return TitanEmbedder()
