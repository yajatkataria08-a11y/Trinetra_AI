import torch
import streamlit as st
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, ClapModel, pipeline

from config import DEVICE


@st.cache_resource(show_spinner="Loading neural models...")
def load_models():
    dtype  = torch.float16 if DEVICE == "cuda" else torch.float32
    clip_m = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", torch_dtype=dtype
    ).to(DEVICE).eval()
    clip_p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clap_m = ClapModel.from_pretrained(
        "laion/clap-htsat-fused", torch_dtype=dtype
    ).to(DEVICE).eval()
    clap_p = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
    return clip_m, clip_p, clap_m, clap_p


@st.cache_resource
def load_toxicity_model():
    return pipeline(
        "text-classification",
        model="unitary/multilingual-toxic-xlm-roberta",
        device=0 if torch.cuda.is_available() else -1,
    )
