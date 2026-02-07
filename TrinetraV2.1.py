import os
import faiss
import torch
import pickle
import librosa
import numpy as np
import streamlit as st
from PIL import Image
from deep_translator import GoogleTranslator
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, ClapModel
import re

# ---------------- CONFIG ---------------- #
BASE_DIR = "trinetra_registry"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# File constraints
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
ALLOWED_AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

st.set_page_config(page_title="Trinetra: Bharat AI", layout="wide", page_icon="👁️")

# Custom CSS for Hackathon Polish
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-card {
        border: 2px solid #667eea;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: #f8f9fa;
        transition: transform 0.2s;
    }
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.2);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ---------------- #
@st.cache_resource
def load_models():
    try:
        clip_m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        clip_p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clap_m = ClapModel.from_pretrained("laion/clap-htsat-fused").to(DEVICE).eval()
        clap_p = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
        return clip_m, clip_p, clap_m, clap_p
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

clip_model, clip_processor, clap_model, clap_processor = load_models()

# ---------------- UTILITY FUNCTIONS ---------------- #
@st.cache_data(ttl=3600)
def cached_translation(text):
    """Cache translations to avoid repeated API calls"""
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        st.warning(f"Translation failed: {str(e)}. Using original text.")
        return text

def sanitize_asset_id(asset_id):
    """Sanitize asset ID to prevent path traversal"""
    return re.sub(r'[^\w\-]', '_', asset_id)

def validate_file_size(uploaded_file):
    """Check if file size is within limits"""
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"File too large ({uploaded_file.size / 1024 / 1024:.1f}MB). Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
    return True, ""

# ---------------- CORE ENGINE ---------------- #
class TrinetraEngine:
    def __init__(self, modality):
        self.modality = modality
        self.db_path = os.path.join(BASE_DIR, modality)
        os.makedirs(self.db_path, exist_ok=True)
        
        self.index_path = os.path.join(self.db_path, "registry.index")
        self.map_path = os.path.join(self.db_path, "id_map.pkl")
        self.dim = 512  # Both CLIP and CLAP use 512
        
        # Load or create index
        if os.path.exists(self.index_path) and os.path.exists(self.map_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.map_path, "rb") as f:
                    self.id_map = pickle.load(f)
            except Exception as e:
                st.warning(f"Failed to load existing index: {str(e)}. Creating new index.")
                self.index = faiss.IndexFlatIP(self.dim)
                self.id_map = []
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.id_map = []
    
    def _normalize(self, v):
        """Normalize vector to unit length"""
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return v
        return v / norm
    
    def get_embedding(self, file_path=None, text=None):
        """Generate embedding for image/audio or text query"""
        try:
            with torch.no_grad():
                if self.modality == "image":
                    if text:
                        inp = clip_processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
                        emb = clip_model.get_text_features(**inp)
                    else:
                        img = Image.open(file_path).convert("RGB")
                        inp = clip_processor(images=img, return_tensors="pt").to(DEVICE)
                        emb = clip_model.get_image_features(**inp)
                else:  # audio
                    if text:
                        inp = clap_processor(text=[text], return_tensors="pt").to(DEVICE)
                        emb = clap_model.get_text_features(**inp)
                    else:
                        audio, _ = librosa.load(file_path, sr=48000, duration=7.0, mono=True)
                        if len(audio) == 0:
                            raise ValueError("Empty or invalid audio file")
                        inp = clap_processor(audios=audio, sampling_rate=48000, return_tensors="pt").to(DEVICE)
                        emb = clap_model.get_audio_features(**inp)
                
                emb = emb.cpu().numpy().flatten()
                return self._normalize(emb).astype("float32")
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {str(e)}")
    
    def register(self, temp_path, asset_id, ext):
        """Register a new asset to the database"""
        try:
            # Sanitize asset ID
            asset_id = sanitize_asset_id(asset_id)
            
            # Check for duplicate IDs
            if any(item["id"] == asset_id for item in self.id_map):
                return False, f"Error: Asset ID '{asset_id}' already exists"
            
            # Create permanent path
            perm_path = os.path.join(STORAGE_DIR, f"{asset_id}{ext}")
            
            # Copy file to storage
            with open(temp_path, "rb") as src, open(perm_path, "wb") as dst:
                dst.write(src.read())
            
            # Generate embedding
            emb = self.get_embedding(file_path=perm_path)
            
            # Add to index
            self.index.add(emb.reshape(1, -1))
            self.id_map.append({"id": asset_id, "path": perm_path})
            
            # Save to disk
            faiss.write_index(self.index, self.index_path)
            with open(self.map_path, "wb") as f:
                pickle.dump(self.id_map, f)
            
            return True, f"Successfully registered: {asset_id}"
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def search(self, file_path=None, text=None, top_k=3):
        """Search for similar assets"""
        if self.index.ntotal == 0:
            return []
        
        try:
            q = self.get_embedding(file_path=file_path, text=text).reshape(1, -1)
            scores, idxs = self.index.search(q, min(top_k, self.index.ntotal))
            
            results = []
            for s, i in zip(scores[0], idxs[0]):
                if i != -1 and i < len(self.id_map):
                    results.append({
                        "id": self.id_map[i]["id"],
                        "path": self.id_map[i]["path"],
                        "score": float(s)
                    })
            return results
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            return []

# ---------------- APP LAYOUT ---------------- #
image_engine = TrinetraEngine("image")
audio_engine = TrinetraEngine("audio")

st.title("Trinetra: Multimodal Asset Registry")
st.write("Cross-modal verification for Bharat's Digital Infrastructure using AI-powered embeddings.")

# Stats Section
m1, m2, m3 = st.columns(3)
m1.markdown(f"""
<div class='metric-card'>
    <div class='metric-label'>Images Registered</div>
    <div class='metric-value'>{image_engine.index.ntotal}</div>
</div>
""", unsafe_allow_html=True)

m2.markdown(f"""
<div class='metric-card'>
    <div class='metric-label'>Audio Registered</div>
    <div class='metric-value'>{audio_engine.index.ntotal}</div>
</div>
""", unsafe_allow_html=True)

m3.markdown(f"""
<div class='metric-card'>
    <div class='metric-label'>Device</div>
    <div class='metric-value'>{DEVICE.upper()}</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# Registration Sidebar
with st.sidebar:
    st.header("Registry Management")
    st.caption("**Created by Team Human**")
    st.divider()
    
    mod = st.selectbox("Modality", ["image", "audio"])
    asset_id = st.text_input("Asset ID (Unique)", help="Use alphanumeric characters, dashes, and underscores only")
    up = st.file_uploader(
        "Upload Source File",
        type=list(ALLOWED_IMAGE_EXTS if mod == "image" else ALLOWED_AUDIO_EXTS)
    )
    
    if st.button("Commit to Registry", use_container_width=True):
        if not up:
            st.error("Please upload a file")
        elif not asset_id or asset_id.strip() == "":
            st.error("Please provide an Asset ID")
        else:
            # Validate file size
            valid_size, size_msg = validate_file_size(up)
            if not valid_size:
                st.error(size_msg)
            else:
                ext = os.path.splitext(up.name)[1].lower()
                
                # Validate file extension
                if mod == "image" and ext not in ALLOWED_IMAGE_EXTS:
                    st.error(f"Invalid image format. Allowed: {', '.join(ALLOWED_IMAGE_EXTS)}")
                elif mod == "audio" and ext not in ALLOWED_AUDIO_EXTS:
                    st.error(f"Invalid audio format. Allowed: {', '.join(ALLOWED_AUDIO_EXTS)}")
                else:
                    # Save temp file
                    temp = f"temp_{up.name}"
                    try:
                        with open(temp, "wb") as f:
                            f.write(up.getbuffer())
                        
                        engine = image_engine if mod == "image" else audio_engine
                        
                        with st.spinner("Generating Neural Embedding..."):
                            success, msg = engine.register(temp, asset_id, ext)
                        
                        if success:
                            st.success(msg)
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(msg)
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
                    finally:
                        # Cleanup temp file
                        if os.path.exists(temp):
                            os.remove(temp)
    
    st.divider()
    st.caption(f"Storage: `{STORAGE_DIR}`")
    st.caption(f"Device: `{DEVICE}`")

# Main Search Interaction
tab_i, tab_a = st.tabs(["Visual Search", "Acoustic Search"])

# ========== IMAGE SEARCH TAB ========== #
with tab_i:
    col_input, col_results = st.columns([1, 2.5])
    
    with col_input:
        mode = st.radio("Search by", ["Regional Text", "Reference Image"], key="i_mode")
        top_k = st.slider("Number of Results", 1, 10, 3, key="i_k")
        
        if mode == "Regional Text":
            query_input = st.text_input("Enter search description (any language)", key="i_text")
        else:
            query_input = st.file_uploader("Upload Query Image", type=list(ALLOWED_IMAGE_EXTS), key="i_file")
    
    with col_results:
        if st.button("Execute Visual Scan", use_container_width=True):
            if not query_input:
                st.warning("Please provide a search query or image")
            elif image_engine.index.ntotal == 0:
                st.info("No images registered yet. Add some images first!")
            else:
                with st.spinner("Searching neural space..."):
                    try:
                        if mode == "Regional Text":
                            eng = cached_translation(query_input)
                            if eng != query_input:
                                st.caption(f"Translated query: *{eng}*")
                            results = image_engine.search(text=eng, top_k=top_k)
                        else:
                            # Validate uploaded image
                            valid_size, size_msg = validate_file_size(query_input)
                            if not valid_size:
                                st.error(size_msg)
                                results = []
                            else:
                                temp = "query_i.png"
                                with open(temp, "wb") as f:
                                    f.write(query_input.getbuffer())
                                results = image_engine.search(file_path=temp, top_k=top_k)
                                os.remove(temp)
                        
                        # Display Results
                        if not results:
                            st.info("No matches found. Try adjusting your query.")
                        else:
                            st.success(f"Found {len(results)} matches")
                            
                            res_cols = st.columns(3)
                            for i, r in enumerate(results):
                                with res_cols[i % 3]:
                                    st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
                                    try:
                                        st.image(r["path"], use_container_width=True)
                                        st.write(f"**ID:** `{r['id']}`")
                                        st.write(f"**Similarity:** {r['score']*100:.1f}%")
                                        st.progress(min(r["score"], 1.0))
                                    except Exception as e:
                                        st.error(f"Failed to load: {r['id']}")
                                    st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Download results
                            result_data = "\n".join([f"{r['id']}: {r['score']:.4f}" for r in results])
                            st.download_button(
                                label="Download Results",
                                data=result_data,
                                file_name="image_search_results.txt",
                                mime="text/plain"
                            )
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")

# ========== AUDIO SEARCH TAB ========== #
with tab_a:
    col_a_in, col_a_res = st.columns([1, 2.5])
    
    with col_a_in:
        mode_a = st.radio("Search by", ["Description", "Audio Sample"], key="a_mode")
        top_k_a = st.slider("Number of Results", 1, 10, 3, key="a_k")
        
        if mode_a == "Description":
            query_a = st.text_input("Describe the sound (any language)", key="a_text")
        else:
            query_a = st.file_uploader("Upload Query Audio", type=list(ALLOWED_AUDIO_EXTS), key="a_file")
    
    with col_a_res:
        if st.button("Execute Acoustic Scan", use_container_width=True):
            if not query_a:
                st.warning("Please provide a description or audio sample")
            elif audio_engine.index.ntotal == 0:
                st.info("No audio registered yet. Add some audio first!")
            else:
                with st.spinner("Analyzing acoustic patterns..."):
                    try:
                        if mode_a == "Description":
                            eng = cached_translation(query_a)
                            if eng != query_a:
                                st.caption(f"Translated query: *{eng}*")
                            results = audio_engine.search(text=eng, top_k=top_k_a)
                        else:
                            # Validate uploaded audio
                            valid_size, size_msg = validate_file_size(query_a)
                            if not valid_size:
                                st.error(size_msg)
                                results = []
                            else:
                                temp = "query_a.wav"
                                with open(temp, "wb") as f:
                                    f.write(query_a.getbuffer())
                                results = audio_engine.search(file_path=temp, top_k=top_k_a)
                                os.remove(temp)
                        
                        # Display Results
                        if not results:
                            st.info("No matches found. Try adjusting your query.")
                        else:
                            st.success(f"Found {len(results)} matches")
                            
                            for r in results:
                                with st.container(border=True):
                                    c1, c2 = st.columns([1, 3])
                                    c1.metric("Similarity", f"{r['score']*100:.1f}%")
                                    c2.write(f"**Asset ID:** `{r['id']}`")
                                    try:
                                        c2.audio(r["path"])
                                    except Exception as e:
                                        c2.error(f"Failed to load audio: {str(e)}")
                            
                            # Download results
                            result_data = "\n".join([f"{r['id']}: {r['score']:.4f}" for r in results])
                            st.download_button(
                                label="Download Results",
                                data=result_data,
                                file_name="audio_search_results.txt",
                                mime="text/plain"
                            )
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong style='font-size: 1.1em; color: #667eea;'>Created by Team Human</strong></p>
    <p>Powered by CLIP & CLAP | Built for Bharat's Digital Future</p>
    <p style='font-size: 0.8em;'>Multimodal embeddings • FAISS indexing • Cross-lingual search</p>
</div>
""", unsafe_allow_html=True)