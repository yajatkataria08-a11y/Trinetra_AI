import os
import torch

class Config:
    HNSW_M               = 32
    HNSW_EF_CONSTRUCTION = 200
    HNSW_EF_SEARCH       = 50
    MAX_FILE_SIZE        = 100 * 1024 * 1024
    ALLOWED_IMAGE_EXTS   = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    ALLOWED_AUDIO_EXTS   = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    SUPPORTED_LANGUAGES  = ['en', 'hi', 'ta', 'te', 'kn', 'ml', 'bn', 'mr', 'gu', 'pa']
    AUDIO_DURATION_S     = 7.0
    EMBEDDING_DIM        = 512
    AUDIO_SAMPLE_RATE    = 48_000
    AUDIO_SILENCE_RMS    = 0.001
    CONFIDENCE_HIGH      = 0.65
    CONFIDENCE_MED       = 0.45
    DUPLICATE_THRESHOLD  = 0.95
    CACHE_SIZE           = 1000
    OTP_COOLDOWN_SECS    = 60


CONFIG      = Config()
BASE_DIR    = "trinetra_registry"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(BASE_DIR,    exist_ok=True)
os.makedirs("logs",      exist_ok=True)
