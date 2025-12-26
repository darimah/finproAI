# config.py
import os
from dataclasses import dataclass

# Base directory = lokasi file config.py ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default folder structure (relative to BASE_DIR)
DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
DEFAULT_HOPE_DIR    = os.path.join(DEFAULT_DATASET_DIR, "HOPE")
DEFAULT_HQC_DIR     = os.path.join(DEFAULT_DATASET_DIR, "High Quality Counseling")

DEFAULT_INDEX_DIR   = os.path.join(BASE_DIR, "indexes")
DEFAULT_TMP_DIR     = os.path.join(BASE_DIR, "tmp")


def _abspath_from_base(path: str) -> str:
    """
    Kalau path sudah absolute -> kembalikan apa adanya.
    Kalau relative -> buat absolute berbasis BASE_DIR.
    """
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


@dataclass
class Config:
    # -------------------------
    # Paths
    # -------------------------
    DATASET_DIR: str = _abspath_from_base(os.getenv("DATASET_DIR", DEFAULT_DATASET_DIR))
    HOPE_DIR: str    = _abspath_from_base(os.getenv("HOPE_DIR", DEFAULT_HOPE_DIR))
    HQC_DIR: str     = _abspath_from_base(os.getenv("HQC_DIR", DEFAULT_HQC_DIR))

    INDEX_DIR: str   = _abspath_from_base(os.getenv("INDEX_DIR", DEFAULT_INDEX_DIR))
    TMP_DIR: str     = _abspath_from_base(os.getenv("TMP_DIR", DEFAULT_TMP_DIR))

    # -------------------------
    # Retrieval
    # -------------------------
    TOP_K: int = int(os.getenv("TOP_K", "3"))

    # -------------------------
    # Models
    # -------------------------
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    CHAT_MODEL: str  = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
    STT_MODEL: str   = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")
    TTS_MODEL: str   = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
    TTS_VOICE: str   = os.getenv("TTS_VOICE", "sage")

    # -------------------------
    # Audio
    # -------------------------
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))

    # Fixed recording (legacy)
    RECORD_SECONDS: int = int(os.getenv("RECORD_SECONDS", "6"))

    # VAD recording (opsional)
    MAX_RECORD_SECONDS: int = int(os.getenv("MAX_RECORD_SECONDS", "60"))
    SILENCE_SECONDS: float  = float(os.getenv("SILENCE_SECONDS", "10.0"))
    RMS_THRESHOLD: float    = float(os.getenv("RMS_THRESHOLD", "0.006"))

    # -------------------------
    # Safety (minimal)
    # -------------------------
    ENABLE_SAFETY: bool = os.getenv("ENABLE_SAFETY", "1") == "1"


# Optional: instance siap pakai (kalau kamu biasa import cfg)
cfg = Config()
