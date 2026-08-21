import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent

INDEX_DIR = BASE_DIR / "data" / "index"


class Settings(BaseSettings):

    APP_NAME: str = "Voice-Enabled Adaptive RAG Engine"
    VERSION: str = "1.0.0"

    DEBUG: bool = False

    PORT: int = 8000

    FRONTEND_URL: str = (
        "https://voice-rag-goa-visora.vercel.app"
    )

    # =====================================================
    # API KEYS
    # =====================================================

    SARVAM_API_KEY: str = ""

    LLM_PROVIDER: str = "groq"

    LLM_API_KEY: str = ""

    # Current Groq model
    LLM_MODEL: str = "llama-3.1-8b-instant"

    # =====================================================
    # RETRIEVAL
    # =====================================================

    EMBEDDING_MODEL: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    TOP_K: int = 10

    FINAL_K: int = 3

    DENSE_WEIGHT: float = 0.6

    BM25_WEIGHT: float = 0.4

    # =====================================================
    # THRESHOLDS
    # =====================================================

    SIMILARITY_THRESHOLD: float = 0.25

    GROUNDING_THRESHOLD: float = 0.40

    # =====================================================
    # INDEX PATHS
    # =====================================================

    INDEX_PATH: str = str(
        INDEX_DIR / "faiss.index"
    )

    METADATA_PATH: str = str(
        INDEX_DIR / "metadata.json"
    )

    BM25_PATH: str = str(
        INDEX_DIR / "bm25.pkl"
    )

    CONFIG_PATH: str = str(
        INDEX_DIR / "index_config.json"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
