import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = BASE_DIR / "backend" / "data" / "index"

class Settings(BaseSettings):
    APP_NAME: str = "Voice-Enabled Adaptive RAG Engine"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    PORT: int = 8000
    FRONTEND_URL: str = "http://localhost:5173"

    # API Keys & Provider Settings
    SARVAM_API_KEY: str = ""
    LLM_PROVIDER: str = "mock"  # Options: groq, openai, sarvam, mock
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "llama-3.3-70b-versatile"  # Default fast model or configurable

    # Vector & Retrieval Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    TOP_K: int = 10
    FINAL_K: int = 3
    DENSE_WEIGHT: float = 0.6  # alpha
    BM25_WEIGHT: float = 0.4   # 1 - alpha

    # Thresholds
    SIMILARITY_THRESHOLD: float = 0.25
    GROUNDING_THRESHOLD: float = 0.40

    # Paths
    INDEX_PATH: str = str(INDEX_DIR / "faiss.index")
    METADATA_PATH: str = str(INDEX_DIR / "metadata.json")
    BM25_PATH: str = str(INDEX_DIR / "bm25.pkl")
    CONFIG_PATH: str = str(INDEX_DIR / "index_config.json")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
