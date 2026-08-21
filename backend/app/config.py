from pathlib import Path
from pydantic_settings import BaseSettings


# =====================================================
# PROJECT PATHS
# =====================================================

# Docker WORKDIR = /app
BASE_DIR = Path(__file__).resolve().parent.parent

# /app/data/index
INDEX_DIR = BASE_DIR / "data" / "index"


class Settings(BaseSettings):

    APP_NAME: str = "Voice-Enabled Adaptive RAG Engine"
    VERSION: str = "1.0.0"

    DEBUG: bool = False

    PORT: int = 8000

    FRONTEND_URL: str = "http://localhost:5173"

    # =====================================================
    # API
    # =====================================================

    SARVAM_API_KEY: str = ""

    LLM_PROVIDER: str = "groq"

    LLM_API_KEY: str = ""

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

    SIMILARITY_THRESHOLD: float = 0.15

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
