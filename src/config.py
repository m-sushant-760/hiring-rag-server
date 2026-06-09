"""
Application configuration — reads from .env via python-dotenv.
All modules import from here; no `os.getenv` calls scattered around.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "hiring-rag")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    # Pinecone Integrated Inference — handles embedding internally
    pinecone_embedding_model: str = os.getenv(
        "PINECONE_EMBEDDING_MODEL", "multilingual-e5-large"
    )

    # Google Gemini (generation)
    # Default: gemini-2.5-flash-lite — stable, fastest, highest free-tier quota.
    # Upgrade to gemini-2.5-flash for better reasoning quality (lower free quota).
    # DO NOT use gemini-2.0-flash or gemini-1.5-flash — both are shut down (2026).
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    # App
    app_env: str = os.getenv("APP_ENV", "development")
    port: int = int(os.getenv("PORT", "8000"))
    max_upload_size_bytes: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")) * 1024 * 1024
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", "10"))
    database_url: str = os.getenv("DATABASE_URL", "")

    # Section-aware chunking (Phase A) — per-section chunk size tuning.
    # Ops can override these via .env without code changes.
    section_chunk_size_skills:      int = int(os.getenv("CHUNK_SIZE_SKILLS",      "400"))
    section_chunk_size_experience:  int = int(os.getenv("CHUNK_SIZE_EXPERIENCE",  "1200"))
    section_chunk_size_default:     int = int(os.getenv("CHUNK_SIZE_DEFAULT",     "1000"))
    section_chunk_overlap_default:  int = int(os.getenv("CHUNK_OVERLAP_DEFAULT",  "100"))


settings = Settings()
