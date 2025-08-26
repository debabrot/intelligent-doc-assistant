"""
Contains configurations
"""

# settings.py
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings for the backend service."""
    # ------------------------------------------------------------------
    # Chroma
    # ------------------------------------------------------------------
    CHROMA_HOST: str
    CHROMA_PORT: int

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    COLLECTION_NAME: str = "pdf_embeddings"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    UPLOAD_DIR: str = "uploads"
    EMBEDDING_API_URL: str

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    """
    Get application settings with environment-specific configuration.

    Returns:
        Settings: Configured settings instance
    """
    return Settings()


settings = get_settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)