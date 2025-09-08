"""
Contains configurations
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import computed_field


class Settings(BaseSettings):
    """Application settings for the backend service."""
    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    CHROMA_HOST: str
    CHROMA_PORT: str

    # ------------------------------------------------------------------
    # File Storage
    # ------------------------------------------------------------------
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET: str

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    COLLECTION_NAME: str
    EMBEDDING_MODEL: str
    UPLOAD_DIR: str
    EMBEDDING_BASE_URL: str

    @computed_field
    @property
    def EMBEDDING_API_URL(self) -> str:
        return f"{self.EMBEDDING_BASE_URL}/embed"

    @computed_field
    @property
    def EMBEDDING_TOKENIZE_URL(self) -> str:
        return f"{self.EMBEDDING_BASE_URL}/tokenize"

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