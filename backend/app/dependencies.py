"""Contains dependencies"""

from backend.app.services.embeddings.file_processor import FileProcessor
from backend.app.services.embeddings.embedding_service import EmbeddingService
from backend.app.services.embeddings.factory import (
    EmbeddingServiceFactory,
    RetrieverServiceFactory
)
from backend.app.services.embeddings.retriever import RetrieverService
from backend.app.services.llm.llm_provider import VLLMProvider
from backend.app.core.config import settings
from backend.app.domain.protocols import LLMProviderProtocol


def get_file_processor() -> FileProcessor:
    return FileProcessor()


def get_embedding_service() -> EmbeddingService:
    return EmbeddingServiceFactory.create()


def get_retriever() -> RetrieverService:
    return RetrieverServiceFactory.create()


def get_llm_provider() -> LLMProviderProtocol:
    return VLLMProvider(
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL
    )