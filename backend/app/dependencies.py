"""Contains dependencies"""

from backend.app.services.embeddings.file_processor import FileProcessor
from backend.app.services.embeddings.embedding_service import EmbeddingService
from backend.app.services.embeddings.factory import (
    EmbeddingServiceFactory,
    RetrieverServiceFactory
)
from backend.app.services.embeddings.retriever import RetrieverService


def get_file_processor() -> FileProcessor:
    return FileProcessor()


def get_embedding_service() -> EmbeddingService:
    return EmbeddingServiceFactory.create()

def get_retriever() -> RetrieverService:
    return RetrieverServiceFactory.create()
