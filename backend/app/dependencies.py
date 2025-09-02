"""Contains dependencies"""

from backend.app.core.config import settings
from backend.app.services.embeddings.file_processor import FileProcessor
from backend.app.services.embeddings.embedding_service import EmbeddingService
from backend.app.services.embeddings.factory import EmbeddingServiceFactory


def get_file_processor() -> FileProcessor:
    return FileProcessor(settings.UPLOAD_DIR)


def get_embedding_service() -> EmbeddingService:
    return EmbeddingServiceFactory.create()
