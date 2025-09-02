"""Factory of embedding service"""

from backend.app.core.config import settings
from backend.app.services.embeddings.tokenizer import TEITokenizer
from backend.app.services.embeddings.document_loader import PDFDocumentLoader
from backend.app.services.embeddings.embedding_provider import TEIEmbeddingProvider
from backend.app.services.embeddings.vector_store import ChromaVectorStore
from backend.app.services.embeddings.embedding_service import EmbeddingService


class EmbeddingServiceFactory:
    @staticmethod
    def create() -> EmbeddingService:
        tokenizer = TEITokenizer(settings.EMBEDDING_TOKENIZE_URL)
        document_loader = PDFDocumentLoader(tokenizer)
        embedding_provider = TEIEmbeddingProvider(settings.EMBEDDING_API_URL)
        vector_store = ChromaVectorStore(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            collection_name=settings.COLLECTION_NAME
        )
        
        return EmbeddingService(
            document_loader=document_loader,
            embedding_provider=embedding_provider,
            vector_store=vector_store
        )