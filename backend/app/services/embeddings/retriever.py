
import logging
from typing import List
from backend.app.domain.protocols import (
    EmbeddingProviderProtocol,
    VectorStoreProtocol,
    DocumentChunk,
)

logger = logging.getLogger(__name__)


class RetrieverService:
    def __init__(
        self,
        embedding_provider: EmbeddingProviderProtocol,
        vector_store: VectorStoreProtocol
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Given a text query, embed it and retrieve top-k relevant chunks.
        """
        logger.info("Generating embedding for query: %s", query[:100] + "...")
        query_embedding = self.embedding_provider.get_embeddings([query])[0]

        logger.info("Querying vector store for top %d results", top_k)
        chunks = self.vector_store.retrieve(query_embedding, top_k)

        logger.info("Retrieved %d chunks", len(chunks))
        return chunks
