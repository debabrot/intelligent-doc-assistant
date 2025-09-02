"""Contains embedding service"""

import os
import logging
from typing import List
from backend.app.domain.protocols import (
    DocumentLoaderProtocol, 
    EmbeddingProviderProtocol, 
    VectorStoreProtocol,
    DocumentChunk
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(
        self,
        document_loader: DocumentLoaderProtocol,
        embedding_provider: EmbeddingProviderProtocol,
        vector_store: VectorStoreProtocol,
        batch_size: int = 16
    ):
        self.document_loader = document_loader
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.batch_size = batch_size

    def process_file(
        self,
        file_path: str,
        chunk_size: int = 256,
        chunk_overlap: int = 50
    ) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info("Loading and splitting PDF: %s", file_path)
        chunks = self.document_loader.load_and_split(file_path, chunk_size, chunk_overlap)
        logger.info("Split into %d chunks", len(chunks))

        if not chunks:
            logger.warning("No documents extracted from PDF.")
            return

        embeddings = self._generate_embeddings_batched(chunks)
        self.vector_store.add_documents(chunks, embeddings)

    def _generate_embeddings_batched(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        texts = [chunk.content for chunk in chunks]
        all_embeddings = []
        
        logger.info("Generating embeddings in batches of %d", self.batch_size)
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                batch_embeddings = self.embedding_provider.get_embeddings(batch)
                if len(batch_embeddings) != len(batch):
                    logger.warning(
                        "Embedding count mismatch: expected %d, got %d", 
                        len(batch), 
                        len(batch_embeddings)
                    )
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error("Failed to get embeddings for batch %d: %s", i, e)
                raise

        return all_embeddings