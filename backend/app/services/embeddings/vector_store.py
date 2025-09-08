"""Contains vector store"""

import logging
from typing import List

from chromadb import HttpClient

from backend.app.domain.protocols import (
    VectorStoreProtocol,
    DocumentChunk,
)

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStoreProtocol):
    def __init__(self, host: str, port: int, collection_name: str):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    @property
    def client(self):
        if self._client is None:
            self._client = HttpClient(host=self.host, port=self.port)
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(name=self.collection_name)
        return self._collection

    def add_documents(
            self,
            chunks: List[DocumentChunk],
            embeddings: List[List[float]],
            batch_size: int=1000
        ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")

        total = len(chunks)
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            self.collection.upsert(
                embeddings=batch_embeddings,
                documents=[chunk.content for chunk in batch_chunks],
                metadatas=[chunk.metadata for chunk in batch_chunks],
                ids=[chunk.id for chunk in batch_chunks]
            )

            logger.info(
                "Inserted batch %d-%d of %d documents into Chroma collection '%s'",
                i + 1,
                min(i + batch_size, total),
                total,
                self.collection_name
            )

        logger.info(
            "Successfully added %d documents to Chroma collection '%s'", 
            len(chunks), 
            self.collection_name
        )

    def delete_by_source(self, source: str) -> None:
        """
        Delete all document chunks associated with a given source file.
        """
        try:
            # Query all chunk IDs that have this source in metadata
            results = self.collection.get(
                where={"source": source},
                include=["metadatas"]
            )

            ids_to_delete = results["ids"]

            if not ids_to_delete:
                logger.info("No chunks found for source: %s", source)
                return

            # Delete by IDs
            self.collection.delete(ids=ids_to_delete)

            logger.info(
                "Deleted %d chunks for source '%s' from collection '%s'",
                len(ids_to_delete),
                source,
                self.collection_name
            )

        except Exception as e:
            logger.error("Error deleting chunks for source %s: %s", source, e)
            raise RuntimeError(f"Failed to delete embeddings for {source}") from e