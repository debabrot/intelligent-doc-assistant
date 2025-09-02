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

    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")

        self.collection.add(
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            ids=[chunk.id for chunk in chunks]
        )

        logger.info(
            "Successfully added %d documents to Chroma collection '%s'", 
            len(chunks), 
            self.collection_name
        )