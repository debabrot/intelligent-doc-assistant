"""Contains protocols"""

from typing import List, Dict, Any, Protocol

from pydantic import BaseModel, Field, ConfigDict


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra='allow')

    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    id: str


class TokenizerProtocol(Protocol):
    def count_tokens(self, text: str) -> int: ...


class EmbeddingProviderProtocol(Protocol):
    def get_embeddings(self, texts: List[str]) -> List[List[float]]: ...


class VectorStoreProtocol(Protocol):
    def add_documents(self,
                      chunks: List[DocumentChunk],
                      embeddings: List[List[float]]) -> None: ...


class DocumentLoaderProtocol(Protocol):
    def load_and_split(self,
                       file_path: str,
                       chunk_size: int,
                       chunk_overlap: int) -> List[DocumentChunk]: ...
