from typing import List, Optional
from pydantic import BaseModel


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class ChunkMetadata(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None

    class Config:
        extra = "allow"


class RetrievedChunk(BaseModel):
    id: str
    content: str
    metadata: ChunkMetadata
    similarity: Optional[float] = None


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    chunks: List[RetrievedChunk]