"""Contains schema for embeddings"""

from typing import List, Dict

from pydantic import BaseModel

class EmbedResponse(BaseModel):
    processed: List[str]
    failed: List[Dict[str, str]]
    message: str
