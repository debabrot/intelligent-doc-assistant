"""Contains embedding provider"""

import logging
import requests
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.domain.protocols import EmbeddingProviderProtocol


logger = logging.getLogger(__name__)


class TEIEmbeddingProvider(EmbeddingProviderProtocol):
    def __init__(self, api_url: str, timeout: int = 30):
        self.api_url = api_url
        self.timeout = timeout

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            json={"inputs": texts},
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()

        return self._parse_embedding_response(result)

    def _parse_embedding_response(self, result) -> List[List[float]]:
        # Parse response: support multiple formats
        if isinstance(result, list):
            if result and isinstance(result[0], (int, float)):  # single flat vector
                return [result]
            return result  # assume list of vectors
        elif isinstance(result, dict):
            return result.get("embeddings", []) or result.get("data", [])
        
        raise ValueError(f"Unknown embedding response format: {result}")
