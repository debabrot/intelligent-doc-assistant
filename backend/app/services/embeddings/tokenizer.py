"""Contains tokenizer services"""

import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from backend.app.domain.protocols import TokenizerProtocol

logger = logging.getLogger(__name__)


class TEITokenizer(TokenizerProtocol):
    def __init__(self, tokenize_url: str, timeout: int = 5):
        self.tokenize_url = tokenize_url
        self.timeout = timeout

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def count_tokens(self, text: str) -> int:
        if not text or not text.strip():
            return 1

        clean_text = text.strip()

        try:
            response = requests.post(
                self.tokenize_url,
                json={"inputs": clean_text},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            return self._parse_tokenize_response(result)
        except Exception as e:
            logger.warning("Tokenization failed (using fallback): %s", e)
            return self._fallback_token_count(clean_text)

    def _parse_tokenize_response(self, result) -> int:
        if isinstance(result, list):
            if result and isinstance(result[0], list):  # [[tokens]]
                return len(result[0])
            elif result and isinstance(result[0], dict):  # [{"id":...}, ...]
                return len(result)
        elif isinstance(result, dict):
            return len(result.get("input_ids", []))

        logger.warning("Unexpected tokenize response: %s", result)
        return 1

    def _fallback_token_count(self, text: str) -> int:
        word_count = len(text.split())
        char_estimate = len(text) / 4.0  # ~4 chars per token
        return int(max(word_count, char_estimate)) + 10  # conservative buffer