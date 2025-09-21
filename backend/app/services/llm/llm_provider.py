"""
    Handles direct communication with vLLM service via its OpenAI-compatible API.
    Manages HTTP client, retries, error handling, and message formatting.
"""

import logging
from typing import List, Dict, AsyncGenerator, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.app.domain.protocols import LLMProviderProtocol


logger = logging.getLogger(__name__)



class LLMProviderError(Exception):
    """Custom exception for LLM provider errors."""
    pass


class VLLMProvider(LLMProviderProtocol):
    """
    Handles direct communication with vLLM service via its OpenAI-compatible API.
    Manages HTTP client, retries, error handling, and message formatting.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        pool_connections: int = 10,
        pool_maxsize: int = 100,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,

    ):
        """
        Initialize the vLLM provider with an async HTTP client.

        :param base_url: Base URL of the vLLM OpenAI-compatible endpoint (e.g., http://vllm:8000/v1)
        :param api_key: Optional API key if vLLM requires authentication
        :param timeout: Request timeout in seconds
        :param max_retries: Max number of retry attempts on failure
        :param pool_connections: Number of pooled connections
        :param pool_maxsize: Max connection pool size
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.model = model
        self.history = history or []
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Set up async HTTP client with connection pooling
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=pool_connections,
                max_connections=pool_maxsize
            )
        )

        # Configure logger
        self.logger = logging.getLogger(__name__)

    async def close(self):
        """Gracefully close the HTTP client."""
        await self._client.aclose()

    def _build_messages(self, prompt: str, history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """
        Format conversation into OpenAI messages format.

        Example:
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]

        :param prompt: The current user prompt
        :param history: List of prior {"role": "...", "content": "..."} messages
        :return: Formatted message list
        """
        messages = history or []
        messages.append({"role": "user", "content": prompt})
        return messages

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
        reraise=True
    )
    async def _handle_error(self, coro):
        """
        Wrapper to handle retries and structured error conversion.

        Retries on HTTP errors or connection issues.
        Converts to LLMProviderError for known failures.
        """
        try:
            return await coro
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise LLMProviderError(f"vLLM API returned HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            self.logger.error(f"Request failed: {str(e)}")
            raise LLMProviderError(f"Failed to reach vLLM service: {str(e)}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise LLMProviderError(f"Unexpected error: {str(e)}") from e

    async def generate_response(
        self,
        prompt: str
    ) -> str:
        """
        Generate a single text response from vLLM.

        :param prompt: User input prompt
        :param model: Model name (e.g., "meta-llama/Llama-2-7b-chat-hf")
        :param history: Conversation history in OpenAI format
        :param max_tokens: Max tokens to generate
        :param temperature: Sampling temperature
        :param top_p: Nucleus sampling parameter
        :return: Generated text response
        """
        messages = self._build_messages(prompt, self.history)

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False
        }

        async def _make_request():
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        try:
            result = await self._handle_error(_make_request())
            return result
        except LLMProviderError:
            raise
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            raise LLMProviderError(f"Generation failed: {str(e)}") from e

    async def stream_response(
        self,
        prompt: str
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens from vLLM as they are generated.

        Yields each token (or chunk) as a string.

        :param prompt: User input prompt
        :param model: Model name
        :param history: Conversation history
        :param max_tokens: Max tokens to generate
        :param temperature: Sampling temperature
        :param top_p: Nucleus sampling parameter
        :yield: Generated text chunks
        """
        messages = self._build_messages(prompt, self.history)

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True
        }

        async def _stream_request():
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]  # strip "data: "
                        if chunk.strip() == "[DONE]":
                            break
                        try:
                            import json
                            data = json.loads(chunk)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            self.logger.warning(f"Invalid JSON chunk: {chunk}")
                            continue

        try:
            async for chunk in _stream_request():
                yield chunk
        except Exception as e:
            self.logger.error(f"Streaming failed: {str(e)}")
            raise LLMProviderError(f"Streaming failed: {str(e)}") from e
