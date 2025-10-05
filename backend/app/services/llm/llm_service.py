import logging
from typing import AsyncGenerator, List, Optional
from fastapi import HTTPException
from backend.app.domain.protocols import LLMProviderProtocol
from backend.app.schemas.chat_schema import ChatResponse


logger = logging.getLogger(__name__)


class LLMService:
    """
    Higher-level LLM operations and prompt management.
    Business logic for different types of generation.
    Prompt templating and context injection.
    Integration point for your chat service.
    Response post-processing.
    """

    def __init__(self, llm_provider: LLMProviderProtocol):
        self.llm_provider = llm_provider
        self.system_prompt_template = (
            "You are a helpful, respectful, and honest AI assistant. "
            "Always answer as helpfully as possible, while being safe. "
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
            "If a question does not make sense or is not factually coherent, explain why instead of answering something incorrect. "
            "If you don't know the answer, do not make up an answer.\n\n"
            "{context_section}"
            "User: {user_prompt}\n"
            "Assistant:"
        )

    async def chat_completion(
        self,
        user_prompt: str,
        context_docs: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> ChatResponse:
        """
        Main chat method with context management.
        """
        try:

            if context_docs:
                context_text = self._format_context(context_docs)
                user_prompt = f"{context_text}\n\nUser Question: {user_prompt}"

            # Pass simple system prompt to provider
            if system_prompt is None:
                system_prompt = "You are a helpful AI assistant. Provide accurate, concise answers."

            raw_response = await self.llm_provider.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt
            )

            # Validate and post-process
            validated_response = self._validate_response(raw_response)

            return ChatResponse(response=validated_response)

        except Exception as e:
            logger.error(f"Error in chat_completion: {str(e)}")
            raise HTTPException(status_code=500, detail="LLM generation failed")

    async def stream_chat_completion(
        self,
        user_prompt: str,
        context_docs: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streaming version of chat completion.
        Yields tokens or chunks as they are generated.
        """
        try:
            prompt = self._apply_prompt_template(
                user_prompt=user_prompt,
                context_docs=context_docs,
                system_prompt=system_prompt
            )

            async for chunk in self.llm_provider.stream_response(prompt=prompt):
                # Optional: validate/filter each chunk if needed
                filtered_chunk = self._validate_response_chunk(chunk)
                if filtered_chunk:  # Only yield non-empty/valid chunks
                    yield filtered_chunk

        except Exception as e:
            logger.error(f"Error in stream_chat_completion: {str(e)}")
            raise HTTPException(status_code=500, detail="Streaming LLM failed")

    async def generate_with_context(
        self,
        user_prompt: str,
        retrieved_docs: List[str],
        system_prompt: Optional[str] = None
    ) -> ChatResponse:
        """
        RAG-specific generation with retrieved documents injected as context.
        """
        if not retrieved_docs:
            logger.warning("generate_with_context called with empty retrieved_docs")
            # Optionally fall back to regular chat without context
            return await self.chat_completion(user_prompt, system_prompt=system_prompt)

        # Inject context into prompt
        context_section = self._format_context(retrieved_docs)
        return await self.chat_completion(
            user_prompt=user_prompt,
            context_docs=retrieved_docs,
            system_prompt=system_prompt
        )

    def _apply_prompt_template(
        self,
        user_prompt: str,
        context_docs: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Apply system prompts and formatting.
        Injects context if provided.
        """
        if system_prompt is None:
            system_prompt = self.system_prompt_template

        context_section = ""
        if context_docs:
            context_section = self._format_context(context_docs)

        # Format the full prompt
        formatted_prompt = system_prompt.format(
            context_section=context_section,
            user_prompt=user_prompt
        )

        logger.debug(f"Formatted prompt: {formatted_prompt[:500]}...")  # Log first 500 chars
        return formatted_prompt

    def _format_context(self, docs: List[str]) -> str:
        """
        Format retrieved documents into a context section for the prompt.
        """
        if not docs:
            return ""

        context_lines = ["Relevant context from retrieved documents:"]
        for i, doc in enumerate(docs, 1):
            context_lines.append(f"[Doc {i}]: {doc.strip()}")

        context_lines.append("\n---\n")
        return "\n".join(context_lines)

    def _validate_response(self, response: str) -> str:
        """
        Response validation and filtering.
        - Strip whitespace
        - Basic safety/content filtering (extend as needed)
        - Placeholder for moderation hooks
        """
        if not isinstance(response, str):
            raise ValueError("Response must be a string")

        cleaned = response.strip()

        # Add basic safety filters (customize based on your needs)
        unsafe_patterns = [
            "<script", "javascript:", "eval(", "base64",
            "rm -rf", "; DROP", "sudo", "chmod", "passwd"
        ]

        for pattern in unsafe_patterns:
            if pattern and pattern.lower() in cleaned.lower():
                logger.warning(f"Unsafe pattern '{pattern}' found in response: {cleaned[:100]}")
                cleaned = "[Filtered due to safety policy]"
                break

        if len(cleaned) == 0:
            cleaned = "I'm sorry, I couldn't generate a valid response."

        return cleaned

    def _validate_response_chunk(self, chunk: str) -> str:
        """
        Validate/filter individual streaming chunks.
        """
        # For now, same logic as full response, but you can make it lighter
        try:
            return self._validate_response(chunk)
        except Exception:
            return ""  # Suppress invalid chunks

    async def close(self):
        """
        Gracefully close underlying LLM provider connection.
        """
        await self.llm_provider.close()