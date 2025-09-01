import os
import uuid
import logging
import requests
from typing import List

from chromadb import HttpClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def tei_token_length(text: str) -> int:
    if not text or not text.strip():
        return 1

    clean_text = text.strip()

    try:
        response = requests.post(
            settings.EMBEDDING_TOKENIZE_URL,
            json={"inputs": clean_text},
            timeout=5
        )
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list):
            if result and isinstance(result[0], list):  # [[tokens]]
                return len(result[0])
            elif result and isinstance(result[0], dict):  # [{"id":...}, ...]
                return len(result)
        elif isinstance(result, dict):
            return len(result.get("input_ids", []))

        logger.warning("Unexpected tokenize response: %s", result)
    except Exception as e:
        logger.warning("Tokenization failed (using fallback): %s", e)

    # Fallback strategy
    word_count = len(clean_text.split())
    char_estimate = len(clean_text) / 4.0  # ~4 chars per token
    return int(max(word_count, char_estimate)) + 10  # conservative buffer


def load_and_split_pdf(
    file_path: str,
    chunk_size: int = 256,
    chunk_overlap: int = 50
) -> List:
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    loader = PyPDFLoader(file_path, extract_images=False)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tei_token_length,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def get_embeddings_batch(texts: List[str], embedding_api_url: str) -> List[List[float]]:
    response = requests.post(
        embedding_api_url,
        json={"inputs": texts},
        timeout=30
    )
    response.raise_for_status()
    result = response.json()

    # Parse response: support multiple formats
    if isinstance(result, list):
        if result and isinstance(result[0], (int, float)):  # single flat vector
            return [result]
        return result  # assume list of vectors
    elif isinstance(result, dict):
        return result.get("embeddings", []) or result.get("data", [])
    raise ValueError(f"Unknown embedding response format: {result}")


def create_embeddings_for_file(
    file_path: str,
    vector_db_host: str,
    vector_db_port: int,
    collection_name: str,
    embedding_api_url: str,
    chunk_size: int = 256,
    chunk_overlap: int = 50,
    batch_size: int = 16
):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info("Loading and splitting PDF: %s", file_path)
    docs = load_and_split_pdf(file_path, chunk_size, chunk_overlap)
    logger.info("Split into %d chunks", len(docs))

    if not docs:
        logger.warning("No documents extracted from PDF.")
        return

    texts = [doc.page_content for doc in docs]

    # Batched embedding generation
    all_embeddings = []
    logger.info("Generating embeddings in batches of %d", batch_size)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            batch_embs = get_embeddings_batch(batch, embedding_api_url)
            if len(batch_embs) != len(batch):
                logger.warning("Embedding count mismatch: expected %d, got %d", len(batch), len(batch_embs))
            all_embeddings.extend(batch_embs)
        except Exception as e:
            logger.error("Failed to get embeddings for batch %d: %s", i, e)
            raise

    # Connect to Chroma
    client = HttpClient(host=vector_db_host, port=vector_db_port)
    collection = client.get_or_create_collection(name=collection_name)

    # Generate unique IDs
    ids = [str(uuid.uuid4()) for _ in texts]
    metadatas = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", -1),
            "chunk_index": i
        }
        for i, doc in enumerate(docs)
    ]

    collection.add(
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

    logger.info("Successfully added %d documents to Chroma collection '%s'", len(texts), collection_name)