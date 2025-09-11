"""Contains chat endpoints"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from backend.app.dependencies import get_retriever
from backend.app.schemas.chat_schema import (
    RetrieveRequest,
    RetrieveResponse,
    RetrievedChunk
)
from backend.app.services.embeddings.retriever import RetrieverService
from backend.app.schemas.chat_schema import ChunkMetadata


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    request: RetrieveRequest,
    retriever: RetrieverService = Depends(get_retriever)
):
    try:
        top_k = max(1, min(request.top_k, 100))
        logger.info(f"Retrieving {top_k} chunks for query: {request.query[:100]}...")

        doc_chunks = retriever.retrieve(request.query, top_k)

        retrieved_chunks = [
            RetrievedChunk(
                id=chunk.id,
                content=chunk.content,
                metadata=ChunkMetadata(**chunk.metadata),
                similarity=None
            )
            for chunk in doc_chunks
        ]

        return RetrieveResponse(
            query=request.query,
            top_k=top_k,
            chunks=retrieved_chunks
        )

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve results. Please try again later."
        )