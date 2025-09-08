# backend/app/routers/embed.py

import logging
import os
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.config import settings
from backend.app.dependencies import get_file_processor, get_embedding_service
from backend.app.services.embeddings.file_processor import FileProcessor
from backend.app.services.embeddings.embedding_service import EmbeddingService
from backend.app.schemas.embedding_schema import EmbedResponse


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/embed", tags=["embed"])


@router.post("/", response_model=EmbedResponse)
def embed_documents(
    file_processor: FileProcessor = Depends(get_file_processor),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> EmbedResponse:
    """
    Process all PDF files in the upload directory and create embeddings.
    
    Returns a summary of processed and failed files.
    """
    return file_processor.process_files(embedding_service)


@router.post("/file/{filename}", response_model=Dict[str, Any])
def embed_specific_file(
    filename: str,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> Dict[str, Any]:
    """
    Process a specific PDF file and create embeddings.
    """
    file_path = os.path.join(settings.UPLOAD_DIR, filename)

    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    
    try:
        embedding_service.process_file(file_path)
        logger.info("Successfully processed file: %s", filename)
        return {
            "filename": filename,
            "status": "success",
            "message": f"Successfully embedded {filename}"
        }
    except Exception as e:
        logger.error("Failed to process %s: %s", filename, str(e))
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process {filename}: {str(e)}"
        )