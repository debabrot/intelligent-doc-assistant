"""Contains endpoints to upload & delete file"""

import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Path

from backend.app.services.file_service import save_uploaded_file
from backend.app.services.embeddings.file_processor import FileProcessor
from backend.app.services.embeddings.embedding_service import EmbeddingService
from backend.app.dependencies import get_file_processor, get_embedding_service

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/files", tags=["upload"])


@router.post("/", summary="Upload a PDF file")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        file_path = await save_uploaded_file(file)
        return {"filename": file.filename, "location": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@router.delete("/{filename}", summary="Delete a file and its embeddings")
async def delete_file(
    filename: str = Path(..., description="Name of the file to delete (e.g., 'report.pdf')"),
    file_processor: FileProcessor = Depends(get_file_processor),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Deletes a file from storage and removes all associated embeddings.
    """
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files can be deleted")
    try:
        success = file_processor.delete_file(filename, embedding_service)

        if not success:
            raise HTTPException(status_code=500, detail="Deletion failed unexpectedly")

        return {
            "message": f"File '{filename}' and its embeddings deleted successfully",
            "filename": filename
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error("Unexpected error deleting file %s: %s", filename, str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
