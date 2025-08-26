# backend/app/routers/embed.py
from fastapi import APIRouter, HTTPException
from backend.app.services.embedding_service import create_embeddings_for_file
from backend.app.core.config import settings
import os

router = APIRouter(prefix="/embed", tags=["embed"])


@router.post("/")
def embed_documents():
    if not os.path.exists(settings.UPLOAD_DIR):
        raise HTTPException(status_code=404, detail="Upload directory not found")

    pdf_files = [f for f in os.listdir(settings.UPLOAD_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        raise HTTPException(status_code=400, detail="No PDF files to embed")

    processed = []
    errors = []

    for filename in pdf_files:
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        try:
            create_embeddings_for_file(
                file_path,
                vector_db_host=settings.CHROMA_HOST,
                vector_db_port=settings.CHROMA_PORT,
                collection_name=settings.COLLECTION_NAME,
                embedding_api_url=settings.EMBEDDING_API_URL
                )
            processed.append(filename)
        except Exception as e:
            errors.append({filename: str(e)})

    return {
        "processed": processed,
        "failed": errors,
        "message": f"Embedded {len(processed)} file(s)"
    }