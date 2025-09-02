"""Contains file processor"""

import os
import logging
from typing import List, Dict

from fastapi import HTTPException
from pydantic import BaseModel

from backend.app.services.embeddings.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class EmbedResponse(BaseModel):
    processed: List[str]
    failed: List[Dict[str, str]]
    message: str


class FileProcessor:
    def __init__(self, upload_dir: str):
        self.upload_dir = upload_dir

    def get_pdf_files(self) -> List[str]:
        if not os.path.exists(self.upload_dir):
            raise HTTPException(status_code=404, detail="Upload directory not found")
        
        pdf_files = [f for f in os.listdir(self.upload_dir) if f.endswith(".pdf")]
        if not pdf_files:
            raise HTTPException(status_code=400, detail="No PDF files to embed")
        
        return pdf_files

    def process_files(self, embedding_service: EmbeddingService) -> EmbedResponse:
        pdf_files = self.get_pdf_files()
        processed = []
        errors = []

        for filename in pdf_files:
            file_path = os.path.join(self.upload_dir, filename)
            try:
                embedding_service.process_file(file_path)
                processed.append(filename)
                logger.info("Successfully processed file: %s", filename)
            except Exception as e:
                error_msg = f"Failed to process {filename}: {str(e)}"
                logger.error(error_msg)
                errors.append({filename: str(e)})

        return EmbedResponse(
            processed=processed,
            failed=errors,
            message=f"Embedded {len(processed)} file(s)"
        )