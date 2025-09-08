"""Contains file processor"""

import os
import logging
import tempfile
from typing import List, Dict

import boto3
from botocore.exceptions import ClientError
from botocore.client import Config
from fastapi import HTTPException
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.services.embeddings.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class EmbedResponse(BaseModel):
    processed: List[str]
    failed: List[Dict[str, str]]
    message: str


class FileProcessor:
    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=settings.MINIO_ENDPOINT,
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
            verify=False,
        )
        self.bucket_name = settings.MINIO_BUCKET

    def get_pdf_files(self) -> List[str]:
        """List all PDF files in the bucket."""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            if "Contents" not in response:
                raise HTTPException(status_code=400, detail="No PDF files to embed")

            pdf_files = [
                obj["Key"] for obj in response["Contents"]
                if obj["Key"].endswith(".pdf")
            ]

            if not pdf_files:
                raise HTTPException(status_code=400, detail="No PDF files to embed")
            return pdf_files

        except ClientError as e:
            logger.error("S3 error: %s", e)
            raise HTTPException(status_code=500, detail="Error accessing S3 bucket")

    def process_files(self, embedding_service: EmbeddingService) -> EmbedResponse:
        pdf_files = self.get_pdf_files()
        processed = []
        errors = []

        # Create a temporary directory to download files
        with tempfile.TemporaryDirectory() as tmp_dir:
            for filename in pdf_files:
                local_path = os.path.join(tmp_dir, filename)
                try:
                    # Download file from S3/MinIO
                    self.s3_client.download_file(self.bucket_name, filename, local_path)
                    logger.info("Downloaded %s to %s", filename, local_path)

                    # Process with existing embedding service (expects local file path)
                    embedding_service.process_file(local_path)
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

    def delete_file(self, filename: str, embedding_service: EmbeddingService) -> bool:
        """
        Delete a file from S3/MinIO and remove its corresponding embeddings.
        """
        try:
            # 1. Delete embeddings by source (filename)
            embedding_service.vector_store.delete_by_source(filename)

            # 2. Delete file from S3/MinIO
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=filename)
            logger.info("Deleted file '%s' from S3 bucket '%s'", filename, self.bucket_name)

            return True

        except Exception as e:
            error_msg = f"Failed to delete file {filename}: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
