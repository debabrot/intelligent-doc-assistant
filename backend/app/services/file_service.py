
import logging
from io import BytesIO

import boto3
from botocore.client import Config

from backend.app.core.config import settings


logger = logging.getLogger(__name__)


async def save_uploaded_file(file):
    # Create MinIO/S3 client
    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.MINIO_ENDPOINT,
        aws_access_key_id=settings.MINIO_ACCESS_KEY,
        aws_secret_access_key=settings.MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
        verify=False,
    )
    logger.info("File storage initialized")

    bucket = settings.MINIO_BUCKET

    # Ensure bucket exists
    try:
        s3_client.head_bucket(Bucket=bucket)
    except Exception:
        s3_client.create_bucket(Bucket=bucket)
    logger.info("Bucket exists")

    # Read file content
    file_content = await file.read()
    logger.info("file read")

    # Upload
    s3_client.upload_fileobj(
        BytesIO(file_content),
        bucket,
        file.filename,
        ExtraArgs={"ContentType": "application/pdf"}
    )
    logger.info("File loaded")

    # Return public or presigned URL
    # In production, use presigned for private access
    file_url = f"{settings.MINIO_ENDPOINT}/{bucket}/{file.filename}"
    return file_url