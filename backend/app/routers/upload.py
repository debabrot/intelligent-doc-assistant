# backend/app/routers/upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.services.file_service import save_uploaded_file
from backend.app.core.config import settings


router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        file_path = await save_uploaded_file(file)
        return {"filename": file.filename, "location": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
