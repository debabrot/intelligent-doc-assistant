import os
from fastapi import UploadFile


async def save_uploaded_file(UPLOAD_DIR:str, file: UploadFile) -> str:
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb+") as f:
        f.write(await file.read())
    return file_location