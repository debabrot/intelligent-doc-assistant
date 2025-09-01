# backend/app/main.py
from fastapi import FastAPI
from backend.app.routers.upload import router as upload_router
from backend.app.routers.embed import router as embed_router
from backend.app.utils.logger import setup_logging


# Set up logging
setup_logging()

app = FastAPI(title="RAG Ingestion Microservice")

app.include_router(upload_router)
app.include_router(embed_router)

@app.get("/")
def read_root():
    return {"message": "RAG Ingestion Service is running", "endpoints": ["/upload", "/embed"]}