# backend/app/main.py
from fastapi import FastAPI
from backend.app.routers.files import router as files_router
from backend.app.routers.embed import router as embed_router
from backend.app.routers.chat import router as chat_router
from backend.app.utils.logger import setup_logging


# Set up logging
setup_logging()

app = FastAPI(title="RAG Ingestion Microservice")

app.include_router(files_router)
app.include_router(embed_router)
app.include_router(chat_router)

@app.get("/")
def read_root():
    return {"message": "RAG Ingestion Service is running", "endpoints": ["/files", "/embed"]}