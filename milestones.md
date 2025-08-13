## 🗓️ Week-by-Week Plan

### 🔹 **Week 1: Project Setup & Ingestion MVP**

**Goal**: Load a PDF and store chunks in Chroma.

**Tasks**:
- Set up Python environment (venv, pip)
- Install: `langchain`, `chromadb`, `unstructured`, `PyPDF2`, `sentence-transformers`
- Write script to:
  - Load a PDF
  - Split into chunks
  - Embed with `all-MiniLM-L6-v2`
  - Store in Chroma (persistent mode)

**Deliverable**: `ingestion-demo.py` that ingests `sample.pdf` → `vector_db/chroma/`

**Validation**:
```python
retriever.similarity_search("What is the API key policy?")
```
→ Returns relevant chunks

---

### 🔹 **Week 2: Build Ingestion Service (FastAPI + Upload)**

**Goal**: Turn ingestion into a web service with file upload.

**Tasks**:
- Create `services/ingestion-service/`
- Build FastAPI app with:
  - `POST /upload` → saves file to `data/raw/`
  - Sync indexing (for now, sync processing)
- Add basic file validation (PDF, MD, TXT)

**Deliverable**: Running FastAPI server at `http://localhost:8001/upload`

**Validation**:
```bash
curl -X POST -F "file=@sample.pdf" http://localhost:8001/upload
```
→ File appears in `data/raw/`, chunks in Chroma

---

### 🔹 **Week 3: Add Async Processing & Chunking Strategies**

**Goal**: Make ingestion async and support multiple formats.

**Tasks**:
- Add **Celery + Redis** (use docker-compose)
- Move processing to `tasks/process_document.py`
- Support:
  - PDF (text + OCR for scanned)
  - Markdown (with header splitting)
  - TXT
- Add meta `source`, `page`, `type`

**Deliverable**: Upload → async job → processed → indexed

**Validation**:
- Upload scanned PDF → OCR extracts text → indexed
- Upload MD → preserves `# API Docs` headers in metadata

---

### 🔹 **Week 4: Build Query Service & LLM Integration**

**Goal**: Answer questions using retrieved context.

**Tasks**:
- Create `services/query-service/`
- Build:
  - `POST /query` → uses Chroma + LLM
  - Use **Ollama + Llama 3** for generation
  - Simple retrieval → prompt → answer
- Test with real questions

**Deliverable**: `http://localhost:8000/query` returns answers

**Validation**:
```json
POST /query
{"question": "How do I reset my password?"}
→ "To reset your password, go to Settings > Account..."
```

---

### 🔹 **Week 5: Add Advanced RAG Features**

**Goal**: Go beyond basic RAG with hybrid search, re-ranking, and query rewrite.

**Tasks**:
- Implement **Hybrid Search**:
  - Vector (Chroma) + BM25 (rank-bm25)
  - Combine with **RRF (Reciprocal Rank Fusion)**
- Add **Re-Ranking**:
  - Use `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Add **Query Transformation**:
  - HyDE: generate hypothetical answer → embed
  - Or simple rewrite: “How to reset?” → “steps to reset password”

**Deliverable**: Better retrieval → better answers

**Validation**:
- Query: “What’s the auth flow?” → retrieves OAuth diagram + text
- Before: 2/5 relevant chunks → After: 5/5

---

### 🔹 **Week 6: Frontend, Feedback & Deployment**

**Goal**: Ship a usable, monitored system.

**Tasks**:
- Build `ui/app.py` with **Gradio**:
  - Chat interface
  - File upload
  - Thumbs up/down
- Add `POST /feedback` endpoint
- Log feedback to `feedback.log` or SQLite
- Dockerize both services
- Deploy to **Fly.io** or **Railway**

**Deliverable**: Publicly accessible demo!

**Validation**:
- Share link with a friend → they upload a doc → ask a question → get answer
- You see feedback logged

---

## 🚀 Bonus (Week 7–8, Optional)

| Feature | Value |
|-------|-------|
| **Self-Correction** | Analyze low-rated queries → improve chunking/query rewrite |
| **Monitoring** | Prometheus + Grafana: track latency, retrieval quality |
| **CI/CD** | GitHub Actions: test + deploy on push |
| **Multi-Modal** | Extract images from PDF → embed with CLIP → retrieve on “see diagram” |
| **Replace Ollama → Bedrock** | Use `BedrockLLM` provider in production |

---

# 📁 Final Project Structure (Updated)

```bash
intelligent-doc-assistant/
├── services/
│   ├── query-service/          # FastAPI: /query, /feedback
│   └── ingestion-service/      # FastAPI: /upload, async processing
├── shared/                     # Reusable models, utils, vectorstore
├── ui/                         # ✅ Frontend: Gradio app
│   └── app.py
├── data/
│   ├── raw/                    # Uploaded files
│   └── processed/              # OCR'd, chunked
├── vector_db/chroma/           # Persistent vector store
├── prefect_flows/              # Optional: sync from GitHub
├── monitoring/                 # Prometheus/Grafana configs
├── docker-compose.yml
├── requirements.txt
└── README.md                   # Project overview, how to run
```

---

# 🛠️ Tools You’ll Use

| Purpose | Tool |
|-------|------|
| Backend | FastAPI |
| Frontend | Gradio |
| LLM | Ollama (Llama 3) |
| Embedding | Sentence Transformers |
| Vector DB | Chroma |
| Async | Celery + Redis |
| Container | Docker + docker-compose |
| Hosting | Fly.io / Railway (free tier) |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana Cloud (free) |

---

# 🏁 Final Deliverables (Portfolio-Ready)

By the end, you’ll have:

✅ A **working RAG system** with advanced features  
✅ A **modular microservices architecture**  
✅ A **public demo** (Gradio + hosted API)  
✅ **Clean, documented code** on GitHub  
✅ **MLOps practices**: logging, monitoring, CI/CD  
✅ A **README** with screenshots, architecture diagram, and setup guide

---

# 💡 Pro Tip: Document Your Journey

As you build:
- Take **screenshots** of UI, retrieval results
- Record a **short Loom video** demo
- Write a **blog post** or LinkedIn post: “How I built an AI document assistant”

This will **amplify your visibility** and help you land interviews.

---

## FULL DIRECTORY STRUCTURE
intelligent-doc-assistant/
│
├── services/
│   ├── query-service/                 # FastAPI: handles /query
│   │   ├── main.py
│   │   ├── routers/
│   │   │   ├── query_router.py
│   │   │   └── feedback_router.py
│   │   ├── services/
│   │   │   ├── retrieval_service.py   # Hybrid search, re-ranking
│   │   │   └── llm_service.py         # LLM abstraction
│   │   ├── models/
│   │   │   ├── request.py
│   │   │   └── response.py
│   │   ├── config/
│   │   │   ├── settings.py            # Config via env vars
│   │   │   └── constants.py
│   │   ├── utils/
│   │   │   ├── logger.py
│   │   │   └── metrics.py
│   │   └── Dockerfile
│   │
│   ├── ingestion-service/             # FastAPI: handles /upload
│   │   ├── main.py
│   │   ├── routers/
│   │   │   └── upload_router.py
│   │   ├── services/
│   │   │   ├── ingestion_service.py   # Load, chunk, embed
│   │   │   └── indexing_service.py    # Update vector DB
│   │   ├── workers/
│   │   │   └── celery_worker.py       # Async processing
│   │   ├── tasks/
│   │   │   └── process_document.py    # Celery task
│   │   ├── config/
│   │   │   └── storage.py             # S3, local, etc.
│   │   └── Dockerfile
│   │
│   └── llm-service/ (optional future) # Unified LLM API
│       └── ...
│
├── shared/
│   ├── vectorstore/                   # Chroma wrapper
│   │   ├── client.py
│   │   └── hybrid_retriever.py
│   ├── models/                        # Shared embedding/LLM logic
│   │   ├── embeddings.py              # BGE, MiniLM
│   │   ├── reranker.py                # Cross-encoder
│   │   └── multimodal.py              # CLIP, OCR
│   ├── schemas/                       # Pydantic models
│   │   ├── document.py
│   │   └── chunk.py
│   └── utils/                         # Shared helpers
│       ├── file_utils.py
│       └── monitoring.py
│
├── data/
│   ├── raw/                           # Uploaded files
│   └── processed/                     # Chunked, OCR'd
│
├── vector_db/
│   └── chroma/                        # Persistent Chroma DB
│
├── prefect_flows/
│   └── scheduled_ingest.py            # Daily sync from GitHub, etc.
│
├── monitoring/
│   ├── prometheus/
│   └── grafana/                       # Dashboard config
│
├── docker-compose.yml                 # Or Kubernetes later
├── requirements.txt
└── README.md

## QWEN CHAT LINK
https://chat.qwen.ai/c/be8c44c9-8ab1-4b0c-93f3-c50114cf33f3
