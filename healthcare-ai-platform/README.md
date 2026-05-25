# Intelligent Healthcare Support System

> A production-grade multi-agent AI platform for healthcare triage, medical Q&A, and patient analytics — built with Azure OpenAI, LangChain, RandomForest ML, and FastAPI.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INTELLIGENT HEALTHCARE SUPPORT SYSTEM            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │              FASTAPI BACKEND (Port 8000)                     │  │
│   │  POST /api/ingest  POST /api/predict  POST /api/search       │  │
│   │                   POST /api/agent   GET /health              │  │
│   └──────────────────────────┬──────────────────────────────────┘  │
│                              │                                      │
│   ┌──────────────────────────▼──────────────────────────────────┐  │
│   │              LANGCHAIN AGENT ORCHESTRATOR                    │  │
│   │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │  │
│   │  │ Dr. Triage  │  │  MedSearch   │  │  HealthAnalyst  │    │  │
│   │  │   Agent 1   │  │   Agent 2    │  │    Agent 3      │    │  │
│   │  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘    │  │
│   └─────────┼────────────────┼───────────────────┼─────────────┘  │
│             │                │                   │                 │
│   ┌─────────▼────┐  ┌────────▼──────┐  ┌────────▼────────────┐   │
│   │  RandomForest│  │  Azure AI     │  │   Azure CosmosDB    │   │
│   │  ML Model    │  │  Search / FAISS│  │   / In-Memory Store │   │
│   │  (132 feats) │  │  (RAG Chunks) │  │   Patient Records   │   │
│   └──────────────┘  └───────────────┘  └─────────────────────┘   │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  Azure OpenAI: GPT-4o-mini + text-embedding-3-small         │  │
│   │  Azure Blob Storage: raw/ → staged/ → curated/ pipeline     │  │
│   │  Azure Key Vault: Secret management (production)             │  │
│   └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| pip | 23+ |
| Git | Any recent version |
| Docker + Docker Compose | Optional — for containerised run |
| Azure account | Optional — required only for cloud features |

---

## Quick Setup — Run Locally (No Azure Required)

> All Azure services have local fallbacks. The app runs **fully offline** — Azure credentials only unlock GPT-powered responses.

### Step 1 — Clone the Repository

```bash
git clone https://github.com/AryanAthreya/Capgemeni-Sprint-Project
cd healthcare-ai-platform
```

### Step 2 — Create & Activate Virtual Environment

```bash
# Create the venv
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\activate

# Activate (macOS / Linux)
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

*(Note: PyTorch and HuggingFace models are large downloads. Ensure you have a stable internet connection).*

### Step 4 — Configure Environment Variables

```bash
cp .env.example .env
# For local development, all Azure values can remain empty.
# The app will use local fallbacks automatically.
```

### Step 5 — Train the ML Model

The model is not committed to git (it's gitignored). Run this **once** to generate `models/model.pkl`:

```bash
# Windows (.venv not activated)
.venv\Scripts\python.exe train_model.py

# Windows (.venv activated) or macOS/Linux
python train_model.py
```

Expected output:
```
============================================================
  DISEASE PREDICTION — TRAINING PIPELINE
============================================================
[1/7] Loading CSV files...
[2/7] Preprocessing data...
...
  Training Accuracy  : 100.00%
  Test Accuracy      : 100.00%
  Macro F1 Score     : 1.0000
[OK] All artifacts saved to models/
  model.pkl                         11653.5 KB
  label_encoder.pkl                    1.1 KB
  symptom_columns.pkl                  2.3 KB
  severity_map.pkl                     3.4 KB
```

### Step 6 — Populate the RAG Vector Store

The RAG agent uses local HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and Azure AI Search. You must ingest the medical text data from the `data/` folder before the RAG search will work.

```bash
# Windows (.venv activated) or macOS/Linux
PYTHONPATH=. python agents/vector_store.py
```

Expected output:
```
INFO:__main__:Azure AI Search client connected to index: medical-knowledge
INFO:__main__:Uploaded 28 chunks from chronic_diseases.txt to Azure AI Search.
...
INFO:__main__:PDF ingestion complete. Total chunks: 110
```

### Step 7 — Start the FastAPI Backend

```bash
# Windows (.venv not activated)
.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Windows (.venv activated) or macOS/Linux
PYTHONPATH=. uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now live at:
- **Swagger UI (interactive docs):** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/health

### Step 8 — Verify Everything is Working

```bash
# Health check (PowerShell)
Invoke-RestMethod -Uri http://localhost:8000/health

# Health check (curl / bash)
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "db_connected": true,
  "version": "1.0.0"
}
```

---

## Running Tests

The test suite covers all 5 API endpoints with **33 tests** — runs against the live local server.

> Make sure the backend server is running on port 8000 before running tests.

```bash
# Run all tests (Windows — .venv not activated)
.venv\Scripts\python.exe -m pytest tests/test_api_endpoints.py -v

# Run all tests (Windows — .venv activated, or macOS/Linux)
pytest tests/test_api_endpoints.py -v

# Run with short failure traces
pytest tests/test_api_endpoints.py -v --tb=short

# Run a specific test class
pytest tests/test_api_endpoints.py::TestPredict -v

# Run a single test
pytest tests/test_api_endpoints.py::TestPredict::test_predict_diabetes_symptoms -v
```

Expected output:
```
============================= test session starts =============================
collected 33 items

tests/test_api_endpoints.py::TestHealth::test_health_returns_200 PASSED
tests/test_api_endpoints.py::TestHealth::test_health_model_loaded PASSED
tests/test_api_endpoints.py::TestPredict::test_predict_diabetes_symptoms PASSED
tests/test_api_endpoints.py::TestPredict::test_predict_malaria_symptoms PASSED
...
======================= 33 passed in 24.05s ==========================
```

### Test Coverage by Endpoint

| Endpoint | Tests | What's Verified |
|---|---|---|
| `GET /health` | 4 | 200 status, model loaded, version present |
| `POST /api/predict` | 10 | Top-3 results, confidence range, sorted order, error handling, Diabetes & Malaria accuracy |
| `POST /api/ingest` | 6 | Patient creation, UUID generation, validation errors |
| `POST /api/search` | 6 | Results structure, query echo, top_k limit |
| `POST /api/agent` | 7 | Intent routing, session IDs, all 3 agents, error handling |

---

## API Reference

All endpoints accept and return `application/json`. Interactive docs at `/docs`.

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true, "db_connected": true, "version": "1.0.0"}
```

---

### POST /api/predict — Disease Prediction

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]}'
```

```json
{
  "predictions": [
    {"disease": "Fungal infection", "confidence": 0.82, "risk_level": "high"},
    {"disease": "Allergy",          "confidence": 0.11, "risk_level": "low"},
    {"disease": "Drug Reaction",    "confidence": 0.04, "risk_level": "low"}
  ],
  "patient_id": null
}
```

PowerShell equivalent:
```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/predict -Method POST `
  -ContentType "application/json" `
  -Body '{"symptoms": ["fever", "chills", "headache", "muscle_pain"]}'
```

---

### POST /api/ingest — Ingest Patient Record

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Alice Johnson",
    "age": 34,
    "gender": "Female",
    "symptoms": ["fever", "headache", "fatigue"]
  }'
```

```json
{
  "patient_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "created",
  "message": "Patient record 'Alice Johnson' ingested successfully."
}
```

---

### POST /api/search — Semantic Medical Search

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "symptoms of malaria", "top_k": 2}'
```

```json
{
  "results": [
    {"content": "Malaria is a life-threatening disease...", "source": "who_infectious_diseases.pdf", "score": 0.92}
  ],
  "query": "symptoms of malaria"
}
```

---

### POST /api/agent — Multi-Agent Chat

```bash
curl -X POST http://localhost:8000/api/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "I have fever and headache", "intent": "triage"}'
```

```json
{
  "response": "Based on the symptoms you described (fever, headache)...",
  "agent_used": "Dr. Triage",
  "session_id": "uuid-here"
}
```

**Available intents:** `triage` | `search` | `analytics` | *(omit for auto-routing)*

**Streaming (SSE):**
```bash
curl -X POST "http://localhost:8000/api/agent?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"message": "what are the symptoms of dengue"}'
```

---

## Running with Azure (Production)

### Required Azure Resources

| Resource | Purpose |
|---|---|
| Azure OpenAI | GPT-4o-mini (chat) + text-embedding-3-small (embeddings) |
| Azure AI Search | Vector store for RAG medical documents |
| Azure CosmosDB (NoSQL) | Patient record database |
| Azure Blob Storage | Data pipeline storage (raw/staged/curated) |
| Azure Key Vault | Secret management (optional) |

### Set Environment Variables in `.env`

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_API_VERSION=2024-02-01

AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_KEY=your-search-key
AZURE_SEARCH_INDEX_NAME=healthcare-docs

COSMOS_ENDPOINT=https://your-cosmos.documents.azure.com:443/
COSMOS_KEY=your-cosmos-key
COSMOS_DATABASE=healthcare
COSMOS_CONTAINER=patients

AZURE_BLOB_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
```

---

## Running with Docker

```bash
# Build and start
docker compose up --build

# API: http://localhost:8000
# API docs: http://localhost:8000/docs
```

---

## Local Fallbacks (No Azure Needed)

| Azure Service | Local Fallback |
|---|---|
| CosmosDB | In-memory dict store |
| Azure AI Search | Mock results / FAISS index |
| Azure OpenAI | Rule-based response generation |
| Azure Blob | Local file copy to `data/blob_local_sim/` |

---

## Folder Structure

```
healthcare-ai-platform/
├── api/                        # FastAPI backend
│   ├── main.py                 # App factory + lifespan hooks
│   ├── routes/                 # 4 REST endpoint routers
│   │   ├── predict.py          # POST /api/predict
│   │   ├── ingest.py           # POST /api/ingest
│   │   ├── search.py           # POST /api/search
│   │   └── agent.py            # POST /api/agent
│   ├── models/schemas.py       # Pydantic v2 request/response models
│   ├── database/               # CosmosDB client with local fallback
│   └── middleware/             # Request logging middleware
├── ml/                         # Machine learning pipeline
│   ├── dataset.csv             # Disease-symptom training data
│   ├── Symptom-severity.csv    # Symptom severity weights
│   ├── feature_engineering.py  # Symptom encoding helpers
│   └── predict.py              # Inference with model caching
├── agents/                     # LangChain multi-agent system
│   ├── orchestrator.py         # Keyword-based intent routing
│   ├── triage_agent.py         # Dr. Triage (ML + GPT fallback)
│   ├── rag_agent.py            # MedSearch (RAG + GPT fallback)
│   ├── analytics_agent.py      # HealthAnalyst (CosmosDB + GPT fallback)
│   └── vector_store.py         # Azure AI Search / FAISS
├── pipeline/                   # Data utilities
│   ├── generate_patients.py    # Faker synthetic patient generator
│   ├── blob_client.py          # Azure Blob upload/download
│   └── adf_trigger.py          # ADF pipeline trigger
├── tests/
│   └── test_api_endpoints.py   # 33 API endpoint tests (pytest + httpx)
├── models/                     # Saved ML artifacts (gitignored)
│   ├── model.pkl               # Trained RandomForestClassifier
│   ├── label_encoder.pkl       # LabelEncoder
│   ├── symptom_columns.pkl     # Ordered symptom feature list
│   └── severity_map.pkl        # Symptom severity weights
├── train_model.py              # Standalone ML training script (run first!)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## GitHub Actions Secrets Required (CI/CD)

| Secret | Description |
|---|---|
| `AZURE_CREDENTIALS` | Azure service principal JSON |
| `ACR_NAME` | Azure Container Registry name (without .azurecr.io) |
| `WEBAPP_NAME` | Azure Web App name |
| `AZURE_RESOURCE_GROUP` | Azure resource group name |
