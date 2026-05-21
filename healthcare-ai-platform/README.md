# Intelligent Healthcare Support System

> A production-grade multi-agent AI platform for healthcare triage, medical Q&A, and patient analytics — built with Azure OpenAI, LangChain, RandomForest ML, FastAPI, and Streamlit.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INTELLIGENT HEALTHCARE SUPPORT SYSTEM            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │              STREAMLIT FRONTEND (Port 8501)                  │  │
│   │  ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐  │  │
│   │  │ Symptom      │  │ Medical Q&A     │  │ Analytics     │  │  │
│   │  │ Checker      │  │ (MedSearch)     │  │ Dashboard     │  │  │
│   │  └──────┬───────┘  └────────┬────────┘  └──────┬────────┘  │  │
│   └─────────┼───────────────────┼──────────────────┼───────────┘  │
│             │    HTTP/REST       │                  │               │
│   ┌─────────▼───────────────────▼──────────────────▼───────────┐  │
│   │              FASTAPI BACKEND (Port 8000)                     │  │
│   │  POST /api/ingest  POST /api/predict  POST /api/search       │  │
│   │                         POST /api/agent                      │  │
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
│   │  ML Model    │  │  Search       │  │   Patient Records   │   │
│   │  (132 feats) │  │  (RAG Chunks) │  │   (NoSQL)           │   │
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
| Python | 3.11+ |
| pip | 23+ |
| Docker + Docker Compose | Any recent version |
| Azure account | Required for cloud features |
| Git | Any recent version |

---

## Quick Setup (Local — No Azure Required)

### 1. Clone and set up the environment

```bash
git clone <your-repo-url>
cd healthcare-ai-platform
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env — for local dev, you can leave Azure values empty
```

### 3. Generate synthetic patient data

```bash
python pipeline/generate_patients.py
# Creates data/raw/patients.csv with 500 records
```

### 4. Download the Kaggle dataset

- Go to: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
- Download `Training.csv`, `symptom_Description.csv`, `symptom_precaution.csv`
- Place `Training.csv` in `data/raw/Training.csv`

### 5. Run the ML data pipeline and train the model

```bash
python ml/data_pipeline.py
# Outputs: data/staged/Training_cleaned.csv, data/curated/features.parquet

python ml/train.py
# Outputs: models/model.pkl, models/label_encoder.pkl, models/confusion_matrix.png
```

### 6. Start the FastAPI backend

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# API docs available at: http://localhost:8000/docs
```

### 7. Start the Streamlit frontend (in a second terminal)

```bash
streamlit run frontend/app.py
# Frontend available at: http://localhost:8501
```

---

## Running with Azure

### Required Azure Resources

| Resource | Purpose |
|---|---|
| Azure OpenAI | GPT-4o-mini (chat) + text-embedding-3-small (RAG) |
| Azure AI Search | Vector store for medical documents |
| Azure CosmosDB (NoSQL) | Patient record database |
| Azure Blob Storage | Data pipeline storage (raw/staged/curated) |
| Azure Key Vault | Secret management (optional, production) |
| Azure Container Registry | Docker image storage (CI/CD) |
| Azure Web App | Hosting (CI/CD) |

### Set all environment variables in `.env`:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_KEY=your-search-key
COSMOS_ENDPOINT=https://your-cosmos.documents.azure.com:443/
COSMOS_KEY=your-cosmos-key
AZURE_BLOB_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
```

### Ingest PDF medical documents into the vector store

```bash
# Place WHO PDF files in data/pdfs/ then run:
python -c "
from agents.vector_store import get_vector_store
vs = get_vector_store()
vs.create_index_if_not_exists()
count = vs.ingest_pdfs('data/pdfs/')
print(f'Ingested {count} document chunks.')
"
```

---

## Running with Docker

```bash
# Build and start all services
docker compose up --build

# API: http://localhost:8000
# Frontend: http://localhost:8501
# API docs: http://localhost:8000/docs
```

---

## API Documentation

All endpoints accept and return `application/json`.

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

**Response:**
```json
{"patient_id": "uuid-here", "status": "created", "message": "Patient record 'Alice Johnson' ingested successfully."}
```

---

### POST /api/predict — Disease Prediction

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]}'
```

**Response:**
```json
{
  "predictions": [
    {"disease": "Fungal infection", "confidence": 0.82, "risk_level": "high"},
    {"disease": "Allergy", "confidence": 0.11, "risk_level": "low"},
    {"disease": "Drug Reaction", "confidence": 0.04, "risk_level": "low"}
  ],
  "disclaimer": "This is not medical advice..."
}
```

---

### POST /api/search — Semantic Document Search

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is malaria", "top_k": 3}'
```

**Response:**
```json
{
  "results": [
    {"content": "Malaria is caused by...", "source": "who_infectious_diseases.pdf", "score": 0.92}
  ],
  "query": "what is malaria"
}
```

---

### POST /api/agent — Multi-Agent Chat

```bash
curl -X POST http://localhost:8000/api/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "I have fever and headache", "intent": "triage"}'
```

**Response:**
```json
{
  "response": "Based on your symptoms... ⚠️ This assessment is not a medical diagnosis...",
  "agent_used": "Dr. Triage",
  "session_id": "uuid-here"
}
```

**Streaming (SSE):**
```bash
curl -X POST "http://localhost:8000/api/agent?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"message": "how many patients have high risk"}'
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v --asyncio-mode=auto

# Run specific test file
pytest tests/test_predict.py -v --asyncio-mode=auto

# Run with coverage
pytest tests/ --cov=api --cov=ml --cov=agents --cov-report=term-missing
```

---

## Power BI Setup

1. Run the export pipeline:
   ```bash
   python exports/powerbi_export.py
   # Creates: exports/healthcare_powerbi.csv
   ```

2. Open Power BI Desktop → **Get Data → Text/CSV**

3. Select `exports/healthcare_powerbi.csv`

4. Build the following visuals:
   - **Bar Chart:** `diagnosis` (X) vs `count` (Y)
   - **Pie Chart:** `risk_level` distribution
   - **Line Chart:** `date` (X) vs `patient count` (Y)
   - **Stacked Bar:** `age_group` vs `risk_level`

---

## Folder Structure

```
healthcare-ai-platform/
├── api/                    # FastAPI backend
│   ├── main.py             # App factory + lifespan
│   ├── routes/             # 4 REST endpoint routers
│   ├── models/schemas.py   # Pydantic v2 request/response models
│   ├── database/           # CosmosDB client
│   └── middleware/         # Request logging middleware
├── ml/                     # Machine learning pipeline
│   ├── data_pipeline.py    # Raw → Staged → Curated ETL
│   ├── feature_engineering.py  # Symptom encoding
│   ├── train.py            # RandomForest training
│   └── predict.py          # Inference with model caching
├── agents/                 # LangChain multi-agent system
│   ├── orchestrator.py     # Intent routing
│   ├── triage_agent.py     # Dr. Triage (ML + GPT)
│   ├── rag_agent.py        # MedSearch (RAG)
│   ├── analytics_agent.py  # HealthAnalyst (DB + GPT)
│   └── vector_store.py     # Azure AI Search / FAISS
├── pipeline/               # Data pipeline utilities
│   ├── generate_patients.py  # Faker data generator
│   ├── blob_client.py      # Azure Blob upload/download
│   └── adf_trigger.py      # ADF pipeline trigger
├── frontend/               # Streamlit UI
│   ├── app.py              # Main app + navigation
│   ├── pages/              # 3 feature pages
│   └── utils/api_client.py # HTTP client
├── exports/
│   └── powerbi_export.py   # Power BI CSV export
├── tests/                  # Pytest test suite
├── data/                   # Data lake: raw/staged/curated/pdfs
├── models/                 # Saved ML artifacts
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/deploy.yml  # CI/CD
└── requirements.txt
```

---

## GitHub Actions Secrets Required

| Secret | Description |
|---|---|
| `AZURE_CREDENTIALS` | Azure service principal JSON |
| `ACR_NAME` | Azure Container Registry name (without .azurecr.io) |
| `WEBAPP_NAME` | Azure Web App name |
| `AZURE_RESOURCE_GROUP` | Azure resource group name |

---

## Local Development Without Azure

All Azure services have graceful local fallbacks:

| Service | Local Fallback |
|---|---|
| CosmosDB | In-memory dict store |
| Azure AI Search | FAISS in-memory index |
| Azure OpenAI | Rule-based response generation |
| Azure Blob | Local file copy to `data/blob_local_sim/` |

The application runs fully offline. Azure credentials unlock production-quality AI responses.
