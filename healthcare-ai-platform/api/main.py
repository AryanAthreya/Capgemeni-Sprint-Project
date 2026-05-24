# stdlib
import logging
from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# third-party
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# local
from api.middleware.logging_middleware import LoggingMiddleware
from api.routes import agent, ingest, predict, search

logger = logging.getLogger(__name__)

# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown lifecycle.
    Pre-loads the ML model into memory on startup so the first prediction
    request does not incur cold-start latency.
    """
    logger.info("Healthcare AI Platform starting up...")

    # Pre-load ML model
    try:
        from ml.predict import _load_artifacts
        _load_artifacts()
        app.state.model_loaded = True
        logger.info("ML model pre-loaded successfully.")
    except Exception as exc:
        app.state.model_loaded = False
        logger.warning("Could not pre-load ML model: %s", exc)

    # Check DB connectivity
    try:
        from api.database.cosmos_client import get_cosmos_client
        client = get_cosmos_client()
        client._get_container()  # triggers lazy init
        app.state.db_connected = True
        logger.info("Database connection initialised.")
    except Exception as exc:
        app.state.db_connected = False
        logger.warning("Database init failed: %s", exc)

    yield  # Application runs here

    logger.info("Healthcare AI Platform shutting down.")


# ─── App Factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(
        title="Healthcare AI Platform",
        version="1.0.0",
        description=(
            "Intelligent Healthcare Support System with ML triage, "
            "RAG medical Q&A, and patient analytics powered by Azure OpenAI."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ─── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Routers ──────────────────────────────────────────────────────────────
    app.include_router(ingest.router, prefix="/api", tags=["Data Ingestion"])
    app.include_router(predict.router, prefix="/api", tags=["ML Inference"])
    app.include_router(search.router, prefix="/api", tags=["RAG Search"])
    app.include_router(agent.router, prefix="/api", tags=["Multi-Agent"])

    # ─── Health Endpoint ──────────────────────────────────────────────────────
    @app.get(
        "/health",
        summary="Health check",
        tags=["System"],
        response_description="Platform health status",
    )
    async def health(request: Request) -> dict:
        """
        GET /health

        Returns platform health status including model and DB connectivity.
        """
        model_loaded = getattr(request.app.state, "model_loaded", False)
        db_connected = getattr(request.app.state, "db_connected", False)
        return {
            "status": "ok",
            "model_loaded": model_loaded,
            "db_connected": db_connected,
            "version": "1.0.0",
        }

    # ─── Custom Exception Handlers ────────────────────────────────────────────
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle 404 Not Found errors with a structured JSON response."""
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "message": f"The path '{request.url.path}' was not found.",
                "docs": "/docs",
            },
        )

    @app.exception_handler(500)
    async def server_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unhandled 500 errors with a structured JSON response."""
        logger.exception("Unhandled server error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again later.",
            },
        )

    return app


# Module-level app instance (used by uvicorn)
app: FastAPI = create_app()
