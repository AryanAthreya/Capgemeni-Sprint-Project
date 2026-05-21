# stdlib
import logging
import os
from typing import Any, Dict, Generator, List, Optional

# third-party
import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 30.0


class HealthcareAPIClient:
    """
    HTTP client for the Healthcare AI Platform FastAPI backend.
    All methods use httpx for synchronous HTTP calls.
    """

    def __init__(self, base_url: str = _DEFAULT_BASE_URL) -> None:
        """
        Initialise the API client.

        Args:
            base_url: Base URL of the FastAPI backend.
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=_TIMEOUT)

    def health_check(self) -> Dict[str, Any]:
        """
        GET /health — Check backend health status.

        Returns:
            Dict with status, model_loaded, db_connected fields.
        """
        try:
            response = self._client.get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error("Health check failed: %s", exc)
            return {"status": "error", "model_loaded": False, "db_connected": False}

    def ingest_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /api/ingest — Ingest a patient record.

        Args:
            patient_data: Dict matching PatientIngest schema.

        Returns:
            Dict with patient_id, status, message.
        """
        try:
            response = self._client.post("/api/ingest", json=patient_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Ingest error %d: %s", exc.response.status_code, exc.response.text)
            return {"error": exc.response.text, "status_code": exc.response.status_code}
        except Exception as exc:
            logger.error("Ingest request failed: %s", exc)
            return {"error": str(exc)}

    def predict(self, symptoms: List[str], patient_id: Optional[str] = None) -> Dict[str, Any]:
        """
        POST /api/predict — Predict diseases from a symptom list.

        Args:
            symptoms: List of symptom strings.
            patient_id: Optional patient ID to link to the prediction.

        Returns:
            Dict with predictions list and disclaimer.
        """
        payload: Dict[str, Any] = {"symptoms": symptoms}
        if patient_id:
            payload["patient_id"] = patient_id

        try:
            response = self._client.post("/api/predict", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Predict error %d: %s", exc.response.status_code, exc.response.text)
            return {"error": exc.response.text, "status_code": exc.response.status_code}
        except Exception as exc:
            logger.error("Predict request failed: %s", exc)
            return {"error": str(exc)}

    def search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        POST /api/search — Semantic document search.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            Dict with results list and query echo.
        """
        try:
            response = self._client.post(
                "/api/search", json={"query": query, "top_k": top_k}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Search error %d: %s", exc.response.status_code, exc.response.text)
            return {"error": exc.response.text}
        except Exception as exc:
            logger.error("Search request failed: %s", exc)
            return {"error": str(exc)}

    def chat_with_agent(
        self,
        message: str,
        session_id: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        POST /api/agent — Send a message to the multi-agent system.

        Args:
            message: User message string.
            session_id: Optional conversation session ID.
            intent: Optional intent hint ("triage", "medical_info", "analytics").

        Returns:
            Dict with response, agent_used, session_id.
        """
        payload: Dict[str, Any] = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        if intent:
            payload["intent"] = intent

        try:
            response = self._client.post("/api/agent", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Agent error %d: %s", exc.response.status_code, exc.response.text)
            return {
                "error": exc.response.text,
                "response": f"Error: {exc.response.status_code}",
                "agent_used": "Error",
                "session_id": session_id or "",
            }
        except Exception as exc:
            logger.error("Agent request failed: %s", exc)
            return {
                "error": str(exc),
                "response": "Backend unavailable. Please check the server is running.",
                "agent_used": "Error",
                "session_id": session_id or "",
            }

    def chat_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        POST /api/agent?stream=true — Stream agent response via SSE.

        Args:
            message: User message string.
            session_id: Optional conversation session ID.

        Yields:
            Text chunks as they arrive.
        """
        payload: Dict[str, Any] = {"message": message}
        if session_id:
            payload["session_id"] = session_id

        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/agent",
                params={"stream": "true"},
                json=payload,
                timeout=60.0,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        if not chunk.startswith("[ERROR]"):
                            yield chunk
        except Exception as exc:
            logger.error("Stream request failed: %s", exc)
            yield f"[Stream error: {exc}]"

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()
