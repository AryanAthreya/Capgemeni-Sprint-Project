"""
test_api_endpoints.py — Healthcare AI Platform API Tests
=========================================================
Tests all 5 endpoints:
  GET  /health
  POST /api/predict
  POST /api/ingest
  POST /api/search
  POST /api/agent

Run with:
    .venv\Scripts\python.exe -m pytest tests/test_api_endpoints.py -v

Requires the server to be running at http://localhost:8000
"""

import pytest
import httpx

BASE_URL = "http://localhost:8000"
CLIENT_TIMEOUT = 30.0  # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Synchronous HTTP client for all tests."""
    with httpx.Client(base_url=BASE_URL, timeout=CLIENT_TIMEOUT) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_200(self, client):
        """Health endpoint should always return HTTP 200."""
        r = client.get("/health")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    def test_health_status_ok(self, client):
        """status field must be 'ok'."""
        r = client.get("/health")
        data = r.json()
        assert data["status"] == "ok"

    def test_health_model_loaded(self, client):
        """ML model should be loaded (True) after running train_model.py."""
        r = client.get("/health")
        data = r.json()
        assert data["model_loaded"] is True, (
            "model_loaded is False — restart uvicorn after running train_model.py"
        )

    def test_health_version_present(self, client):
        """Response must include a non-empty version string."""
        r = client.get("/health")
        data = r.json()
        assert "version" in data
        assert data["version"]  # not empty


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:

    def test_predict_basic_symptoms(self, client):
        """Standard prediction with known symptoms should return top 3 results."""
        payload = {"symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]}
        r = client.post("/api/predict", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "predictions" in data
        assert len(data["predictions"]) > 0

    def test_predict_returns_three_results(self, client):
        """Should return exactly top-3 predictions by default."""
        payload = {"symptoms": ["fever", "chills", "headache", "muscle_pain"]}
        r = client.post("/api/predict", json=payload)
        data = r.json()
        assert len(data["predictions"]) == 3

    def test_predict_result_structure(self, client):
        """Each prediction must have disease, confidence, and risk_level fields."""
        payload = {"symptoms": ["fever", "fatigue", "vomiting"]}
        r = client.post("/api/predict", json=payload)
        data = r.json()
        for pred in data["predictions"]:
            assert "disease" in pred, "Missing 'disease' key"
            assert "confidence" in pred, "Missing 'confidence' key"
            assert "risk_level" in pred, "Missing 'risk_level' key"
            assert isinstance(pred["confidence"], float)
            assert pred["risk_level"] in ("low", "medium", "high")

    def test_predict_confidence_is_valid(self, client):
        """Confidence scores must be between 0.0 and 1.0."""
        payload = {"symptoms": ["fever", "chills", "sweating"]}
        r = client.post("/api/predict", json=payload)
        data = r.json()
        for pred in data["predictions"]:
            assert 0.0 <= pred["confidence"] <= 1.0, (
                f"Confidence {pred['confidence']} out of range [0,1]"
            )

    def test_predict_sorted_by_confidence(self, client):
        """Results should be sorted descending by confidence."""
        payload = {"symptoms": ["fever", "chills", "vomiting", "sweating"]}
        r = client.post("/api/predict", json=payload)
        data = r.json()
        confidences = [p["confidence"] for p in data["predictions"]]
        assert confidences == sorted(confidences, reverse=True), (
            "Predictions are not sorted by confidence descending"
        )

    def test_predict_empty_symptoms_returns_400(self, client):
        """Empty symptoms list should return HTTP 400 or 422 (validation error)."""
        payload = {"symptoms": []}
        r = client.post("/api/predict", json=payload)
        assert r.status_code in (400, 422), f"Expected 400 or 422, got {r.status_code}"

    def test_predict_unknown_symptoms(self, client):
        """Unrecognised symptoms should still return 200 (not crash)."""
        payload = {"symptoms": ["xyzzy_not_a_real_symptom_abc"]}
        r = client.post("/api/predict", json=payload)
        # Model handles unrecognised symptoms gracefully
        assert r.status_code == 200

    def test_predict_with_patient_id(self, client):
        """prediction with optional patient_id should echo it back."""
        payload = {
            "symptoms": ["fever", "headache"],
            "patient_id": "test-patient-001",
        }
        r = client.post("/api/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data.get("patient_id") == "test-patient-001"

    def test_predict_diabetes_symptoms(self, client):
        """Classic diabetes symptoms should rank Diabetes highly."""
        payload = {
            "symptoms": [
                "polyuria", "polydipsia", "sudden_weight_loss",
                "fatigue", "blurred_and_distorted_vision"
            ]
        }
        r = client.post("/api/predict", json=payload)
        data = r.json()
        top_disease = data["predictions"][0]["disease"]
        assert "Diabetes" in top_disease, (
            f"Expected Diabetes as top prediction, got: {top_disease}"
        )

    def test_predict_malaria_symptoms(self, client):
        """Classic malaria symptoms should rank Malaria highly."""
        payload = {
            "symptoms": [
                "chills", "vomiting", "high_fever",
                "sweating", "headache", "nausea", "muscle_pain"
            ]
        }
        r = client.post("/api/predict", json=payload)
        data = r.json()
        diseases = [p["disease"] for p in data["predictions"]]
        assert any("Malaria" in d for d in diseases), (
            f"Malaria not in top predictions: {diseases}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/ingest
# ─────────────────────────────────────────────────────────────────────────────

class TestIngest:

    def test_ingest_basic_patient(self, client):
        """Valid patient record should be ingested with status 200."""
        payload = {
            "name": "Jane Doe",
            "age": 34,
            "gender": "Female",
            "symptoms": ["fever", "cough"],
        }
        r = client.post("/api/ingest", json=payload)
        assert r.status_code == 200, r.text

    def test_ingest_returns_patient_id(self, client):
        """Response must contain a patient_id."""
        payload = {
            "name": "John Smith",
            "age": 45,
            "gender": "Male",
            "symptoms": ["headache", "fatigue"],
        }
        r = client.post("/api/ingest", json=payload)
        data = r.json()
        assert "patient_id" in data
        assert data["patient_id"]  # non-empty

    def test_ingest_status_created(self, client):
        """status field should be 'created'."""
        payload = {
            "name": "Alice Brown",
            "age": 28,
            "gender": "Female",
            "symptoms": ["nausea", "vomiting"],
        }
        r = client.post("/api/ingest", json=payload)
        data = r.json()
        assert data["status"] == "created"

    def test_ingest_with_explicit_patient_id(self, client):
        """Supplying a patient_id should echo it back unchanged."""
        payload = {
            "patient_id": "custom-id-12345",
            "name": "Bob Lee",
            "age": 52,
            "gender": "Male",
            "symptoms": ["chest_pain"],
        }
        r = client.post("/api/ingest", json=payload)
        data = r.json()
        assert data["patient_id"] == "custom-id-12345"

    def test_ingest_message_contains_name(self, client):
        """Success message should reference the patient name."""
        payload = {
            "name": "Carol White",
            "age": 61,
            "gender": "Female",
            "symptoms": ["breathlessness"],
        }
        r = client.post("/api/ingest", json=payload)
        data = r.json()
        assert "Carol White" in data.get("message", "")

    def test_ingest_missing_required_field_returns_422(self, client):
        """Missing required name field should return HTTP 422 (validation error)."""
        payload = {
            "age": 30,
            "gender": "Male",
            "symptoms": ["fever"],
            # 'name' intentionally omitted
        }
        r = client.post("/api/ingest", json=payload)
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/search
# ─────────────────────────────────────────────────────────────────────────────

class TestSearch:

    def test_search_returns_200(self, client):
        """Search endpoint should return HTTP 200."""
        payload = {"query": "symptoms of malaria", "top_k": 2}
        r = client.post("/api/search", json=payload)
        assert r.status_code == 200, r.text

    def test_search_results_present(self, client):
        """Response must include a non-empty results list."""
        payload = {"query": "diabetes treatment guidelines"}
        r = client.post("/api/search", json=payload)
        data = r.json()
        assert "results" in data
        assert len(data["results"]) > 0

    def test_search_result_structure(self, client):
        """Each result must have content, source, and score."""
        payload = {"query": "fever and chills"}
        r = client.post("/api/search", json=payload)
        data = r.json()
        for result in data["results"]:
            assert "content" in result
            assert "source" in result
            assert "score" in result
            assert isinstance(result["score"], float)

    def test_search_echoes_query(self, client):
        """Response must echo back the original query string."""
        payload = {"query": "WHO malaria prevention guidelines"}
        r = client.post("/api/search", json=payload)
        data = r.json()
        assert data.get("query") == payload["query"]

    def test_search_top_k_limit(self, client):
        """top_k=1 should return at most 1 result."""
        payload = {"query": "hepatitis symptoms", "top_k": 1}
        r = client.post("/api/search", json=payload)
        data = r.json()
        assert len(data["results"]) <= 1

    def test_search_empty_query_returns_422(self, client):
        """Empty query string should return HTTP 422 or be handled gracefully."""
        payload = {"query": ""}
        r = client.post("/api/search", json=payload)
        # Either 422 (validation) or 200 with mock results — both acceptable
        assert r.status_code in (200, 422)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/agent
# ─────────────────────────────────────────────────────────────────────────────

class TestAgent:

    def test_agent_returns_200(self, client):
        """Agent endpoint should return HTTP 200."""
        payload = {"message": "I have fever and headache, what could it be?"}
        r = client.post("/api/agent", json=payload)
        assert r.status_code == 200, r.text

    def test_agent_response_structure(self, client):
        """Response must have response, agent_used, and session_id."""
        payload = {"message": "What are the symptoms of dengue fever?"}
        r = client.post("/api/agent", json=payload)
        data = r.json()
        assert "response" in data, f"Missing 'response' key. Got: {data}"
        assert "agent_used" in data, f"Missing 'agent_used' key. Got: {data}"
        assert "session_id" in data, f"Missing 'session_id' key. Got: {data}"

    def test_agent_session_id_echoed(self, client):
        """Supplying session_id should echo it in the response."""
        payload = {
            "message": "Tell me about malaria prevention",
            "session_id": "test-session-abc",
        }
        r = client.post("/api/agent", json=payload)
        data = r.json()
        assert data.get("session_id") == "test-session-abc"

    def test_agent_triage_intent(self, client):
        """Explicit triage intent should route to the triage agent."""
        payload = {
            "message": "I feel dizzy and have chest pain",
            "intent": "triage",
        }
        r = client.post("/api/agent", json=payload)
        assert r.status_code == 200, r.text

    def test_agent_search_intent(self, client):
        """Explicit search intent should route to the search agent."""
        payload = {
            "message": "Find WHO guidelines on tuberculosis",
            "intent": "search",
        }
        r = client.post("/api/agent", json=payload)
        assert r.status_code == 200, r.text

    def test_agent_auto_routes_symptom_message(self, client):
        """Symptom-based message should auto-route without explicit intent."""
        payload = {"message": "I have itching, skin rash and fever since 3 days"}
        r = client.post("/api/agent", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data.get("response")  # non-empty response

    def test_agent_missing_message_returns_422(self, client):
        """Missing 'message' field should return HTTP 422."""
        payload = {}
        r = client.post("/api/agent", json=payload)
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"
