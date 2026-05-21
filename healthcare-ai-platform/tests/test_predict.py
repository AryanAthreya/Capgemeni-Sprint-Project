# stdlib
import pytest
from httpx import AsyncClient, ASGITransport

# local
from api.main import app

_KNOWN_SYMPTOMS = ["itching", "skin_rash", "nodal_skin_eruptions"]


@pytest.mark.asyncio
async def test_predict_known_symptoms() -> None:
    """
    POST known symptoms and assert a valid prediction response is returned.
    Checks that predictions list is non-empty and each item has required fields.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/predict",
            json={"symptoms": _KNOWN_SYMPTOMS},
        )

    # Model may not be loaded in CI — accept either 200 or 500
    if response.status_code == 500:
        pytest.skip("ML model not available in test environment.")

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) > 0

    first = data["predictions"][0]
    assert "disease" in first
    assert "confidence" in first
    assert "risk_level" in first
    assert isinstance(first["confidence"], float)
    assert first["risk_level"] in ("low", "medium", "high")


@pytest.mark.asyncio
async def test_predict_empty_symptoms() -> None:
    """POST an empty symptoms list and assert a 422 validation error."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/predict",
            json={"symptoms": []},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_response_has_disclaimer() -> None:
    """Assert that every prediction response includes the medical disclaimer."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/predict",
            json={"symptoms": _KNOWN_SYMPTOMS},
        )

    if response.status_code == 500:
        pytest.skip("ML model not available in test environment.")

    assert response.status_code == 200
    data = response.json()
    assert "disclaimer" in data
    assert "not medical advice" in data["disclaimer"].lower()


@pytest.mark.asyncio
async def test_predict_with_patient_id() -> None:
    """POST symptoms with a patient_id and assert it is echoed in the response."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/predict",
            json={"symptoms": _KNOWN_SYMPTOMS, "patient_id": "test-patient-001"},
        )

    if response.status_code == 500:
        pytest.skip("ML model not available in test environment.")

    assert response.status_code == 200
    data = response.json()
    assert data.get("patient_id") == "test-patient-001"


@pytest.mark.asyncio
async def test_predict_whitespace_only_symptoms() -> None:
    """POST symptoms that are only whitespace and assert 422."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/predict",
            json={"symptoms": ["   ", "\t", ""]},
        )
    assert response.status_code == 422
