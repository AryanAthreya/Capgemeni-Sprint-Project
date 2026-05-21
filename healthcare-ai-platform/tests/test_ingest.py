# stdlib
import pytest
from httpx import AsyncClient, ASGITransport

# local
from api.main import app


@pytest.mark.asyncio
async def test_ingest_valid_patient() -> None:
    """POST a valid patient payload and assert a 200 response with patient_id."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/ingest",
            json={
                "name": "Alice Johnson",
                "age": 34,
                "gender": "Female",
                "symptoms": ["fever", "headache", "fatigue"],
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert "patient_id" in data
    assert len(data["patient_id"]) > 0
    assert data["status"] == "created"


@pytest.mark.asyncio
async def test_ingest_missing_required_field() -> None:
    """POST a patient without the required 'name' field and assert 422."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/ingest",
            json={
                # name is intentionally missing
                "age": 45,
                "gender": "Male",
                "symptoms": ["cough"],
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ingest_invalid_age() -> None:
    """POST a patient with age=150 (out of 1-120 range) and assert 422."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/ingest",
            json={
                "name": "Bob Smith",
                "age": 150,
                "gender": "Male",
                "symptoms": ["fever"],
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ingest_empty_symptoms() -> None:
    """POST a patient with an empty symptoms list and assert 422."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/ingest",
            json={
                "name": "Carol White",
                "age": 28,
                "gender": "Female",
                "symptoms": [],
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ingest_custom_patient_id() -> None:
    """POST a patient with a pre-set patient_id and assert it is echoed back."""
    custom_id = "test-pid-12345"
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/ingest",
            json={
                "patient_id": custom_id,
                "name": "Dave Brown",
                "age": 52,
                "gender": "Male",
                "symptoms": ["nausea", "vomiting"],
            },
        )
    assert response.status_code == 200
    assert response.json()["patient_id"] == custom_id
