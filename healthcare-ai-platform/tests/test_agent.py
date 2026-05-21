# stdlib
import pytest
from httpx import AsyncClient, ASGITransport

# local
from api.main import app


@pytest.mark.asyncio
async def test_agent_triage_intent() -> None:
    """
    POST a symptom-description message and assert Dr. Triage handles it.
    The agent_used field should equal 'Dr. Triage'.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/agent",
            json={"message": "I have fever and headache"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["agent_used"] == "Dr. Triage"
    assert "response" in data
    assert len(data["response"]) > 0


@pytest.mark.asyncio
async def test_agent_analytics_intent() -> None:
    """
    POST a data analytics question and assert HealthAnalyst handles it.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/agent",
            json={"message": "how many patients do we have"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["agent_used"] == "HealthAnalyst"


@pytest.mark.asyncio
async def test_agent_returns_session_id() -> None:
    """Assert that every agent response includes a non-empty session_id."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/agent",
            json={"message": "tell me about malaria"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert len(data["session_id"]) > 0


@pytest.mark.asyncio
async def test_agent_session_id_preserved() -> None:
    """Assert that a provided session_id is preserved in the response."""
    custom_session = "test-session-abc-123"
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/agent",
            json={"message": "I feel nauseous", "session_id": custom_session},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == custom_session


@pytest.mark.asyncio
async def test_agent_intent_override() -> None:
    """
    POST with explicit intent='analytics' and assert HealthAnalyst is used
    even if the message text would normally classify as triage.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/agent",
            json={
                "message": "give me statistics",
                "intent": "analytics",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["agent_used"] == "HealthAnalyst"


@pytest.mark.asyncio
async def test_agent_empty_message() -> None:
    """POST an empty message and assert 422 validation error."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/agent",
            json={"message": ""},
        )
    assert response.status_code == 422
