# stdlib
import pytest
from httpx import AsyncClient, ASGITransport

# local
from api.main import app


@pytest.mark.asyncio
async def test_search_returns_results() -> None:
    """
    POST a search query and assert a 200 response with a non-empty results list.
    When Azure AI Search is not configured, mock results should be returned.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/search",
            json={"query": "what is malaria", "top_k": 3},
        )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0

    first = data["results"][0]
    assert "content" in first
    assert "source" in first
    assert "score" in first
    assert isinstance(first["score"], float)


@pytest.mark.asyncio
async def test_search_empty_query() -> None:
    """POST an empty query string and assert a 422 validation error."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/search",
            json={"query": "", "top_k": 3},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_query_is_echoed() -> None:
    """Assert that the query field in the response matches the request query."""
    query = "symptoms of diabetes"
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/search",
            json={"query": query, "top_k": 2},
        )
    assert response.status_code == 200
    assert response.json()["query"] == query


@pytest.mark.asyncio
async def test_search_top_k_respected() -> None:
    """Assert that the results list length does not exceed top_k."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/search",
            json={"query": "chronic disease prevention", "top_k": 2},
        )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_search_top_k_out_of_range() -> None:
    """POST top_k=0 (below minimum of 1) and assert 422."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/search",
            json={"query": "fever", "top_k": 0},
        )
    assert response.status_code == 422
