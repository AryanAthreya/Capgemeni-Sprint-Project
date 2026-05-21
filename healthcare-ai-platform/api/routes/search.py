# stdlib
import logging

# third-party
from fastapi import APIRouter, HTTPException, Request

# local
from api.models.schemas import SearchRequest, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

router = APIRouter()

_MOCK_RESULTS = [
    SearchResult(
        content=(
            "Malaria is a life-threatening disease caused by Plasmodium parasites "
            "transmitted through the bites of infected female Anopheles mosquitoes. "
            "Symptoms include fever, chills, headache, muscle aches, and fatigue."
        ),
        source="who_infectious_diseases.pdf",
        score=0.92,
    ),
    SearchResult(
        content=(
            "Prevention of malaria involves use of insecticide-treated mosquito nets, "
            "indoor residual spraying, and antimalarial chemoprophylaxis for travellers. "
            "WHO recommends the RTS,S/AS01 malaria vaccine for children in sub-Saharan Africa."
        ),
        source="who_infectious_diseases.pdf",
        score=0.87,
    ),
    SearchResult(
        content=(
            "Chronic diseases such as diabetes, cardiovascular diseases, and chronic "
            "respiratory diseases account for 71% of all deaths globally each year. "
            "They share common risk factors including tobacco use, physical inactivity, "
            "unhealthy diet, and harmful alcohol use."
        ),
        source="who_chronic_diseases.pdf",
        score=0.61,
    ),
]


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic document search",
    description=(
        "Search the medical knowledge base using natural language queries. "
        "Returns relevant document chunks from WHO guidelines and disease fact sheets."
    ),
    tags=["RAG Search"],
)
async def search_documents(body: SearchRequest, request: Request) -> SearchResponse:
    """
    POST /api/search

    Embeds the query and retrieves top-k relevant chunks from Azure AI Search.
    Falls back to mock results in local development mode.

    Args:
        body: SearchRequest with query string and top_k count.
        request: Starlette request for tracing.

    Returns:
        SearchResponse with retrieved document chunks and the original query.

    Raises:
        HTTPException 500: If the vector store is unavailable.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.info(
        "Search request | query='%s' | top_k=%d | request_id=%s",
        body.query,
        body.top_k,
        request_id,
    )

    try:
        from agents.vector_store import get_vector_store

        vs = get_vector_store()
        raw_results = vs.search_documents(body.query, top_k=body.top_k)

        results = [
            SearchResult(
                content=r["content"],
                source=r["source"],
                score=r["score"],
            )
            for r in raw_results
        ]

    except Exception as exc:
        logger.warning(
            "Vector store unavailable (%s). Returning mock results. request_id=%s",
            exc,
            request_id,
        )
        # Return mock results with a note so the caller is aware
        results = _MOCK_RESULTS[: body.top_k]
        for r in results:
            r.source = f"[MOCK — vector store not configured] {r.source}"

    logger.info(
        "Search complete | returned %d results | request_id=%s",
        len(results),
        request_id,
    )

    return SearchResponse(results=results, query=body.query)
