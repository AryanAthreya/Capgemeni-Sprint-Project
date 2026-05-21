# stdlib
import logging
import uuid
from typing import AsyncGenerator

# third-party
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

# local
from api.models.schemas import AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

router = APIRouter()


async def _stream_agent_response(
    message: str, session_id: str
) -> AsyncGenerator[str, None]:
    """
    Yield agent response tokens as Server-Sent Events (SSE).

    Args:
        message: User message to pass to the agent orchestrator.
        session_id: Session ID for conversation continuity.

    Yields:
        SSE-formatted string chunks.
    """
    from agents.orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    try:
        agent_response = orchestrator.route_to_agent(message, session_id)
        # Simulate token-by-token streaming by splitting on words
        words = agent_response.response.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {chunk}\n\n"
        yield f"data: [DONE]\n\n"
    except Exception as exc:
        logger.error("Streaming error: %s", exc)
        yield f"data: [ERROR] {str(exc)}\n\n"


@router.post(
    "/agent",
    response_model=AgentResponse,
    summary="Multi-agent chat endpoint",
    description=(
        "Routes user messages to the appropriate AI agent "
        "(Dr. Triage, MedSearch, or HealthAnalyst). "
        "Add ?stream=true for Server-Sent Events streaming."
    ),
    tags=["Multi-Agent"],
)
async def agent_chat(
    body: AgentRequest,
    request: Request,
    stream: bool = Query(default=False, description="Enable SSE streaming response"),
) -> AgentResponse | StreamingResponse:
    """
    POST /api/agent

    Classifies user intent and routes to the appropriate LangChain agent.
    Optionally streams the response via SSE when stream=true.

    Args:
        body: AgentRequest with message, optional session_id, and intent hint.
        request: Starlette request for tracing.
        stream: Query param to enable streaming.

    Returns:
        AgentResponse with response text, agent name, and session_id.
        Or a StreamingResponse when stream=true.

    Raises:
        HTTPException 500: If the agent orchestrator fails.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    session_id = body.session_id or str(uuid.uuid4())

    logger.info(
        "Agent request | session_id=%s | intent_hint=%s | request_id=%s | message='%s'",
        session_id,
        body.intent,
        request_id,
        body.message[:100],
    )

    if stream:
        return StreamingResponse(
            _stream_agent_response(body.message, session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        from agents.orchestrator import get_orchestrator

        orchestrator = get_orchestrator()

        # Pass intent hint from request body to override auto-classification
        agent_response = orchestrator.route_to_agent(
            message=body.message,
            session_id=session_id,
            intent_override=body.intent,
        )

    except Exception as exc:
        logger.error(
            "Agent orchestrator error | session_id=%s | error=%s | request_id=%s",
            session_id,
            exc,
            request_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing failed: {str(exc)}",
        ) from exc

    logger.info(
        "Agent response | agent=%s | session_id=%s | request_id=%s",
        agent_response.agent_used,
        session_id,
        request_id,
    )

    return agent_response
