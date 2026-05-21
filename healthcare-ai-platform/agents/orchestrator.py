# stdlib
import logging
import uuid
from typing import Optional

# local
from api.models.schemas import AgentResponse

logger = logging.getLogger(__name__)

# ─── Keyword sets for intent classification ────────────────────────────────────
_TRIAGE_KEYWORDS = frozenset([
    "i have", "i feel", "i am experiencing", "my symptoms", "feeling",
    "suffering from", "hurts", "aching", "swollen", "fever", "pain",
    "headache", "cough", "vomiting", "nausea", "rash", "itching",
    "breathless", "chills", "diarrhea", "fatigue", "tired", "dizzy",
    "sore throat", "runny nose", "chest pain", "bleeding", "swelling",
    "burning", "numbness", "weakness", "stomach", "back pain",
])

_ANALYTICS_KEYWORDS = frozenset([
    "how many", "statistics", "report", "trend", "average", "most common",
    "percentage", "distribution", "count", "total patients", "analytics",
    "data", "chart", "summary", "breakdown", "proportion", "high risk",
    "low risk", "patient data", "dashboard", "metrics",
])

_MEDICAL_INFO_KEYWORDS = frozenset([
    "what is", "what are", "tell me about", "explain", "how to treat",
    "what causes", "symptoms of", "treatment for", "cure for",
    "prevention of", "how does", "disease information", "medication",
    "about diabetes", "about malaria", "about covid", "guidelines",
    "who recommends", "risk factors", "diagnosis of",
])


def classify_intent(message: str) -> str:
    """
    Classify the intent of a user message into one of four categories.

    Uses keyword matching against known vocabulary sets. Falls back to
    "general" if no strong signal is found.

    Priority order: triage > analytics > medical_info > general

    Args:
        message: Raw user message string.

    Returns:
        Intent string: "triage", "medical_info", "analytics", or "general".
    """
    text = message.lower().strip()

    # Score each intent by counting matched keywords
    triage_score = sum(1 for kw in _TRIAGE_KEYWORDS if kw in text)
    analytics_score = sum(1 for kw in _ANALYTICS_KEYWORDS if kw in text)
    medical_score = sum(1 for kw in _MEDICAL_INFO_KEYWORDS if kw in text)

    # Triage takes priority when personal symptoms are mentioned
    if triage_score >= 1:
        logger.debug("Intent classified as 'triage' (score=%d).", triage_score)
        return "triage"

    if analytics_score >= 1:
        logger.debug("Intent classified as 'analytics' (score=%d).", analytics_score)
        return "analytics"

    if medical_score >= 1:
        logger.debug("Intent classified as 'medical_info' (score=%d).", medical_score)
        return "medical_info"

    logger.debug("Intent classified as 'general' (no strong signal).")
    return "general"


class AgentOrchestrator:
    """
    Multi-agent orchestrator that initialises all three agents once and
    routes incoming messages to the appropriate agent based on intent.
    """

    def __init__(self) -> None:
        """Initialise agent instances (lazy — agents init their LLMs on first use)."""
        from agents.triage_agent import TriageAgent
        from agents.rag_agent import RAGAgent
        from agents.analytics_agent import AnalyticsAgent

        self._triage = TriageAgent()
        self._rag = RAGAgent()
        self._analytics = AnalyticsAgent()
        logger.info("AgentOrchestrator: all agents initialised.")

    def route_to_agent(
        self,
        message: str,
        session_id: str,
        intent_override: Optional[str] = None,
    ) -> AgentResponse:
        """
        Classify intent and route the message to the appropriate agent.

        Args:
            message: User message string.
            session_id: Conversation session ID.
            intent_override: Optional explicit intent from the API request.

        Returns:
            AgentResponse with response text, agent_used, and session_id.
        """
        intent = intent_override if intent_override else classify_intent(message)
        logger.info(
            "Routing | session_id=%s | intent=%s | override=%s",
            session_id,
            intent,
            bool(intent_override),
        )

        if intent == "triage":
            agent_name = "Dr. Triage"
            response_text = self._triage.run(message, session_id=session_id)

        elif intent == "analytics":
            agent_name = "HealthAnalyst"
            response_text = self._analytics.run(message, session_id=session_id)

        elif intent == "medical_info":
            agent_name = "MedSearch"
            response_text = self._rag.run(message, session_id=session_id)

        else:
            # General fallback — use RAG to attempt an answer
            agent_name = "MedSearch"
            response_text = self._rag.run(message, session_id=session_id)
            if not response_text or "don't have information" in response_text.lower():
                response_text = (
                    "I'm here to help with medical questions, symptom triage, "
                    "and healthcare analytics. Please describe your symptoms or "
                    "ask a health-related question."
                )

        logger.info(
            "Agent '%s' responded | session_id=%s | length=%d chars",
            agent_name,
            session_id,
            len(response_text),
        )

        return AgentResponse(
            response=response_text,
            agent_used=agent_name,
            session_id=session_id,
        )


# ─── Module-level singleton ────────────────────────────────────────────────────
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """Return the module-level singleton AgentOrchestrator (created on first call)."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
