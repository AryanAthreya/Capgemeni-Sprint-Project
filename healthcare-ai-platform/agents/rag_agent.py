# AZURE SETUP REQUIRED:
# 1. Deploy gpt-4o-mini in Azure OpenAI Studio
# 2. Ingest PDF documents via agents/vector_store.py before using this agent
# 3. Set env vars: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
#    AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION
# 4. Set Azure AI Search vars for vector retrieval
# LOCAL FALLBACK: Returns an answer from FAISS index if Azure Search is not configured.
#                 If Azure OpenAI is also absent, returns retrieved context directly.

# stdlib
import logging
from typing import Any, Dict, List, Optional

# local
from api.models.schemas import AgentResponse

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are MedSearch, a medical information assistant.

You answer questions about diseases, symptoms, treatments, and health guidelines 
STRICTLY based on the provided medical documents. 

Rules:
- If the answer is clearly present in the context, answer it accurately and cite the source.
- If the answer is NOT in the provided context, say exactly: 
  "I don't have information about that in my knowledge base."
- Always cite which document your answer comes from using the format: [Source: filename]
- Never make up medical information not present in the context.
- Never recommend specific prescription medications by name.
- Keep answers concise (under 300 words) and use plain language.
- Use bullet points for lists of symptoms, treatments, or steps."""


class RAGAgent:
    """
    MedSearch RAG agent that retrieves relevant medical document chunks
    and generates grounded answers using GPT-4o-mini.
    """

    def __init__(self) -> None:
        """Initialise lazily — LLM and vector store connected on first run."""
        self._llm: Any = None

    def _get_llm(self) -> Any:
        """
        Return the Azure ChatOpenAI LLM (initialised once).

        Returns:
            AzureChatOpenAI instance or None if not configured.
        """
        if self._llm is not None:
            return self._llm

        from config import settings

        if settings.azure_openai_configured:
            from langchain_openai import AzureChatOpenAI

            self._llm = AzureChatOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                azure_deployment=settings.azure_openai_deployment_name,
                openai_api_version=settings.azure_openai_api_version,
                temperature=0.1,
                max_tokens=600,
            )
            logger.info("RAGAgent: Azure ChatOpenAI LLM initialised.")
        else:
            logger.warning("RAGAgent: Azure OpenAI not configured.")
            self._llm = None

        return self._llm

    def _retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant document chunks from the vector store.

        Args:
            query: Natural language query string.
            top_k: Number of chunks to retrieve.

        Returns:
            List of result dicts with content, source, score.
        """
        from agents.vector_store import get_vector_store

        vs = get_vector_store()
        try:
            return vs.search_documents(query, top_k=top_k)
        except Exception as exc:
            logger.error("Vector store search failed: %s", exc)
            return []

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a numbered context block for the prompt.

        Args:
            results: List of vector search result dicts.

        Returns:
            Formatted context string.
        """
        if not results:
            return "No relevant documents found."

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[Document {i} | Source: {r['source']} | Score: {r['score']:.3f}]\n"
                f"{r['content']}"
            )
        return "\n\n---\n\n".join(parts)

    def run(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Answer a medical information query using retrieved document context.

        Args:
            query: Natural language medical question.
            session_id: Optional session ID for logging.

        Returns:
            Grounded answer string with source citations.
        """
        logger.info(
            "RAGAgent.run | session_id=%s | query='%s'",
            session_id,
            query[:80],
        )

        results = self._retrieve_context(query, top_k=3)
        context = self._format_context(results)
        llm = self._get_llm()

        if llm is None:
            # Fallback: return raw retrieved context if GPT is unavailable
            if results:
                return (
                    "📄 **Retrieved from knowledge base** (Azure OpenAI not configured):\n\n"
                    + context
                )
            return "I don't have information about that in my knowledge base."

        from langchain_core.messages import HumanMessage, SystemMessage

        user_prompt = (
            f"Medical Documents Context:\n\n{context}\n\n"
            f"---\n\nUser Question: {query}\n\n"
            "Please answer based only on the documents above."
        )

        try:
            messages = [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)
            answer = response.content

            # Append source list at the end for clarity
            if results:
                sources = list({r["source"] for r in results})
                source_line = "\n\n📚 **Sources:** " + ", ".join(sources)
                answer += source_line

            return answer

        except Exception as exc:
            logger.error("LLM call failed in RAGAgent: %s", exc)
            return (
                "I encountered an error while processing your question. "
                "Here is the raw context retrieved:\n\n" + context
            )
