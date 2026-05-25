# AZURE SETUP REQUIRED:
# 1. Deploy gpt-4o-mini in Azure OpenAI Studio
# 2. CosmosDB must be populated with patient records (run pipeline/generate_patients.py
#    then POST to /api/ingest, or run the data pipeline)
# 3. Set env vars: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
#    COSMOS_ENDPOINT, COSMOS_KEY
# LOCAL FALLBACK: Reads from exports/healthcare_powerbi.csv or data/raw/patients.csv
#                 if CosmosDB is not configured. Answers are generated without GPT
#                 when Azure OpenAI is absent.

# stdlib
import json
import logging
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ANALYTICS_SYSTEM_PROMPT = """You are HealthAnalyst, a medical data analytics assistant.

You have access to patient data from the healthcare database. When asked analytics questions:
1. Examine the data provided to you carefully
2. Compute accurate numerical statistics
3. Present clear insights with specific numbers and percentages
4. Highlight any concerning trends (e.g., high proportion of high-risk patients)
5. Always round percentages to 1 decimal place
6. Present data in a structured format using bullet points or numbered lists

Rules:
- Only report numbers that are present in the data provided
- Clearly state the sample size (N=X patients) at the start
- If the data is insufficient to answer the question, say so explicitly
- Never fabricate statistics"""


class AnalyticsAgent:
    """
    HealthAnalyst agent that queries patient data and generates
    natural-language analytics insights using GPT-4o-mini.
    """

    def __init__(self) -> None:
        """Initialise lazily — LLM and DB client connected on first run."""
        self._llm: Any = None

    def _get_llm(self) -> Any:
        """
        Return the Azure ChatOpenAI LLM (initialised once).

        Returns:
            AzureChatOpenAI or None if not configured.
        """
        if self._llm is not None:
            return self._llm

        from langchain_openai import AzureChatOpenAI
        from config import settings

        self._llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            azure_deployment=settings.azure_openai_deployment_name,
            api_version=settings.azure_openai_api_version,
            temperature=0.3,
        )

        return self._llm

    # ─── Tool functions ────────────────────────────────────────────────────────

    def get_all_patients_tool(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Retrieve all patient records from CosmosDB or local CSV fallback.

        Args:
            limit: Maximum number of records to fetch.

        Returns:
            List of patient record dicts.
        """
        try:
            from api.database.cosmos_client import get_cosmos_client
            cosmos = get_cosmos_client()
            records = cosmos.get_all_patients(limit=limit)
            if records:
                logger.info("Fetched %d patients from CosmosDB.", len(records))
                return records
        except Exception as exc:
            logger.warning("CosmosDB query failed: %s. Trying CSV fallback.", exc)

        # CSV fallback
        import pandas as pd
        from pathlib import Path

        for csv_path in [
            "exports/healthcare_powerbi.csv",
            "data/raw/patients.csv",
        ]:
            if Path(csv_path).exists():
                df = pd.read_csv(csv_path).head(limit)
                logger.info("Loaded %d patients from %s.", len(df), csv_path)
                return df.to_dict(orient="records")

        logger.warning("No patient data source available.")
        return []

    def get_disease_distribution_tool(self) -> Dict[str, int]:
        """
        Count how many patients have each diagnosis.

        Returns:
            Dict mapping disease name to patient count.
        """
        patients = self.get_all_patients_tool()
        diseases = [p.get("diagnosis", p.get("latest_prediction", {}).get("disease", "Unknown"))
                    for p in patients]
        return dict(Counter(diseases).most_common(20))

    def get_risk_summary_tool(self) -> Dict[str, Any]:
        """
        Summarise patient counts by risk level.

        Returns:
            Dict with counts and percentages per risk level.
        """
        patients = self.get_all_patients_tool()
        total = len(patients)
        if total == 0:
            return {"total": 0, "low": 0, "medium": 0, "high": 0}

        risk_counts: Counter = Counter()
        for p in patients:
            risk = p.get("risk_level", "unknown").lower()
            risk_counts[risk] += 1

        return {
            "total": total,
            "low": risk_counts.get("low", 0),
            "medium": risk_counts.get("medium", 0),
            "high": risk_counts.get("high", 0),
            "low_pct": round(risk_counts.get("low", 0) / total * 100, 1),
            "medium_pct": round(risk_counts.get("medium", 0) / total * 100, 1),
            "high_pct": round(risk_counts.get("high", 0) / total * 100, 1),
        }

    def _compute_stats(self, patients: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute summary statistics from a patient list.

        Args:
            patients: List of patient record dicts.

        Returns:
            Dict of computed statistics.
        """
        total = len(patients)
        if total == 0:
            return {"total": 0}

        ages = [p.get("age", 0) for p in patients if isinstance(p.get("age"), (int, float))]
        avg_age = round(sum(ages) / len(ages), 1) if ages else 0

        disease_dist = self.get_disease_distribution_tool()
        top_disease = max(disease_dist, key=disease_dist.get) if disease_dist else "N/A"

        risk_summary = self.get_risk_summary_tool()

        return {
            "total_patients": total,
            "average_age": avg_age,
            "top_disease": top_disease,
            "disease_distribution": dict(list(disease_dist.items())[:10]),
            "risk_summary": risk_summary,
        }

    def _rule_based_response(self, query: str, stats: Dict[str, Any]) -> str:
        """
        Generate a structured analytics response without GPT.

        Args:
            query: Original user question.
            stats: Computed statistics dict.

        Returns:
            Formatted analytics response string.
        """
        total = stats.get("total_patients", 0)
        risk = stats.get("risk_summary", {})
        lines = [
            f"📊 **Healthcare Analytics Report** (N={total} patients)",
            "",
            f"- **Average Age:** {stats.get('average_age', 'N/A')} years",
            f"- **Most Common Diagnosis:** {stats.get('top_disease', 'N/A')}",
            "",
            "**Risk Level Distribution:**",
            f"  🟢 Low: {risk.get('low', 0)} patients ({risk.get('low_pct', 0)}%)",
            f"  🟡 Medium: {risk.get('medium', 0)} patients ({risk.get('medium_pct', 0)}%)",
            f"  🔴 High: {risk.get('high', 0)} patients ({risk.get('high_pct', 0)}%)",
            "",
            "**Top 5 Diagnoses:**",
        ]
        for disease, count in list(stats.get("disease_distribution", {}).items())[:5]:
            pct = round(count / total * 100, 1) if total else 0
            lines.append(f"  - {disease}: {count} patients ({pct}%)")
        return "\n".join(lines)

    def run(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Answer an analytics question about the patient database.

        Args:
            query: Natural language analytics question.
            session_id: Optional session ID for logging.

        Returns:
            Natural language analytics insight string.
        """
        logger.info(
            "AnalyticsAgent.run | session_id=%s | query='%s'",
            session_id,
            query[:80],
        )

        patients = self.get_all_patients_tool()
        stats = self._compute_stats(patients)
        llm = self._get_llm()

        if llm is None:
            return self._rule_based_response(query, stats)

        from langchain_core.messages import HumanMessage, SystemMessage

        context = json.dumps(stats, indent=2)
        user_prompt = (
            f"Patient Analytics Data:\n```json\n{context}\n```\n\n"
            f"User Question: {query}\n\n"
            "Please provide a clear, data-driven answer."
        )

        try:
            messages = [
                SystemMessage(content=ANALYTICS_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)
            return response.content
        except Exception as exc:
            logger.error("LLM call failed in AnalyticsAgent: %s", exc)
            return self._rule_based_response(query, stats)
