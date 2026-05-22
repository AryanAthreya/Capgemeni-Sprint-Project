# AZURE SETUP REQUIRED:
# 1. Deploy gpt-4o-mini model in Azure OpenAI Studio
# 2. Set env vars: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
#    AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION
# LOCAL FALLBACK: If Azure OpenAI is not configured, returns a rule-based
#                 response using only the ML model predictions (no GPT needed).

# stdlib
import logging
import re
from typing import Any, Dict, List, Optional

# local
from api.models.schemas import AgentResponse

logger = logging.getLogger(__name__)

TRIAGE_SYSTEM_PROMPT = """You are Dr. Triage, a medical triage assistant AI. Your role is to:
1. Analyze patient symptoms provided to you
2. Interpret the ML prediction results in plain, empathetic language
3. Assign a risk level (low/medium/high) with a clear, understandable explanation
4. Recommend appropriate next steps:
   - Low risk: home care advice, rest, hydration, monitor symptoms
   - Medium risk: schedule a doctor's appointment within 24-48 hours
   - High risk: seek immediate medical attention or go to an emergency department
5. Be compassionate and avoid alarming language unnecessarily

Always end every response with exactly this line:
'⚠️ This assessment is not a medical diagnosis. Please consult a licensed healthcare professional for proper diagnosis and treatment.'

Never diagnose definitively. Always express clinical uncertainty appropriately.
Never recommend specific prescription medications by name.
If symptoms suggest a life-threatening emergency (chest pain, difficulty breathing, stroke signs), always direct to emergency services immediately."""


class TriageAgent:
    """
    Dr. Triage agent that combines ML disease prediction with GPT-4o-mini
    to generate empathetic, actionable triage assessments.
    """

    def __init__(self) -> None:
        """Initialise the agent lazily — LLM client created on first run() call."""
        self._llm: Any = None
        self._memory: Any = None
        self._chain: Any = None

    def _get_llm(self) -> Any:
        """
        Return the Azure ChatOpenAI LLM instance (initialised once).

        Returns:
            AzureChatOpenAI instance or a mock LLM for local development.
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
                temperature=0.3,
                max_tokens=800,
            )
            logger.info("TriageAgent: Azure ChatOpenAI LLM initialised.")
        else:
            logger.warning("TriageAgent: Azure OpenAI not configured — using rule-based fallback.")
            self._llm = None

        return self._llm

    def _extract_symptoms(self, text: str) -> List[str]:
        """
        Extract symptom keywords from free-text using known symptom column names.

        Args:
            text: Natural language symptom description.

        Returns:
            List of matched symptom column name strings.
        """
        try:
            import pandas as pd
            from ml.feature_engineering import get_symptom_columns, extract_symptom_keywords

            staged_path = "data/staged/Training_cleaned.csv"
            df = pd.read_csv(staged_path, nrows=1)
            all_cols = get_symptom_columns(df)
            matched = extract_symptom_keywords(text, all_cols)
            if matched:
                return matched
        except Exception as exc:
            logger.warning("Symptom extraction from dataset failed: %s", exc)

        # Simple keyword fallback
        common = [
            "fever", "headache", "cough", "fatigue", "vomiting", "nausea",
            "pain", "rash", "diarrhea", "chills", "swelling", "itching",
            "breathlessness", "chest_pain", "weight_loss", "jaundice",
        ]
        text_lower = text.lower()
        return [kw for kw in common if kw.replace("_", " ") in text_lower or kw in text_lower]

    def _call_predict(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """
        Call the ML predict function to get top-3 disease predictions.

        Args:
            symptoms: List of symptom strings.

        Returns:
            List of prediction dicts with disease, confidence, risk_level.
        """
        from ml.predict import predict_disease

        try:
            return predict_disease(symptoms, top_k=3)
        except Exception as exc:
            logger.error("ML prediction failed in TriageAgent: %s", exc)
            return []

    def _rule_based_response(
        self, symptoms: List[str], predictions: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a structured triage response without GPT when Azure is unavailable.

        Args:
            symptoms: List of symptom strings.
            predictions: Top-3 ML predictions.

        Returns:
            Formatted triage assessment string.
        """
        if not predictions:
            return (
                "Based on the symptoms described, I was unable to generate a specific "
                "assessment at this time. Please consult a healthcare professional.\n\n"
                "⚠️ This assessment is not a medical diagnosis. Please consult a licensed "
                "healthcare professional for proper diagnosis and treatment."
            )

        top = predictions[0]
        disease = top["disease"]
        confidence = top["confidence"]
        risk = top["risk_level"]

        risk_advice = {
            "high": "🔴 HIGH RISK — Please seek immediate medical attention or visit an emergency department.",
            "medium": "🟡 MEDIUM RISK — Schedule a doctor's appointment within the next 24–48 hours.",
            "low": "🟢 LOW RISK — Monitor your symptoms at home. Rest, stay hydrated, and seek help if symptoms worsen.",
        }

        lines = [
            f"Based on the symptoms you described ({', '.join(symptoms[:5])}), "
            f"the most likely condition is **{disease}** "
            f"(model confidence: {confidence * 100:.0f}%).",
            "",
            f"**Risk Level:** {risk_advice.get(risk, risk)}",
            "",
            "**Other possible conditions:**",
        ]
        for p in predictions[1:]:
            lines.append(f"  - {p['disease']} ({p['confidence'] * 100:.0f}% confidence)")

        lines += [
            "",
            "⚠️ This assessment is not a medical diagnosis. Please consult a licensed "
            "healthcare professional for proper diagnosis and treatment.",
        ]
        return "\n".join(lines)

    def run(self, symptoms_text: str, session_id: Optional[str] = None) -> str:
        """
        Run the triage assessment for a given symptom description.

        Args:
            symptoms_text: Natural language description of symptoms.
            session_id: Optional session identifier for logging.

        Returns:
            Formatted triage assessment as a string.
        """
        logger.info(
            "TriageAgent.run | session_id=%s | input='%s'",
            session_id,
            symptoms_text[:80],
        )

        symptoms = self._extract_symptoms(symptoms_text)
        if not symptoms:
            symptoms = ["fever"]  # minimum default to avoid empty-array prediction errors

        predictions = self._call_predict(symptoms)
        llm = self._get_llm()

        if llm is None:
            return self._rule_based_response(symptoms, predictions)

        # Build a structured prompt for GPT
        pred_text = "\n".join(
            f"  {i+1}. {p['disease']} — confidence: {p['confidence']*100:.0f}%, risk: {p['risk_level']}"
            for i, p in enumerate(predictions)
        )

        user_prompt = (
            f"Patient reports: \"{symptoms_text}\"\n\n"
            f"Detected symptoms: {', '.join(symptoms)}\n\n"
            f"ML Model Predictions:\n{pred_text}\n\n"
            "Please provide a compassionate, clear triage assessment with recommended next steps."
        )

        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            messages = [
                SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)
            return response.content
        except Exception as exc:
            logger.error("LLM call failed in TriageAgent: %s", exc)
            return self._rule_based_response(symptoms, predictions)
