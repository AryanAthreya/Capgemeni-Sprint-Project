# stdlib
import uuid
from typing import List

# third-party
import streamlit as st

# local (relative import works when run via streamlit)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from frontend.utils.api_client import HealthcareAPIClient
from frontend.utils.theme import apply_notion_theme

_RISK_COLORS = {"high": "🔴", "medium": "🟡", "low": "🟢", "unknown": "⚪"}
_DISCLAIMER = (
    "⚠️ **Disclaimer:** This tool provides AI-generated assessments for informational "
    "purposes only. It is **not a substitute** for professional medical advice, "
    "diagnosis, or treatment."
)


def _get_client() -> HealthcareAPIClient:
    """Return a cached API client stored in session state."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = HealthcareAPIClient()
    return st.session_state.api_client


def _init_session() -> None:
    """Initialise chat history and session ID in Streamlit session state."""
    if "triage_history" not in st.session_state:
        st.session_state.triage_history = []
    if "triage_session_id" not in st.session_state:
        st.session_state.triage_session_id = str(uuid.uuid4())


def _render_chat_bubble(role: str, content: str, risk: str = "unknown") -> None:
    """
    Render a styled chat bubble for user or assistant messages.

    Args:
        role: "user" or "assistant".
        content: Message text.
        risk: Risk level for coloured badge on assistant messages.
    """
    if role == "user":
        st.markdown(
            f"""
            <div style="
                background: #ffffff;
                color: #1f1f1d;
                border-radius: 18px 18px 4px 18px;
                padding: 12px 16px;
                margin: 6px 0 6px 20%;
                text-align: right;
                font-size: 0.95rem;
                border: 1px solid #e7e2da;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            ">
                🧑 <strong>You:</strong><br>{content}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        risk_badge = _RISK_COLORS.get(risk.lower(), "⚪")
        st.markdown(
            f"""
            <div style="
                background: #fbfaf8;
                color: #1f1f1d;
                border-radius: 18px 18px 18px 4px;
                padding: 12px 16px;
                margin: 6px 20% 6px 0;
                font-size: 0.95rem;
                border: 1px solid #e7e2da;
                border-left: 4px solid #2f6feb;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            ">
                🩺 <strong>Dr. Triage</strong> {risk_badge}<br><br>{content.replace(chr(10), "<br>")}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render() -> None:
    """Render the Symptom Checker page."""
    _init_session()
    apply_notion_theme()

    st.markdown(
        """
        <div class="notion-card" style="padding:18px 20px; margin-bottom:16px;">
        <h2 class="notion-page-title" style="margin-top:0;">🩺 Symptom Checker</h2>
        <p class="notion-page-subtitle" style="margin-bottom:0;">
            Describe your symptoms in plain language. Dr. Triage will assess them using AI + ML.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Chat history ──────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.triage_history:
            _render_chat_bubble(
                msg["role"],
                msg["content"],
                risk=msg.get("risk", "unknown"),
            )

    st.markdown("---")

    # ─── Input form ───────────────────────────────────────────────────────────
    with st.form("triage_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_area(
                "Describe your symptoms:",
                placeholder="e.g. I have a high fever, headache, and body aches for the last 2 days...",
                height=90,
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Send 🚀", use_container_width=True)

    if submitted and user_input.strip():
        # Add user message to history
        st.session_state.triage_history.append(
            {"role": "user", "content": user_input.strip(), "risk": "unknown"}
        )

        with st.spinner("Dr. Triage is reviewing your symptoms..."):
            client = _get_client()
            result = client.chat_with_agent(
                message=user_input.strip(),
                session_id=st.session_state.triage_session_id,
                intent="triage",
            )

        response_text = result.get("response", "An error occurred. Please try again.")
        risk_level = "unknown"

        # Detect risk level from response keywords
        rl = response_text.lower()
        if "high risk" in rl or "🔴" in response_text:
            risk_level = "high"
        elif "medium risk" in rl or "🟡" in response_text:
            risk_level = "medium"
        elif "low risk" in rl or "🟢" in response_text:
            risk_level = "low"

        st.session_state.triage_history.append(
            {"role": "assistant", "content": response_text, "risk": risk_level}
        )

        # Update session_id from server if returned
        if result.get("session_id"):
            st.session_state.triage_session_id = result["session_id"]

        st.rerun()

    # ─── Risk legend ──────────────────────────────────────────────────────────
    with st.expander("📊 Risk Level Guide", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🟢 **Low Risk**\nMonitor at home, rest & hydrate")
        with col2:
            st.markdown("🟡 **Medium Risk**\nSee a doctor within 24–48 hours")
        with col3:
            st.markdown("🔴 **High Risk**\nSeek immediate medical attention")

    # ─── Clear chat button ────────────────────────────────────────────────────
    if st.session_state.triage_history:
        if st.button("🗑️ Clear conversation", key="clear_triage"):
            st.session_state.triage_history = []
            st.session_state.triage_session_id = str(uuid.uuid4())
            st.rerun()

    # ─── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<p style='font-size:0.8rem; color:#aaa;'>{_DISCLAIMER}</p>",
        unsafe_allow_html=True,
    )
