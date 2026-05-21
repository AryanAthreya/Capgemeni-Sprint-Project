# stdlib
import sys
import os
import uuid

# third-party
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from frontend.utils.api_client import HealthcareAPIClient


def _get_client() -> HealthcareAPIClient:
    """Return a cached API client."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = HealthcareAPIClient()
    return st.session_state.api_client


def _init_session() -> None:
    """Initialise Q&A history and session ID."""
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    if "qa_session_id" not in st.session_state:
        st.session_state.qa_session_id = str(uuid.uuid4())


def _render_answer(answer: str) -> None:
    """
    Render the agent's answer with source citations highlighted.

    Args:
        answer: Full answer text (may include source citations).
    """
    # Split off source lines if present
    lines = answer.split("\n")
    main_lines = []
    source_lines = []

    for line in lines:
        if line.strip().startswith("📚") or "[Source:" in line or "Source:" in line.lower():
            source_lines.append(line)
        else:
            main_lines.append(line)

    main_text = "\n".join(main_lines).strip()
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0d2137, #1e3a5f);
            color: #e8f4f8;
            border-radius: 12px;
            padding: 16px 20px;
            margin: 8px 0;
            border-left: 4px solid #00bcd4;
            font-size: 0.95rem;
            line-height: 1.6;
        ">
            📖 <strong>MedSearch Answer:</strong><br><br>
            {main_text.replace(chr(10), "<br>")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if source_lines:
        st.markdown("---")
        st.markdown("**📚 Sources:**")
        for src in source_lines:
            src_clean = (
                src.replace("📚 **Sources:**", "")
                   .replace("📚", "")
                   .strip()
                   .lstrip(",")
                   .strip()
            )
            if src_clean:
                parts = [s.strip() for s in src_clean.split(",") if s.strip()]
                for part in parts:
                    st.markdown(
                        f"""
                        <span style="
                            background:#1a3a4f;
                            color:#00bcd4;
                            border-radius:6px;
                            padding:3px 10px;
                            font-size:0.82rem;
                            margin:2px;
                            display:inline-block;
                        ">📄 {part}</span>
                        """,
                        unsafe_allow_html=True,
                    )


def render() -> None:
    """Render the Medical Q&A page."""
    _init_session()

    st.markdown(
        """
        <h2 style="color:#00bcd4; margin-bottom:4px;">🔬 Medical Q&A — MedSearch</h2>
        <p style="color:#888; font-size:0.9rem;">
            Ask questions about diseases, symptoms, or treatments.
            Answers are grounded in WHO medical guidelines and health documents.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ─── Example questions ─────────────────────────────────────────────────────
    with st.expander("💡 Example questions to try", expanded=False):
        examples = [
            "What is malaria and how is it transmitted?",
            "What are the symptoms of diabetes?",
            "How can chronic diseases be prevented?",
            "What are the WHO guidelines for tuberculosis treatment?",
            "What causes hepatitis B?",
        ]
        for q in examples:
            if st.button(q, key=f"ex_{q[:20]}"):
                st.session_state["prefill_qa"] = q
                st.rerun()

    # ─── Q&A history ───────────────────────────────────────────────────────────
    for item in st.session_state.qa_history:
        st.markdown(
            f"""
            <div style="
                background:#1a1a2e;
                border-radius:8px;
                padding:10px 16px;
                margin:4px 0;
                color:#ccc;
                font-size:0.9rem;
            ">
                ❓ <strong>You asked:</strong> {item["question"]}
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_answer(item["answer"])
        st.markdown("")

    # ─── Input form ───────────────────────────────────────────────────────────
    prefill = st.session_state.pop("prefill_qa", "")
    with st.form("qa_form", clear_on_submit=True):
        query = st.text_input(
            "Your medical question:",
            value=prefill,
            placeholder="e.g. What are the symptoms of malaria?",
        )
        submitted = st.form_submit_button("🔍 Search Knowledge Base")

    if submitted and query.strip():
        with st.spinner("MedSearch is retrieving information..."):
            client = _get_client()
            result = client.chat_with_agent(
                message=query.strip(),
                session_id=st.session_state.qa_session_id,
                intent="medical_info",
            )

        answer = result.get("response", "No answer found.")
        st.session_state.qa_history.append(
            {"question": query.strip(), "answer": answer}
        )
        if result.get("session_id"):
            st.session_state.qa_session_id = result["session_id"]
        st.rerun()

    # ─── Clear button ──────────────────────────────────────────────────────────
    if st.session_state.qa_history:
        if st.button("🗑️ Clear Q&A history"):
            st.session_state.qa_history = []
            st.session_state.qa_session_id = str(uuid.uuid4())
            st.rerun()

    # ─── Footer note ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.8rem; color:#888;'>"
        "ℹ️ MedSearch answers are based on WHO guidelines and medical knowledge documents. "
        "Always verify with a qualified healthcare provider.</p>",
        unsafe_allow_html=True,
    )
