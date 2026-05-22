"""
Healthcare AI Platform — Streamlit Frontend
Main entry point. Run with: streamlit run frontend/app.py
"""

# stdlib
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# third-party
import streamlit as st

# local
from frontend.utils.api_client import HealthcareAPIClient
from frontend.utils.theme import apply_notion_theme
from frontend.pages import symptom_checker, medical_qa, dashboard

# ─── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare AI Platform",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_notion_theme()

# ─── Disclaimer banner ────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #7c1515, #5c0d0d);
        color: #ffcccc;
        padding: 8px 20px;
        border-radius: 6px;
        text-align: center;
        font-size: 0.82rem;
        margin-bottom: 12px;
        border: 1px solid #9e2020;
    ">
        ⚠️ <strong>For demonstration and educational purposes only.</strong>
        This platform is <strong>NOT</strong> a substitute for professional medical advice.
        Always consult a qualified healthcare provider for medical decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:left; padding: 12px 0 18px 0;">
            <div class="notion-pill">Workspace</div>
            <h1 style="color:#1f1f1d; font-size:1.5rem; margin:10px 0 0 0; letter-spacing:-0.03em;">Healthcare AI</h1>
            <p style="color:#6f6b66; font-size:0.84rem; margin:6px 0 0 0; line-height:1.5;">
                Intelligent Healthcare Support System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        options=["🩺 Symptom Checker", "🔬 Medical Q&A", "📊 Analytics Dashboard"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Backend connection status
    st.markdown("**⚡ Backend Status**")
    if "api_client" not in st.session_state:
        st.session_state.api_client = HealthcareAPIClient()

    health = st.session_state.api_client.health_check()
    is_ok = health.get("status") == "ok"
    model_ok = health.get("model_loaded", False)
    db_ok = health.get("db_connected", False)

    status_color = "#4caf50" if is_ok else "#f44336"
    st.markdown(
        f"""
        <div class="notion-card-muted" style="font-size:0.84rem; color:#1f1f1d; padding:12px 14px;">
            <span style="color:{status_color};">●</span>
            {"Connected" if is_ok else "Offline"}<br>
            <span style="color:#6f6b66;">{"Model loaded" if model_ok else "Model not loaded"}</span><br>
            <span style="color:#6f6b66;">{"Database connected" if db_ok else "Database not connected"}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.75rem; color:#6f6b66; text-align:left;">
            v1.0.0 · Azure OpenAI + LangChain<br>
            RandomForest + RAG · CosmosDB
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Page routing ─────────────────────────────────────────────────────────────
if page == "🩺 Symptom Checker":
    symptom_checker.render()
elif page == "🔬 Medical Q&A":
    medical_qa.render()
elif page == "📊 Analytics Dashboard":
    dashboard.render()
