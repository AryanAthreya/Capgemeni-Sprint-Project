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
from frontend.pages import symptom_checker, medical_qa, dashboard

# ─── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare AI Platform",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #050d1a 0%, #0d1b2e 50%, #091627 100%);
        color: #e0e0e0;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 100%);
        border-right: 1px solid #1a3a5f;
    }

    /* Sidebar radio buttons */
    .stRadio label {
        color: #c0d4e8 !important;
        font-size: 0.95rem;
    }

    /* Hide default Streamlit hamburger menu */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        background: #0d1f3c !important;
        color: #e0e0e0 !important;
        border: 1px solid #1a3a5f !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2196f3, #1565c0);
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.4);
        transform: translateY(-1px);
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: #0d1f3c !important;
        color: #c0d4e8 !important;
        border-radius: 8px !important;
    }

    /* Dividers */
    hr { border-color: #1a3a5f; }

    /* Spinner */
    .stSpinner > div { border-top-color: #1a73e8 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <h1 style="color:#1a73e8; font-size:1.6rem; margin:0;">⚕️ Healthcare AI</h1>
            <p style="color:#7a9cc4; font-size:0.78rem; margin:4px 0 0 0;">
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
        <div style="font-size:0.82rem; color:#c0d4e8;">
            <span style="color:{status_color};">●</span>
            {"🟢 Connected" if is_ok else "🔴 Offline"}<br>
            {"✅" if model_ok else "❌"} ML Model loaded<br>
            {"✅" if db_ok else "❌"} Database connected
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.75rem; color:#556; text-align:center;">
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
