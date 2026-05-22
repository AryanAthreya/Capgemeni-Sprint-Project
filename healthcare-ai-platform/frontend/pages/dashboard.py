# stdlib
import os
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# third-party
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from frontend.utils.api_client import HealthcareAPIClient
from frontend.utils.theme import apply_notion_theme


def _get_client() -> HealthcareAPIClient:
    """Return a cached API client."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = HealthcareAPIClient()
    return st.session_state.api_client


def _load_patient_data() -> pd.DataFrame:
    """
    Load patient data from the Power BI export CSV, falling back to raw patients.csv.

    Returns:
        Patient DataFrame or empty DataFrame if no data is available.
    """
    for path in ["exports/healthcare_powerbi.csv", "data/raw/patients.csv"]:
        if Path(path).exists():
            df = pd.read_csv(path)
            # Parse symptoms from pipe-separated string if needed
            if "symptoms" in df.columns:
                df["symptom_count"] = df["symptoms"].apply(
                    lambda x: len(str(x).split("|")) if pd.notna(x) else 0
                )
            if "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.date
            return df
    return pd.DataFrame()


def _metric_card(label: str, value: str, delta: str = "", color: str = "#1a73e8") -> None:
    """
    Render a styled metric card.

    Args:
        label: Metric label.
        value: Display value string.
        delta: Optional sub-label or delta text.
        color: Accent color.
    """
    st.markdown(
        f"""
        <div style="
            background: #ffffff;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #e7e2da;
            border-left: 5px solid {color};
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            text-align: center;
        ">
            <p style="color:#6f6b66; font-size:0.82rem; margin:0;">{label}</p>
            <h2 style="color:#1f1f1d; margin:6px 0; font-size:2rem;">{value}</h2>
            <p style="color:{color}; font-size:0.78rem; margin:0;">{delta}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    """Render the Analytics Dashboard page."""
    apply_notion_theme()
    st.markdown(
        """
        <div class="notion-card" style="padding:18px 20px; margin-bottom:16px;">
        <h2 class="notion-page-title" style="margin-top:0;">📊 Patient Analytics Dashboard</h2>
        <p class="notion-page-subtitle" style="margin-bottom:0;">
            Real-time insights from the patient database.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading patient data..."):
        df = _load_patient_data()

    if df.empty:
        st.warning(
            "No patient data found. Run `python pipeline/generate_patients.py` "
            "and `python exports/powerbi_export.py` first."
        )
        return

    # ─── Top metrics row ──────────────────────────────────────────────────────
    total = len(df)
    high_risk = int((df["risk_level"].str.lower() == "high").sum()) if "risk_level" in df.columns else 0
    avg_age = round(df["age"].mean(), 1) if "age" in df.columns else "N/A"

    top_disease = "N/A"
    if "diagnosis" in df.columns:
        top_disease = df["diagnosis"].mode()[0] if not df["diagnosis"].mode().empty else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("Total Patients", str(total), "in database", "#1a73e8")
    with col2:
        high_pct = round(high_risk / total * 100, 1) if total else 0
        _metric_card("High Risk Patients", str(high_risk), f"{high_pct}% of total", "#f44336")
    with col3:
        _metric_card("Most Common Disease", top_disease[:20], "top diagnosis", "#ff9800")
    with col4:
        _metric_card("Average Patient Age", str(avg_age), "years", "#4caf50")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Row 1: Disease bar + Risk pie ───────────────────────────────────────
    col_bar, col_pie = st.columns([3, 2])

    with col_bar:
        st.markdown("#### 🦠 Top 10 Diagnoses")
        if "diagnosis" in df.columns:
            disease_counts = df["diagnosis"].value_counts().head(10).reset_index()
            disease_counts.columns = ["Disease", "Count"]
            fig_bar = px.bar(
                disease_counts,
                x="Count",
                y="Disease",
                orientation="h",
                color="Count",
                color_continuous_scale="Blues",
                template="plotly_dark",
            )
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_pie:
        st.markdown("#### ⚠️ Risk Level Distribution")
        if "risk_level" in df.columns:
            risk_counts = df["risk_level"].str.lower().value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            fig_pie = px.pie(
                risk_counts,
                values="Count",
                names="Risk Level",
                color="Risk Level",
                color_discrete_map={
                    "low": "#4caf50",
                    "medium": "#ff9800",
                    "high": "#f44336",
                    "unknown": "#9e9e9e",
                },
                template="plotly_dark",
                hole=0.4,
            )
            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ─── Row 2: Patients per day line chart ──────────────────────────────────
    st.markdown("#### 📅 Patient Intake Over the Last 30 Days")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        date_counts = df.groupby("date").size().reset_index(name="Patients")
        date_counts = date_counts.sort_values("date")

        fig_line = px.line(
            date_counts,
            x="date",
            y="Patients",
            markers=True,
            template="plotly_dark",
            color_discrete_sequence=["#7c4dff"],
        )
        fig_line.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Date",
            yaxis_title="New Patients",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Timestamp data not available for timeline chart.")

    # ─── Analytics Agent Q&A panel ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🤖 Ask HealthAnalyst")
    analytics_q = st.text_input(
        "Ask a question about the patient data:",
        placeholder="e.g. How many high-risk patients do we have?",
        key="analytics_question",
    )
    if st.button("📈 Analyse", key="analytics_submit") and analytics_q.strip():
        with st.spinner("HealthAnalyst is computing..."):
            client = _get_client()
            result = client.chat_with_agent(
                message=analytics_q.strip(),
                intent="analytics",
            )
        answer = result.get("response", "Unable to compute analytics.")
        st.markdown(
            f"""
            <div style="
                background:#ffffff;
                border-radius:10px;
                padding:14px 18px;
                border:1px solid #e7e2da;
                border-left:4px solid #2f6feb;
                color:#1f1f1d;
                font-size:0.9rem;
                line-height:1.6;
                margin-top:8px;
            ">
                🤖 <strong>HealthAnalyst:</strong><br><br>
                {answer.replace(chr(10), "<br>")}
            </div>
            """,
            unsafe_allow_html=True,
        )
