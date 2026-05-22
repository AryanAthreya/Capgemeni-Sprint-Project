"""Shared Streamlit theme helpers for the frontend."""

from __future__ import annotations

import streamlit as st


def apply_notion_theme() -> None:
    """Inject a light, Notion-inspired visual theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg: #f7f5f2;
            --surface: #ffffff;
            --surface-2: #fbfaf8;
            --border: #e7e2da;
            --text: #1f1f1d;
            --muted: #6f6b66;
            --accent: #2f6feb;
            --accent-soft: rgba(47, 111, 235, 0.12);
            --shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 255, 255, 0.9), transparent 30%),
                linear-gradient(180deg, #fbfaf8 0%, #f3f0ea 100%);
            color: var(--text);
        }

        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.82);
            backdrop-filter: blur(18px);
            border-right: 1px solid var(--border);
        }

        #MainMenu, footer, header { visibility: hidden; }

        .stMarkdown, .stText, .stRadio, .stSelectbox, .stTextInput, .stTextArea {
            color: var(--text);
        }

        .stRadio label, .stCheckbox label, label, p, li {
            color: var(--text) !important;
        }

        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div,
        .stNumberInput input {
            background: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }

        .stButton > button, .stForm button {
            background: linear-gradient(180deg, #2f6feb 0%, #2557c8 100%);
            color: white;
            border: 1px solid rgba(37, 87, 200, 0.15);
            border-radius: 14px;
            font-weight: 600;
            box-shadow: 0 6px 20px rgba(47, 111, 235, 0.18);
        }

        .stButton > button:hover, .stForm button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 28px rgba(47, 111, 235, 0.24);
        }

        .streamlit-expanderHeader {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            color: var(--text) !important;
        }

        hr {
            border-color: var(--border);
        }

        .notion-shell {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow);
        }

        .notion-page-title {
            color: var(--text);
            font-size: 2rem;
            line-height: 1.15;
            margin-bottom: 0.25rem;
            letter-spacing: -0.03em;
        }

        .notion-page-subtitle {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .notion-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: var(--shadow);
        }

        .notion-card-muted {
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 18px;
        }

        .notion-pill {
            display: inline-block;
            background: var(--accent-soft);
            color: var(--accent);
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
