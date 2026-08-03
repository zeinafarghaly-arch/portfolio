"""
Content-Refresh Triage Agent
-----------------------------
Reuses the feature set and relative feature importance from the
Logistic Regression model built during the FlyRank ML internship
(see scoring.py for the full explanation of what's real vs. re-derived).

Two modes:
  1. Batch (CSV upload) -- scores every page, ranks them, and lets you
     ask the agent to explain any row.
  2. Single page (manual entry) -- score one page against fixed assumed
     ranges and get an immediate explanation.

Run locally:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from scoring import score_batch, score_single, REQUIRED_COLUMNS
from agent import explain_page

st.set_page_config(page_title="Content-Refresh Triage Agent", page_icon="📈", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background-color: #070b14; color: #eef0f3; }
    h1, h2, h3 { color: #e2a03f !important; font-family: 'JetBrains Mono', monospace; }
    .stButton>button {
        background-color: #e2a03f; color: #070b14; font-weight: 700; border-radius: 8px; border: none;
    }
    .stDataFrame { background-color: #131c30; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Content-Refresh Triage Agent")
st.caption(
    "Reuses the real feature set and feature weighting from my FlyRank ML internship "
    "Logistic Regression model. Precision@500 on the original model: 0.490 → 0.528 "
    "over the baseline rule. [Read the full case study](case-flyrank.html)"
)

api_key = st.sidebar.text_input("Anthropic API key", type="password", help="Get one at console.anthropic.com")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Required columns for CSV upload:**\n\n"
    + "\n".join(f"- `{c}`" for c in ["content_id"] + REQUIRED_COLUMNS)
)

mode = st.radio("Mode", ["Batch (CSV upload)", "Single page (manual entry)"], horizontal=True)

if mode == "Batch (CSV upload)":
    uploaded = st.file_uploader("Upload a CSV of page metrics", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        try:
            scored = score_batch(df)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        st.subheader("Ranked pages")
        st.dataframe(
            scored[["content_id", "priority_score"] + REQUIRED_COLUMNS].style.format(
                {"priority_score": "{:.3f}"}
            ),
            use_container_width=True,
        )

        st.subheader("Ask the agent to explain a page")
        options = scored["content_id"].tolist()
        pick = st.selectbox("Pick a page from the ranked list above", options)

        if st.button("Explain this triage call"):
            if not api_key:
                st.warning("Add your Anthropic API key in the sidebar first.")
            else:
                row = scored[scored["content_id"] == pick].iloc[0].to_dict()
                with st.spinner("Agent is reasoning..."):
                    explanation = explain_page(row, api_key=api_key)
                st.info(explanation)

else:
    st.subheader("Enter one page's metrics")
    st.caption("Scored against fixed assumed ranges since there's no batch to normalize against — see scoring.py.")

    col1, col2 = st.columns(2)
    with col1:
        content_id = st.text_input("Page URL or ID", "example-page")
        days_since_last_update = st.number_input("Days since last update", 0, 3650, 200)
        content_age_days = st.number_input("Content age (days)", 0, 3650, 600)
        search_volume = st.number_input("Search volume", 0, 200000, 500)
    with col2:
        ctr = st.number_input("CTR", 0.0, 1.0, 0.02, format="%.4f")
        engagement_rate = st.number_input("Engagement rate", 0.0, 1.0, 0.3, format="%.3f")
        avg_position = st.number_input("Average position", 1.0, 100.0, 15.0)

    if st.button("Score this page"):
        metrics = {
            "days_since_last_update": days_since_last_update,
            "content_age_days": content_age_days,
            "search_volume": search_volume,
            "ctr": ctr,
            "engagement_rate": engagement_rate,
            "avg_position": avg_position,
        }
        priority = score_single(metrics)
        st.metric("Priority score", f"{priority:.3f}")

        if api_key:
            row = {**metrics, "content_id": content_id, "priority_score": priority}
            with st.spinner("Agent is reasoning..."):
                explanation = explain_page(row, api_key=api_key)
            st.info(explanation)
        else:
            st.warning("Add your Anthropic API key in the sidebar to get the agent's explanation.")
