# Content-Refresh Triage Agent

A personal agent that reuses the real feature set and feature weighting from
the Logistic Regression model I built during my FlyRank ML internship
(see `case-flyrank.html` on my portfolio for the full case study).

## How it works

1. **Scoring tool** (`scoring.py`) — a deterministic, transparent function
   that combines 6 page-health features using the same signed relative
   weights my real model learned from real data (content age hurts most,
   recent updates help, low CTR is a warning sign, etc).
2. **Agent reasoning layer** (`agent.py`) — takes the tool's score and asks
   Claude to explain *why* a page landed where it did, in plain language,
   without letting the model re-invent the score itself.
3. **UI** (`app.py`) — a Streamlit app with two modes: upload a CSV of pages
   and get a ranked shortlist, or score one page manually.

## Why the scores aren't identical to my original notebook's output

The original model was trained inside a Colab session on a private,
anonymized 9.8M-row warehouse, and its exact scaler/imputer statistics were
never saved outside that session. Rather than fake reloading a model I don't
actually have, this agent is upfront about reusing the **feature set and
relative feature importance** as a transparent rule — which is honestly what
it is. Full explanation is in the `scoring.py` docstring.

## Run it locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # from console.anthropic.com, separate from claude.ai
streamlit run app.py
```

Try it with the included `sample_pages.csv` in batch mode.

## Deploy it (free)

1. Push this folder to a GitHub repo (or a subfolder of your portfolio repo).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Point it at `app.py` in this repo.
4. Add `ANTHROPIC_API_KEY` as a secret in the app's settings (not in code).
5. You get a public URL — that's your shipped agent link for the capstone.
