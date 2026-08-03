"""
Content-refresh priority scoring.

The feature set and relative weighting here are taken directly from the
Logistic Regression model trained during the FlyRank ML internship
(work/notebooks/w05_model.ipynb, ML-08). That model was trained on a
private, anonymized 9.8M-row warehouse and its exact scaler/imputer
statistics live only in that Colab session -- they were never saved to
the repo, so this agent can't reload the literal serialized pipeline.

What it DOES reuse honestly: the same 6 features, and the same signed,
relative importance the real model learned from real data --

    content_age_days         -0.396   (older content -> LESS likely declining --
                                        counterintuitive; possibly long-lived
                                        content has already proven it's stable)
    days_since_last_update    0.237   (not touched in a while -> more likely declining)
    ctr                       -0.202  (higher CTR -> less likely declining)
    engagement_rate          -0.015   (weak effect)
    search_volume              0.008  (weak effect)
    avg_position                0.002  (weak effect)

This module turns that into a transparent 0-1 priority score. It is a
decision-support shortlist for a human editor, not a certified model
output -- exactly the same framing used in the original notebook's
error analysis.
"""

import pandas as pd

# Signed relative weights, taken directly from the trained model's
# standardized coefficients (see docstring above).
WEIGHTS = {
    "content_age_days": -0.396152,
    "days_since_last_update": 0.236994,
    "ctr": -0.202318,
    "engagement_rate": -0.015001,
    "search_volume": 0.007540,
    "avg_position": 0.001573,
}

REQUIRED_COLUMNS = list(WEIGHTS.keys())


def score_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a batch of pages. df must contain the 6 required feature columns
    plus an identifying column (content_id or url).
    Returns df with a new 'priority_score' column (0-1, higher = review sooner),
    sorted descending.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()

    # Normalize each feature to 0-1 across the batch (min-max), so the
    # signed weights combine on a comparable scale. With only one row,
    # normalization is skipped (see score_single below) since there's
    # no distribution to normalize against.
    normed = pd.DataFrame(index=work.index)
    for col in REQUIRED_COLUMNS:
        col_min, col_max = work[col].min(), work[col].max()
        if col_max > col_min:
            normed[col] = (work[col] - col_min) / (col_max - col_min)
        else:
            normed[col] = 0.5  # no spread in this batch, neutral

    raw_score = sum(normed[col] * WEIGHTS[col] for col in REQUIRED_COLUMNS)

    # Min-max normalize the combined score to a clean 0-1 range for display.
    rs_min, rs_max = raw_score.min(), raw_score.max()
    if rs_max > rs_min:
        work["priority_score"] = (raw_score - rs_min) / (rs_max - rs_min)
    else:
        work["priority_score"] = 0.5

    return work.sort_values("priority_score", ascending=False).reset_index(drop=True)


# Reasonable fixed ranges for single-page scoring, where there's no batch
# to normalize against. These are rough, documented assumptions -- not
# learned from data -- and are clearly labeled as such in the UI.
SINGLE_PAGE_RANGES = {
    "content_age_days": (0, 1000),
    "days_since_last_update": (0, 365),
    "ctr": (0.0, 0.30),
    "engagement_rate": (0.0, 1.0),
    "search_volume": (0, 50000),
    "avg_position": (1, 100),
}


def score_single(metrics: dict) -> float:
    """Score one page against fixed assumed ranges (see SINGLE_PAGE_RANGES)."""
    normed = {}
    for col, (lo, hi) in SINGLE_PAGE_RANGES.items():
        v = max(lo, min(hi, metrics[col]))  # clamp into range
        normed[col] = (v - lo) / (hi - lo) if hi > lo else 0.5

    raw = sum(normed[col] * WEIGHTS[col] for col in REQUIRED_COLUMNS)

    # Rough rescale of the raw signed sum into 0-1 using the theoretical
    # min/max of the weighted sum given the ranges above.
    theoretical_min = sum(min(0, w) for w in WEIGHTS.values())
    theoretical_max = sum(max(0, w) for w in WEIGHTS.values())
    span = theoretical_max - theoretical_min
    return (raw - theoretical_min) / span if span > 0 else 0.5
