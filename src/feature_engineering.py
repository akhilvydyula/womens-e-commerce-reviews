"""Fuse NLP, vision-text, image CV, and survey signals into one feature matrix."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.nlp_features import extract_nlp_features
from src.survey import augment_review_text, default_survey_frame
from src.vision_features import (
    IMAGE_FEATURE_NAMES,
    VISION_TEXT_FEATURE_NAMES,
    extract_image_features,
    extract_vision_text_features,
)


def enrich_dataframe(
    df: pd.DataFrame,
    *,
    survey: dict[str, float] | None = None,
    image_bytes: bytes | None = None,
) -> pd.DataFrame:
    """
    Add engineered columns. Training: survey=None, image_bytes=None (zeros).
    Inference: pass user survey + optional product image.
    """
    out = df.copy()
    n = len(out)

    nlp = extract_nlp_features(out["Title"], out["Review Text"])
    vis = extract_vision_text_features(out["Title"], out["Review Text"])

    if survey is None:
        survey_df = default_survey_frame(n)
    else:
        survey_df = pd.DataFrame([survey] * n)

    if image_bytes is None:
        img_row = {k: 0.0 for k in IMAGE_FEATURE_NAMES}
        img_row["img_has_image"] = 0.0
        img_df = pd.DataFrame([img_row] * n)
    else:
        img_df = pd.DataFrame([extract_image_features(image_bytes)] * n)

    eng = pd.concat([nlp.reset_index(drop=True), vis.reset_index(drop=True), img_df, survey_df.reset_index(drop=True)], axis=1)
    out = pd.concat([out.reset_index(drop=True), eng], axis=1)

    # Augment combined text with survey tokens for TF-IDF
    texts = []
    for i in range(n):
        row_survey = survey_df.iloc[i].to_dict() if survey is None else survey
        texts.append(
            augment_review_text(
                str(out.loc[i, "Title"]),
                str(out.loc[i, "Review Text"]),
                row_survey,
            )
        )
    out["text"] = texts
    return out


def explain_features(row: pd.Series) -> dict[str, Any]:
    """Human-readable feature breakdown for the Research panel."""
    groups = {
        "NLP linguistics": [c for c in row.index if c.startswith("nlp_")],
        "Vision (text)": [c for c in row.index if c.startswith("vis_")],
        "Vision (image)": [c for c in row.index if c.startswith("img_")],
        "Survey / behavior": [c for c in row.index if c.startswith("survey_")],
    }
    return {
        group: {c: float(row[c]) for c in cols if c in row.index}
        for group, cols in groups.items()
        if cols
    }
