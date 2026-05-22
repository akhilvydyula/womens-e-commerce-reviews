"""Research-grade shopper questionnaire → model features + augmented text."""
from __future__ import annotations

from typing import Any

import pandas as pd

# Neutral defaults used at training time (historical CSV has no survey).
DEFAULT_SURVEY: dict[str, Any] = {
    "survey_purchase_channel": 0.0,  # 0 online, 1 in-store, 2 gift
    "survey_days_since_purchase": 14.0,
    "survey_first_time_buyer": 0.5,
    "survey_compared_alternatives": 0.5,
    "survey_fit_vs_expectation": 3.0,
    "survey_size_vs_usual": 0.0,  # -1 small, 0 TTS, 1 large
    "survey_alterations_needed": 0.0,
    "survey_comfort_hours": 4.0,
    "survey_material_vs_price": 3.0,
    "survey_durability_expectation": 3.0,
    "survey_color_accuracy": 3.0,
    "survey_odor_issue": 0.0,
    "survey_shipping_satisfaction": 3.0,
    "survey_packaging_quality": 3.0,
    "survey_support_contact": 0.0,
    "survey_return_likelihood": 0.2,
    "survey_repurchase_likelihood": 0.5,
    "survey_occasion_formality": 0.5,
    "survey_season_match": 0.5,
    "survey_care_difficulty": 0.3,
    "survey_sustainability_importance": 0.5,
    "survey_worn_before_review": 0.5,
    "survey_gift_purchase": 0.0,
    "survey_price_sensitivity": 0.5,
    "survey_brand_loyalty": 0.5,
    "survey_photo_uploaded": 0.0,
}

SURVEY_FEATURE_NAMES = list(DEFAULT_SURVEY.keys())


def survey_from_ui(answers: dict[str, Any]) -> dict[str, float]:
    """Map UI widget values to numeric survey_* columns."""
    out = dict(DEFAULT_SURVEY)
    mapping = {
        "purchase_channel": ("survey_purchase_channel", {"Online": 0.0, "In-store": 1.0, "Gift": 2.0}),
        "days_since_purchase": ("survey_days_since_purchase", None),
        "first_time_buyer": ("survey_first_time_buyer", None),
        "compared_alternatives": ("survey_compared_alternatives", None),
        "fit_vs_expectation": ("survey_fit_vs_expectation", None),
        "size_vs_usual": (
            "survey_size_vs_usual",
            {"Runs small": -1.0, "True to size": 0.0, "Runs large": 1.0},
        ),
        "alterations_needed": ("survey_alterations_needed", None),
        "comfort_hours": ("survey_comfort_hours", None),
        "material_vs_price": ("survey_material_vs_price", None),
        "color_accuracy": ("survey_color_accuracy", None),
        "odor_issue": ("survey_odor_issue", None),
        "shipping_satisfaction": ("survey_shipping_satisfaction", None),
        "packaging_quality": ("survey_packaging_quality", None),
        "support_contact": ("survey_support_contact", None),
        "return_likelihood": ("survey_return_likelihood", None),
        "repurchase_likelihood": ("survey_repurchase_likelihood", None),
        "occasion_formality": ("survey_occasion_formality", None),
        "season_match": ("survey_season_match", None),
        "care_difficulty": ("survey_care_difficulty", None),
        "sustainability_importance": ("survey_sustainability_importance", None),
        "worn_before_review": ("survey_worn_before_review", None),
        "gift_purchase": ("survey_gift_purchase", None),
        "price_sensitivity": ("survey_price_sensitivity", None),
        "brand_loyalty": ("survey_brand_loyalty", None),
        "durability_expectation": ("survey_durability_expectation", None),
        "photo_uploaded": ("survey_photo_uploaded", None),
    }
    for key, val in answers.items():
        if key not in mapping:
            continue
        col, lookup = mapping[key]
        if lookup is None:
            if isinstance(val, bool):
                out[col] = 1.0 if val else 0.0
            else:
                out[col] = float(val) if isinstance(val, (int, float)) else (1.0 if val else 0.0)
        else:
            out[col] = float(lookup.get(val, 0.0))
    return out


def augment_review_text(title: str, review: str, survey: dict[str, float]) -> str:
    """Inject structured answers as tokens for TF-IDF (multimodal fusion)."""
    tags = []
    if survey.get("survey_return_likelihood", 0) >= 0.7:
        tags.append("high_return_intent")
    if survey.get("survey_repurchase_likelihood", 0) >= 0.7:
        tags.append("high_repurchase_intent")
    if survey.get("survey_fit_vs_expectation", 3) <= 2:
        tags.append("poor_fit_experience")
    if survey.get("survey_color_accuracy", 3) <= 2:
        tags.append("color_mismatch_reported")
    if survey.get("survey_size_vs_usual", 0) < 0:
        tags.append("runs_small")
    elif survey.get("survey_size_vs_usual", 0) > 0:
        tags.append("runs_large")
    if survey.get("survey_photo_uploaded", 0) >= 1:
        tags.append("product_photo_provided")
    suffix = " ".join(tags)
    base = f"{title} {review}".strip()
    return f"{base} {suffix}".strip() if suffix else base


def default_survey_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame([DEFAULT_SURVEY] * n)
