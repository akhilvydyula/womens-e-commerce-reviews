"""Local business-facing insights generated from model outputs and features."""
from __future__ import annotations

from typing import Any

import pandas as pd


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _feature_label(name: str) -> str:
    return (
        str(name)
        .replace("survey_", "")
        .replace("nlp_", "")
        .replace("vis_", "")
        .replace("img_", "")
        .replace("_", " ")
    )


def _top_signals(
    breakdown: dict[str, dict[str, float]],
    limit: int = 3,
) -> list[tuple[str, str, float]]:
    ranked: list[tuple[str, str, float]] = []
    for group, feats in (breakdown or {}).items():
        for feat, val in feats.items():
            if feat == "img_has_image":
                continue
            ranked.append((group, feat, float(val)))
    ranked.sort(key=lambda x: abs(x[2]), reverse=True)
    return ranked[:limit]


def build_review_emoji_summary(
    core: dict[str, Any],
    pred: int,
    score: float,
    answers: dict[str, Any] | None = None,
) -> str:
    """Compact emoji summary line for business users."""
    answers = answers or {}
    tags: list[str] = []

    rating = _f(core.get("Rating"), 0.0)
    helpful = _f(core.get("Positive Feedback Count"), 0.0)
    repurchase = _f(answers.get("repurchase_likelihood"), 0.5)
    returns = _f(answers.get("return_likelihood"), 0.2)

    if pred == 1 and score >= 0.8:
        tags.append("✅ Strong recommend signal")
    elif score >= 0.5:
        tags.append("🤔 Mixed recommend signal")
    else:
        tags.append("⚠️ Low recommend signal")

    if rating >= 4:
        tags.append("⭐ High rating")
    elif rating <= 2.5:
        tags.append("⭐ Low rating")

    if repurchase >= 0.7:
        tags.append("🔁 High repurchase intent")
    elif repurchase <= 0.35:
        tags.append("🧍 Weak repurchase intent")

    if returns >= 0.55:
        tags.append("↩️ High return risk")
    elif returns <= 0.25:
        tags.append("📦 Low return risk")

    if helpful >= 8:
        tags.append("👍 Helpful-vote traction")

    if bool(answers.get("photo_uploaded")):
        tags.append("📷 Photo-verified review")

    return " | ".join(tags[:5])


def build_local_ai_insights(
    core: dict[str, Any],
    pred: int,
    score: float,
    answers: dict[str, Any] | None = None,
    breakdown: dict[str, dict[str, float]] | None = None,
) -> list[str]:
    """Actionable local insights for business users (no external LLM/API)."""
    answers = answers or {}
    insights: list[str] = []

    label = "recommended" if pred == 1 else "not recommended"
    insights.append(f"Model outcome: {label} with {score:.1%} confidence.")

    rating = _f(core.get("Rating"), 0.0)
    if rating >= 4 and score < 0.65:
        insights.append("High rating but weaker model confidence suggests hidden friction in review language.")
    elif rating <= 2.5 and score >= 0.55:
        insights.append("Low star rating with moderate confidence indicates mixed signals worth manual review.")

    repurchase = _f(answers.get("repurchase_likelihood"), 0.5)
    returns = _f(answers.get("return_likelihood"), 0.2)
    fit = _f(answers.get("fit_vs_expectation"), 3.0)
    material = _f(answers.get("material_vs_price"), 3.0)
    shipping = _f(answers.get("shipping_satisfaction"), 3.0)

    if returns >= 0.55:
        insights.append("Risk watch: return likelihood is high; prioritize fit clarity and pre-purchase guidance.")
    if repurchase >= 0.7:
        insights.append("Growth signal: repurchase intent is strong; candidate for loyalty or upsell campaigns.")
    if fit <= 2.5 or material <= 2.5:
        insights.append("Product signal: fit/material perception is weak; review sizing content and fabric expectations.")
    if shipping <= 2.5:
        insights.append("Operations signal: shipping experience is below target; investigate fulfillment/last-mile issues.")

    top = _top_signals(breakdown or {})
    if top:
        terms = ", ".join(
            f"{group.split('(')[0].strip()}: {_feature_label(name)}={value:.2f}"
            for group, name, value in top
        )
        insights.append(f"Strongest measured signals: {terms}.")

    if pred == 1 and score >= 0.75:
        insights.append("Recommended action: amplify this review in merchandising and social proof placements.")
    elif pred == 0 and score >= 0.65:
        insights.append("Recommended action: route to quality CX queue and trigger root-cause follow-up.")
    else:
        insights.append("Recommended action: monitor additional reviews before making category-level changes.")

    return insights[:5]


def attach_batch_business_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add emoji summary + local AI insight columns for batch scoring output."""
    out = df.copy()

    def _row_summary(row: pd.Series) -> str:
        score = _f(row.get("score"), 0.0)
        pred = int(_f(row.get("prediction"), 0.0))
        rating = _f(row.get("Rating"), 0.0)
        tags: list[str] = []

        if pred == 1 and score >= 0.8:
            tags.append("✅ Strong buy signal")
        elif score >= 0.5:
            tags.append("🤔 Mixed signal")
        else:
            tags.append("⚠️ Risk signal")

        if rating >= 4:
            tags.append("⭐ High rating")
        elif 0 < rating <= 2.5:
            tags.append("⭐ Low rating")

        if _f(row.get("Positive Feedback Count"), 0.0) >= 8:
            tags.append("👍 Helpful traction")

        return " | ".join(tags)

    def _row_insight(row: pd.Series) -> str:
        score = _f(row.get("score"), 0.0)
        pred = int(_f(row.get("prediction"), 0.0))
        rating = _f(row.get("Rating"), 0.0)

        if pred == 1 and score >= 0.8:
            return "Promote in high-conversion slots; confidence and sentiment are aligned."
        if pred == 0 and score >= 0.65:
            return "Prioritize remediation workflow; strong non-recommend pattern."
        if rating >= 4 and score < 0.6:
            return "Star rating is high but text signal is mixed; validate review consistency."
        return "Monitor and collect more evidence before large-scale merchandising changes."

    out["business_emoji"] = out.apply(_row_summary, axis=1)
    out["local_ai_insight"] = out.apply(_row_insight, axis=1)
    return out
