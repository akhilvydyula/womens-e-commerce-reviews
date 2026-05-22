"""Format and render the 24-question survey protocol for the Streamlit report."""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard import _apply, chart_funnel_journey, chart_image_cv_radar, plotly_show
from src.survey import survey_from_ui
from src.vision_features import IMAGE_FEATURE_NAMES, extract_image_features

LIKERT_5 = {1: "Terrible", 2: "Poor", 3: "OK", 4: "Good", 5: "Perfect"}
DURABILITY = {1: "6+ months", 2: "12+ months", 3: "18+ months", 4: "24+ months", 5: "30+ months"}

SURVEY_SECTIONS: list[dict[str, Any]] = [
    {
        "id": "purchase",
        "title": "Purchase context",
        "icon": "🛒",
        "fields": [
            ("purchase_channel", "Where did you purchase?", "choice"),
            ("days_since_purchase", "Days since purchase", "days"),
            ("first_time_buyer", "First purchase from this brand?", "bool"),
            ("compared_alternatives", "Compared alternatives before buying?", "bool"),
            ("gift_purchase", "Bought as a gift?", "bool"),
        ],
    },
    {
        "id": "fit",
        "title": "Fit, sizing & wear",
        "icon": "👗",
        "fields": [
            ("fit_vs_expectation", "Fit vs expectation", "likert5"),
            ("size_vs_usual", "Size vs your usual", "choice"),
            ("alterations_needed", "Alterations needed?", "bool"),
            ("comfort_hours", "Comfortable wear-time (hours)", "hours"),
            ("worn_before_review", "Times worn before reviewing", "count"),
        ],
    },
    {
        "id": "quality",
        "title": "Material, color & quality",
        "icon": "✨",
        "fields": [
            ("material_vs_price", "Material quality vs price", "likert5"),
            ("durability_expectation", "Expected durability", "durability"),
            ("color_accuracy", "Color accuracy vs photos", "likert5"),
            ("odor_issue", "Chemical odor on arrival?", "bool"),
        ],
    },
    {
        "id": "logistics",
        "title": "Logistics & post-purchase intent",
        "icon": "📦",
        "fields": [
            ("shipping_satisfaction", "Shipping speed satisfaction", "likert5"),
            ("packaging_quality", "Packaging quality", "likert5"),
            ("support_contact", "Contacted customer support?", "bool"),
            ("return_likelihood", "Likelihood of return", "pct"),
            ("repurchase_likelihood", "Likelihood to buy again", "pct"),
        ],
    },
    {
        "id": "values",
        "title": "Usage context & values",
        "icon": "🌱",
        "fields": [
            ("occasion_formality", "Occasion formality (casual → formal)", "pct"),
            ("season_match", "Season appropriateness", "pct"),
            ("care_difficulty", "Care / maintenance difficulty", "pct"),
            ("sustainability_importance", "Sustainability importance", "pct"),
            ("price_sensitivity", "Price sensitivity", "pct"),
            ("brand_loyalty", "Brand loyalty", "pct"),
        ],
    },
    {
        "id": "vision",
        "title": "Computer vision (product photo)",
        "icon": "📷",
        "fields": [
            ("photo_uploaded", "Product photo provided?", "bool"),
        ],
    },
]

UI_KEY_TO_FEATURE = {
    "purchase_channel": "survey_purchase_channel",
    "days_since_purchase": "survey_days_since_purchase",
    "first_time_buyer": "survey_first_time_buyer",
    "compared_alternatives": "survey_compared_alternatives",
    "gift_purchase": "survey_gift_purchase",
    "fit_vs_expectation": "survey_fit_vs_expectation",
    "size_vs_usual": "survey_size_vs_usual",
    "alterations_needed": "survey_alterations_needed",
    "comfort_hours": "survey_comfort_hours",
    "worn_before_review": "survey_worn_before_review",
    "material_vs_price": "survey_material_vs_price",
    "durability_expectation": "survey_durability_expectation",
    "color_accuracy": "survey_color_accuracy",
    "odor_issue": "survey_odor_issue",
    "shipping_satisfaction": "survey_shipping_satisfaction",
    "packaging_quality": "survey_packaging_quality",
    "support_contact": "survey_support_contact",
    "return_likelihood": "survey_return_likelihood",
    "repurchase_likelihood": "survey_repurchase_likelihood",
    "occasion_formality": "survey_occasion_formality",
    "season_match": "survey_season_match",
    "care_difficulty": "survey_care_difficulty",
    "sustainability_importance": "survey_sustainability_importance",
    "price_sensitivity": "survey_price_sensitivity",
    "brand_loyalty": "survey_brand_loyalty",
    "photo_uploaded": "survey_photo_uploaded",
}

IMAGE_LABELS = {
    "img_brightness": "Brightness",
    "img_contrast": "Contrast",
    "img_saturation": "Saturation",
    "img_aspect_ratio": "Aspect ratio",
    "img_warm_tone_ratio": "Warm tone ratio",
    "img_edge_density": "Edge / texture density",
    "img_color_entropy": "Color entropy",
    "img_dominant_red": "Mean red channel",
    "img_dominant_green": "Mean green channel",
    "img_dominant_blue": "Mean blue channel",
    "img_has_image": "Image present",
}


def format_answer(key: str, value: Any, fmt: str) -> str:
    if value is None:
        return "—"
    if fmt == "bool":
        return "Yes" if value else "No"
    if fmt == "likert5":
        return LIKERT_5.get(int(value), str(value))
    if fmt == "durability":
        return DURABILITY.get(int(value), str(value))
    if fmt == "pct":
        return f"{float(value):.0%}"
    if fmt == "hours":
        return f"{float(value):.1f} h"
    if fmt == "days":
        return f"{int(value)} days"
    if fmt == "count":
        return f"{int(value)} times"
    if fmt == "choice" and key == "purchase_channel":
        return str(value)
    if fmt == "choice" and key == "size_vs_usual":
        return str(value)
    return str(value)


def format_feature_value(col: str, val: float) -> str:
    if col == "survey_purchase_channel":
        return {0.0: "Online", 1.0: "In-store", 2.0: "Gift"}.get(val, f"{val:.0f}")
    if col == "survey_size_vs_usual":
        return {-1.0: "Runs small", 0.0: "True to size", 1.0: "Runs large"}.get(val, f"{val:.1f}")
    if col in {
        "survey_return_likelihood",
        "survey_repurchase_likelihood",
        "survey_occasion_formality",
        "survey_season_match",
        "survey_care_difficulty",
        "survey_sustainability_importance",
        "survey_price_sensitivity",
        "survey_brand_loyalty",
        "survey_first_time_buyer",
        "survey_compared_alternatives",
        "survey_alterations_needed",
        "survey_odor_issue",
        "survey_support_contact",
        "survey_gift_purchase",
        "survey_photo_uploaded",
    }:
        if col.endswith("_likelihood") or col.endswith("_importance") or col.endswith("_loyalty") or col.endswith("_sensitivity") or col in ("survey_occasion_formality", "survey_season_match", "survey_care_difficulty"):
            return f"{val:.0%}" if val <= 1 else f"{val:.2f}"
        return "Yes" if val >= 0.5 else "No"
    if col.startswith("survey_") and col not in ("survey_days_since_purchase", "survey_comfort_hours", "survey_worn_before_review"):
        if val in (1.0, 2.0, 3.0, 4.0, 5.0):
            return LIKERT_5.get(int(val), f"{val:.1f}")
    if col == "survey_days_since_purchase":
        return f"{val:.0f} days"
    if col == "survey_comfort_hours":
        return f"{val:.1f} h"
    if col == "survey_worn_before_review":
        return f"{val:.0f} wears"
    return f"{val:.3f}"


def build_report_table(
    answers: dict[str, Any],
    encoded: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for section in SURVEY_SECTIONS:
        for key, label, fmt in section["fields"]:
            feat = UI_KEY_TO_FEATURE.get(key)
            rows.append(
                {
                    "Section": f"{section['icon']} {section['title']}",
                    "Question": label,
                    "Your answer": format_answer(key, answers.get(key), fmt),
                    "Model signal": format_feature_value(feat, encoded[feat]) if feat else "—",
                    "Feature": feat or "",
                }
            )
    return pd.DataFrame(rows)


def render_research_report(
    *,
    answers: dict[str, Any],
    core: dict[str, Any],
    breakdown: dict[str, dict[str, float]],
    feat_row: pd.Series | None,
    image_bytes: bytes | None,
    pred: int,
    score: float,
    emoji_summary: str | None = None,
    ai_insights: list[str] | None = None,
) -> None:
    """Full-width report: review + all survey answers + CV + modality summary."""
    encoded = survey_from_ui(answers)

    st.markdown("---")
    st.markdown(
        '<p class="section-title">Survey report · Complete response profile</p>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    n_questions = sum(len(s["fields"]) for s in SURVEY_SECTIONS)
    m1.metric("Survey questions answered", str(n_questions))
    m2.metric("Model verdict", "Recommended" if pred == 1 else "Not recommended")
    m3.metric("P(recommended)", f"{score:.1%}")
    m4.metric("Photo analyzed", "Yes" if answers.get("photo_uploaded") else "No")

    if emoji_summary or ai_insights:
        st.markdown("### 🧠 Local AI business insights")
        if emoji_summary:
            st.write(emoji_summary)
        if ai_insights:
            for item in ai_insights:
                st.markdown(f"- {item}")
        st.caption("Generated locally from model outputs and engineered features (no external API).")

    st.markdown("### ✍️ Written review & catalog")
    rc1, rc2 = st.columns([2, 1])
    with rc1:
        st.markdown(f"**{core.get('Title', '') or '(no title)'}**")
        st.write(core.get("Review Text", ""))
    with rc2:
        st.markdown(
            f"- **Rating:** {core.get('Rating', 0):.1f} ★\n"
            f"- **Age:** {core.get('Age', 0):.0f}\n"
            f"- **Helpful votes:** {core.get('Positive Feedback Count', 0)}\n"
            f"- **Division:** {core.get('Division Name', '')}\n"
            f"- **Department:** {core.get('Department Name', '')}\n"
            f"- **Class:** {core.get('Class Name', '')}"
        )

    # Intent gauges
    st.markdown("### 🎯 Post-purchase intent signals")
    g1, g2 = st.columns(2)
    with g1:
        ret = float(answers.get("return_likelihood", 0))
        st.progress(min(1.0, ret), text="Return likelihood")
        st.caption(format_answer("return_likelihood", ret, "pct"))
    with g2:
        rep = float(answers.get("repurchase_likelihood", 0))
        st.progress(min(1.0, rep), text="Repurchase likelihood")
        st.caption(format_answer("repurchase_likelihood", rep, "pct"))

    likert_keys = [
        ("fit_vs_expectation", "Fit vs expectation"),
        ("material_vs_price", "Material vs price"),
        ("color_accuracy", "Color accuracy"),
        ("shipping_satisfaction", "Shipping"),
        ("packaging_quality", "Packaging"),
    ]
    likert_df = pd.DataFrame(
        [
            {"Dimension": label, "Score": int(answers.get(key, 3))}
            for key, label in likert_keys
        ]
    )
    c_a, c_b = st.columns(2)
    with c_a:
        fig = px.bar(
            likert_df,
            x="Score",
            y="Dimension",
            orientation="h",
            range_x=[0, 5.5],
            color="Score",
            color_continuous_scale=["#7f1d1d", "#a78bfa", "#34d399"],
        )
        _apply(fig, "Satisfaction profile (1–5)", 280)
        plotly_show(fig, "dossier_likert_profile")
    with c_b:
        chart_funnel_journey(answers, score, "dossier_report_funnel")
        chart_image_cv_radar(image_bytes, "dossier_report_cv")

    report_df = build_report_table(answers, encoded)

    for section in SURVEY_SECTIONS:
        sid = section["id"]
        icon = section["icon"]
        title = section["title"]
        section_label = f"{icon} {title}"
        block = report_df[report_df["Section"] == section_label]

        with st.expander(f"{icon} {title} — {len(block)} questions", expanded=sid in ("purchase", "fit")):
            if sid == "vision" and image_bytes:
                vc1, vc2 = st.columns([1, 1.2])
                with vc1:
                    st.image(image_bytes, caption="Uploaded product photo", width="stretch")
                with vc2:
                    img_feats = extract_image_features(image_bytes)
                    st.markdown("**Extracted CV descriptors**")
                    for k in IMAGE_FEATURE_NAMES:
                        if k == "img_has_image":
                            continue
                        label = IMAGE_LABELS.get(k, k)
                        val = img_feats.get(k, 0.0)
                        if k in ("img_brightness", "img_contrast", "img_saturation", "img_dominant_red", "img_dominant_green", "img_dominant_blue"):
                            st.metric(label, f"{val:.3f}")
                        else:
                            st.metric(label, f"{val:.2f}")

            if not block.empty:
                st.dataframe(
                    block[["Question", "Your answer", "Model signal"]].rename(
                        columns={"Model signal": "Encoded for ML"}
                    ),
                    width="stretch",
                    hide_index=True,
                )

    if feat_row is not None and breakdown:
        st.markdown("### 🔬 How your answers became model features")
        t1, t2, t3, t4 = st.tabs(["NLP", "Vision (text)", "Vision (image)", "Survey"])
        groups = [
            ("NLP", "NLP linguistics"),
            ("Vision (text)", "Vision (text)"),
            ("Vision (image)", "Vision (image)"),
            ("Survey", "Survey / behavior"),
        ]
        for tab, (_, gname) in zip([t1, t2, t3, t4], groups):
            with tab:
                feats = breakdown.get(gname, {})
                if not feats:
                    st.caption("No features in this group.")
                    continue
                df = pd.DataFrame(
                    [{"Feature": k.replace("_", " "), "Value": v} for k, v in sorted(feats.items(), key=lambda x: -abs(x[1]))]
                )
                st.dataframe(df, width="stretch", hide_index=True)

    with st.expander("📊 Full survey table (export-friendly)", expanded=False):
        st.dataframe(report_df, width="stretch", hide_index=True)
