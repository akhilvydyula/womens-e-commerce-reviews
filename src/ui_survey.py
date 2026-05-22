"""Streamlit research questionnaire widgets."""
from __future__ import annotations

from typing import Any

import streamlit as st


def render_research_survey() -> dict[str, Any]:
    """26 structured questions — purchase, fit, quality, logistics, values, multimodal."""
    answers: dict[str, Any] = {}
    q = 0

    def _q(label: str) -> str:
        nonlocal q
        q += 1
        return f"**Q{q}.** {label}"

    with st.expander("1 · Purchase context (5 questions)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            answers["purchase_channel"] = st.selectbox(
                _q("Where did you purchase?"),
                ["Online", "In-store", "Gift"],
                help="Channel effects return friction and expectation calibration.",
            )
            answers["days_since_purchase"] = st.slider(
                _q("Days since purchase"),
                0,
                90,
                14,
                help="Recency bias: honeymoon vs defect-discovery window.",
            )
        with c2:
            answers["first_time_buyer"] = st.toggle(_q("First purchase from this brand?"))
            answers["compared_alternatives"] = st.toggle(_q("Compared alternatives before buying?"))
            answers["gift_purchase"] = st.toggle(_q("Bought as a gift?"))

    with st.expander("2 · Fit, sizing & wear experience (5 questions)"):
        c1, c2 = st.columns(2)
        with c1:
            answers["fit_vs_expectation"] = st.select_slider(
                _q("Fit vs expectation"),
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: ["Terrible", "Poor", "OK", "Good", "Perfect"][x - 1],
            )
            answers["size_vs_usual"] = st.radio(
                _q("Size vs your usual"),
                ["Runs small", "True to size", "Runs large"],
                horizontal=True,
            )
            answers["alterations_needed"] = st.toggle(_q("Alterations needed?"))
        with c2:
            answers["comfort_hours"] = st.slider(
                _q("Comfortable wear-time (hours) before discomfort"),
                0.0,
                12.0,
                4.0,
                0.5,
            )
            answers["worn_before_review"] = st.slider(
                _q("Times worn before reviewing"),
                0,
                20,
                2,
            )

    with st.expander("3 · Material, color & perceived quality (4 questions)"):
        c1, c2 = st.columns(2)
        with c1:
            answers["material_vs_price"] = st.select_slider(
                _q("Material quality vs price paid"),
                options=[1, 2, 3, 4, 5],
                value=3,
            )
            answers["durability_expectation"] = st.select_slider(
                _q("Expected durability (months)"),
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: f"{x * 6}+ months",
            )
        with c2:
            answers["color_accuracy"] = st.select_slider(
                _q("Color accuracy vs website photos"),
                options=[1, 2, 3, 4, 5],
                value=3,
            )
            answers["odor_issue"] = st.toggle(_q("Chemical odor / off-smell on arrival?"))

    with st.expander("4 · Logistics & post-purchase intent (5 questions)"):
        c1, c2 = st.columns(2)
        with c1:
            answers["shipping_satisfaction"] = st.select_slider(
                _q("Shipping speed satisfaction"),
                options=[1, 2, 3, 4, 5],
                value=3,
            )
            answers["packaging_quality"] = st.select_slider(
                _q("Packaging quality"),
                options=[1, 2, 3, 4, 5],
                value=3,
            )
            answers["support_contact"] = st.toggle(_q("Contacted customer support?"))
        with c2:
            answers["return_likelihood"] = st.slider(
                _q("Likelihood of return (0–1)"),
                0.0,
                1.0,
                0.15,
                0.05,
            )
            answers["repurchase_likelihood"] = st.slider(
                _q("Likelihood to buy again (0–1)"),
                0.0,
                1.0,
                0.55,
                0.05,
            )

    with st.expander("5 · Usage context & values (6 questions)"):
        c1, c2 = st.columns(2)
        with c1:
            answers["occasion_formality"] = st.slider(
                _q("Occasion formality (casual → formal)"),
                0.0,
                1.0,
                0.5,
                0.05,
            )
            answers["season_match"] = st.slider(
                _q("Season appropriateness"),
                0.0,
                1.0,
                0.5,
                0.05,
            )
            answers["care_difficulty"] = st.slider(
                _q("Care / maintenance difficulty"),
                0.0,
                1.0,
                0.3,
                0.05,
            )
        with c2:
            answers["sustainability_importance"] = st.slider(
                _q("Sustainability importance to you"),
                0.0,
                1.0,
                0.5,
                0.05,
            )
            answers["price_sensitivity"] = st.slider(
                _q("Price sensitivity"),
                0.0,
                1.0,
                0.5,
                0.05,
            )
            answers["brand_loyalty"] = st.slider(
                _q("Brand loyalty"),
                0.0,
                1.0,
                0.5,
                0.05,
            )

    with st.expander("6 · Computer vision — product photo (1 question)"):
        st.caption(
            "Upload the listing or your photo. We extract brightness, color distribution, "
            "texture complexity, and warmth — aligned with vision-language features at training."
        )
        uploaded = st.file_uploader(
            "Product image (optional)",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        answers["photo_uploaded"] = uploaded is not None
        if uploaded is not None:
            st.image(uploaded, width="stretch")
        answers["_image_bytes"] = uploaded.getvalue() if uploaded else None

    return answers
