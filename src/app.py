"""ReviewSense Streamlit application for multimodal recommendation analysis."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from src.config import CATEGORICAL_COLUMNS, RAW_FILE_PATH, TARGET_COLUMN
from src.dashboard import (
    FEATURE_COUNT,
    chart_gauge_triplet,
    chart_rating_sensitivity,
    chart_radar_modalities,
    inject_executive_css,
    render_feature_diagnostics,
    render_journey_analysis,
    render_live_snapshot,
    render_market_analysis,
    plotly_show,
    render_executive_hero,
    render_executive_kpis,
)
from src.data import basic_cleaning, load_raw_data
from src.inference import load_model, predict, predict_batch
from src.insights import (
    attach_batch_business_signals,
    build_local_ai_insights,
    build_review_emoji_summary,
)
from src.nlp_features import NLP_FEATURE_NAMES
from src.survey_display import render_research_report
from src.ui_survey import render_research_survey
from src.survey import SURVEY_FEATURE_NAMES
from src.vision_features import IMAGE_FEATURE_NAMES, VISION_TEXT_FEATURE_NAMES

st.set_page_config(
    page_title="ReviewSense — Multimodal Analytics",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRESETS: dict[str, dict[str, Any]] = {
    "Glowing review": {
        "Title": "Perfect fit and beautiful fabric",
        "Review Text": (
            "I absolutely love this piece. The quality exceeded my expectations, "
            "fits true to size, and I have already recommended it to friends."
        ),
        "Rating": 5.0,
        "Age": 34.0,
        "Positive Feedback Count": 12,
        "Division Name": "General",
        "Department Name": "Dresses",
        "Class Name": "Dresses",
    },
    "Disappointed": {
        "Title": "Not worth the price",
        "Review Text": (
            "Material feels cheap and sizing runs small. Color was different "
            "from the photos. Very disappointed and would not recommend."
        ),
        "Rating": 2.0,
        "Age": 28.0,
        "Positive Feedback Count": 2,
        "Division Name": "General",
        "Department Name": "Tops",
        "Class Name": "Blouses",
    },
    "Mixed feelings": {
        "Title": "Okay but nothing special",
        "Review Text": "Decent for the sale price. Comfortable for casual wear.",
        "Rating": 3.0,
        "Age": 41.0,
        "Positive Feedback Count": 0,
        "Division Name": "General Petite",
        "Department Name": "Bottoms",
        "Class Name": "Pants",
    },
}


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame | None:
    if not RAW_FILE_PATH.exists():
        return None
    return basic_cleaning(load_raw_data())


@st.cache_resource(show_spinner="Loading prediction model…")
def get_model():
    return load_model()


@st.cache_data
def categorical_options(_df: pd.DataFrame | None) -> dict[str, list[str]]:
    opts = {col: ["Unknown"] for col in CATEGORICAL_COLUMNS}
    if _df is None:
        return opts
    for col in CATEGORICAL_COLUMNS:
        if col in _df.columns:
            opts[col] = sorted(_df[col].dropna().astype(str).unique().tolist())
    return opts


def core_row(f: dict[str, Any]) -> dict[str, Any]:
    return {
        "Title": f["title"],
        "Review Text": f["review"],
        "Age": f["age"],
        "Positive Feedback Count": f["feedback"],
        "Rating": f["rating"],
        "Division Name": f["division"],
        "Department Name": f["department"],
        "Class Name": f["class_name"],
    }


def render_verdict(pred: int, score: float, actual: int | None = None) -> None:
    label = "Recommended" if pred == 1 else "Not recommended"
    css = "verdict-yes" if pred == 1 else "verdict-no"
    st.markdown(
        f'<div class="verdict-card {css}"><p class="verdict-title">{label}</p>'
        f'<p class="verdict-sub">P(recommended) = {score:.1%}</p></div>',
        unsafe_allow_html=True,
    )
    if actual is not None:
        act = "Recommended" if actual == 1 else "Not recommended"
        (st.success if int(actual) == pred else st.warning)(f"Actual: **{act}**")


def tab_command_center(df: pd.DataFrame | None) -> None:
    report = st.session_state.get("last_report")
    render_executive_kpis(df, report)
    if report and (report.get("emoji_summary") or report.get("ai_insights")):
        st.markdown('<p class="section-title">Business summary · Local AI</p>', unsafe_allow_html=True)
        if report.get("emoji_summary"):
            st.write(report["emoji_summary"])
        for item in report.get("ai_insights", []):
            st.markdown(f"- {item}")
    if report:
        render_live_snapshot(report, prefix="command")
    else:
        st.info("Run **Predict** to generate live session charts.")


def tab_predict(model, df: pd.DataFrame | None, cat_opts: dict[str, list[str]]) -> None:
    if "form" not in st.session_state:
        st.session_state.form = {
            "title": "",
            "review": "",
            "rating": 5.0,
            "age": 35.0,
            "feedback": 0,
            "division": cat_opts["Division Name"][0],
            "department": cat_opts["Department Name"][0],
            "class_name": cat_opts["Class Name"][0],
        }

    st.markdown('<p class="section-title">Quick scenarios</p>', unsafe_allow_html=True)
    cols = st.columns(len(PRESETS))
    for col, (name, preset) in zip(cols, PRESETS.items()):
        with col:
            if st.button(name, width="stretch", key=f"p_{name}"):
                st.session_state.form.update(
                    {
                        "title": preset["Title"],
                        "review": preset["Review Text"],
                        "rating": preset["Rating"],
                        "age": preset["Age"],
                        "feedback": preset["Positive Feedback Count"],
                        "division": preset["Division Name"],
                        "department": preset["Department Name"],
                        "class_name": preset["Class Name"],
                    }
                )
                st.rerun()

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        f = st.session_state.form
        st.markdown("#### Review & catalog")
        f["title"] = st.text_input("Title", value=f["title"])
        f["review"] = st.text_area("Review", value=f["review"], height=100)
        c1, c2, c3 = st.columns(3)
        f["rating"] = c1.slider("Rating", 1.0, 5.0, float(f["rating"]), 0.5)
        f["age"] = c2.number_input("Age", 0.0, 120.0, float(f["age"]))
        f["feedback"] = c3.number_input("Helpful votes", 0, value=int(f["feedback"]))
        f["division"] = st.selectbox("Division", cat_opts["Division Name"])
        c4, c5 = st.columns(2)
        f["department"] = c4.selectbox("Department", cat_opts["Department Name"])
        f["class_name"] = c5.selectbox("Class", cat_opts["Class Name"])

        st.markdown('<p class="section-title">Survey form · 26 questions</p>', unsafe_allow_html=True)
        survey_answers = render_research_survey()
        image_bytes = survey_answers.pop("_image_bytes", None)
        analyze = st.button("Run analysis →", type="primary", width="stretch")
        if df is not None and st.button("Random review", width="stretch"):
            s = df.sample(1).iloc[0]
            st.session_state.form.update(
                {
                    "title": str(s.get("Title", "")),
                    "review": str(s.get("Review Text", "")),
                    "rating": float(s.get("Rating", 5) or 5),
                    "age": float(s.get("Age", 35) or 35),
                    "feedback": int(s.get("Positive Feedback Count", 0) or 0),
                    "division": str(s.get("Division Name", "Unknown")),
                    "department": str(s.get("Department Name", "Unknown")),
                    "class_name": str(s.get("Class Name", "Unknown")),
                }
            )
            st.session_state["_sample_actual"] = int(s[TARGET_COLUMN]) if TARGET_COLUMN in s.index else None
            st.rerun()

    with right:
        st.markdown("#### AI verdict")
        box = st.empty()
        if analyze:
            f = st.session_state.form
            if not f["review"].strip() and not f["title"].strip():
                box.warning("Enter a title or review.")
            else:
                try:
                    pred, score, feat_df, breakdown = predict(
                        model, core_row(f), survey_answers, image_bytes
                    )
                except (ValueError, Exception) as exc:
                    box.error(str(exc))
                    st.code("make train", language="bash")
                    return
                actual = st.session_state.pop("_sample_actual", None)
                st.session_state.last_breakdown = breakdown
                core = core_row(f)
                emoji_summary = build_review_emoji_summary(core, pred, score, survey_answers)
                ai_insights = build_local_ai_insights(core, pred, score, survey_answers, breakdown)
                st.session_state.last_report = {
                    "answers": dict(survey_answers),
                    "core": core,
                    "breakdown": breakdown,
                    "feat_row": feat_df.iloc[0],
                    "image_bytes": image_bytes,
                    "pred": pred,
                    "score": score,
                    "emoji_summary": emoji_summary,
                    "ai_insights": ai_insights,
                }
                with box.container():
                    render_verdict(pred, score, actual)
                    st.markdown("#### Emoji summary")
                    st.write(emoji_summary)
                    st.markdown("#### Local AI insights")
                    for item in ai_insights:
                        st.markdown(f"- {item}")
        elif st.session_state.get("last_report"):
            box.info("Session cached — see **Overview** & **Survey report**.")
        else:
            box.markdown(
                '<div class="glass-card" style="text-align:center;color:#a1a1aa;">'
                "Complete inputs and run <b>analysis</b>.</div>",
                unsafe_allow_html=True,
            )

    report = st.session_state.get("last_report")
    if report:
        st.markdown('<p class="section-title">Session analytics</p>', unsafe_allow_html=True)
        render_journey_analysis(report, prefix="predict_session")
        f = st.session_state.form
        curve = []
        base = core_row(f)
        for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
            try:
                _, sc, _, _ = predict(model, {**base, "Rating": r})
                curve.append({"Rating": r, "P(recommended)": sc})
            except Exception:
                pass
        chart_rating_sensitivity(curve, "predict_sensitivity")


def tab_science() -> None:
    breakdown = st.session_state.get("last_breakdown")
    report = st.session_state.get("last_report")
    if not breakdown:
        st.info("Run analysis on **Predict** to unlock feature diagnostics.")
        return
    st.markdown('<p class="section-title">Multimodal feature diagnostics</p>', unsafe_allow_html=True)
    st.caption(
        f"{len(NLP_FEATURE_NAMES)} NLP · {len(VISION_TEXT_FEATURE_NAMES)} vision-text · "
        f"{len(IMAGE_FEATURE_NAMES)} CV · {len(SURVEY_FEATURE_NAMES)} survey · TF-IDF"
    )
    if report:
        render_feature_diagnostics(report, prefix="science")


def tab_explore(df: pd.DataFrame) -> None:
    render_market_analysis(df, prefix="explore")


def tab_batch(model) -> None:
    st.markdown('<p class="section-title">Batch scoring</p>', unsafe_allow_html=True)
    up = st.file_uploader("Upload CSV", type=["csv"])
    if not up:
        return
    raw = pd.read_csv(up)
    st.dataframe(raw.head(8), width="stretch")
    if st.button("Score portfolio", type="primary"):
        out = predict_batch(model, raw)
        out = attach_batch_business_signals(out)
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{len(out):,}")
        m2.metric("Recommend rate", f"{(out['prediction']==1).mean():.1%}")
        m3.metric("Avg confidence", f"{out['score'].mean():.1%}")
        import plotly.express as px
        from src.dashboard import _apply, plotly_show

        fig = px.histogram(out, x="score", color="recommended_label", barmode="overlay", opacity=0.75)
        _apply(fig, "Score distribution · Batch run", 360)
        plotly_show(fig, "batch_hist")
        if "Rating" in out.columns:
            fig3 = px.scatter_3d(
                out.sample(min(500, len(out)), random_state=0),
                x="Rating",
                y="score",
                z="prediction",
                color="recommended_label",
                opacity=0.7,
            )
            _apply(fig3, "3D · Batch portfolio (rating × confidence × class)", 420)
            plotly_show(fig3, "batch_scatter3d")
        st.markdown("#### Batch business summary (local AI)")
        cols = ["recommended_label", "score", "business_emoji", "local_ai_insight"]
        show_cols = [c for c in cols if c in out.columns]
        st.dataframe(out[show_cols].head(20), width="stretch")
        st.download_button(
            "Download",
            out.to_csv(index=False).encode("utf-8"),
            "scored.csv",
            "text/csv",
            width="stretch",
        )


def main() -> None:
    inject_executive_css()
    try:
        model = get_model()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.code("make train", language="bash")
        st.stop()

    df = load_dataset()
    cat_opts = categorical_options(df)
    render_executive_hero(
        feature_count=FEATURE_COUNT,
        corpus_size=(len(df) if df is not None else None),
        has_session=st.session_state.get("last_report") is not None,
    )

    with st.sidebar:
        st.markdown("### ◆ App status")
        st.success("Model online")
        if df is not None:
            st.metric("Corpus", f"{len(df):,}")
        st.caption(f"{FEATURE_COUNT}+ engineered features")
        st.divider()
        st.markdown("**Feature stack**")
        st.markdown("NLP · Vision-L · CV · Survey · 3D analytics")
        if st.button("Reset session", width="stretch"):
            for k in ("history", "last_breakdown", "last_report", "form"):
                st.session_state.pop(k, None)
            st.rerun()

    tabs = st.tabs(
        [
            "◆ Overview",
            "🔮 Predict",
            "📋 Survey report",
            "🔬 Feature analysis",
            "📊 Market view",
            "📁 Batch",
        ]
    )
    with tabs[0]:
        tab_command_center(df)
    with tabs[1]:
        tab_predict(model, df, cat_opts)
    with tabs[2]:
        report = st.session_state.get("last_report")
        if report:
            render_research_report(
                answers=report["answers"],
                core=report["core"],
                breakdown=report["breakdown"],
                feat_row=report["feat_row"],
                image_bytes=report.get("image_bytes"),
                pred=report["pred"],
                score=report["score"],
                emoji_summary=report.get("emoji_summary"),
                ai_insights=report.get("ai_insights"),
            )
        else:
            st.info("Run **Predict** first.")
    with tabs[3]:
        tab_science()
    with tabs[4]:
        tab_explore(df) if df is not None else st.warning("Add CSV to data/raw/")
    with tabs[5]:
        tab_batch(model)


if __name__ == "__main__":
    main()
