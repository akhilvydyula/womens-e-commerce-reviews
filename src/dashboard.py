"""Multimodal analytics charts and shared Streamlit UI components."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import TARGET_COLUMN
from src.nlp_features import NLP_FEATURE_NAMES
from src.survey import SURVEY_FEATURE_NAMES, survey_from_ui
from src.vision_features import IMAGE_FEATURE_NAMES, VISION_TEXT_FEATURE_NAMES, extract_image_features

FEATURE_COUNT = (
    len(NLP_FEATURE_NAMES)
    + len(VISION_TEXT_FEATURE_NAMES)
    + len(IMAGE_FEATURE_NAMES)
    + len(SURVEY_FEATURE_NAMES)
)

# Plotly dark application theme
_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0f0f14",
    font=dict(family="DM Sans, sans-serif", color="#e4e4e7", size=12),
    margin=dict(l=12, r=12, t=48, b=12),
)
_COLORS = ["#818cf8", "#a78bfa", "#34d399", "#f472b6", "#38bdf8", "#fbbf24"]
_EXEC_SCALE = [
    [0.0, "#1e1e24"],
    [0.25, "#5b21b6"],
    [0.5, "#7c3aed"],
    [0.75, "#c4b5fd"],
    [1.0, "#34d399"],
]
_MODALITY_ORDER = ["nlp", "vis", "img", "survey"]

_SURVEY_PROFILE = [
    ("survey_fit_vs_expectation", "Fit satisfaction", 5.0),
    ("survey_material_vs_price", "Material vs price", 5.0),
    ("survey_color_accuracy", "Color accuracy", 5.0),
    ("survey_shipping_satisfaction", "Shipping", 5.0),
    ("survey_packaging_quality", "Packaging", 5.0),
    ("survey_repurchase_likelihood", "Repurchase intent", 1.0),
    ("survey_return_likelihood", "Return risk", 1.0),
    ("survey_brand_loyalty", "Brand loyalty", 1.0),
    ("survey_days_since_purchase", "Days since purchase", 60.0),
    ("survey_comfort_hours", "Comfort (hours)", 12.0),
]


def _normalize_display_value(name: str, val: float) -> float:
    """Map heterogeneous feature scales to 0–1 for comparable heatmaps."""
    n = str(name).lower()
    v = float(val)
    if "char_count" in n or "word_count" in n or "sentence_len" in n:
        return float(np.clip(v / 400.0, 0.0, 1.0))
    if "days_since" in n:
        return float(np.clip(v / 60.0, 0.0, 1.0))
    if "comfort_hours" in n:
        return float(np.clip(v / 12.0, 0.0, 1.0))
    if n.startswith("survey_"):
        if any(
            x in n
            for x in (
                "likelihood",
                "loyalty",
                "sensitivity",
                "match",
                "importance",
                "formality",
                "difficulty",
                "buyer",
                "alternatives",
                "worn",
                "gift",
                "odor",
                "contact",
                "alterations",
                "photo",
                "channel",
            )
        ):
            return float(np.clip(v, 0.0, 1.0))
        return float(np.clip(v / 5.0, 0.0, 1.0))
    if n.startswith("img_"):
        return float(np.clip(abs(v), 0.0, 1.0))
    if abs(v) <= 1.0:
        return float(np.clip(abs(v), 0.0, 1.0))
    return float(np.clip(abs(v) / 10.0, 0.0, 1.0))


def _short_feature_label(name: str) -> str:
    return str(name).replace("survey_", "").replace("nlp_", "").replace("vis_", "").replace("img_", "")


def plotly_show(fig: go.Figure, key: str, height: int | None = None) -> None:
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": True, "displaylogo": False},
        key=key,
    )


def inject_executive_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,600;9..40,700&family=Instrument+Serif&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }
        .block-container { padding-top: 1rem; max-width: 1400px; }
        .exec-hero {
            background: linear-gradient(160deg, #030712 0%, #1e1b4b 35%, #4c1d95 70%, #7c3aed 100%);
            border-radius: 24px; padding: 2.5rem 3rem; margin-bottom: 1.5rem;
            border: 1px solid rgba(167,139,250,0.35);
            box-shadow: 0 32px 64px rgba(0,0,0,0.45);
        }
        .exec-hero h1 {
            font-family: 'Instrument Serif', Georgia, serif;
            color: #fafafa; font-size: 2.75rem; font-weight: 400; margin: 0 0 0.5rem 0;
            letter-spacing: -0.02em;
        }
        .exec-hero p { color: rgba(250,250,250,0.82); font-size: 1.05rem; line-height: 1.6; margin: 0; }
        .exec-pill {
            display: inline-block; background: rgba(255,255,255,0.08);
            backdrop-filter: blur(12px); color: #e9d5ff;
            padding: 0.35rem 0.9rem; border-radius: 999px; font-size: 0.75rem;
            font-weight: 600; margin: 0 0.35rem 0.35rem 0;
            border: 1px solid rgba(255,255,255,0.12);
        }
        .glass-card {
            background: rgba(24,24,27,0.85); border: 1px solid #3f3f46;
            border-radius: 16px; padding: 1.25rem 1.5rem; margin-bottom: 1rem;
        }
        .kpi-value { font-size: 1.75rem; font-weight: 700; color: #fafafa; }
        .kpi-label { font-size: 0.8rem; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.06em; }
        .verdict-card { border-radius: 20px; padding: 1.75rem; text-align: center; }
        .verdict-yes {
            background: linear-gradient(145deg, #064e3b 0%, #047857 50%, #10b981 100%);
            border: 1px solid #34d399; box-shadow: 0 0 40px rgba(52,211,153,0.25);
        }
        .verdict-no {
            background: linear-gradient(145deg, #7f1d1d 0%, #991b1b 50%, #dc2626 100%);
            border: 1px solid #f87171; box-shadow: 0 0 40px rgba(248,113,113,0.2);
        }
        .verdict-title { font-size: 1.85rem; font-weight: 700; color: #fff; margin: 0; }
        .verdict-sub { color: rgba(255,255,255,0.92); font-size: 1rem; margin-top: 0.35rem; }
        .section-title {
            font-size: 1.15rem; font-weight: 700; color: #fafafa;
            border-left: 4px solid #7c3aed; padding-left: 0.75rem; margin: 1.5rem 0 1rem 0;
        }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_executive_hero(
    feature_count: int,
    corpus_size: int | None,
    has_session: bool,
) -> None:
    data_label = f"Data: {corpus_size:,} reviews" if corpus_size is not None else "Data: not loaded"
    session_label = "Session: analyzed" if has_session else "Session: no analysis yet"
    st.markdown(
        f"""
        <div class="exec-hero">
            <span class="exec-pill">Model: multimodal classifier</span>
            <span class="exec-pill">{data_label}</span>
            <span class="exec-pill">{session_label}</span>
            <span class="exec-pill">Features: {feature_count}</span>
            <h1>ReviewSense Analytics Workspace</h1>
            <p>Prediction workspace for review recommendations using text, structured survey responses,
            and optional image signals with 2D/3D diagnostics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _apply(fig: go.Figure, title: str = "", height: int | None = 380) -> go.Figure:
    layout = {**_LAYOUT, "title": dict(text=title, x=0.02, font=dict(size=14))}
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    return fig


def chart_3d_market_space(df: pd.DataFrame, key: str) -> None:
    sample = df.sample(min(2500, len(df)), random_state=42)
    sample = sample[sample["Age"] > 0]
    fig = px.scatter_3d(
        sample,
        x="Age",
        y="Rating",
        z="Positive Feedback Count",
        color=TARGET_COLUMN,
        color_discrete_map={0: "#f87171", 1: "#34d399"},
        opacity=0.65,
        size_max=6,
        labels={TARGET_COLUMN: "Recommended"},
        title="",
    )
    fig.update_traces(marker=dict(size=3))
    _apply(fig, "3D · Shopper space (Age × Rating × Engagement)", 520)
    plotly_show(fig, key)


def chart_3d_modality_surface(breakdown: dict[str, dict[str, float]], key: str) -> None:
    groups = []
    strengths = []
    for g, feats in breakdown.items():
        if feats:
            groups.append(g.replace(" ", "\n"))
            strengths.append(sum(abs(v) for v in feats.values()) / len(feats))
    if len(groups) < 1:
        return
    fig = go.Figure(
        data=go.Scatter3d(
            x=list(range(len(groups))),
            y=[0] * len(groups),
            z=strengths,
            mode="markers+text",
            marker=dict(
                size=[max(8, s * 25) for s in strengths],
                color=strengths,
                colorscale="Plasma",
                showscale=True,
            ),
            text=groups,
            textposition="top center",
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Modality",
            yaxis_title="",
            zaxis_title="Signal strength",
            bgcolor="#0f0f14",
        )
    )
    _apply(fig, "3D · Modality signal strength", 480)
    plotly_show(fig, key)


def chart_3d_risk_repurchase(answers: dict[str, Any], key: str) -> None:
    ret = float(answers.get("return_likelihood", 0.15))
    rep = float(answers.get("repurchase_likelihood", 0.5))
    fit = float(answers.get("fit_vs_expectation", 3))
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[ret],
                y=[rep],
                z=[fit],
                mode="markers+text",
                marker=dict(size=18, color=rep, colorscale="RdYlGn", cmin=0, cmax=1),
                text=["Your profile"],
                textposition="top center",
                showlegend=False,
            ),
            go.Scatter3d(
                x=[0, 1, 0.5],
                y=[0, 1, 0.5],
                z=[3, 3, 3],
                mode="markers",
                marker=dict(size=4, color="rgba(120,120,140,0.4)"),
                showlegend=False,
            ),
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Return risk",
            yaxis_title="Repurchase intent",
            zaxis_title="Fit satisfaction",
            bgcolor="#0f0f14",
            camera=dict(eye=dict(x=1.5, y=1.45, z=0.95)),
        ),
        **_LAYOUT,
        height=480,
        title=dict(text="3D · Retention risk cube", x=0.02),
    )
    plotly_show(fig, key)


def chart_parallel_survey(answers: dict[str, Any], key: str) -> None:
    """Horizontal survey profile (replaces cramped parallel-coordinates layout)."""
    enc = survey_from_ui(answers)
    rows = []
    for col, label, cap in _SURVEY_PROFILE:
        if col not in enc:
            continue
        raw = float(enc[col])
        pct = min(100.0, max(0.0, raw / cap * 100.0))
        rows.append({"Dimension": label, "Strength": pct})
    if len(rows) < 3:
        return
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Strength",
        y="Dimension",
        orientation="h",
        color="Strength",
        color_continuous_scale=_EXEC_SCALE,
        range_color=[0, 100],
        range_x=[0, 100],
    )
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis=dict(categoryorder="array", categoryarray=df["Dimension"].tolist()[::-1]),
        xaxis=dict(ticksuffix="%", title="Normalized strength"),
        height=max(360, 44 * len(rows)),
    )
    fig.update_traces(marker_line=dict(color="rgba(228,228,231,0.35)", width=0.5))
    _apply(fig, "Survey profile · normalized dimensions", None)
    plotly_show(fig, key)


def chart_sunburst_catalog(df: pd.DataFrame, key: str) -> None:
    agg = (
        df.groupby(["Division Name", "Department Name", "Class Name"], observed=True)
        .agg(count=(TARGET_COLUMN, "count"), recommend=(TARGET_COLUMN, "mean"))
        .reset_index()
    )
    fig = px.sunburst(
        agg,
        path=["Division Name", "Department Name", "Class Name"],
        values="count",
        color="recommend",
        color_continuous_scale="RdYlGn",
    )
    _apply(fig, "Sunburst · Catalog hierarchy & recommend rate", 480)
    plotly_show(fig, key)


def chart_treemap_departments(df: pd.DataFrame, key: str) -> None:
    agg = df.groupby("Department Name", observed=False).agg(
        reviews=(TARGET_COLUMN, "count"),
        recommend=(TARGET_COLUMN, "mean"),
    ).reset_index()
    fig = px.treemap(
        agg,
        path=["Department Name"],
        values="reviews",
        color="recommend",
        color_continuous_scale="Purples",
    )
    _apply(fig, "Treemap · Department volume", 400)
    plotly_show(fig, key)


def chart_heatmap_age_rating(df: pd.DataFrame, key: str) -> None:
    tmp = df[df["Age"] > 0].copy()
    tmp["age_bin"] = pd.cut(tmp["Age"], bins=[0, 25, 35, 45, 55, 100], labels=["<25", "25-35", "35-45", "45-55", "55+"])
    pivot = tmp.pivot_table(
        index="age_bin",
        columns="Rating",
        values=TARGET_COLUMN,
        aggfunc="mean",
        observed=False,
    )
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        labels=dict(color="P(recommend)"),
    )
    _apply(fig, "Heatmap · Age cohort × Star rating", 380)
    plotly_show(fig, key)


def chart_feature_heatmap(feat_row: pd.Series, key: str) -> None:
    has_image = float(feat_row.get("img_has_image", 0)) > 0
    prefixes = ("nlp_", "vis_", "img_", "survey_")
    if not has_image:
        prefixes = tuple(p for p in prefixes if p != "img_")

    rows = []
    for p in prefixes:
        cols = [(c, feat_row[c]) for c in feat_row.index if str(c).startswith(p)]
        if not cols:
            continue
        top = sorted(cols, key=lambda x: abs(x[1]), reverse=True)[:6]
        for name, val in top:
            norm = _normalize_display_value(name, float(val))
            if norm < 0.02 and p == "img_":
                continue
            rows.append(
                {
                    "Modality": p.rstrip("_"),
                    "Feature": _short_feature_label(name),
                    "Strength": norm,
                }
            )
    if not rows:
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="Feature", columns="Modality", values="Strength", aggfunc="first")
    pivot = pivot.reindex(columns=[c for c in _MODALITY_ORDER if c in pivot.columns])
    pivot = pivot.loc[[i for i in pivot.index if pivot.loc[i].max() > 0.02]]
    if pivot.empty:
        return

    fig = px.imshow(
        pivot.fillna(0),
        color_continuous_scale=_EXEC_SCALE,
        zmin=0,
        zmax=1,
        aspect="auto",
        labels=dict(color="Signal", x="Modality", y="Feature"),
    )
    fig.update_coloraxes(colorbar=dict(title="Signal", tickformat=".0%"))
    fig.update_yaxes(tickfont=dict(size=11))
    fig.update_xaxes(side="top", tickfont=dict(size=12))
    row_h = max(28, 32 * len(pivot))
    _apply(fig, "Feature matrix · Normalized multimodal signals", min(520, row_h + 120))
    plotly_show(fig, key)


def chart_funnel_journey(answers: dict[str, Any], score: float, key: str) -> None:
    stages = [
        "Purchase",
        "Fit",
        "Quality",
        "Logistics",
        "Recommend",
    ]
    vals = [
        100,
        float(answers.get("fit_vs_expectation", 3)) / 5 * 100,
        float(answers.get("material_vs_price", 3)) / 5 * 100,
        float(answers.get("shipping_satisfaction", 3)) / 5 * 100,
        score * 100,
    ]
    fig = go.Figure(
        go.Funnel(
            y=stages,
            x=vals,
            textinfo="value+percent initial",
            marker=dict(color=["#6366f1", "#818cf8", "#a78bfa", "#c4b5fd", "#34d399"]),
        )
    )
    _apply(fig, "Funnel · Shopper journey → recommendation", 400)
    plotly_show(fig, key)


def chart_radar_modalities(breakdown: dict[str, dict[str, float]], key: str) -> None:
    scores = {}
    for group, feats in breakdown.items():
        if feats:
            if group == "NLP linguistics":
                label = "NLP"
            elif group == "Vision (text)":
                label = "Vision text"
            elif group == "Vision (image)":
                label = "Vision image"
            elif group == "Survey / behavior":
                label = "Survey"
            else:
                label = group
            scores[label] = sum(abs(v) for v in feats.values()) / len(feats)
    if not scores:
        return
    max_score = max(scores.values()) if scores else 1.0
    fig = go.Figure(
        data=go.Scatterpolar(
            r=list(scores.values()),
            theta=list(scores.keys()),
            fill="toself",
            fillcolor="rgba(124,58,237,0.35)",
            line_color="#a78bfa",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="#18181b",
            radialaxis=dict(visible=True, range=[0, max(max_score * 1.2, 1.0)]),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        paper_bgcolor=_LAYOUT["paper_bgcolor"],
        plot_bgcolor=_LAYOUT["plot_bgcolor"],
        font=_LAYOUT["font"],
        margin=dict(l=40, r=40, t=48, b=40),
        height=420,
        title=dict(text="Radar · Multimodal fusion", x=0.02),
    )
    plotly_show(fig, key)


def chart_gauge_triplet(score: float, pred: int, answers: dict[str, Any], key: str) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            title={"text": "P(Recommend)"},
            number={"suffix": "%", "font": {"size": 28}},
            gauge={"bar": {"color": "#34d399" if pred == 1 else "#f87171"}, "bgcolor": "#27272a"},
            domain={"x": [0.0, 0.30], "y": [0, 1]},
        )
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=float(answers.get("repurchase_likelihood", 0.5)) * 100,
            title={"text": "Repurchase"},
            number={"suffix": "%", "font": {"size": 28}},
            gauge={"bar": {"color": "#818cf8"}, "bgcolor": "#27272a"},
            domain={"x": [0.35, 0.65], "y": [0, 1]},
        )
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=(1 - float(answers.get("return_likelihood", 0.2))) * 100,
            title={"text": "Retention"},
            number={"suffix": "%", "font": {"size": 28}},
            gauge={"bar": {"color": "#38bdf8"}, "bgcolor": "#27272a"},
            domain={"x": [0.70, 1.0], "y": [0, 1]},
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=48, b=10))
    _apply(fig, "Session gauges", 320)
    plotly_show(fig, key)


def chart_image_cv_radar(image_bytes: bytes | None, key: str) -> None:
    if not image_bytes:
        return
    feats = extract_image_features(image_bytes)
    labels = []
    vals = []
    for k in IMAGE_FEATURE_NAMES:
        if k == "img_has_image":
            continue
        labels.append(k.replace("img_", ""))
        vals.append(float(feats.get(k, 0)))
    if not labels:
        return
    vmax = max(vals) or 1
    norm = [v / vmax for v in vals]
    fig = go.Figure(
        data=go.Scatterpolar(r=norm, theta=labels, fill="toself", line_color="#38bdf8")
    )
    fig.update_layout(
        polar=dict(bgcolor="#18181b"),
        **_LAYOUT,
        height=340,
        title=dict(text="CV descriptor radar (product photo)", x=0.02),
    )
    plotly_show(fig, key)


def chart_rating_sensitivity(curve: list[dict], key: str) -> None:
    if not curve:
        return
    fig = px.area(
        pd.DataFrame(curve),
        x="Rating",
        y="P(recommended)",
        markers=True,
        color_discrete_sequence=["#a78bfa"],
    )
    fig.update_layout(yaxis_tickformat=".0%")
    _apply(fig, "Sensitivity · Star rating elasticity", 300)
    plotly_show(fig, key)


def chart_sankey_research_journey(answers: dict[str, Any], score: float, key: str) -> None:
    """Sankey: structured research path → recommendation probability."""
    fit = float(answers.get("fit_vs_expectation", 3)) / 5
    qual = float(answers.get("material_vs_price", 3)) / 5
    ship = float(answers.get("shipping_satisfaction", 3)) / 5
    rep = float(answers.get("repurchase_likelihood", 0.5))
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=[
                        "Purchase",
                        "Fit",
                        "Quality",
                        "Logistics",
                        "Intent",
                        "Recommend",
                    ],
                    color=["#6366f1", "#818cf8", "#a78bfa", "#c4b5fd", "#38bdf8", "#34d399"],
                ),
                link=dict(
                    source=[0, 0, 1, 2, 3, 4],
                    target=[1, 2, 3, 4, 4, 5],
                    value=[
                        max(0.1, fit),
                        max(0.1, qual),
                        max(0.1, ship),
                        max(0.1, rep),
                        max(0.1, rep),
                        max(0.05, score),
                    ],
                    color="rgba(124,58,237,0.35)",
                ),
            )
        ]
    )
    _apply(fig, "Sankey · Survey path to recommendation", 400)
    plotly_show(fig, key)


def chart_rating_violin_by_dept(df: pd.DataFrame, key: str) -> None:
    top = df["Department Name"].value_counts().head(8).index.tolist()
    sub = df[df["Department Name"].isin(top)]
    fig = px.violin(
        sub,
        x="Department Name",
        y="Rating",
        color=TARGET_COLUMN,
        color_discrete_map={0: "#f87171", 1: "#34d399"},
        box=True,
        points="outliers",
    )
    fig.update_layout(xaxis_tickangle=-25)
    _apply(fig, "Violin · Rating distribution by department", 400)
    plotly_show(fig, key)


def chart_modality_waterfall(breakdown: dict[str, dict[str, float]], key: str) -> None:
    rows = []
    for group, feats in breakdown.items():
        if not feats:
            continue
        top = sorted(feats.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
        for name, val in top:
            rows.append({"Modality": group, "Feature": name.replace("_", " ")[:20], "Impact": float(val)})
    if len(rows) < 2:
        return
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Impact",
        y="Feature",
        color="Modality",
        orientation="h",
        color_discrete_sequence=_COLORS,
    )
    _apply(fig, "Waterfall · Top feature drivers by modality", 380)
    plotly_show(fig, key)


def chart_department_recommend_3d(df: pd.DataFrame, key: str) -> None:
    agg = (
        df.groupby("Department Name", observed=False)
        .agg(avg_rating=("Rating", "mean"), avg_age=("Age", "mean"), recommend=(TARGET_COLUMN, "mean"), n=(TARGET_COLUMN, "count"))
        .reset_index()
    )
    agg = agg[agg["avg_age"] > 0].head(12)
    fig = px.scatter_3d(
        agg,
        x="avg_age",
        y="avg_rating",
        z="recommend",
        size="n",
        color="recommend",
        text="Department Name",
        color_continuous_scale="Viridis",
    )
    _apply(fig, "3D · Department performance landscape", 500)
    plotly_show(fig, key)


def render_executive_kpis(df: pd.DataFrame | None, report: dict | None) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    if df is not None:
        c1.metric("Corpus", f"{len(df):,}", "reviews indexed")
        c2.metric("Baseline NPS proxy", f"{df[TARGET_COLUMN].mean():.1%}", "recommend rate")
        c3.metric("Avg rating", f"{df['Rating'].mean():.2f}", "stars")
    else:
        c1.metric("Corpus", "—", "add data/raw CSV")
        c2.metric("Baseline", "—", "")
        c3.metric("Avg rating", "—", "")
    if report:
        c4.metric("Live prediction", "Yes" if report["pred"] == 1 else "No", f"{report['score']:.1%} conf")
        c5.metric("Modalities", "4", "NLP·Vis·CV·Survey")
    else:
        c4.metric("Live prediction", "—", "run analysis")
        c5.metric("Modalities", "4", "ready")


def render_live_snapshot(report: dict, prefix: str) -> None:
    """Overview tab: high-level live session state only."""
    st.markdown('<p class="section-title">Live session · snapshot</p>', unsafe_allow_html=True)
    chart_gauge_triplet(report["score"], report["pred"], report["answers"], f"{prefix}_gauges")
    g1, g2 = st.columns(2)
    with g1:
        chart_radar_modalities(report["breakdown"], f"{prefix}_radar")
    with g2:
        chart_3d_risk_repurchase(report["answers"], f"{prefix}_risk3d")


def render_journey_analysis(report: dict, prefix: str) -> None:
    """Predict tab: customer journey and intent analysis."""
    st.markdown('<p class="section-title">Journey analysis · decision path</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        chart_funnel_journey(report["answers"], report["score"], f"{prefix}_funnel")
    with c2:
        chart_sankey_research_journey(report["answers"], report["score"], f"{prefix}_sankey")
    chart_parallel_survey(report["answers"], f"{prefix}_parallel")


def render_feature_diagnostics(report: dict, prefix: str) -> None:
    """Feature analysis tab: engineered-feature diagnostics."""
    st.markdown('<p class="section-title">Feature diagnostics · engineered signals</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        chart_3d_modality_surface(report["breakdown"], f"{prefix}_mod3d")
    with c2:
        chart_modality_waterfall(report["breakdown"], f"{prefix}_waterfall")
    if report.get("feat_row") is not None:
        chart_feature_heatmap(report["feat_row"], f"{prefix}_heatmap")
    chart_image_cv_radar(report.get("image_bytes"), f"{prefix}_cv_radar")


def render_market_analysis(df: pd.DataFrame, prefix: str) -> None:
    """Market view tab: dataset-level behavior and segment analytics."""
    st.markdown('<p class="section-title">Dataset view · market behavior</p>', unsafe_allow_html=True)
    chart_3d_market_space(df, f"{prefix}_market3d")
    r1, r2 = st.columns(2)
    with r1:
        chart_sunburst_catalog(df, f"{prefix}_sunburst")
        chart_heatmap_age_rating(df, f"{prefix}_ageheat")
    with r2:
        chart_treemap_departments(df, f"{prefix}_treemap")
        chart_department_recommend_3d(df, f"{prefix}_dept3d")
    chart_rating_violin_by_dept(df, f"{prefix}_violin")
