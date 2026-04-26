"""
Model interpretability helpers for stakeholder-facing explanations.

Primary focus: `better` pipeline (TF-IDF + LogisticRegression) — linear weights
map directly to log-odds contributions and are easy to discuss with business.

Tree/boosting models: use permutation importance or SHAP in a notebook for deeper dives
(see docs/ML_PRODUCT_RAW_TO_PRODUCTION.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, PROCESSED_CLEAN_CSV, RAW_FILE_PATH


def summarize_logistic_tfidf_pipeline(
    pipeline: Pipeline,
    *,
    top_k: int = 25,
):
    """Return (top_positive, top_negative) TF-IDF/feature weights for class 1."""
    if "prep" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError("Expected a Pipeline with steps 'prep' and 'model'.")
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    if not isinstance(model, LogisticRegression):
        raise TypeError(
            "This summarizer expects LogisticRegression as the final estimator "
            f"(got {type(model).__name__}). Use permutation importance or SHAP for trees."
        )

    names = prep.get_feature_names_out()
    coef = model.coef_.ravel()
    pairs = list(zip(names, coef))
    positive = sorted([p for p in pairs if p[1] > 0], key=lambda x: x[1], reverse=True)[:top_k]
    negative = sorted([p for p in pairs if p[1] < 0], key=lambda x: x[1])[:top_k]
    return positive, negative


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print top logistic regression weights (interpretable NLP + tabular features)."
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "better_pipeline.joblib",
        help="Trained sklearn Pipeline (expects better-style TF-IDF + logistic).",
    )
    p.add_argument("--top-k", type=int, default=25, help="How many weights per direction to show.")
    p.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional CSV to verify schema (defaults to processed clean then raw).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ref_path = args.data_path or PROCESSED_CLEAN_CSV
    if not ref_path.exists():
        ref_path = RAW_FILE_PATH
    print(f"Reference data path (schema): {ref_path} (exists={ref_path.exists()})")

    pipeline = joblib.load(args.model_path)
    pos, neg = summarize_logistic_tfidf_pipeline(pipeline, top_k=args.top_k)

    print("\n=== Top weights pushing prediction toward Recommended (positive class) ===")
    for name, w in pos:
        print(f"  {w:+.4f}  {name}")
    print("\n=== Top weights pushing away from Recommended ===")
    for name, w in neg:
        print(f"  {w:+.4f}  {name}")
    print(
        "\nBusiness note: positive TF-IDF terms are phrases associated with recommending the product; "
        "negatives often align with complaints. Combine with error analysis on false positives."
    )
