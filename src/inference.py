"""
Batch inference CLI + shared helpers used by `src.api` (HTTP service).

Design goals for students:
  - Same cleaning rules as training, but **no requirement** for the label column.
  - One place to build the `text` feature so API + batch jobs stay consistent.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    CATEGORICAL_COLUMNS,
    MODELS_DIR,
    NUMERIC_COLUMNS,
    SUBMISSIONS_DIR,
    TARGET_COLUMN,
    TEXT_COLUMNS,
    ensure_dirs,
)
from src.data import make_text_feature


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def clean_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean input data for inference without requiring target column.

    Training cleaning drops rows missing target labels, but inference inputs
    may not include labels at all, so we handle columns independently.
    """
    clean = df.copy()

    for col in TEXT_COLUMNS:
        if col not in clean.columns:
            clean[col] = ""
        clean[col] = clean[col].fillna("").astype(str)

    for col in NUMERIC_COLUMNS:
        if col not in clean.columns:
            clean[col] = 0
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0)

    for col in CATEGORICAL_COLUMNS:
        if col not in clean.columns:
            clean[col] = "Unknown"
        clean[col] = clean[col].fillna("Unknown").astype(str)

    return clean


def prepare_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """Apply inference cleaning, drop labels if present, add combined text column."""
    df = clean_inference_data(df)
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])
    # Rebuild text from Title + Review Text (ignore any client-supplied `text` column).
    if "text" in df.columns:
        df = df.drop(columns=["text"])
    df = df.copy()
    df["text"] = make_text_feature(df)
    return df


def predict_scores(model, features_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return parallel Series of class predictions and positive-class scores."""
    preds = model.predict(features_df)
    pred_series = pd.Series(preds, dtype=int, name="prediction")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_df)
        # Some estimators (e.g., single-class DummyClassifier) return one column.
        if proba.shape[1] >= 2:
            scores = proba[:, 1]
        else:
            scores = proba[:, 0]
    else:
        scores = preds
    score_series = pd.Series(scores, name="score")
    return pred_series, score_series


def run_batch_inference(input_csv: Path, model_path: Path, output_csv: Path | None = None) -> Path:
    """
    Run batch inference for a raw review CSV.

    Expected behavior:
    - input format is similar to the training CSV.
    - target column is optional during inference; if present we ignore it.
    """
    ensure_dirs()
    model = load_model(model_path)

    df = pd.read_csv(input_csv)
    feat = prepare_features_for_model(df)
    pred_series, score_series = predict_scores(model, feat)

    result = pd.DataFrame(
        {
            "Id": range(len(pred_series)),
            "prediction": pred_series.values,
            "score": score_series.values,
        }
    )

    if output_csv is None:
        output_csv = SUBMISSIONS_DIR / f"inference_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        output_csv = Path(output_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to CSV for inference.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "better_pipeline.joblib",
        help="Path to trained model artifact.",
    )
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = run_batch_inference(
        input_csv=args.input_csv, model_path=args.model_path, output_csv=args.output_csv
    )
    print(f"Inference output written to: {out}")
