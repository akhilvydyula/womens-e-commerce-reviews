"""
Fast leaderboard over existing saved sklearn models (no retraining).

Goal:
- Reuse already trained `models/*_pipeline.joblib` files.
- Score all available models on the same holdout split.
- Finish quickly (usually under a couple minutes, often seconds).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from src.config import MODELS_DIR, PROCESSED_DIR, RAW_FILE_PATH, TARGET_COLUMN
from src.data import basic_cleaning, load_raw_data, make_text_feature, split_data


def _positive_class_proba(model: Any, X: pd.DataFrame, y_pred: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba[:, 0], dtype=float)
    return np.asarray(y_pred, dtype=float)


def _evaluate_model(model_path: Path, X_valid: pd.DataFrame, y_valid: pd.Series) -> Dict[str, Any]:
    model = joblib.load(model_path)
    y_pred = model.predict(X_valid)
    y_proba = _positive_class_proba(model, X_valid, y_pred)
    y_proba_clip = np.clip(y_proba, 1e-15, 1.0 - 1e-15)
    return {
        "model_file": model_path.name,
        "accuracy": float(accuracy_score(y_valid, y_pred)),
        "f1": float(f1_score(y_valid, y_pred)),
        "roc_auc": float(roc_auc_score(y_valid, y_proba)),
        "log_loss": float(log_loss(y_valid, y_proba_clip)),
    }


def run(args: argparse.Namespace) -> None:
    data_path = args.data_path or RAW_FILE_PATH
    df = load_raw_data(data_path)
    df = basic_cleaning(df)
    df["text"] = make_text_feature(df)
    _, X_valid, _, y_valid = split_data(df, TARGET_COLUMN)

    model_dir = args.models_dir or MODELS_DIR
    model_paths = sorted(model_dir.glob("*_pipeline.joblib"))
    if not model_paths:
        raise FileNotFoundError(
            f"No *_pipeline.joblib files found in {model_dir}. Train at least one model first."
        )

    rows: List[Dict[str, Any]] = []
    for path in model_paths:
        try:
            rows.append(_evaluate_model(path, X_valid, y_valid))
        except Exception as exc:
            rows.append({"model_file": path.name, "error": repr(exc)})

    ok_rows = [r for r in rows if "error" not in r]
    ok_rows.sort(key=lambda r: (r["accuracy"], r["f1"], r["roc_auc"]), reverse=True)
    bad_rows = [r for r in rows if "error" in r]
    report = {
        "data_file": str(Path(data_path).resolve()),
        "n_holdout_rows": int(len(y_valid)),
        "ranked_models": ok_rows,
        "failed_models": bad_rows,
    }

    out_dir = PROCESSED_DIR / "leaderboards"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"existing_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nFast leaderboard (existing models only)")
    print("-" * 58)
    for i, r in enumerate(ok_rows, start=1):
        print(
            f"{i:>2}. {r['model_file']:<30} "
            f"acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['roc_auc']:.4f} ll={r['log_loss']:.4f}"
        )
    if bad_rows:
        print("\nSkipped models with errors:")
        for r in bad_rows:
            print(f"- {r['model_file']}: {r['error']}")
    print(f"\nWrote leaderboard JSON to: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank existing saved models quickly (no retraining).")
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--models-dir", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
