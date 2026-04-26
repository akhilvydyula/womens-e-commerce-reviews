from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, RAW_FILE_PATH, SUBMISSIONS_DIR, TARGET_COLUMN, ensure_dirs
from src.data import basic_cleaning, load_raw_data, make_text_feature, split_data
from src.download_data import download_dataset
from src.features import build_tabular_preprocessor, build_text_tabular_preprocessor


def get_model_pipeline(model_type: str) -> Pipeline:
    if model_type == "baseline":
        preprocessor = build_tabular_preprocessor()
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        return Pipeline([("prep", preprocessor), ("model", model)])

    if model_type == "better":
        preprocessor = build_text_tabular_preprocessor(text_col="text")
        model = LogisticRegression(
            C=2.0, max_iter=2000, class_weight="balanced", solver="liblinear"
        )
        return Pipeline([("prep", preprocessor), ("model", model)])

    if model_type == "advanced":
        preprocessor = build_text_tabular_preprocessor(text_col="text")
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        return Pipeline([("prep", preprocessor), ("model", model)])

    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_binary(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    return metrics


def print_metrics(metrics: Dict[str, float], y_true: pd.Series, y_pred: np.ndarray) -> None:
    print("\nValidation Metrics")
    print("-" * 50)
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")

    print("\nClassification Report")
    print("-" * 50)
    print(classification_report(y_true, y_pred, digits=4))


def maybe_tune_advanced(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    try:
        import optuna
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except Exception:
        print("Optuna not installed or unavailable; skipping tuning.")
        return pipeline

    def objective(trial):
        candidate = Pipeline(
            steps=[
                ("prep", build_text_tabular_preprocessor(text_col="text")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=trial.suggest_int("n_estimators", 200, 700, step=100),
                        max_depth=trial.suggest_int("max_depth", 8, 30),
                        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                        random_state=42,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        score = cross_val_score(candidate, X_train, y_train, cv=cv, scoring="f1", n_jobs=1).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=12)
    print("Best tuning trial:", study.best_trial.params)

    best = study.best_trial.params
    tuned = Pipeline(
        steps=[
            ("prep", build_text_tabular_preprocessor(text_col="text")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=best["n_estimators"],
                    max_depth=best["max_depth"],
                    min_samples_leaf=best["min_samples_leaf"],
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    return tuned


def make_submission_file(model: Pipeline, source_df: pd.DataFrame, model_name: str) -> Path:
    inference_df = source_df.copy()
    inference_df["text"] = make_text_feature(inference_df)
    preds = model.predict(inference_df)

    submission = pd.DataFrame(
        {
            "Id": np.arange(len(preds)),
            TARGET_COLUMN: preds.astype(int),
        }
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SUBMISSIONS_DIR / f"submission_{model_name}_{timestamp}.csv"
    submission.to_csv(out_path, index=False)
    return out_path


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    if args.download_data:
        path = download_dataset(args.dataset_slug)
        print(f"Downloaded dataset CSV to: {path}")

    print(f"Loading data from: {RAW_FILE_PATH}")
    df = load_raw_data(RAW_FILE_PATH)
    df = basic_cleaning(df)
    df["text"] = make_text_feature(df)

    X_train, X_valid, y_train, y_valid = split_data(df, TARGET_COLUMN)

    pipeline = get_model_pipeline(args.model)
    if args.model == "advanced" and args.tune:
        pipeline = maybe_tune_advanced(pipeline, X_train, y_train)

    print(f"\nTraining model: {args.model}")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)
    if hasattr(pipeline[-1], "predict_proba"):
        y_proba = pipeline.predict_proba(X_valid)[:, 1]
    else:
        y_proba = y_pred

    metrics = evaluate_binary(y_valid, y_pred, y_proba)
    print_metrics(metrics, y_valid, y_pred)

    model_path = MODELS_DIR / f"{args.model}_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved model to: {model_path}")

    if args.make_submission:
        sub_path = make_submission_file(pipeline, df.drop(columns=[TARGET_COLUMN]), args.model)
        print(f"Submission written to: {sub_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "better", "advanced"],
        help="Model complexity level for teaching progression.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable lightweight hyperparameter tuning (advanced model only).",
    )
    parser.add_argument(
        "--make-submission",
        action="store_true",
        help="Create a Kaggle-style prediction CSV.",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download dataset directly from Kaggle before training.",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default="nicapotato/womens-ecommerce-clothing-reviews",
        help="Kaggle dataset slug for direct download.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
