"""
Training entrypoint for teaching + production-like workflows.

Flow (matches how many teams work):
  1) Load and clean data
  2) Holdout split for final offline metrics
  3) Optional cross-validation on the *training* split for stability estimates
  4) Fit the final model on the full training split
  5) Report metrics on the holdout validation split
  6) Save a joblib artifact for batch inference and the FastAPI service

Use `--cv-f1` to log stratified K-fold F1 on `X_train` (mean + std).
Use `--mlflow` to send params/metrics/artifact to MLflow (Databricks or local `mlruns`).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import (
    LOGS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_FILE_PATH,
    SUBMISSIONS_DIR,
    TARGET_COLUMN,
    ensure_dirs,
)
from src.data import basic_cleaning, load_raw_data, make_text_feature, split_data
from src.download_data import download_dataset
from src.features import build_tabular_preprocessor, build_text_tabular_preprocessor

LOGGER = logging.getLogger(__name__)


def _xgb_device_kwargs() -> Dict[str, str]:
    """Prefer GPU if XGBoost can run a tiny fit; otherwise CPU (portable classrooms)."""
    try:
        from xgboost import XGBClassifier

        probe = XGBClassifier(
            n_estimators=1,
            eval_metric="logloss",
            tree_method="hist",
            device="cuda",
        )
        probe.fit(np.array([[0.0], [1.0]], dtype=np.float32), np.array([0, 1]))
        return {"tree_method": "hist", "device": "cuda"}
    except Exception:
        return {"tree_method": "hist", "device": "cpu"}


def cross_val_f1_report(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, n_splits: int
) -> Dict[str, Any]:
    """
    Stratified K-fold F1 on the training split only.

    Why train-only CV?
    - The holdout `X_valid` must remain untouched until final reporting.
    - Mean/std across folds gives students a sense of variance vs a single split.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        clone(pipeline), X_train, y_train, cv=cv, scoring="f1", n_jobs=1
    )
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "folds": [float(x) for x in scores],
        "n_splits": n_splits,
    }


def persist_cv_report(model_name: str, report: Dict[str, Any]) -> Path:
    """Write CV summary next to other processed artifacts (easy to diff across runs)."""
    ensure_dirs()
    out_dir = PROCESSED_DIR / "cv_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cv_f1_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


def mlflow_log_run(
    args: argparse.Namespace,
    *,
    train_params: Dict[str, Any],
    val_metrics: Dict[str, float],
    cv_report: Optional[Dict[str, Any]],
    model_path: Path,
) -> None:
    """Optional MLflow tracking (local file store or Databricks tracking URI)."""
    if not args.mlflow:
        return
    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = (
        args.mlflow_experiment
        or os.environ.get("MLFLOW_EXPERIMENT", "womens-ecommerce-reviews")
    )
    mlflow.set_experiment(experiment)

    with mlflow.start_run():
        mlflow.log_params({k: str(v) for k, v in train_params.items()})
        for name, value in val_metrics.items():
            mlflow.log_metric(f"val_{name}", float(value))
        if cv_report is not None:
            mlflow.log_metric("cv_f1_mean", cv_report["mean"])
            mlflow.log_metric("cv_f1_std", cv_report["std"])
        mlflow.log_artifact(str(model_path))


def setup_logging(model_name: str) -> Path:
    """Configure console + file logging for this run."""
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"train_{model_name}_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    return log_path


def get_model_pipeline(model_type: str) -> Pipeline:
    # Baseline is intentionally simple for teaching and fast iteration.
    if model_type == "baseline":
        preprocessor = build_tabular_preprocessor()
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        return Pipeline([("prep", preprocessor), ("model", model)])

    if model_type == "better":
        # Better model combines text with tabular/categorical context.
        preprocessor = build_text_tabular_preprocessor(text_col="text")
        model = LogisticRegression(
            C=2.0, max_iter=2000, class_weight="balanced", solver="liblinear"
        )
        return Pipeline([("prep", preprocessor), ("model", model)])

    if model_type == "advanced":
        # Advanced tier prioritizes non-linear interactions across mixed features.
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

    if model_type == "advanced_xgb":
        # Strong default for tabular + sparse text: gradient boosting on tree_method hist.
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "The 'advanced_xgb' model requires xgboost. Install with: pip install xgboost"
            ) from exc

        preprocessor = build_text_tabular_preprocessor(text_col="text")
        model = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            **_xgb_device_kwargs(),
        )
        return Pipeline([("prep", preprocessor), ("model", model)])

    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_binary(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    # log_loss uses predicted probability of the positive class (binary).
    proba = np.clip(np.asarray(y_proba, dtype=float), 1e-15, 1.0 - 1e-15)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "log_loss": float(log_loss(y_true, proba)),
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
    except Exception:
        print("Optuna not installed or unavailable; skipping tuning.")
        return pipeline

    def objective(trial):
        # Keep search-space small for classroom runtime constraints.
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
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
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
    log_path = setup_logging(args.model)
    LOGGER.info("Starting training run with args: %s", vars(args))
    LOGGER.info("Logs will be written to: %s", log_path)
    ensure_dirs()
    if args.download_data:
        LOGGER.info("Downloading data from Kaggle dataset: %s", args.dataset_slug)
        path = download_dataset(args.dataset_slug)
        LOGGER.info("Downloaded dataset CSV to: %s", path)

    LOGGER.info("Loading data from: %s", RAW_FILE_PATH)
    df = load_raw_data(RAW_FILE_PATH)
    LOGGER.debug("Raw dataset shape: %s", df.shape)
    df = basic_cleaning(df)
    LOGGER.debug("Post-cleaning dataset shape: %s", df.shape)
    df["text"] = make_text_feature(df)
    LOGGER.debug("Text feature generated.")

    X_train, X_valid, y_train, y_valid = split_data(df, TARGET_COLUMN)
    LOGGER.info(
        "Split complete: train_rows=%s, valid_rows=%s", len(X_train), len(X_valid)
    )

    pipeline = get_model_pipeline(args.model)
    # Optuna tuning is implemented for RandomForest advanced tier only (classroom runtime).
    if args.model == "advanced" and args.tune:
        LOGGER.info("Running lightweight tuning for advanced model.")
        pipeline = maybe_tune_advanced(pipeline, X_train, y_train)

    cv_report: Optional[Dict[str, Any]] = None
    if args.cv_f1:
        LOGGER.info("Running %s-fold CV (F1) on training split...", args.cv_splits)
        cv_report = cross_val_f1_report(
            pipeline, X_train, y_train, n_splits=args.cv_splits
        )
        LOGGER.info(
            "CV F1 mean=%.4f std=%.4f folds=%s",
            cv_report["mean"],
            cv_report["std"],
            cv_report["folds"],
        )
        print(
            f"\nCV F1 (train split only): mean={cv_report['mean']:.4f} "
            f"std={cv_report['std']:.4f} folds={cv_report['folds']}"
        )
        cv_path = persist_cv_report(args.model, cv_report)
        LOGGER.info("Wrote CV report JSON to: %s", cv_path)

    LOGGER.info("Training model: %s", args.model)
    pipeline.fit(X_train, y_train)
    LOGGER.info("Training complete.")

    y_pred = pipeline.predict(X_valid)
    # Some estimators return a single-column proba matrix; handle both shapes.
    if hasattr(pipeline[-1], "predict_proba"):
        proba = pipeline.predict_proba(X_valid)
        if proba.shape[1] >= 2:
            y_proba = proba[:, 1]
        else:
            y_proba = proba[:, 0]
    else:
        y_proba = y_pred

    metrics = evaluate_binary(y_valid, y_pred, y_proba)
    LOGGER.info("Validation metrics: %s", metrics)
    print_metrics(metrics, y_valid, y_pred)

    model_path = MODELS_DIR / f"{args.model}_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    LOGGER.info("Saved model to: %s", model_path)

    train_params = {
        "model": args.model,
        "tune": args.tune,
        "cv_splits": args.cv_splits,
        "cv_f1": args.cv_f1,
        "random_state": RANDOM_STATE,
        "raw_file": str(RAW_FILE_PATH),
    }
    mlflow_log_run(
        args,
        train_params=train_params,
        val_metrics=metrics,
        cv_report=cv_report,
        model_path=model_path,
    )

    if args.make_submission:
        sub_path = make_submission_file(pipeline, df.drop(columns=[TARGET_COLUMN]), args.model)
        LOGGER.info("Submission written to: %s", sub_path)

    LOGGER.info("Run completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "better", "advanced", "advanced_xgb"],
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
    parser.add_argument(
        "--cv-f1",
        action="store_true",
        help="Run stratified K-fold F1 on the training split (mean/std printed + JSON report).",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Number of CV folds when --cv-f1 is set.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log params, metrics, and model artifact to MLflow (tracking URI from env).",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (else MLFLOW_EXPERIMENT env or default).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
