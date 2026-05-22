from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any
import warnings

import joblib
import pandas as pd
import requests
from sklearn.exceptions import InconsistentVersionWarning

from src.config import MODEL_PATH, ensure_dirs
from src.data import prepare_model_frame
from src.feature_engineering import explain_features
from src.survey import survey_from_ui


def _validate_artifact(artifact):
    if isinstance(artifact, dict) and "pipeline" in artifact:
        if artifact.get("version", 1) < 2:
            raise ValueError("Stale model artifact. Run: make train")
        return artifact["pipeline"]
    # Legacy single-pipeline joblib from older train runs
    return artifact


def _download_model(url: str, target: Path) -> None:
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()
    target.write_bytes(resp.content)


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        model_url = os.getenv("MODEL_URL", "").strip()
        if model_url:
            ensure_dirs()
            try:
                _download_model(model_url, path)
            except Exception as exc:
                raise FileNotFoundError(
                    f"Model not found at {path} and MODEL_URL download failed: {exc}"
                ) from exc
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                "Set MODEL_URL env var to a public model.joblib URL or run: make train"
            )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        artifact = joblib.load(path)
    if any(isinstance(w.message, InconsistentVersionWarning) for w in caught):
        raise ValueError(
            "Model artifact was trained with a different scikit-learn version. "
            "Run: make train"
        )
    return _validate_artifact(artifact)


def load_model_from_bytes(model_bytes: bytes):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        artifact = joblib.load(io.BytesIO(model_bytes))
    if any(isinstance(w.message, InconsistentVersionWarning) for w in caught):
        raise ValueError(
            "Uploaded model was trained with a different scikit-learn version."
        )
    return _validate_artifact(artifact)


def build_row(
    core: dict[str, Any],
    survey_answers: dict[str, Any] | None = None,
    image_bytes: bytes | None = None,
) -> pd.DataFrame:
    survey = survey_from_ui(survey_answers) if survey_answers else None
    row = {**core}
    return prepare_model_frame(pd.DataFrame([row]), survey=survey, image_bytes=image_bytes)


def predict(
    model,
    core: dict[str, Any],
    survey_answers: dict[str, Any] | None = None,
    image_bytes: bytes | None = None,
) -> tuple[int, float, pd.DataFrame, dict[str, Any]]:
    """Returns prediction, score, feature row, and grouped feature explanation."""
    feat_df = build_row(core, survey_answers, image_bytes)
    pred = int(model.predict(feat_df)[0])
    score = float(model.predict_proba(feat_df)[0, 1])
    breakdown = explain_features(feat_df.iloc[0])
    return pred, score, feat_df, breakdown


def predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    feat = prepare_model_frame(df)
    preds = model.predict(feat)
    proba = model.predict_proba(feat)[:, 1]
    out = df.copy()
    out["prediction"] = preds.astype(int)
    out["score"] = proba
    out["recommended_label"] = out["prediction"].map(
        {1: "Recommended", 0: "Not recommended"}
    )
    return out
