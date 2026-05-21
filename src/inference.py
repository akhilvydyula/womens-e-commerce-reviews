from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    CATEGORICAL_COLUMNS,
    MODEL_PATH,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    TEXT_COLUMNS,
)
from src.data import make_text_feature


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}\nRun: make train")
    return joblib.load(path)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TEXT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    for col in NUMERIC_COLUMNS:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    for col in CATEGORICAL_COLUMNS:
        if col not in out.columns:
            out[col] = "Unknown"
        out[col] = out[col].fillna("Unknown").astype(str)
    if TARGET_COLUMN in out.columns:
        out = out.drop(columns=[TARGET_COLUMN])
    if "text" in out.columns:
        out = out.drop(columns=["text"])
    out["text"] = make_text_feature(out)
    return out


def predict(model, row: dict) -> tuple[int, float]:
    df = prepare_features(pd.DataFrame([row]))
    pred = int(model.predict(df)[0])
    score = float(model.predict_proba(df)[0, 1])
    return pred, score
