from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    RAW_FILE_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    TEXT_COLUMNS,
)


def load_raw_data(path=RAW_FILE_PATH) -> pd.DataFrame:
    # Keep this function thin so loading logic stays reusable across scripts/notebooks.
    df = pd.read_csv(path)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Always copy incoming frames to avoid side-effects in notebooks.
    clean = df.copy()

    # Text columns should be safe for vectorizers and string operations.
    for col in TEXT_COLUMNS:
        clean[col] = clean[col].fillna("").astype(str)

    # Numeric columns are coerced to numeric for robust model inputs.
    for col in NUMERIC_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0)

    # Categorical nulls become a dedicated bucket for consistency.
    for col in CATEGORICAL_COLUMNS:
        clean[col] = clean[col].fillna("Unknown").astype(str)

    # Training target must exist and be integer typed for classification.
    clean = clean.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    clean[TARGET_COLUMN] = clean[TARGET_COLUMN].astype(int)
    return clean


def make_text_feature(df: pd.DataFrame) -> pd.Series:
    # Single text field keeps feature pipelines simpler and reproducible.
    return (df["Title"] + " " + df["Review Text"]).str.strip()


def split_data(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Keep target separate from features before splitting.
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Stratified split preserves class ratio in train/validation.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_valid, y_train, y_valid
