from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
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


def stratified_subsample(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    sample_frac: Optional[float] = None,
    max_rows: Optional[int] = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Draw a smaller stratified random subset for fast iteration (debug / classroom).

    Preserves approximate class balance when sklearn stratified split succeeds.
    If ``max_rows`` and ``sample_frac`` are both None, returns ``df`` unchanged.
    If both are set, ``max_rows`` wins (and ``sample_frac`` is ignored).
    """
    if sample_frac is None and max_rows is None:
        return df

    y = df[target_column]
    n_total = len(df)
    if n_total < 2:
        return df

    if max_rows is not None:
        n = min(int(max_rows), n_total)
    else:
        assert sample_frac is not None
        if sample_frac >= 1.0:
            return df
        if sample_frac <= 0.0:
            raise ValueError("sample_frac must be in (0, 1) when provided")
        n = max(2, int(n_total * sample_frac))

    if n < 2:
        return df

    idx = np.arange(n_total)
    try:
        keep_idx, _ = train_test_split(
            idx,
            train_size=n,
            stratify=y,
            random_state=random_state,
        )
    except ValueError:
        # Too few rows per class for stratify — fall back to simple random rows.
        return df.sample(n=min(n, n_total), random_state=random_state).reset_index(
            drop=True
        )

    return df.iloc[keep_idx].reset_index(drop=True)


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
