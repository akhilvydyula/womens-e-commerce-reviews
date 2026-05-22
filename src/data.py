from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_COLUMNS,
    CORE_NUMERIC_COLUMNS,
    RANDOM_STATE,
    RAW_FILE_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    TEXT_COLUMNS,
)
from src.feature_engineering import enrich_dataframe


def load_raw_data(path=RAW_FILE_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    for col in TEXT_COLUMNS:
        clean[col] = clean[col].fillna("").astype(str)
    for col in CORE_NUMERIC_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0)
    for col in CATEGORICAL_COLUMNS:
        clean[col] = clean[col].fillna("Unknown").astype(str)
    if TARGET_COLUMN in clean.columns:
        clean = clean.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        clean[TARGET_COLUMN] = clean[TARGET_COLUMN].astype(int)
    return clean


def prepare_model_frame(df: pd.DataFrame, **enrich_kwargs) -> pd.DataFrame:
    """Clean + NLP/vision/survey/image feature enrichment."""
    return enrich_dataframe(basic_cleaning(df), **enrich_kwargs)


def split_data(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
