from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_tabular_preprocessor() -> ColumnTransformer:
    # Median imputation + scaling is a practical baseline for mixed numeric quality.
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    # OHE with unknown handling is robust for real-world category drift.
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_COLUMNS),
            ("cat", cat_pipe, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    return preprocessor


def build_text_tabular_preprocessor(text_col: str = "text") -> ColumnTransformer:
    # Reuse the same numeric pipeline for consistency across model tiers.
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    # Reuse the same categorical strategy for deterministic training behavior.
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    # TF-IDF gives an interpretable and strong baseline for review text.
    text_pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=30000,
                    ngram_range=(1, 2),
                    min_df=2,
                    stop_words="english",
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_pipe, text_col),
            ("num", num_pipe, NUMERIC_COLUMNS),
            ("cat", cat_pipe, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor
