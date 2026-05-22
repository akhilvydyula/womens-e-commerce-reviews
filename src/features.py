from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_multimodal_preprocessor(text_col: str = "text") -> ColumnTransformer:
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    text_pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=25000,
                    ngram_range=(1, 2),
                    min_df=2,
                    stop_words="english",
                ),
            )
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("text", text_pipe, text_col),
            ("num", num_pipe, NUMERIC_COLUMNS),
            ("cat", cat_pipe, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
