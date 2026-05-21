"""Train TF-IDF + logistic model. Saves models/model.joblib."""
from __future__ import annotations

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, RAW_FILE_PATH, TARGET_COLUMN, ensure_dirs
from src.data import basic_cleaning, load_raw_data, make_text_feature, split_data
from src.features import build_text_tabular_preprocessor


def main() -> None:
    if not RAW_FILE_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE_PATH}\n"
            "Download from Kaggle and place the CSV in data/raw/"
        )

    ensure_dirs()
    df = basic_cleaning(load_raw_data())
    df["text"] = make_text_feature(df)

    X_train, X_valid, y_train, y_valid = split_data(df, TARGET_COLUMN)
    pipeline = Pipeline(
        [
            ("prep", build_text_tabular_preprocessor()),
            (
                "model",
                LogisticRegression(
                    C=2.0, max_iter=2000, class_weight="balanced", solver="liblinear"
                ),
            ),
        ]
    )

    print("Training…")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)
    y_proba = pipeline.predict_proba(X_valid)[:, 1]
    print(f"Accuracy: {accuracy_score(y_valid, y_pred):.4f}")
    print(f"F1:         {f1_score(y_valid, y_pred):.4f}")
    print(f"ROC-AUC:    {roc_auc_score(y_valid, y_proba):.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
