"""Train multimodal model (TF-IDF + NLP + vision + survey defaults)."""
from __future__ import annotations

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, RAW_FILE_PATH, TARGET_COLUMN, ensure_dirs
from src.data import load_raw_data, prepare_model_frame, split_data
from src.features import build_multimodal_preprocessor


def main() -> None:
    if not RAW_FILE_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE_PATH}\n"
            "Download from Kaggle and place the CSV in data/raw/"
        )

    ensure_dirs()
    print("Engineering NLP + vision-language + survey features...")
    df = prepare_model_frame(load_raw_data())

    X_train, X_valid, y_train, y_valid = split_data(df, TARGET_COLUMN)
    pipeline = Pipeline(
        [
            ("prep", build_multimodal_preprocessor()),
            (
                "model",
                LogisticRegression(
                    C=1.5, max_iter=3000, class_weight="balanced", solver="liblinear"
                ),
            ),
        ]
    )

    print(f"Training on {len(X_train):,} rows, {len(X_train.columns)} input columns...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)
    y_proba = pipeline.predict_proba(X_valid)[:, 1]
    print(f"Accuracy: {accuracy_score(y_valid, y_pred):.4f}")
    print(f"F1:         {f1_score(y_valid, y_pred):.4f}")
    print(f"ROC-AUC:    {roc_auc_score(y_valid, y_proba):.4f}")

    joblib.dump(
        {"pipeline": pipeline, "version": 2, "feature_count": len(X_train.columns)},
        MODEL_PATH,
    )
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
