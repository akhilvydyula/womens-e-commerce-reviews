"""
Fine-tune a compact transformer for review text -> Recommended IND (binary).

Why this exists
  TF-IDF + trees/logistic are strong baselines; a small BERT-family model can
  sometimes improve holdout accuracy/F1 by capturing phrasing and negation.
  This is the most direct "organic" upgrade path when you have a GPU.

Not wired into FastAPI by default
  The HTTP service loads sklearn `*.joblib` pipelines. HF artifacts live under
  `models/hf_recommender/` for offline benchmarking and optional custom serving.

Install
  1) PyTorch for your CUDA stack: https://pytorch.org/get-started/locally/
  2) pip install -r requirements-transformers.txt

Run
  python -m src.train_transformer
  make train-transformer
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    HF_RECOMMENDER_DIR,
    RANDOM_STATE,
    RAW_FILE_PATH,
    TARGET_COLUMN,
    ensure_dirs,
)
from src.data import (
    basic_cleaning,
    load_raw_data,
    make_text_feature,
    split_data,
    stratified_subsample,
)

LOGGER = logging.getLogger(__name__)


def _require_hf():
    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise ImportError(
            "Transformers stack not installed. "
            "Install PyTorch, then: pip install -r requirements-transformers.txt"
        ) from exc
    return (
        Dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )


def _build_weight_tensor(y: pd.Series):
    import torch

    yv = y.values
    classes = np.unique(yv)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=yv)
    by_c = {int(c): float(w) for c, w in zip(classes, weights)}
    return torch.tensor([by_c[0], by_c[1]], dtype=torch.float32)


class WeightedTrainer:
    """Factory for a Trainer subclass with balanced cross-entropy (defined lazily)."""

    @staticmethod
    def build(class_weights, *trainer_args: Any, **trainer_kwargs: Any):
        import torch
        from transformers import Trainer

        cw = class_weights

        class _WT(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                w = cw.to(logits.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=w)
                loss = loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss

        return _WT(*trainer_args, **trainer_kwargs)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _compute_metrics_builder():
    def compute_metrics(eval_pred) -> Dict[str, float]:
        if hasattr(eval_pred, "predictions"):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        proba = _softmax_rows(logits)[:, 1]
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds)),
            "roc_auc": float(roc_auc_score(labels, proba)),
        }

    return compute_metrics


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    (
        Dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    ) = _require_hf()

    import torch

    set_seed(RANDOM_STATE)
    ensure_dirs()

    data_path = args.data_path or RAW_FILE_PATH
    LOGGER.info("Loading %s", data_path)
    df = load_raw_data(data_path)
    df = basic_cleaning(df)
    df["text"] = make_text_feature(df)

    if args.max_rows is not None or args.sample_frac is not None:
        if (
            args.sample_frac is not None
            and args.max_rows is None
            and (args.sample_frac <= 0 or args.sample_frac > 1)
        ):
            raise ValueError("--sample-frac must be in (0, 1]")
        if args.max_rows is not None and args.sample_frac is not None:
            LOGGER.warning(
                "Both --max-rows and --sample-frac set; using --max-rows only."
            )
        n0 = len(df)
        df = stratified_subsample(
            df,
            sample_frac=None if args.max_rows is not None else args.sample_frac,
            max_rows=args.max_rows,
        )
        LOGGER.info("Stratified subsample: rows %s -> %s", n0, len(df))

    X_train, X_valid, y_train, y_valid = split_data(df, TARGET_COLUMN)
    LOGGER.info("train=%s valid=%s", len(X_train), len(X_valid))

    train_df = pd.DataFrame({"text": X_train["text"].values, "label": y_train.values})
    valid_df = pd.DataFrame({"text": X_valid["text"].values, "label": y_valid.values})

    raw_ds_train = Dataset.from_pandas(train_df)
    raw_ds_valid = Dataset.from_pandas(valid_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    max_length = args.max_length

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    tok_train = raw_ds_train.map(
        tokenize,
        batched=True,
        remove_columns=raw_ds_train.column_names,
    )
    tok_valid = raw_ds_valid.map(
        tokenize,
        batched=True,
        remove_columns=raw_ds_valid.column_names,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    class_weights = _build_weight_tensor(y_train)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    out_dir = Path(args.output_dir)
    ckpt_root = out_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    if args.fp16 and not torch.cuda.is_available():
        LOGGER.warning("fp16 requested but CUDA unavailable; training on CPU in fp32.")

    training_args = TrainingArguments(
        output_dir=str(ckpt_root),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        save_total_limit=2,
        fp16=use_fp16,
        logging_steps=max(10, len(tok_train) // (args.batch_size * 8)),
        report_to="none",
        seed=RANDOM_STATE,
        warmup_ratio=args.warmup_ratio,
    )

    trainer = WeightedTrainer.build(
        class_weights,
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_valid,
        data_collator=collator,
        compute_metrics=_compute_metrics_builder(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    LOGGER.info(
        "Starting fine-tune (model=%s, epochs=%s, batch=%s, fp16=%s)",
        args.model_name,
        args.epochs,
        args.batch_size,
        use_fp16,
    )
    trainer.train()

    metrics = trainer.evaluate()
    LOGGER.info("Eval metrics: %s", metrics)

    print("\nHoldout (same 20% stratified split as src.train)")
    print("-" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    preds_out = trainer.predict(tok_valid)
    logits = preds_out.predictions
    y_hat = np.argmax(logits, axis=-1)
    print("\nClassification report (holdout)")
    print("-" * 50)
    print(classification_report(y_valid.values, y_hat, digits=4))

    final_dir = out_dir.resolve()
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    summary = {
        "model_name": args.model_name,
        "data_file": str(Path(data_path).resolve()),
        "output_dir": str(final_dir),
        "random_state": RANDOM_STATE,
        "sample_frac": args.sample_frac,
        "max_rows": args.max_rows,
        "trainer_eval": {k: float(v) for k, v in metrics.items() if isinstance(v, float)},
        "holdout_accuracy": float(accuracy_score(y_valid.values, y_hat)),
        "holdout_f1": float(f1_score(y_valid.values, y_hat)),
        "holdout_roc_auc": float(roc_auc_score(y_valid.values, _softmax_rows(logits)[:, 1])),
    }
    (final_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOGGER.info("Saved model + tokenizer + train_summary.json to %s", final_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT/BERT for review recommendation.")
    p.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Hugging Face model id (try distilbert-base-uncased or bert-base-uncased).",
    )
    p.add_argument("--data-path", type=Path, default=None, help="CSV path (default: raw Kaggle file).")
    p.add_argument("--output-dir", type=Path, default=HF_RECOMMENDER_DIR, help="Save model here.")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument(
        "--metric-for-best-model",
        default="eval_accuracy",
        help="HF metric name for best checkpoint (e.g. eval_accuracy, eval_f1, eval_roc_auc).",
    )
    p.add_argument("--early-stopping-patience", type=int, default=2)
    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mixed precision on CUDA (default: on when CUDA available).",
    )
    p.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Stratified fraction of rows (0,1] before split; for quick experiments.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stratified row cap before split; overrides --sample-frac if both set.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
