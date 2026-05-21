# Pushing accuracy higher (GPU transformers + tuned XGBoost)

## First: what is that ~0.94 number?

If you saw **0.9387** in the **`/ui` playground**, that value is **`score` = P(recommended)** for a **single** review, not **accuracy** on thousands of labeled rows. Always report **holdout accuracy / F1 / ROC-AUC** from training logs, not a one-off probability.

## What “99% accuracy” really means here

On this dataset, **strong classical models already land in the mid‑high 90s** for accuracy on a **single random 80/20 stratified split** (see `docs/GENERALIZATION_AND_ACCURACY.md`). Moving from, say, **96% → 99%** may or may not be realistic because:

- Some errors are **inherent label noise** (ambiguous reviews).
- **Accuracy** can look “good enough” while hiding poor performance on the minority class—watch **F1** and the confusion matrix too.
- Chasing a round number by **re-tuning on the holdout** inflates metrics; use **CV on the training split** or accept a **fresh** locked test set.

Treat **99%** as an aspiration to **earn with a clean protocol**, not a guarantee.

---

## Path A — Tuned XGBoost (same stack as production API)

Your repo already uses **GPU XGBoost** when CUDA works (`tree_method=hist`, `device=cuda` in `src/train.py`).

**One-command tuning** (Optuna search + then full fit + CV F1; takes longer):

```bash
make train-xgb-tune
```

Fewer trials (faster):

```bash
make train-xgb-tune TUNE_XGB_TRIALS=8
```

Or directly:

```bash
python -m src.train --model advanced_xgb --cv-f1 --tune-xgb --tune-xgb-trials 16
```

Artifacts: still `models/advanced_xgb_pipeline.joblib` — **compatible with `make api`** after you point `MODEL_PATH` if needed.

### Extra boosting / bagging-style experiments

```bash
make train-lgbm
make train-ensemble
```

- `train-lgbm`: LightGBM boosting baseline (GPU if available).
- `train-ensemble`: soft-voting blend of Logistic + XGBoost + LightGBM.

---

## Path B — Transformer fine-tuning (maximum “organic” text upgrade)

This uses a **DistilBERT**-style encoder (default `distilbert-base-uncased`) with:

- The **same** stratified train/holdout split as `src.train` (fair comparison).
- **Class-weighted** cross-entropy (similar spirit to `class_weight="balanced"`).
- **fp16** on CUDA when available.
- **Early stopping** on the holdout metric you choose (default **eval_accuracy**).

### Install (Windows + NVIDIA GPU typical flow)

1. Install **PyTorch with CUDA** using the command from [pytorch.org](https://pytorch.org/get-started/locally/) (pick your CUDA version).
2. From the repo root:

```bash
pip install -r requirements-transformers.txt
```

### Train

```bash
make train-transformer
```

Or:

```bash
python -m src.train_transformer --epochs 4 --batch-size 16
```

Stronger but slower / more VRAM:

```bash
python -m src.train_transformer --model-name bert-base-uncased --batch-size 8 --epochs 3
```

### Outputs

- Model + tokenizer under **`models/hf_recommender/`** (gitignored except `.gitkeep` pattern).
- **`models/hf_recommender/train_summary.json`** — holdout accuracy, F1, ROC-AUC.

**Important:** the FastAPI service in this course repo loads **sklearn pipelines** (`.joblib`). The HF folder is for **research / benchmarking / custom serving**. Wiring transformers into `/predict` is a possible extension project.

---

## Recommended evaluation habit

1. Run **`make train-debug`** on your sklearn model (train vs holdout gap).
2. Run **`make train-xgb-tune`** or **`make train-transformer`**.
3. Compare **holdout** metrics using the **same split** (same `RANDOM_STATE` / `TEST_SIZE` in `src/config.py`).
4. If train ≫ holdout, you are likely **overfitting**—add regularization, simpler model, or more data—not just more epochs.

For the conceptual background, see **`docs/GENERALIZATION_AND_ACCURACY.md`**.
