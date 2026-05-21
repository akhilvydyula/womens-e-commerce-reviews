# Student assignments: beat the baselines (CV F1 + holdout metrics)

This document defines **reproducible** benchmarks and homework-style challenges.

## Interpreting numbers (accuracy vs probability vs generalization)

- If you use the **`/ui` playground** or **`POST /predict`**, the **`score`** field is **P(recommended)** for that single row—not “accuracy.” **Accuracy** needs many labeled rows and compares predictions to true labels.
- **Overfitting** means the model fits training noise: **train metrics much better than holdout**. **Underfitting** means both are weak. See **`docs/GENERALIZATION_AND_ACCURACY.md`** and run training with **`--fit-gap`** for a train vs holdout table.
- A fixed **99%** target is not always meaningful; prefer **honest holdout + F1/ROC-AUC** and avoid **tuning on the holdout** (that makes the score optimistic). Use **`make train-debug`** (or **`--save-holdout-indices`**) to record which rows are your locked evaluation set.

## Metric definitions

- **CV F1 (mean ± std)**: stratified K-fold F1 computed on the **training split only** (`--cv-f1`, default 3 folds). This estimates stability without touching the holdout set.
- **Holdout F1 / ROC-AUC / log-loss**: computed once on the **validation split** after training on the full training split (see `src/train.py`).

## How to record your numbers

```powershell
python -m src.train --model baseline --cv-f1
python -m src.train --model better --cv-f1
python -m src.train --model advanced --cv-f1
python -m src.train --model advanced_xgb --cv-f1
```

Each run prints CV F1 and validation metrics, and writes a JSON summary to:

`data/processed/cv_reports/cv_f1_<model>_<timestamp>.json`

(That folder is gitignored when empty of tracked files; runs still create it locally.)

## Reference scores (example run on the full Kaggle CSV)

Recorded with: `RANDOM_STATE=42`, `TEST_SIZE=0.2`, **3-fold** stratified CV on the **training** split only (`--cv-f1`), then final fit + **holdout** metrics on the validation split.

These numbers are **machine-dependent** (CPU/GPU, library versions). Students should reproduce on their environment before treating any value as a strict leaderboard.

| Model | CV F1 mean | CV F1 std | Val F1 | Val ROC-AUC | Val log-loss |
|-------|------------|-----------|--------|-------------|----------------|
| baseline | 0.9601 | 0.0010 | 0.9605 | 0.9722 | 0.1947 |
| better | 0.9612 | 0.0009 | 0.9618 | 0.9780 | 0.1616 |
| advanced | 0.9611 | 0.0031 | 0.9588 | 0.9694 | 0.2866 |
| advanced_xgb | 0.9636 | 0.0025 | 0.9633 | 0.9764 | 0.1495 |

**Teaching note:** here, `advanced_xgb` improves **CV F1**, **holdout F1**, and especially **log-loss** vs `better`, while the default `advanced` RandomForest is **not** well-calibrated (higher log-loss) even though CV F1 is similar. This is a great discussion topic for calibration, thresholding, and model choice.

**Instructor note:** re-run the four `python -m src.train ...` commands each term and refresh this table if library versions change materially.

## Assignment 1 — Beat `better` without new libraries

- Change only **hyperparameters** and **TF-IDF settings** in `src/features.py` / `src/train.py` (logistic `C`, `max_features`, `ngram_range`, `min_df`).
- Deliverable: table with old vs new CV F1 mean/std and holdout metrics; 5-bullet memo on what changed and why.

## Assignment 2 — Beat `advanced_xgb`

- Tune XGBoost (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`) with Optuna or grid search.
- Deliverable: MLflow run link or screenshot; final registered model name.

## Assignment 3 — Deployment

- Train `better` or `advanced_xgb`, run `src.inference` batch job, and bring up `src.api` locally.
- Deliverable: sample `curl` or PowerShell `Invoke-RestMethod` call to `/predict`.

## Integrity rules for grading

- Document random seed (`RANDOM_STATE` in `src/config.py`).
- Never tune on the holdout set; use CV or a nested strategy.
- Keep raw data immutable; store derived artifacts under `data/processed/`.
