# Make commands (copy-paste cheat sheet)

Run these from the **repository root** (folder that contains `Makefile`).

## If `make` is not found (Windows)

1. **Git Bash:** usually includes `make` — open Git Bash in the project folder.
2. **PowerShell (no Make):** use the same flows via  
   `.\scripts\run-workflow.ps1 <command>`  
   (see script for the list; mirrors this Makefile.)
3. **Install Make:** e.g. Chocolatey `choco install make`, or use **WSL**.

## Daily use

| Goal | Command |
|------|---------|
| See all targets | `make` or `make help` |
| First-time setup + train + test | `make quickstart` |
| Check data + run tests | `make check` |
| Check data + tests + coverage | `make check-cov` |
| Data quality only | `make validate` |
| **ETL (raw → clean file + manifest)** | `make etl` |
| **Interpretability (linear model weights)** | `make explain` (train `better` first) |
| Train default strong model | `make train-better` |
| **Student debug train** (CV + train vs holdout + holdout JSON manifest) | `make train-debug` |
| Train XGBoost tier | `make train-xgb` |
| Train LightGBM tier | `make train-lgbm` |
| Train soft-voting ensemble (LogReg + XGB + LGBM) | `make train-ensemble` |
| **XGBoost + Optuna tuning** (GPU if available; longer) | `make train-xgb-tune` |
| **Transformer fine-tune** (DistilBERT; GPU; separate from API `.joblib`) | `make train-transformer` (see `docs/TRANSFORMER_AND_HIGH_ACCURACY.md`) |
| **Fast compare existing saved models** (no retraining) | `make leaderboard-fast` |
| **Fast crux run** (25% stratified sample + 2-fold CV) | `make train-quick` |
| Train every tier (long) | `make train-all` |
| Unit tests | `make test` |
| Tests + coverage | `make test-cov` |
| **Local CI mirror** (audit + bandit + tests, no gitleaks) | `make ci-local` |
| Batch score CSV | `make inference` (default `MODEL=better`) |
| REST API | `make api` (browser form: `http://127.0.0.1:8000/ui`) |

## Examples with variables

```bash
# Use project venv interpreter
make train-better PYTHON=.venv/Scripts/python

# Log training to MLflow for any train-* target
make train-better TRAIN_EXTRA=--mlflow

# Debug generalization: same as python -m src.train --cv-f1 --fit-gap --save-holdout-indices
make train-debug
make train-debug MODEL=advanced_xgb

# Score with a different saved model
make inference MODEL=advanced_xgb
```

PowerShell (no Make): `.\scripts\run-workflow.ps1 train-debug` — optional `$env:MODEL = "advanced_xgb"`.

## Requirements files

| Command | Installs |
|---------|----------|
| `make install` | `requirements.txt` (full) |
| `make install-train` | `requirements_train.txt` (lighter) |
| `make install-dev` | `requirements-dev.txt` (coverage, etc.) |
| `make install-transformer` | `requirements-transformers.txt` (HF stack; install torch separately first) |

## Related docs

- `README.md` — full project overview
- `docs/TRANSFORMER_AND_HIGH_ACCURACY.md` — GPU transformers + tuned XGBoost for higher holdout metrics
- `docs/FAST_ITERATION_AND_SAMPLING.md` — sampling, smaller CV, and other speed tricks
- `docs/DATABRICKS_MLFLOW.md` — MLflow on Databricks
- `docs/DEPLOYMENT_RUNBOOK.md` — validate → train → inference → API
