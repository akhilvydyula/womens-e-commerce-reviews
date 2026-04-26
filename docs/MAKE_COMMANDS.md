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
| Train XGBoost tier | `make train-xgb` |
| Train every tier (long) | `make train-all` |
| Unit tests | `make test` |
| Tests + coverage | `make test-cov` |
| **Local CI mirror** (audit + bandit + tests, no gitleaks) | `make ci-local` |
| Batch score CSV | `make inference` (default `MODEL=better`) |
| REST API | `make api` |

## Examples with variables

```bash
# Use project venv interpreter
make train-better PYTHON=.venv/Scripts/python

# Log training to MLflow for any train-* target
make train-better TRAIN_EXTRA=--mlflow

# Score with a different saved model
make inference MODEL=advanced_xgb
```

## Requirements files

| Command | Installs |
|---------|----------|
| `make install` | `requirements.txt` (full) |
| `make install-train` | `requirements_train.txt` (lighter) |
| `make install-dev` | `requirements-dev.txt` (coverage, etc.) |

## Related docs

- `README.md` — full project overview
- `docs/DATABRICKS_MLFLOW.md` — MLflow on Databricks
- `docs/DEPLOYMENT_RUNBOOK.md` — validate → train → inference → API
