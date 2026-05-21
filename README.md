# Women's E-Commerce Reviews — ML Pipeline

Binary classification on the [Kaggle Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) dataset: tabular baselines → TF-IDF + logistic → gradient boosting, with validation, batch inference, and a FastAPI service.

**Default target:** `Recommended IND` (change `TARGET_COLUMN` in `src/config.py` if needed).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
make quickstart                 # install + validate + train + tests
```

**Windows without Make:** `.\scripts\run-workflow.ps1 quickstart`

### Data

Place the CSV at `data/raw/Womens Clothing E-Commerce Reviews.csv`, or download via Kaggle API:

```bash
python -m src.train --download-data --model baseline
```

See [data/README.md](data/README.md) for folder layout.

## Repository structure

```text
womens-e-commerce-reviews/
├── data/
│   ├── raw/              # Kaggle CSV (gitignored)
│   ├── processed/        # ETL, quality reports, CV outputs (generated)
│   └── submissions/      # Kaggle-style predictions
├── docs/                 # Guides (see docs/README.md)
├── models/               # Saved *.joblib pipelines (generated)
├── notebooks/            # EDA and experiments
├── scripts/              # PowerShell workflow (mirrors Makefile)
├── src/                  # Train, inference, API, ETL, validation
│   ├── pipeline/         # validate, etl, audit
│   ├── train.py          # Main training CLI
│   ├── inference.py      # Batch scoring
│   └── api.py            # REST API + UI
├── tests/
├── Makefile              # Primary task runner
└── requirements*.txt     # Full / train-only / dev / CI / transformers
```

## Common commands

| Goal | Command |
|------|---------|
| Help / all targets | `make` or `make help` |
| Data quality gate | `make validate` |
| Train recommended model | `make train-better` |
| XGBoost / ensemble | `make train-xgb` / `make train-ensemble` |
| Unit tests | `make test` |
| Batch scoring | `make inference` |
| REST API + UI | `make api` → http://127.0.0.1:8000/ui |
| Local CI mirror | `make ci-local` |

Full cheat sheet: [docs/MAKE_COMMANDS.md](docs/MAKE_COMMANDS.md).

## Models

| Tier | Command | Artifact |
|------|---------|----------|
| Tabular baseline | `make train-baseline` | `models/baseline_pipeline.joblib` |
| TF-IDF + logistic (default) | `make train-better` | `models/better_pipeline.joblib` |
| Random forest | `make train-advanced` | `models/advanced_pipeline.joblib` |
| XGBoost | `make train-xgb` | `models/advanced_xgb_pipeline.joblib` |
| Ensemble | `make train-ensemble` | `models/ultra_ensemble_pipeline.joblib` |

Submission file: `python -m src.train --model better --make-submission` → `data/submissions/submission_better.csv`

Cross-validation + MLflow: `python -m src.train --model better --cv-f1 --mlflow`

## Serve

```bash
make test
make inference MODEL=better
make api
```

- Health: `GET /health`
- Predict: `POST /predict` (Swagger: `/docs`)
- Override model: `MODEL_PATH=models/baseline_pipeline.joblib`

## Notebooks

| Notebook | Focus |
|----------|--------|
| `01_end_to_end_womens_reviews.ipynb` | Full workflow |
| `02_eda_visualizations.ipynb` | EDA |
| `03_advanced_tuning_experiments.ipynb` | Tuning experiments |

## Documentation

| Topic | Guide |
|-------|--------|
| All guides | [docs/README.md](docs/README.md) |
| Make / PowerShell | [docs/MAKE_COMMANDS.md](docs/MAKE_COMMANDS.md) |
| Deploy (validate → API) | [docs/DEPLOYMENT_RUNBOOK.md](docs/DEPLOYMENT_RUNBOOK.md) |
| Metrics & overfitting | [docs/GENERALIZATION_AND_ACCURACY.md](docs/GENERALIZATION_AND_ACCURACY.md) |
| CI/CD (GitLab) | [docs/CI_CD.md](docs/CI_CD.md) |

## Suggested workflow

1. Explore data in `02_eda_visualizations.ipynb`
2. `make train-baseline` → inspect metrics
3. `make train-better` → compare text features
4. `make train-xgb` or `make train-ensemble` → check overfitting with `make train-debug`
5. `make etl` + `make api` → deployment-style flow
