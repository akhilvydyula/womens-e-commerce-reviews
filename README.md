# Women's E-Commerce Reviews - End-to-End ML Case Study

This project is a teaching-friendly codebase built around the Kaggle dataset:
`Women's E-Commerce Clothing Reviews`.

It is designed for students to learn in stages:

1. Baseline model (simple tabular features)
2. Better model (text + tabular)
3. Advanced model (boosting + hyperparameter tuning)
4. Generate prediction CSV in Kaggle format

## 1) Project structure

```text
.
├─ data/
│  ├─ raw/                 # place downloaded CSV here (see data/README.md)
│  ├─ processed/           # cleaned/intermediate outputs
│  └─ submissions/         # final prediction CSV files
├─ models/                 # saved models
├─ notebooks/
│  ├─ 01_end_to_end_womens_reviews.ipynb
│  ├─ 02_eda_visualizations.ipynb
│  └─ 03_advanced_tuning_experiments.ipynb
├─ src/
│  ├─ config.py
│  ├─ data.py
│  ├─ download_data.py
│  ├─ features.py
│  ├─ inference.py
│  ├─ api.py
│  └─ train.py
├─ requirements.txt
└─ README.md
```

## 2) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Automation (one-command workflows)

- **Make** (Git Bash, WSL, macOS, Linux, or Windows with `make` installed): see `Makefile` — e.g. `make train-better`, `make validate`, `make mlflow-better`.
- **PowerShell (Windows):** `.\scripts\run-workflow.ps1 train-better` (same curated commands; no `make` required).

Databricks + MLflow: `docs/DATABRICKS_MLFLOW.md`. Student baselines / homework: `docs/STUDENT_ASSIGNMENTS_AND_BASELINES.md`.

## 3) Prepare data (Direct from Kaggle Web/API)

Dataset page:
`https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews/data`

### Option A (recommended): direct download from code

1. Authenticate Kaggle once:
   - Go to Kaggle Account -> Create New API Token
   - Save `kaggle.json` to `%USERPROFILE%/.kaggle/kaggle.json`
2. Run:

```bash
python -m src.train --download-data --model baseline
```

This downloads the dataset and copies the CSV to:
`data/raw/Womens Clothing E-Commerce Reviews.csv`

### Option B: manual web download

Download from Kaggle website and place the CSV in:
`data/raw/Womens Clothing E-Commerce Reviews.csv`

## 4) Train models

### Baseline

```bash
python -m src.train --model baseline
```

### Better model (text + tabular)

```bash
python -m src.train --model better
```

### Advanced model (boosting + optional tuning)

```bash
python -m src.train --model advanced
```

### Strong gradient boosting (XGBoost; GPU if available)

```bash
python -m src.train --model advanced_xgb
```

With tuning:

```bash
python -m src.train --model advanced --tune
```

**Cross-validation F1** on the training split (mean/std) + JSON report:

```bash
python -m src.train --model better --cv-f1
```

**MLflow** (local `mlruns` or Databricks tracking):

```bash
python -m src.train --model better --cv-f1 --mlflow
```

## 5) Create submission

```bash
python -m src.train --model better --make-submission
```

All-in-one (download + train + submission):

```bash
python -m src.train --download-data --model better --make-submission
```

This writes:

`data/submissions/submission_<model>.csv`

## 5b) Data validation gate (industry-style)

Run a data-quality check before training:

```bash
python -m src.pipeline.validate
```

This writes a JSON quality report to:

`data/processed/quality/`

## 5c) Batch inference (deployment MVP)

After training a model, run batch inference:

```bash
python -m src.inference --input-csv "data/raw/Womens Clothing E-Commerce Reviews.csv" --model-path "models/better_pipeline.joblib"
```

This writes:

`data/submissions/inference_output_<timestamp>.csv`

## 5d) Run starter tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## 5e) HTTP API (FastAPI)

After training (so `models/better_pipeline.joblib` exists):

```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/predict` with JSON fields matching one data row (`Title`, `Review Text`, `Age`, …)

Override model path:

```powershell
$env:MODEL_PATH = "models/baseline_pipeline.joblib"
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

## 6) Learning roadmap for students

- Start with EDA in notebook.
- Train `baseline` and inspect metrics.
- Add text features in `better`; compare gains.
- Try `advanced` and explain when it helps.
- Run small tuning loops and discuss overfitting.
- Export model + inference function.

## 7) Target options

Default target is `Recommended IND` (binary classification).

If your Kaggle challenge target differs:
- Change `TARGET_COLUMN` in `src/config.py`
- Adjust model objective in `src/train.py` if needed

## 8) Suggested classroom exercises

- Add n-grams to TF-IDF and compare.
- Try class weighting for imbalance.
- Evaluate by age segments.
- Build error analysis report for false positives/negatives.
- Replace TF-IDF with sentence embeddings and compare.

## 9) Industry-ready student pathway

For a practical end-to-end learning track (data engineering -> KPI analytics -> ML -> deployment/MLOps), see:

- `docs/INDUSTRY_LEARNING_PATH.md`
- `docs/KPI_DEFINITIONS.md`
- `docs/DEPLOYMENT_RUNBOOK.md`
- `docs/LLM_VS_CLASSICAL_ML.md`
- `data/README.md` (what each folder is for)

