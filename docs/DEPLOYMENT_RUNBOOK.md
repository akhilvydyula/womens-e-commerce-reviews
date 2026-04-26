# Deployment Runbook (MVP -> Production)

This runbook is designed for students and early-career engineers to operate the model like a real team.

## 1) Pre-deploy checklist

- Data file exists at `data/raw/Womens Clothing E-Commerce Reviews.csv`
- Quality gate passes
- Model trains successfully
- Inference smoke-test succeeds
- Logs and artifacts are generated

## 2) Validate data quality

```bash
python -m src.pipeline.validate
```

Or with Make: `make validate` (see repository `Makefile`).

Output:
- JSON report in `data/processed/quality/`

If you want stricter gate:

```bash
python -m src.pipeline.validate --min-rows 5000
```

## 3) Train model artifact

```bash
python -m src.train --model better
```

Artifact:
- `models/better_pipeline.joblib`

## 4) Batch inference

```bash
python -m src.inference --input-csv "data/raw/Womens Clothing E-Commerce Reviews.csv" --model-path "models/better_pipeline.joblib"
```

Output:
- `data/submissions/inference_output_<timestamp>.csv`

## 5) Smoke tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## 6) Operational best practices

- Keep all run logs in `logs/`
- Keep model artifacts versioned by timestamp/commit
- Keep quality reports for auditing
- Record metric regressions between releases

## 7) HTTP API (FastAPI)

Install API dependencies (included in `requirements.txt`):

```bash
pip install -r requirements.txt
```

Train a model artifact first (example: `models/better_pipeline.joblib`), then:

```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

- `GET /health` — process + model path probe
- `POST /predict` — JSON body with the same fields as one CSV row (see `PredictRequest` in `src/api.py`)

Optional: point to a different artifact:

```powershell
$env:MODEL_PATH = "models/baseline_pipeline.joblib"
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

## 8) Suggested production upgrades

- Add authentication and rate limiting in front of `POST /predict`
- Add CI pipeline to run quality + tests on every commit
- Add scheduled retraining
- Add drift monitoring and alert thresholds
