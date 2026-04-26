# =============================================================================
# Womens E-Commerce Reviews — curated Make targets (no need to remember CLI flags)
# =============================================================================
#
# First time on Windows?
#   - Option A: Git Bash or WSL → run `make` here.
#   - Option B: Install Make: `choco install make` or use `scripts\run-workflow.ps1` (same flows).
#
# From repo root, run:
#   make              → list all targets (default)
#   make quickstart   → install deps, validate data, train "better", run tests
#   make check        → validate + unit tests (after you have data + models optional)
#
# Override Python:  make train-better PYTHON=.venv\Scripts\python
#
# Extra train flags:  make train-better TRAIN_EXTRA="--mlflow"
# =============================================================================

.DEFAULT_GOAL := help

PYTHON ?= python
PIP := $(PYTHON) -m pip
# Optional extra args for all train-* targets, e.g. TRAIN_EXTRA=--mlflow
TRAIN_EXTRA ?=
MODEL ?= better

.PHONY: help install install-train install-dev setup quickstart check check-cov ci-local \
	validate etl explain train-baseline train-better train-advanced train-xgb \
	train-all mlflow-better download-better test test-cov inference api databricks-help

help:
	@echo ""
	@echo "=== Most used ==="
	@echo "  make help             - this list (also: run plain  make  with no target)"
	@echo "  make setup            - pip install full requirements.txt"
	@echo "  make quickstart       - setup + validate + train-better + test"
	@echo "  make check            - validate + test (quick quality gate)"
	@echo "  make check-cov        - validate + tests with coverage report"
	@echo ""
	@echo "=== Install ==="
	@echo "  make install          - pip install -r requirements.txt"
	@echo "  make install-train    - pip install -r requirements_train.txt (lighter; no Jupyter)"
	@echo "  make install-dev      - pip install -r requirements-dev.txt (tests + coverage)"
	@echo ""
	@echo "=== Data + train (each run writes models/<name>_pipeline.joblib) ==="
	@echo "  make validate         - data quality JSON under data/processed/quality/"
	@echo "  make etl              - ETL: raw CSV → clean_reviews.csv + manifest + audit log"
	@echo "  make explain          - print top logistic/Tf-idf weights (needs models/better_pipeline.joblib)"
	@echo "  make train-baseline   - baseline model + CV F1"
	@echo "  make train-better     - TF-IDF + logistic + CV F1"
	@echo "  make train-advanced   - RandomForest + CV F1"
	@echo "  make train-xgb        - XGBoost + CV F1 (needs xgboost)"
	@echo "  make train-all        - baseline, better, advanced, xgb in order (long)"
	@echo "  make download-better  - Kaggle download + train better"
	@echo "  make mlflow-better    - train better + log to MLflow (--mlflow)"
	@echo ""
	@echo "=== Test + serve ==="
	@echo "  make test             - unit tests"
	@echo "  make test-cov         - coverage + report (installs coverage if needed)"
	@echo "  make ci-local          - pip-audit + bandit + tests (no gitleaks; run before push)"
	@echo "  make inference        - score raw CSV (MODEL=$(MODEL) → models/$(MODEL)_pipeline.joblib)"
	@echo "  make api              - http://127.0.0.1:8000  (set MODEL_PATH=... if not using better)"
	@echo ""
	@echo "=== Docs ==="
	@echo "  make databricks-help  - Databricks + MLflow pointer"
	@echo ""
	@echo "Optional environment variables:"
	@echo "  TRAIN_EXTRA           - e.g. TRAIN_EXTRA=--mlflow"
	@echo "  MODEL                 - for inference: better | baseline | advanced | advanced_xgb"
	@echo "  MLFLOW_TRACKING_URI   - e.g. file:./mlruns"
	@echo "  MLFLOW_EXPERIMENT     - experiment name"
	@echo "  MODEL_PATH            - for api default artifact (see src/api.py)"
	@echo ""

install:
	$(PIP) install -r requirements.txt

install-train:
	$(PIP) install -r requirements_train.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

# One-shot environment bootstrap (full stack).
setup: install

# Typical first run: deps, check data file passes gate, train default model, verify tests.
quickstart: install validate train-better test

check: validate test

check-cov: validate test-cov

validate:
	$(PYTHON) -m src.pipeline.validate

etl:
	$(PYTHON) -m src.pipeline.etl

explain:
	$(PYTHON) -m src.interpretability

train-baseline:
	$(PYTHON) -m src.train --model baseline --cv-f1 $(TRAIN_EXTRA)

train-better:
	$(PYTHON) -m src.train --model better --cv-f1 $(TRAIN_EXTRA)

train-advanced:
	$(PYTHON) -m src.train --model advanced --cv-f1 $(TRAIN_EXTRA)

train-xgb:
	$(PYTHON) -m src.train --model advanced_xgb --cv-f1 $(TRAIN_EXTRA)

# Train every tier in sequence (use for benchmarking / assignment baselines).
train-all: train-baseline train-better train-advanced train-xgb

mlflow-better:
	$(PYTHON) -m src.train --model better --cv-f1 --mlflow

download-better:
	$(PYTHON) -m src.train --download-data --model better --cv-f1

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

test-cov:
	$(PIP) install -q coverage
	$(PYTHON) -m coverage run -m unittest discover -s tests -p "test_*.py"
	$(PYTHON) -m coverage report -m
	@echo ""
	@echo "Optional HTML: $(PYTHON) -m coverage html  → open htmlcov/index.html"

# Mirrors CI security + test gates (except gitleaks — install gitleaks CLI locally if needed).
ci-local:
	$(PIP) install -q -r requirements-ci.txt
	$(PYTHON) -m pip_audit -r requirements.txt --desc on
	$(PYTHON) -m pip_audit -r requirements_train.txt --desc on
	$(PYTHON) -m bandit -r src -ll -f txt
	$(PYTHON) -m coverage run -m unittest discover -s tests -p "test_*.py"
	$(PYTHON) -m coverage report -m --fail-under=70

inference:
	$(PYTHON) -m src.inference --input-csv "data/raw/Womens Clothing E-Commerce Reviews.csv" --model-path "models/$(MODEL)_pipeline.joblib"

# Interactive API docs: http://127.0.0.1:8000/docs  (GET / alone is not an error — use /docs for POST /predict)
api:
	$(PYTHON) -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

databricks-help:
	@echo See docs/DATABRICKS_MLFLOW.md for cluster / job / MLflow tracking setup.
