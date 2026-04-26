# Curated commands for local development and teaching workflows.
# Windows: install "make" via Git for Windows, Chocolatey (gmake), or WSL; or use scripts/run-workflow.ps1
#
# Override Python:  make train-better PYTHON=python3

PYTHON ?= python
PIP := $(PYTHON) -m pip
# Optional extra args for all train-* targets, e.g. TRAIN_EXTRA=--mlflow
TRAIN_EXTRA ?=

.PHONY: help install install-train validate train-baseline train-better train-advanced train-xgb test inference api mlflow-better databricks-help download-better

help:
	@echo "Targets:"
	@echo "  make install          - pip install -r requirements.txt"
	@echo "  make install-train    - pip install -r requirements_train.txt (no Jupyter stack)"
	@echo "  make validate         - data quality gate"
	@echo "  make train-baseline   - train baseline + optional CV F1 + MLflow (see env below)"
	@echo "  make train-better     - train better (TF-IDF + logistic)"
	@echo "  make train-advanced   - train RandomForest advanced"
	@echo "  make train-xgb        - train XGBoost advanced (GPU if available)"
	@echo "  make test             - unit tests"
	@echo "  make inference        - batch score raw CSV (set MODEL= better|baseline|advanced|advanced_xgb)"
	@echo "  make api              - uvicorn API (needs trained model)"
	@echo "  make mlflow-better    - train better with MLflow + CV F1 logged"
	@echo "  make download-better  - download Kaggle data then train better"
	@echo "  make databricks-help  - print path to Databricks + MLflow doc"
	@echo ""
	@echo "Optional env:"
	@echo "  MLFLOW_TRACKING_URI   - e.g. file:./mlruns or Databricks tracking URI"
	@echo "  MLFLOW_EXPERIMENT     - experiment name (default: womens-ecommerce-reviews)"
	@echo "  MODEL_PATH            - for 'make inference' output model (default by MODEL)"

install:
	$(PIP) install -r requirements.txt

install-train:
	$(PIP) install -r requirements_train.txt

validate:
	$(PYTHON) -m src.pipeline.validate

train-baseline:
	$(PYTHON) -m src.train --model baseline --cv-f1 $(TRAIN_EXTRA)

train-better:
	$(PYTHON) -m src.train --model better --cv-f1 $(TRAIN_EXTRA)

train-advanced:
	$(PYTHON) -m src.train --model advanced --cv-f1 $(TRAIN_EXTRA)

train-xgb:
	$(PYTHON) -m src.train --model advanced_xgb --cv-f1 $(TRAIN_EXTRA)

# Example: make mlflow-better MLFLOW_TRACKING_URI=file:./mlruns
mlflow-better:
	$(PYTHON) -m src.train --model better --cv-f1 --mlflow

download-better:
	$(PYTHON) -m src.train --download-data --model better --cv-f1

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

# Batch inference: MODEL=better (default artifact models/better_pipeline.joblib)
MODEL ?= better
inference:
	$(PYTHON) -m src.inference --input-csv "data/raw/Womens Clothing E-Commerce Reviews.csv" --model-path "models/$(MODEL)_pipeline.joblib"

api:
	$(PYTHON) -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

databricks-help:
	@echo See docs/DATABRICKS_MLFLOW.md for cluster / job / MLflow tracking setup.
