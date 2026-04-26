# Databricks + MLflow (this repository)

This project logs runs with **MLflow** when you pass `--mlflow` to training. On Databricks, MLflow Tracking is already integrated with the workspace.

## 1) Local MLflow (quick check)

```powershell
pip install -r requirements.txt
$env:MLFLOW_TRACKING_URI = "file:./mlruns"
python -m src.train --model better --cv-f1 --mlflow
```

Open the run artifacts under `./mlruns` or use `mlflow ui` (optional).

## 2) Databricks: recommended teaching path (Notebook job)

1. Upload this repo to a Databricks Repo, or import as a Git folder.
2. Create / open a cluster (ML runtime includes MLflow).
3. In a notebook first cell:

```python
%pip install -r requirements.txt
```

4. Set the experiment (optional) and tracking (usually automatic on Databricks):

```python
import mlflow
mlflow.set_experiment("/Shared/womens-ecommerce-reviews")
```

5. Run training with MLflow logging:

```python
!python -m src.train --model better --cv-f1 --mlflow
```

**What gets logged**

- Params: model name, CV settings, split seed, raw file path string
- Metrics: `val_*` (holdout), and `cv_f1_mean` / `cv_f1_std` when `--cv-f1` is on
- Artifact: the saved `*_pipeline.joblib` file

## 3) Environment variables (local or cluster)

| Variable | Purpose |
|----------|---------|
| `MLFLOW_TRACKING_URI` | On local dev: `file:./mlruns`. On Databricks you typically **do not** need to set this manually. |
| `MLFLOW_EXPERIMENT` | Experiment name (default: `womens-ecommerce-reviews`) |

CLI override:

```text
python -m src.train --model advanced_xgb --cv-f1 --mlflow --mlflow-experiment my-team-exp
```

## 4) “One command” on your laptop

- **Make** (Git Bash / WSL / macOS / Linux): `make mlflow-better`
- **PowerShell**: `.\scripts\run-workflow.ps1 mlflow-better`

## 5) Production-style next steps on Databricks

- Register the model from the MLflow Model Registry UI.
- Schedule a **Jobs** workflow: validate → train → register → smoke inference.
- Add separate dev/staging/prod experiments.

For a local deployment checklist, see `docs/DEPLOYMENT_RUNBOOK.md`.
