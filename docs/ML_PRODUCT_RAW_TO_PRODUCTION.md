# From raw data to ML product (industry-style map for this repo)

This guide ties **data engineering**, **analytics**, **modeling**, **interpretability**, **production**, **environments**, **CI/CD**, **monitoring**, and **audit** to the actual code paths in this project. Use it for teaching and interview storytelling.

---

## 1. End-to-end flow in one picture

```text
Extract (raw CSV)
    → Validate (quality gate + JSON report)
    → Transform (cleaning + optional `text` column)
    → Load (processed/clean_reviews.csv + ETL manifest)
    → Train (sklearn/XGBoost pipelines → models/*.joblib)
    → Evaluate (holdout + optional CV F1)
    → Register / track (MLflow optional)
    → Serve (batch inference + FastAPI)
    → Monitor + audit (logs, drift, alerts — your platform choice)
```

**Code touchpoints**

| Stage | Where |
|-------|--------|
| Extract / Transform / Load | `src/pipeline/etl.py` → `data/processed/clean_reviews.csv`, `data/processed/etl_manifests/` |
| Validate only | `src/pipeline/validate.py` |
| Train | `src/train.py` (`--data-path` can point at ETL output) |
| Batch score | `src/inference.py` |
| Online API | `src/api.py` |
| Interpretability (linear model) | `python -m src.interpretability` |
| Audit trail (JSONL) | `src/pipeline/audit.py` → `logs/audit/audit_YYYYMMDD.jsonl` |
| Automation | `Makefile`, `scripts/run-workflow.ps1` |

---

## 2. Data engineering (this repository)

### What “raw” means

- **Raw**: immutable landing file under `data/raw/` (Kaggle export). Never overwrite; re-download if needed.

### What ETL does here

`python -m src.pipeline.etl` (or `make etl`):

1. **Extract**: read CSV.
2. **Validate**: same gate as training (`validate_data_gate`) — row counts, required columns, target presence.
3. **Transform**: `basic_cleaning` + optional combined **`text`** field for NLP.
4. **Load**: write **`data/processed/clean_reviews.csv`** and a **manifest JSON** (lineage, row counts, paths, elapsed time).

### Why both raw and processed?

- **Analytics / BI** can consume `clean_reviews.csv` without re-running notebooks.
- **ML** can train from raw (current default) **or** from processed (`--data-path data/processed/clean_reviews.csv`) to mimic a governed feature store / curated layer.

---

## 3. Productionization for end users

| User need | Pattern in this repo | Typical “real” upgrade |
|-----------|----------------------|------------------------|
| Batch scoring nightly | `src.inference` | Airflow / Databricks Jobs / Azure Data Factory |
| Real-time score | `src.api` (FastAPI) | API behind API Management / Ingress + autoscaling |
| Model artifact | `models/*_pipeline.joblib` | Registry (MLflow Model Registry, Azure ML, SageMaker) |
| Secrets | Kaggle `kaggle.json` | Key Vault / Databricks secrets scopes |

**Low latency**: prefer **vectorized batch** or **small linear / tree** models; avoid LLM-in-the-loop for per-row scoring at massive QPS. For **sub-10ms** SLAs, teams often use **ONNX**, **Triton**, or **embedded scoring** in the database — out of scope here but mention in interviews.

---

## 4. Business value & NLP (women’s e-commerce reviews)

**Business questions this dataset supports**

- Which reviews predict **recommendation** vs **not recommended**?
- Which **departments** or **ratings** drive negative outcomes?
- What **language patterns** correlate with returns / dissatisfaction?

**NLP in this codebase**

- **TF-IDF** on `Title` + `Review Text` (via combined `text` in training pipeline).
- **n-grams**, `max_features`, `min_df` are levers students tune (`src/features.py`).

**Features (config-driven)**

- Text: `Title`, `Review Text` → combined `text`.
- Numeric: `Age`, `Positive Feedback Count`, `Rating`.
- Categorical: `Division Name`, `Department Name`, `Class Name`.
- Target: `Recommended IND`.

**Models & parameters (CLI)**

- `baseline`: logistic on tabular only.
- `better`: TF-IDF + logistic (`C`, `solver`, class weights).
- `advanced`: RandomForest (`n_estimators`, depth, leaf).
- `advanced_xgb`: XGBoost (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, GPU optional).

---

## 5. Interpretability & stakeholder storytelling

| Audience | What to show |
|----------|----------------|
| Product / ops | KPI dashboards (rates by department, rating mix) — extend `notebooks/02_eda_visualizations.ipynb`. |
| Risk / compliance | Audit JSONL + ETL manifests (what data, how many rows, when). |
| Data science | CV vs holdout metrics, confusion by segment, calibration plots. |

**Linear model narrative (fast)**

```bash
python -m src.train --model better
python -m src.interpretability --model-path models/better_pipeline.joblib
```

Weights highlight **phrases** pushing prediction toward/away from recommendation — good for slide decks. For **RandomForest / XGBoost**, add **permutation importance** or **SHAP** in a notebook (extra dependency).

---

## 6. Environments (lower vs higher)

Typical pattern:

| Environment | Data | Models | Purpose |
|-------------|------|--------|---------|
| **Dev** | sample / synthetic | ephemeral | fast iteration |
| **Staging** | production-like snapshot | candidate build | integration + load tests |
| **Prod** | governed sources | registered + versioned | live traffic |

**How to mirror in this repo**

- Set `DEPLOY_ENV=dev|staging|prod` (stored in **audit** lines).
- Use different `MLFLOW_EXPERIMENT` or tracking URIs per env.
- Never run `etl --no-fail-on-gate` in prod (that flag is for messy classrooms only).

---

## 7. CI/CD (conceptual + practical hooks)

**Minimal CI pipeline**

1. Lint / format (optional).
2. `python -m unittest` or `make test-cov`.
3. `python -m src.pipeline.validate` on a **fixed sample** or schema contract.
4. Train **smoke** job on small data (optional) or skip in CI for cost.
5. Build container for `src.api` (optional).

**Azure**

- Azure DevOps pipelines or GitHub Actions → Azure ML / ACR deploy.
- Wire **MLflow** to Azure Databricks or Azure ML tracking.

**Databricks**

- Jobs run `src.pipeline.etl` then `src.train --mlflow` on a schedule.
- See `docs/DATABRICKS_MLFLOW.md`.

This repo does not ship a full GitHub Actions YAML by default (workspace-specific); students should add one as an exercise.

---

## 8. Monitoring (local vs cloud)

**What to monitor**

| Signal | Why |
|--------|-----|
| Row counts / null rates | upstream breakage |
| Prediction distribution drift | model staleness |
| Latency p95 / error rate | serving health |
| Business KPIs | real value |

**Local**: log files + `logs/audit/*.jsonl` + MLflow UI.

**Cloud**: Azure Monitor, Databricks system tables, Datadog, etc. — export the same **structured events** you already append in `audit.py`.

---

## 9. Audit logs & debugging

**Files**

- **Training / ETL**: `logs/audit/audit_YYYYMMDD.jsonl` (append-only JSON lines).
- **ETL lineage**: `data/processed/etl_manifests/etl_manifest_<run_id>.json` (gitignored when under `data/processed/*`).

**Debugging workflow**

1. Find `run_id` in audit line for failed window.
2. Open matching ETL manifest (input path, row counts, gate errors).
3. Cross-check `logs/train_*.log` for model stack traces.

Set `DEPLOY_ENV` in each environment so audit lines show **where** the run executed.

---

## 10. Beginner → “cracking interviews” checklist

1. **Tell the story**: business metric → data contract → model → deployment → monitoring.
2. **Show trade-offs**: accuracy vs latency vs interpretability vs cost.
3. **Own the data path**: raw vs curated; why ETL; what validation gates.
4. **Own the model path**: baseline → better → boosting; **when** each wins.
5. **Own production**: batch vs online, rollback, registry, audit.

This codebase is a **curated spine** for that narrative; extend with your cloud accounts and one real CI YAML when you are ready.

---

## 11. Quick commands

```bash
make etl              # raw → clean_reviews.csv + manifest + audit
make validate         # quality gate only
make train-better     # train default strong linear+TF-IDF model
python -m src.interpretability
make inference
make api
```

PowerShell: `.\scripts\run-workflow.ps1 etl` (after adding to script).
