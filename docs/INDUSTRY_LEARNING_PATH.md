# Industry-Ready Learning Path (Data -> KPI -> ML -> Deployment)

This guide helps students learn the same flow used in real projects:

1. Data engineering foundations
2. Data analysis and KPI design
3. ML experimentation and model selection
4. Deployment and production pipelines
5. Monitoring and iterative improvement

The goal is to build an MVP fast, then improve quality in controlled iterations.

---

## 1) Problem Framing (Business First)

Start every project with a clear business question.

- Business goal: Increase product recommendation quality.
- ML goal: Predict `Recommended IND` accurately.
- Success metrics:
  - Primary: F1 score (imbalanced classification friendly)
  - Secondary: ROC-AUC, precision, recall
  - Business KPI proxy: precision on positive class (fewer false recommendations)

### Student checklist

- Define stakeholders (product manager, analyst, ML engineer).
- Write 3 measurable success criteria.
- Document assumptions and risks.

---

## 2) Data Engineering Foundations

Students should understand how raw data becomes trusted training data.

### Industry concepts to teach

- **Ingestion**: source systems, file drops, APIs
- **Validation**: schema checks, null checks, type checks
- **Versioning**: keep immutable raw data snapshots
- **Lineage**: know where each feature came from
- **Reproducibility**: same input -> same output

### MVP implementation in this project

- Raw data: `data/raw/`
- Processed data: `data/processed/`
- Feature creation: `src/data.py`, `src/features.py`
- Add a data contract file later (schema + required columns)

### Student exercise

- Create `data/processed/data_quality_report.json` with:
  - row count
  - missing values by column
  - duplicate count
  - target distribution

---

## 3) Data Analysis and KPI Building

Students should connect EDA to business decisions.

### KPI examples (analytics layer)

- Recommendation rate by department
- Average rating by department
- Positive feedback count percentiles
- Review length by recommended vs not recommended
- Data freshness KPI (if pipeline is periodic)

### Dashboard-style outputs to generate

- Department recommendation leaderboard
- Drift checks (current month vs historical baseline)
- Segment analysis: age bands, class name, division

### Student exercise

- Build a simple KPI table in notebook output and export to CSV:
  - `data/processed/kpi_snapshot.csv`

---

## 4) Feature Engineering and Baseline ML

Teach students that baseline models are essential.

### Minimum progression

1. Baseline tabular model
2. Text + tabular model
3. Tuned advanced model

### Why this matters

- Baselines provide performance floor.
- Simple models are easier to debug.
- Complexity should be justified by measurable gain.

---

## 5) Experiment Tracking (MLOps Mindset)

Every run should be traceable and comparable.

### What to log every run

- timestamp
- dataset version/hash
- feature set version
- model type + parameters
- train/validation split seed
- metrics (F1, ROC-AUC, precision, recall)
- artifact paths

This project already logs run details in notebook and training logs. Next step is to standardize with a run registry file.

---

## 6) Deployment Path (Industry MVP)

Students should learn a practical deployment route, not only notebook code.

### Recommended staged path

1. **Batch inference MVP** (fastest)
   - Input CSV -> scored output CSV
2. **API inference service**
   - `POST /predict` returns recommendation probability/class
3. **Scheduled pipeline**
   - Daily retrain or weekly retrain + validation checks

### Production components

- Model artifact store (`models/`)
- Inference script/service
- Input validation layer
- Logging + error handling
- Monitoring and alerts

---

## 7) Pipeline Design for Real Projects

Use a modular pipeline with clear stages:

1. `extract`
2. `validate`
3. `transform`
4. `train`
5. `evaluate`
6. `register_model`
7. `deploy`
8. `monitor`

Each stage should:

- have a clear input/output contract
- be testable independently
- produce logs and metrics

---

## 8) Recommended Repo Structure (Industry-Friendly)

Use this direction as the project grows:

```text
.
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ submissions/
├─ docs/
│  ├─ INDUSTRY_LEARNING_PATH.md
│  ├─ KPI_DEFINITIONS.md
│  └─ DEPLOYMENT_RUNBOOK.md
├─ logs/
│  └─ notebooks/
├─ models/
├─ notebooks/
├─ src/
│  ├─ data.py
│  ├─ features.py
│  ├─ train.py
│  ├─ inference.py
│  └─ pipeline/
│     ├─ validate.py
│     ├─ train_step.py
│     └─ monitor.py
├─ tests/
│  ├─ test_data_quality.py
│  ├─ test_features.py
│  └─ test_inference.py
└─ README.md
```

---

## 9) MVP Timeline (4 Weeks)

### Week 1 - Data + KPI

- ingest + validate data
- build first KPI report
- complete EDA notebook

### Week 2 - Baseline ML

- train baseline and better model
- compare with clear metric table
- write experiment notes

### Week 3 - Advanced + Tuning

- run advanced model/tuning
- track experiments systematically
- choose candidate model for deployment

### Week 4 - Deployment + Monitoring

- create batch/API inference path
- add smoke tests
- define monitoring KPIs and retraining trigger

---

## 10) Code Commenting Standard for Students

Use comments to explain **why**, not obvious syntax.

Good:

- why a feature is created
- why a metric is chosen
- why a threshold is selected
- assumptions and trade-offs

Avoid:

- comments that repeat code literally
- stale comments not updated with code changes

---

## 11) Next Repo Improvements (Suggested)

1. Add `src/inference.py` for reusable prediction flow.
2. Add `src/pipeline/validate.py` for schema and quality checks.
3. Add `tests/` with basic unit tests.
4. Add lightweight API (`FastAPI`) for deployment demo — see `src/api.py`.
5. Add `DEPLOYMENT_RUNBOOK.md` for operating the MVP.

These steps move students from notebook work to production thinking.

For LLM vs classical ML tradeoffs at MVP scale, see `docs/LLM_VS_CLASSICAL_ML.md`.

For Makefile / PowerShell automation, see the repository `Makefile` and `scripts/run-workflow.ps1`.

For Databricks + MLflow and homework baselines, see `docs/DATABRICKS_MLFLOW.md` and `docs/STUDENT_ASSIGNMENTS_AND_BASELINES.md`.

