# Data directories (how this repo uses them)

Paths are defined in `src/config.py`. Code creates these folders automatically when you run training, validation, or inference.

| Folder | Purpose |
|--------|---------|
| `data/raw/` | **Source of truth**: place the Kaggle CSV here (`Womens Clothing E-Commerce Reviews.csv`). Training and notebooks read from this path. |
| `data/processed/` | **Derived artifacts**: **`clean_reviews.csv`** (output of `python -m src.pipeline.etl` / `make etl`), ETL manifests (`etl_manifests/`), quality reports (`quality/`), CV reports (`cv_reports/`), experiment logs (`experiments/`). |
| `data/submissions/` | **Outputs**: Kaggle-style submission CSVs and batch inference results. |

**Why you might not see files in Git:** large or generated files are listed in `.gitignore` (for example processed outputs and `.joblib` models). Empty folders are kept with `.gitkeep` so the layout stays visible in the repository.

**Data pipeline (engineering):** `make etl` runs **Extract → Validate → Transform → Load** and writes `clean_reviews.csv` plus a JSON manifest under `etl_manifests/`. Train from that file with `python -m src.train --model better --data-path data/processed/clean_reviews.csv`.

**Quick check locally:** after `python -m src.train --model baseline`, you should see `models/baseline_pipeline.joblib`. After `python -m src.pipeline.validate`, look under `data/processed/quality/`. Audit-style events append to `logs/audit/*.jsonl` (see `docs/ML_PRODUCT_RAW_TO_PRODUCTION.md`).
