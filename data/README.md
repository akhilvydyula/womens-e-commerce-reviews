# Data directories (how this repo uses them)

Paths are defined in `src/config.py`. Code creates these folders automatically when you run training, validation, or inference.

| Folder | Purpose |
|--------|---------|
| `data/raw/` | **Source of truth**: place the Kaggle CSV here (`Womens Clothing E-Commerce Reviews.csv`). Training and notebooks read from this path. |
| `data/processed/` | **Derived artifacts**: quality reports (`quality/`), experiment logs (`experiments/`), KPI exports you add in class. |
| `data/submissions/` | **Outputs**: Kaggle-style submission CSVs and batch inference results. |

**Why you might not see files in Git:** large or generated files are listed in `.gitignore` (for example processed outputs and `.joblib` models). Empty folders are kept with `.gitkeep` so the layout stays visible in the repository.

**Quick check locally:** after `python -m src.train --model baseline`, you should see `models/baseline_pipeline.joblib` and logs under `logs/`. After `python -m src.pipeline.validate`, look under `data/processed/quality/`.
