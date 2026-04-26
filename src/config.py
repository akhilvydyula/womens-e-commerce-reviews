"""
Central configuration for paths, feature column names, and train/val split.

Students: keep "magic strings" here so notebooks, CLI training, and the API
all agree on column names and file locations.
"""
from pathlib import Path

# Repo root = parent of the `src/` package (works whether you run from project root or import as a package).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SUBMISSIONS_DIR = DATA_DIR / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

RAW_FILE_NAME = "Womens Clothing E-Commerce Reviews.csv"
RAW_FILE_PATH = RAW_DIR / RAW_FILE_NAME

# Output of the ETL pipeline (`src.pipeline.etl`): ML-ready, cleaned tabular file.
PROCESSED_CLEAN_CSV = PROCESSED_DIR / "clean_reviews.csv"
# One JSON manifest per ETL run (lineage + row counts + paths).
ETL_MANIFEST_DIR = PROCESSED_DIR / "etl_manifests"
# Append-only JSONL for pipeline/training audit (who/what/when style events).
AUDIT_LOG_DIR = LOGS_DIR / "audit"
KAGGLE_DATASET_SLUG = "nicapotato/womens-ecommerce-clothing-reviews"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews/data"

# Default supervised label for this teaching dataset (binary classification).
TARGET_COLUMN = "Recommended IND"
ID_COLUMN = "Clothing ID"

# Feature groups drive `src/features.py` ColumnTransformer definitions.
TEXT_COLUMNS = ["Title", "Review Text"]
NUMERIC_COLUMNS = ["Age", "Positive Feedback Count", "Rating"]
CATEGORICAL_COLUMNS = ["Division Name", "Department Name", "Class Name"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_dirs() -> None:
    """Create on-disk folders so scripts never fail on first write."""
    for path in [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        SUBMISSIONS_DIR,
        MODELS_DIR,
        LOGS_DIR,
        ETL_MANIFEST_DIR,
        AUDIT_LOG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
