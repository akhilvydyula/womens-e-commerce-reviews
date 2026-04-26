from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SUBMISSIONS_DIR = DATA_DIR / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_FILE_NAME = "Womens Clothing E-Commerce Reviews.csv"
RAW_FILE_PATH = RAW_DIR / RAW_FILE_NAME
KAGGLE_DATASET_SLUG = "nicapotato/womens-ecommerce-clothing-reviews"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews/data"

TARGET_COLUMN = "Recommended IND"
ID_COLUMN = "Clothing ID"

TEXT_COLUMNS = ["Title", "Review Text"]
NUMERIC_COLUMNS = ["Age", "Positive Feedback Count", "Rating"]
CATEGORICAL_COLUMNS = ["Division Name", "Department Name", "Class Name"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_dirs() -> None:
    for path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, SUBMISSIONS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
