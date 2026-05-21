from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_FILE_NAME = "Womens Clothing E-Commerce Reviews.csv"
RAW_FILE_PATH = RAW_DIR / RAW_FILE_NAME
MODEL_PATH = MODELS_DIR / "model.joblib"

TARGET_COLUMN = "Recommended IND"

TEXT_COLUMNS = ["Title", "Review Text"]
NUMERIC_COLUMNS = ["Age", "Positive Feedback Count", "Rating"]
CATEGORICAL_COLUMNS = ["Division Name", "Department Name", "Class Name"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
