from pathlib import Path

from src.nlp_features import NLP_FEATURE_NAMES
from src.survey import SURVEY_FEATURE_NAMES
from src.vision_features import IMAGE_FEATURE_NAMES, VISION_TEXT_FEATURE_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_FILE_NAME = "Womens Clothing E-Commerce Reviews.csv"
RAW_FILE_PATH = RAW_DIR / RAW_FILE_NAME
MODEL_PATH = MODELS_DIR / "model.joblib"

TARGET_COLUMN = "Recommended IND"

TEXT_COLUMNS = ["Title", "Review Text"]
CORE_NUMERIC_COLUMNS = ["Age", "Positive Feedback Count", "Rating"]
ENGINEERED_NUMERIC_COLUMNS = (
    list(NLP_FEATURE_NAMES)
    + list(VISION_TEXT_FEATURE_NAMES)
    + list(IMAGE_FEATURE_NAMES)
    + list(SURVEY_FEATURE_NAMES)
)
NUMERIC_COLUMNS = CORE_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS
CATEGORICAL_COLUMNS = ["Division Name", "Department Name", "Class Name"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
