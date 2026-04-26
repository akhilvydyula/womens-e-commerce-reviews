"""
Minimal HTTP inference API for teaching deployment patterns.

Run locally:
  uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

Model path defaults to models/better_pipeline.joblib; override with env MODEL_PATH
(e.g. models/advanced_xgb_pipeline.joblib after training that tier).
Relative MODEL_PATH is resolved against the project root (repo root).
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from starlette.requests import Request

from src.config import MODELS_DIR, PROJECT_ROOT
from src.inference import load_model, prepare_features_for_model, predict_scores

LOGGER = logging.getLogger(__name__)

_DEFAULT_MODEL = MODELS_DIR / "better_pipeline.joblib"
_PLAYGROUND_HTML_PATH = Path(__file__).with_name("playground.html")


def _playground_html() -> str:
    return _PLAYGROUND_HTML_PATH.read_text(encoding="utf-8")


def _resolve_model_path() -> Path:
    """Resolve MODEL_PATH: absolute paths as-is; relative paths from project root."""
    raw = os.environ.get("MODEL_PATH", str(_DEFAULT_MODEL))
    p = Path(raw)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


class PredictRequest(BaseModel):
    """
    One review row. JSON can use either Kaggle CSV names (e.g. \"Review Text\")
    or common snake_case variants (e.g. review_text) thanks to AliasChoices.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    Title: str = Field(
        default="",
        validation_alias=AliasChoices("Title", "title"),
        serialization_alias="Title",
    )
    Review_Text: str = Field(
        default="",
        validation_alias=AliasChoices("Review Text", "Review_Text", "review_text"),
        serialization_alias="Review Text",
    )
    Age: float = Field(
        default=0.0,
        validation_alias=AliasChoices("Age", "age"),
        serialization_alias="Age",
    )
    Positive_Feedback_Count: int = Field(
        default=0,
        validation_alias=AliasChoices(
            "Positive Feedback Count",
            "Positive_Feedback_Count",
            "positive_feedback_count",
        ),
        serialization_alias="Positive Feedback Count",
    )
    Rating: float = Field(
        default=0.0,
        validation_alias=AliasChoices("Rating", "rating"),
        serialization_alias="Rating",
    )
    Division_Name: str = Field(
        default="Unknown",
        validation_alias=AliasChoices("Division Name", "Division_Name", "division_name"),
        serialization_alias="Division Name",
    )
    Department_Name: str = Field(
        default="Unknown",
        validation_alias=AliasChoices("Department Name", "Department_Name", "department_name"),
        serialization_alias="Department Name",
    )
    Class_Name: str = Field(
        default="Unknown",
        validation_alias=AliasChoices("Class Name", "Class_Name", "class_name"),
        serialization_alias="Class Name",
    )

    def to_dataframe_row(self) -> dict:
        """Exact Kaggle column names for sklearn ColumnTransformer + pandas."""
        d = self.model_dump(by_alias=True)
        return d


class PredictResponse(BaseModel):
    prediction: int
    score: float


_model_cache: dict[str, object] = {}


def get_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = _resolve_model_path()
        _model_cache["path"] = str(path)
        _model_cache["startup_error"] = None
        try:
            _model_cache["model"] = load_model(path)
            LOGGER.info("Loaded model from %s", path)
        except FileNotFoundError as exc:
            _model_cache["model"] = None
            _model_cache["startup_error"] = str(exc)
            LOGGER.error(
                "Model not found at %s. Train first: make train-better", path
            )
        except Exception as exc:  # pragma: no cover - defensive
            _model_cache["model"] = None
            _model_cache["startup_error"] = repr(exc)
            LOGGER.exception("Failed to load model")
        yield

    app = FastAPI(
        title="Womens Reviews Inference",
        version="0.1.0",
        lifespan=lifespan,
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.exception_handler(RequestValidationError)
    async def validation_errors(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "request_validation_failed",
                "issues": exc.errors(),
                "hint": (
                    'Use keys like the Kaggle CSV: "Title", "Review Text", "Age", '
                    '"Positive Feedback Count", "Rating", "Division Name", '
                    '"Department Name", "Class Name". '
                    "Snake_case variants (e.g. review_text) also work. "
                    "Open /ui for a browser form, or /docs and use Try it out."
                ),
            },
        )

    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    def playground() -> str:
        """Browser form that calls POST /predict (teaching: UIs are optional API clients)."""
        return _playground_html()

    @app.get("/")
    def root() -> dict:
        return {
            "service": "Womens E-Commerce Reviews — inference API",
            "health_check": "/health",
            "browser_playground": "/ui",
            "predict": "POST /predict (see /docs for JSON schema)",
            "interactive_docs": "/docs",
            "alternate_docs": "/redoc",
        }

    @app.get("/health")
    def health() -> dict:
        err = _model_cache.get("startup_error")
        loaded = _model_cache.get("model") is not None
        body: dict = {
            "status": "ok" if loaded else "degraded",
            "model_path": _model_cache.get("path"),
            "model_loaded": loaded,
        }
        if err:
            body["error"] = err
            body["fix"] = (
                "Run: make train-better   (or set MODEL_PATH to an existing .joblib file)"
            )
        return body

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest) -> PredictResponse:
        model = _model_cache.get("model")
        if model is None:
            err = _model_cache.get("startup_error") or "Model not loaded"
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Model unavailable",
                    "error": err,
                    "fix": "Train a model (make train-better) or set MODEL_PATH to a valid .joblib",
                },
            )

        row = body.to_dataframe_row()
        df = pd.DataFrame([row])
        feat = prepare_features_for_model(df)
        pred_series, score_series = predict_scores(model, feat)
        return PredictResponse(
            prediction=int(pred_series.iloc[0]),
            score=float(score_series.iloc[0]),
        )

    return app


app = get_app()
