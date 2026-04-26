"""
Minimal HTTP inference API for teaching deployment patterns.

Run locally:
  uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

Model path defaults to models/better_pipeline.joblib; override with env MODEL_PATH
(e.g. models/advanced_xgb_pipeline.joblib after training that tier).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import MODELS_DIR
from src.inference import load_model, prepare_features_for_model, predict_scores

_DEFAULT_MODEL = MODELS_DIR / "better_pipeline.joblib"


class PredictRequest(BaseModel):
    """One review row; field names match the training CSV where possible."""

    model_config = ConfigDict(populate_by_name=True)

    Title: str = ""
    Review_Text: str = Field(default="", alias="Review Text")
    Age: float = 0.0
    Positive_Feedback_Count: int = Field(default=0, alias="Positive Feedback Count")
    Rating: float = 0.0
    Division_Name: str = Field(default="Unknown", alias="Division Name")
    Department_Name: str = Field(default="Unknown", alias="Department Name")
    Class_Name: str = Field(default="Unknown", alias="Class Name")


class PredictResponse(BaseModel):
    prediction: int
    score: float


def _model_path_from_env() -> Path:
    raw = os.environ.get("MODEL_PATH", str(_DEFAULT_MODEL))
    return Path(raw)


_model_cache: dict[str, object] = {}


def get_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = _model_path_from_env()
        _model_cache["path"] = str(path)
        _model_cache["model"] = load_model(path)
        yield

    app = FastAPI(title="Womens Reviews Inference", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_path": _model_cache.get("path"),
            "model_loaded": "model" in _model_cache and _model_cache["model"] is not None,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest) -> PredictResponse:
        model = _model_cache.get("model")
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        row = body.model_dump(by_alias=True)
        df = pd.DataFrame([row])
        feat = prepare_features_for_model(df)
        pred_series, score_series = predict_scores(model, feat)
        return PredictResponse(
            prediction=int(pred_series.iloc[0]),
            score=float(score_series.iloc[0]),
        )

    return app


app = get_app()
