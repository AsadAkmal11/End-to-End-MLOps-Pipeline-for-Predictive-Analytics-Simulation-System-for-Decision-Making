from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from statsmodels.tsa.arima.model import ARIMA

from src.recommend import recommend as hybrid_recommend


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    root.setLevel(logging.INFO)


_configure_logging()
LOG = logging.getLogger("ml_api")

app = FastAPI(title="MLOps Pipeline API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path("models")
YIELD_MODEL_PATH = MODELS_DIR / "yield_model.pkl"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "classifier.pkl"
FORECAST_MODEL_PATH = MODELS_DIR / "time_series.pkl"
CLUSTER_MODEL_PATH = MODELS_DIR / "clustering.pkl"
RECOMMENDER_MODEL_PATH = MODELS_DIR / "crop_recommender.pkl"


class FeaturePayload(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Feature dictionary expected by the stored model",
        example={"num__Soil_pH": 6.5, "num__Temperature": 24.1},
    )


class ForecastPoint(BaseModel):
    date: str
    yield_value: float = Field(..., alias="yield")


class ForecastRequest(BaseModel):
    periods: int = Field(12, ge=1, le=60)
    order: tuple[int, int, int] = (1, 1, 1)
    historical_data: list[ForecastPoint] | None = None


class ClusterRequest(BaseModel):
    samples: list[dict[str, float]] = Field(
        ...,
        description="List of samples each containing rainfall, temperature, N, P, K",
        example=[
            {"rainfall": 200.0, "temperature": 22.5, "N": 90.0, "P": 40.0, "K": 43.0}
        ],
    )


class RecommendRequest(BaseModel):
    N: float
    P: float
    K: float
    rainfall: float
    temperature: float
    min_confidence: float = Field(0.40, ge=0.0, le=1.0)


def _load_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj
    return {"model": obj}


@lru_cache(maxsize=1)
def get_yield_artifact() -> dict[str, Any]:
    LOG.info("Loading yield model from %s", YIELD_MODEL_PATH.resolve())
    return _load_artifact(YIELD_MODEL_PATH)


@lru_cache(maxsize=1)
def get_classifier_artifact() -> dict[str, Any]:
    LOG.info("Loading classifier model from %s", CLASSIFIER_MODEL_PATH.resolve())
    return _load_artifact(CLASSIFIER_MODEL_PATH)


@lru_cache(maxsize=1)
def get_forecast_artifact() -> dict[str, Any]:
    LOG.info("Loading time-series model from %s", FORECAST_MODEL_PATH.resolve())
    return _load_artifact(FORECAST_MODEL_PATH)


@lru_cache(maxsize=1)
def get_cluster_artifact() -> dict[str, Any]:
    LOG.info("Loading clustering model from %s", CLUSTER_MODEL_PATH.resolve())
    return _load_artifact(CLUSTER_MODEL_PATH)


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "ML API running"}


@app.post("/predict-yield")
def predict_yield(payload: FeaturePayload) -> dict[str, Any]:
    """
    Predict continuous yield using models/yield_model.pkl artifact.
    """
    try:
        artifact = get_yield_artifact()
        model = artifact["model"]
        X = pd.DataFrame([payload.features])
        pred = float(model.predict(X)[0])
        LOG.info("/predict-yield success")
        return {
            "prediction": pred,
            "model_name": artifact.get("model_name", "unknown"),
            "target_column": artifact.get("target_column"),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        LOG.exception("/predict-yield failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.post("/classify-yield")
def classify_yield(payload: FeaturePayload) -> dict[str, Any]:
    """
    Predict yield category (low / medium / high) using models/classifier.pkl.
    """
    try:
        artifact = get_classifier_artifact()
        model = artifact["model"]
        X = pd.DataFrame([payload.features])
        pred = str(model.predict(X)[0])
        response: dict[str, Any] = {
            "prediction": pred,
            "model_name": artifact.get("model_name", "unknown"),
            "class_labels": artifact.get("class_labels"),
        }
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = list(getattr(model, "classes_", []))
            if classes:
                response["probabilities"] = {
                    str(c): float(p) for c, p in zip(classes, proba)
                }
        LOG.info("/classify-yield success")
        return response
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        LOG.exception("/classify-yield failed")
        raise HTTPException(status_code=400, detail=f"Classification failed: {exc}") from exc


def _series_from_payload(points: list[ForecastPoint]) -> pd.Series:
    df = pd.DataFrame(
        {"date": [p.date for p in points], "yield": [p.yield_value for p in points]}
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    df = df.dropna(subset=["date", "yield"]).sort_values("date")
    if df.empty:
        raise ValueError("No valid historical_data points.")
    monthly = (
        df.set_index("date")["yield"]
        .resample("MS")
        .mean()
        .asfreq("MS")
        .interpolate(method="linear")
    )
    return monthly


@app.post("/forecast")
def forecast(request: ForecastRequest) -> dict[str, Any]:
    """
    Forecast future yield values.
    - If historical_data is provided: fit ARIMA on provided data.
    - Else: load pre-trained artifact from models/time_series.pkl.
    """
    try:
        periods = int(request.periods)
        order = tuple(int(x) for x in request.order)

        if request.historical_data:
            history = _series_from_payload(request.historical_data)
            fitted = ARIMA(history, order=order).fit()
            fc = fitted.forecast(steps=periods)
            source = "request_historical_data"
        else:
            artifact = get_forecast_artifact()
            model = artifact.get("model")
            history = artifact.get("history")
            if model is not None and hasattr(model, "forecast"):
                fc = model.forecast(steps=periods)
            elif history is not None:
                fitted = ARIMA(history, order=order).fit()
                fc = fitted.forecast(steps=periods)
            else:
                raise ValueError("Invalid forecast artifact: missing model/history.")
            source = "saved_model"

        forecast_items = [
            {"date": str(idx.date()), "prediction": float(val)}
            for idx, val in fc.items()
        ]
        LOG.info("/forecast success")
        return {"source": source, "periods": periods, "order": order, "forecast": forecast_items}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        LOG.exception("/forecast failed")
        raise HTTPException(status_code=400, detail=f"Forecast failed: {exc}") from exc


@app.post("/cluster")
def cluster(request: ClusterRequest) -> dict[str, Any]:
    """
    Predict KMeans cluster labels for one or more samples.
    """
    try:
        artifact = get_cluster_artifact()
        model = artifact["model"]
        feature_cols = artifact.get(
            "feature_columns", ["rainfall", "temperature", "N", "P", "K"]
        )
        X = pd.DataFrame(request.samples)
        missing = [c for c in feature_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required clustering fields: {missing}")
        X = X[feature_cols]
        labels = model.predict(X)
        LOG.info("/cluster success for %d samples", len(X))
        return {
            "clusters": [int(x) for x in labels],
            "feature_columns": feature_cols,
            "n_samples": int(len(X)),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        LOG.exception("/cluster failed")
        raise HTTPException(status_code=400, detail=f"Clustering failed: {exc}") from exc


@app.post("/recommend")
def recommend(request: RecommendRequest) -> dict[str, Any]:
    """
    Hybrid recommendation endpoint (trained model + rule-based fallback).
    """
    try:
        payload = {
            "N": float(request.N),
            "P": float(request.P),
            "K": float(request.K),
            "rainfall": float(request.rainfall),
            "temperature": float(request.temperature),
        }
        result = hybrid_recommend(
            payload,
            model_path=RECOMMENDER_MODEL_PATH,
            min_model_confidence=float(request.min_confidence),
        )
        LOG.info("/recommend success")
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        LOG.exception("/recommend failed")
        raise HTTPException(status_code=400, detail=f"Recommendation failed: {exc}") from exc


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
