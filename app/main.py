from __future__ import annotations

import json
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
FORECAST_MODEL_PATH = MODELS_DIR / "forecast_model.pkl"
CLUSTER_MODEL_PATH = MODELS_DIR / "clustering.pkl"
RECOMMENDER_MODEL_PATH = MODELS_DIR / "crop_recommender.pkl"
RAINFALL_DATA_PATH = Path("data") / "Rain_fall_in_Pakistan.csv"
METRICS_HISTORY_PATH = MODELS_DIR / "metrics_history.json"


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
    context_features: dict[str, float] | None = None
    baseline: float | None = Field(
        None,
        description="Optional baseline value to use for a naive forecast when no model/history exists.",
        examples=[3.25],
    )


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


def _predict_yield_value(features: dict[str, float]) -> tuple[float, dict[str, Any]]:
    """
    Return a numeric value for the dashboard.

    Primary: regression artifact at models/yield_model.pkl.
    Fallback: crop recommender model confidence (0..1) as a proxy score when the
    yield regressor feature schema doesn't match the incoming payload.
    """
    artifact = get_yield_artifact()
    model = artifact["model"]
    X = pd.DataFrame([features])
    try:
        pred = float(model.predict(X)[0])
        return pred, {
            "source": "yield_regression_artifact",
            "model_name": artifact.get("model_name", "unknown"),
            "target_column": artifact.get("target_column"),
        }
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Yield regressor prediction failed; falling back. Error=%s", exc)

    payload = {
        "N": float(features.get("N", 0.0)),
        "P": float(features.get("P", 0.0)),
        "K": float(features.get("K", 0.0)),
        "rainfall": float(features.get("rainfall", 0.0)),
        "temperature": float(features.get("temperature", 0.0)),
    }
    rec = hybrid_recommend(payload, model_path=RECOMMENDER_MODEL_PATH, min_model_confidence=0.0)
    conf = rec.get("diagnostics", {}).get("model_confidence")
    score = float(conf) if conf is not None else 0.0
    return score, {
        "source": "recommender_confidence_proxy",
        "note": "yield_model.pkl feature schema mismatch; using recommender confidence as proxy score",
        "best_crop": rec.get("best_crop"),
        "diagnostics": rec.get("diagnostics"),
    }


def _risk_from_score(score: float) -> str:
    if score <= 0.33:
        return "low"
    if score >= 0.66:
        return "high"
    return "medium"


def _artifact_metrics(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        return {"artifact": label, "path": str(path), "available": False, "metrics": {}}
    try:
        artifact = _load_artifact(path)
        metrics = artifact.get("metrics", {})
        return {
            "artifact": label,
            "path": str(path),
            "available": True,
            "metrics": metrics if isinstance(metrics, dict) else {"raw": metrics},
            "updated_at": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "artifact": label,
            "path": str(path),
            "available": False,
            "metrics": {},
            "error": str(exc),
        }


def _metrics_history() -> list[dict[str, Any]]:
    if not METRICS_HISTORY_PATH.is_file():
        return []
    try:
        raw = json.loads(METRICS_HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
    except Exception:  # noqa: BLE001
        return []
    return []


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
    LOG.info("Loading rainfall forecast model from %s", FORECAST_MODEL_PATH.resolve())
    return _load_artifact(FORECAST_MODEL_PATH)


@lru_cache(maxsize=1)
def get_cluster_artifact() -> dict[str, Any]:
    LOG.info("Loading clustering model from %s", CLUSTER_MODEL_PATH.resolve())
    return _load_artifact(CLUSTER_MODEL_PATH)


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "ML API running"}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """
    Return currently available model metrics from saved artifacts.
    """
    payload = {
        "yield": _artifact_metrics(YIELD_MODEL_PATH, "yield"),
        "classifier": _artifact_metrics(CLASSIFIER_MODEL_PATH, "classifier"),
        "recommender": _artifact_metrics(RECOMMENDER_MODEL_PATH, "crop_recommender"),
        "clustering": _artifact_metrics(CLUSTER_MODEL_PATH, "clustering"),
        "rainfall_forecast": _artifact_metrics(FORECAST_MODEL_PATH, "forecast_model"),
    }
    return {
        "message": "Model metrics snapshot",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "artifacts": payload,
    }


@app.get("/metrics/history")
def metrics_history() -> dict[str, Any]:
    history = _metrics_history()
    return {
        "message": "Model metrics history",
        "count": len(history),
        "history": history,
    }


@app.post("/predict-yield")
def predict_yield(payload: FeaturePayload) -> dict[str, Any]:
    """
    Return crop suitability score.

    Note: the score may come from a yield model or a confidence proxy fallback.
    """
    try:
        pred, meta = _predict_yield_value(payload.features)
        LOG.info("/predict-yield success")
        return {
            "prediction": pred,
            "crop_suitability_score": pred,
            **meta,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        LOG.exception("/predict-yield failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.post("/classify-yield")
def classify_yield(payload: FeaturePayload) -> dict[str, Any]:
    """
    Predict agronomic risk level (low / medium / high).

    Preferred: models/classifier.pkl (trained yield-category classifier).
    Fallback: derive category from yield regression prediction using thresholds.
    """
    try:
        X = pd.DataFrame([payload.features])

        # Preferred path: dedicated classifier artifact.
        if CLASSIFIER_MODEL_PATH.is_file():
            artifact = get_classifier_artifact()
            model = artifact["model"]
            pred = str(model.predict(X)[0])
            response: dict[str, Any] = {
                "prediction": pred,
                "risk_level": pred,
                "source": "classifier_artifact",
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
            LOG.info("/classify-yield success (artifact)")
            return response

        # Fallback path: map regression (or proxy score) to a category.
        pred_yield, meta = _predict_yield_value(payload.features)

        # Default thresholds for proxy score (0..1). If you're using a real yield
        # regressor, adjust these or store quantiles in the artifact.
        low_thr = 0.33
        high_thr = 0.66
        threshold_source = "fixed_default"

        if pred_yield <= low_thr:
            label = "low"
        elif pred_yield >= high_thr:
            label = "high"
        else:
            label = "medium"

        LOG.info("/classify-yield success (fallback)")
        return {
            "prediction": label,
            "risk_level": label,
            "source": "yield_regression_fallback",
            "predicted_yield": pred_yield,
            "thresholds": {
                "low": low_thr,
                "high": high_thr,
                "source": threshold_source,
            },
            "details": meta,
        }
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


def _load_rainfall_history() -> pd.Series:
    if not RAINFALL_DATA_PATH.is_file():
        raise FileNotFoundError(f"Rainfall dataset not found: {RAINFALL_DATA_PATH}")
    df = pd.read_csv(RAINFALL_DATA_PATH, skiprows=[1])
    if "date" not in df.columns or "rfh" not in df.columns:
        raise ValueError("Rainfall dataset must include date and rfh columns.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rfh"] = pd.to_numeric(df["rfh"], errors="coerce")
    df = df.dropna(subset=["date", "rfh"])
    if df.empty:
        raise ValueError("Rainfall dataset has no valid rows.")
    monthly = (
        df.groupby("date", as_index=False)["rfh"]
        .mean()
        .set_index("date")["rfh"]
        .resample("MS")
        .mean()
        .asfreq("MS")
        .interpolate(method="linear")
    )
    return monthly


@app.post("/forecast")
def forecast(request: ForecastRequest) -> dict[str, Any]:
    """
    Forecast future rainfall values and optionally derive suitability/risk using
    the first forecasted rainfall value with provided context features.
    """
    try:
        periods = int(request.periods)
        order = tuple(int(x) for x in request.order)

        if request.historical_data:
            history = _series_from_payload(request.historical_data).rename("rainfall")
            fitted = ARIMA(history, order=order).fit()
            fc = fitted.forecast(steps=periods)
            source = "request_historical_data_arima"
        else:
            if FORECAST_MODEL_PATH.is_file():
                artifact = get_forecast_artifact()
                model = artifact.get("model")
                history = artifact.get("history")
                model_type = str(artifact.get("model_type", "")).lower()
                if model is not None and hasattr(model, "forecast"):
                    fc = model.forecast(steps=periods)
                elif model_type == "rolling_mean":
                    window = int(artifact.get("window", 6))
                    base = float(pd.Series(history).tail(window).mean())
                    idx = pd.date_range(
                        start=pd.to_datetime(pd.Series(history).index[-1]) + pd.offsets.MonthBegin(1),
                        periods=periods,
                        freq="MS",
                    )
                    fc = pd.Series([base] * periods, index=idx)
                elif history is not None:
                    fitted = ARIMA(history, order=order).fit()
                    fc = fitted.forecast(steps=periods)
                else:
                    raise ValueError("Invalid forecast artifact: missing model/history.")
                source = "saved_model"
            else:
                history = _load_rainfall_history()
                base = (
                    float(request.baseline)
                    if request.baseline is not None
                    else float(history.tail(6).mean())
                )
                idx = pd.date_range(
                    start=history.index[-1] + pd.offsets.MonthBegin(1),
                    periods=periods,
                    freq="MS",
                )
                fc = pd.Series([base] * periods, index=idx)
                source = "dataset_history_baseline_fallback"

        forecast_items = [
            {"date": str(idx.date()), "prediction": float(val), "rainfall": float(val)}
            for idx, val in fc.items()
        ]
        historical_tail = []
        if "history" in locals() and history is not None:
            h = pd.Series(history).dropna().tail(24)
            historical_tail = [{"date": str(i.date()), "rainfall": float(v)} for i, v in h.items()]

        suitability = None
        if request.context_features and len(forecast_items) > 0:
            ctx = dict(request.context_features)
            ctx["rainfall"] = float(forecast_items[0]["prediction"])
            score, meta = _predict_yield_value(ctx)
            suitability = {
                "crop_suitability_score": float(score),
                "risk_level": _risk_from_score(float(score)),
                "features_used": ctx,
                "source": meta.get("source"),
            }

        LOG.info("/forecast success")
        return {
            "source": source,
            "periods": periods,
            "order": order,
            "historical": historical_tail,
            "forecast": forecast_items,
            "suitability_projection": suitability,
        }
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
