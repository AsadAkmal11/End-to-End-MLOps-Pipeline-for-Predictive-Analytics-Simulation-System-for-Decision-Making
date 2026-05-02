"""
Prefect orchestration for crop/agri model training.

This flow wraps existing training logic from:
  - src/train_recommender.py
  - src/cluster_kmeans.py

Artifacts are written under models/ by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from prefect import flow, task

try:
    from src.cluster_kmeans import (
        DEFAULT_FEATURES,
        evaluate_clusters,
        fit_kmeans,
        load_features,
        plot_clusters,
        save_artifact,
    )
    from src.train_recommender import train_and_save
    from src.train_rainfall_forecast import train_and_save as train_rainfall_forecast_and_save
except ModuleNotFoundError:
    from cluster_kmeans import (  # type: ignore
        DEFAULT_FEATURES,
        evaluate_clusters,
        fit_kmeans,
        load_features,
        plot_clusters,
        save_artifact,
    )
    from train_recommender import train_and_save  # type: ignore
    from train_rainfall_forecast import train_and_save as train_rainfall_forecast_and_save  # type: ignore

METRICS_HISTORY_PATH = Path("models") / "metrics_history.json"


def _load_artifact_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "metrics": {}, "path": str(path)}
    try:
        artifact = joblib.load(path)
        if isinstance(artifact, dict):
            metrics = artifact.get("metrics", {})
        else:
            metrics = {}
        return {
            "available": True,
            "metrics": metrics if isinstance(metrics, dict) else {"raw": metrics},
            "path": str(path),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "metrics": {}, "path": str(path), "error": str(exc)}


def _append_metrics_history(snapshot: dict[str, Any]) -> None:
    METRICS_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if METRICS_HISTORY_PATH.is_file():
        try:
            history = json.loads(METRICS_HISTORY_PATH.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except Exception:  # noqa: BLE001
            history = []
    history.append(snapshot)
    # Keep recent history bounded.
    history = history[-50:]
    METRICS_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


@task(name="train_recommender_model")
def train_recommender_task(
    csv_path: str = "data/Crop_recommendation.csv",
    output_path: str = "models/crop_recommender.pkl",
    target_col: str = "label",
) -> str:
    out = train_and_save(Path(csv_path), Path(output_path), target_col)
    return str(out)


@task(name="train_kmeans_cluster_model")
def train_cluster_task(
    csv_path: str = "data/Crop_recommendation.csv",
    model_output: str = "models/clustering.pkl",
    plot_output: str = "models/clusters.png",
    k: int = 3,
    random_state: int = 42,
) -> str:
    feature_cols = tuple(DEFAULT_FEATURES)
    X = load_features(Path(csv_path), feature_cols)
    model, labels = fit_kmeans(X, k=k, random_state=random_state)
    metrics = evaluate_clusters(X, labels)
    save_artifact(model, feature_cols, Path(model_output), metrics=metrics)
    plot_clusters(X, labels, Path(plot_output))
    return model_output


@task(name="train_rainfall_forecast_model")
def train_rainfall_forecast_task(
    csv_path: str = "data/Rain_fall_in_Pakistan.csv",
    output_path: str = "models/forecast_model.pkl",
) -> str:
    out = train_rainfall_forecast_and_save(Path(csv_path), Path(output_path))
    return str(out)


@flow(name="crop_agri_training_flow")
def crop_agri_training_flow(
    csv_path: str = "data/Crop_recommendation.csv",
    rainfall_csv_path: str = "data/Rain_fall_in_Pakistan.csv",
    recommender_output: str = "models/crop_recommender.pkl",
    clustering_output: str = "models/clustering.pkl",
    clustering_plot_output: str = "models/clusters.png",
    rainfall_forecast_output: str = "models/forecast_model.pkl",
    target_col: str = "label",
    k: int = 3,
    random_state: int = 42,
) -> dict[str, str]:
    recommender_model = train_recommender_task(
        csv_path=csv_path,
        output_path=recommender_output,
        target_col=target_col,
    )
    cluster_model = train_cluster_task(
        csv_path=csv_path,
        model_output=clustering_output,
        plot_output=clustering_plot_output,
        k=k,
        random_state=random_state,
    )
    rainfall_model = train_rainfall_forecast_task(
        csv_path=rainfall_csv_path,
        output_path=rainfall_forecast_output,
    )
    snapshot = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "artifacts": {
            "yield": _load_artifact_metrics(Path("models") / "yield_model.pkl"),
            "classifier": _load_artifact_metrics(Path("models") / "classifier.pkl"),
            "recommender": _load_artifact_metrics(Path(recommender_output)),
            "clustering": _load_artifact_metrics(Path(clustering_output)),
            "rainfall_forecast": _load_artifact_metrics(Path(rainfall_forecast_output)),
        },
    }
    _append_metrics_history(snapshot)
    return {
        "crop_recommender_model": recommender_model,
        "clustering_model": cluster_model,
        "clustering_plot": clustering_plot_output,
        "rainfall_forecast_model": rainfall_model,
        "metrics_history": str(METRICS_HISTORY_PATH),
    }


if __name__ == "__main__":
    crop_agri_training_flow()
