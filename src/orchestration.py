"""
Prefect orchestration for crop/agri model training.

This flow wraps existing training logic from:
  - src/train_recommender.py
  - src/cluster_kmeans.py

Artifacts are written under models/ by default.
"""

from __future__ import annotations

from pathlib import Path

from prefect import flow, task

from src.cluster_kmeans import (
    DEFAULT_FEATURES,
    fit_kmeans,
    load_features,
    plot_clusters,
    save_artifact,
)
from src.train_recommender import train_and_save


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
    save_artifact(model, feature_cols, Path(model_output))
    plot_clusters(X, labels, Path(plot_output))
    return model_output


@flow(name="crop_agri_training_flow")
def crop_agri_training_flow(
    csv_path: str = "data/Crop_recommendation.csv",
    recommender_output: str = "models/crop_recommender.pkl",
    clustering_output: str = "models/clustering.pkl",
    clustering_plot_output: str = "models/clusters.png",
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
    return {
        "crop_recommender_model": recommender_model,
        "clustering_model": cluster_model,
        "clustering_plot": clustering_plot_output,
    }


if __name__ == "__main__":
    crop_agri_training_flow()
