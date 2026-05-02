"""
KMeans clustering on agro features (rainfall, temperature, soil nutrients).

Loads a CSV, selects the requested columns, imputes missing values, scales
features, fits KMeans, saves model artifact, and visualizes clusters via PCA.

Defaults match data/Crop_recommendation.csv which contains:
N, P, K, temperature, rainfall, ...

Usage:
    python src/cluster_kmeans.py
    python src/cluster_kmeans.py --csv data/Crop_recommendation.csv --k 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("cluster_kmeans")

DEFAULT_CSV = Path("data") / "Crop_recommendation.csv"
DEFAULT_MODEL_PATH = Path("models") / "clustering.pkl"
DEFAULT_PLOT_PATH = Path("models") / "clusters.png"

DEFAULT_FEATURES = ("rainfall", "temperature", "N", "P", "K")


def _configure_logging(level: int = logging.INFO) -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def load_features(csv_path: Path, feature_cols: tuple[str, ...]) -> pd.DataFrame:
    LOG.info("Loading dataset from %s", csv_path.resolve())
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. Available: {list(df.columns)}"
        )

    X = df.loc[:, list(feature_cols)].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    LOG.info("Selected features: %s", list(X.columns))
    LOG.info("Raw feature shape: %s rows, %s cols", X.shape[0], X.shape[1])
    LOG.info("Total missing values in selected features: %d", int(X.isna().sum().sum()))
    return X


def fit_kmeans(X: pd.DataFrame, k: int, random_state: int) -> tuple[Pipeline, pd.Series]:
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init="auto",
                ),
            ),
        ]
    )
    LOG.info("Fitting KMeans (k=%d)", k)
    labels = pipe.fit_predict(X)
    LOG.info("Fit complete. Inertia=%.6f", float(pipe.named_steps["kmeans"].inertia_))
    return pipe, pd.Series(labels, index=X.index, name="cluster")


def _print_evaluation_matrix(rows: list[dict[str, float | str]]) -> None:
    print("\n## Evaluation Matrix")
    print("| Model | Accuracy | F1-Score | RMSE |")
    print("|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['model']} | {float(row['accuracy']):.4f} | {float(row['f1_score']):.4f} | {float(row['rmse']):.4f} |"
        )


def evaluate_clusters(X: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    """
    Create pseudo-labels from rainfall tertiles and score cluster alignment.
    """
    rain = pd.to_numeric(X["rainfall"], errors="coerce").fillna(X["rainfall"].median())
    bins = np.quantile(rain, [0.33, 0.66])
    pseudo = pd.Series(
        np.where(rain <= bins[0], 0, np.where(rain >= bins[1], 2, 1)),
        index=X.index,
        name="pseudo_label",
    )
    map_cluster_to_pseudo: dict[int, int] = {}
    df = pd.DataFrame({"cluster": labels, "pseudo": pseudo})
    for cid, grp in df.groupby("cluster"):
        map_cluster_to_pseudo[int(cid)] = int(grp["pseudo"].mode().iloc[0])
    pred = labels.map(lambda x: map_cluster_to_pseudo[int(x)])
    acc = float(accuracy_score(pseudo, pred))
    f1 = float(f1_score(pseudo, pred, average="weighted", zero_division=0))
    rmse = float(mean_squared_error(pseudo, pred) ** 0.5)
    _print_evaluation_matrix(
        [{"model": "kmeans_cluster_alignment", "accuracy": acc, "f1_score": f1, "rmse": rmse}]
    )
    return {"accuracy": acc, "f1_score": f1, "rmse": rmse}


def plot_clusters(X: pd.DataFrame, labels: pd.Series, out_path: Path) -> None:
    # Visualize in 2D via PCA on the scaled+imputed matrix for stable axes.
    imputer: SimpleImputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)
    pca = PCA(n_components=2, random_state=0)
    X_2d = pca.fit_transform(X_scaled)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels.to_numpy(),
        cmap="tab10",
        s=18,
        alpha=0.85,
    )
    plt.title("KMeans Clusters (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    LOG.info("Saved cluster plot to %s", out_path.resolve())


def save_artifact(
    model: Pipeline,
    feature_cols: tuple[str, ...],
    model_path: Path,
    metrics: dict[str, float] | None = None,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "model_name": "KMeans",
        "feature_columns": list(feature_cols),
        "metrics": metrics or {},
    }
    joblib.dump(artifact, model_path)
    LOG.info("Saved clustering artifact to %s", model_path.resolve())


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KMeans clustering on agro features")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path")
    p.add_argument("--k", type=int, default=3, help="Number of clusters (default: 3)")
    p.add_argument(
        "--features",
        type=str,
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature columns (default: rainfall,temperature,N,P,K)",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="Model output path")
    p.add_argument("--plot-out", type=Path, default=DEFAULT_PLOT_PATH, help="Plot output path")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    _configure_logging()
    args = parse_args(argv)
    feature_cols = tuple([c.strip() for c in args.features.split(",") if c.strip()])
    X = load_features(args.csv, feature_cols)
    model, labels = fit_kmeans(X, k=args.k, random_state=args.random_state)
    metrics = evaluate_clusters(X, labels)
    save_artifact(model, feature_cols, args.output, metrics=metrics)
    plot_clusters(X, labels, args.plot_out)
    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

