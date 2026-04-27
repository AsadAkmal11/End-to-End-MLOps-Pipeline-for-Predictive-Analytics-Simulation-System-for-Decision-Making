"""
Train a crop recommendation model on Crop_recommendation.csv.

Inputs:
    N, P, K, rainfall, temperature  (and optionally other columns if present)

Target:
    label (crop name)

Outputs:
    models/crop_recommender.pkl  (joblib artifact with model + rule stats)

Usage:
    python src/train_recommender.py
    python src/train_recommender.py --csv data/Crop_recommendation.csv --output models/crop_recommender.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("train_recommender")

DEFAULT_CSV = Path("data") / "Crop_recommendation.csv"
DEFAULT_OUT = Path("models") / "crop_recommender.pkl"

# Required inputs for the user-facing recommender
CORE_FEATURES = ("N", "P", "K", "rainfall", "temperature")
DEFAULT_TARGET = "label"


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


def _load_dataset(csv_path: Path) -> pd.DataFrame:
    LOG.info("Loading dataset from %s", csv_path.resolve())
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    LOG.info("Loaded shape: %s rows, %s columns", df.shape[0], df.shape[1])
    return df


def _compute_rule_stats(df: pd.DataFrame, feature_cols: tuple[str, ...], target_col: str) -> dict:
    """
    Rule-based fallback uses crop centroids (mean feature vector) and nutrient thresholds
    (quantiles across the training set) for fertilizer suggestions.
    """
    stats: dict = {}

    # Nutrient thresholds (low/high) used by fertilizer suggestion rules.
    q = df.loc[:, list(feature_cols)].quantile([0.25, 0.75], numeric_only=True)
    thresholds = {
        col: {"low": float(q.loc[0.25, col]), "high": float(q.loc[0.75, col])}
        for col in feature_cols
        if col in q.columns
    }
    stats["thresholds"] = thresholds

    # Crop centroids for rule-based nearest-centroid recommendation.
    grp = df.groupby(target_col)[list(feature_cols)].mean(numeric_only=True)
    stats["centroids"] = grp.to_dict(orient="index")
    stats["centroid_features"] = list(feature_cols)
    stats["classes"] = sorted(df[target_col].dropna().unique().tolist())
    return stats


def train_and_save(csv_path: Path, output_path: Path, target_col: str) -> Path:
    df = _load_dataset(csv_path)

    missing = [c for c in CORE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col!r}")

    X = df.loc[:, list(CORE_FEATURES)].copy()
    y = df[target_col].astype(str)

    # Coerce to numeric and impute in pipeline
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    LOG.info("Train/test split: train=%d test=%d", len(X_train), len(X_test))

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    LOG.info("Training RandomForestClassifier on %s", list(CORE_FEATURES))
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, pred))
    LOG.info("Holdout accuracy: %.4f", acc)
    LOG.info("Classification report (holdout):\n%s", classification_report(y_test, pred))

    # Rule stats computed on TRAIN only to avoid leakage.
    rule_stats = _compute_rule_stats(
        pd.concat([X_train, y_train.rename(target_col)], axis=1),
        feature_cols=CORE_FEATURES,
        target_col=target_col,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": list(CORE_FEATURES),
        "target_column": target_col,
        "rule_stats": rule_stats,
        "metrics": {"holdout_accuracy": acc},
    }
    joblib.dump(artifact, output_path)
    LOG.info("Saved recommender artifact to %s", output_path.resolve())
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train crop recommender model")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output model path")
    p.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target column name")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    _configure_logging()
    args = parse_args(argv)
    train_and_save(args.csv, args.output, args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

