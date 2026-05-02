"""
Train a crop-type classifier on Crop_recommendation.csv.

Features: temperature, rainfall, humidity, region_type (derived)
Target: label (22 crop classes)
Model: sklearn Pipeline -> OneHotEncoder + StandardScaler + RandomForestClassifier

Usage:
    python src/train_crop_classifier.py

Output:
    models/classifier.pkl
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data") / "Crop_recommendation.csv"
MODEL_PATH = Path("models") / "classifier.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def _configure_logging() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
        )
    root.setLevel(logging.INFO)


LOG = logging.getLogger("train_crop_classifier")


def derive_region_type(rainfall: pd.Series) -> pd.Series:
    """Map rainfall (mm) to climate zone: arid / temperate / tropical."""
    conditions = [rainfall < 250, (rainfall >= 250) & (rainfall <= 800), rainfall > 800]
    choices = ["arid", "temperate", "tropical"]
    return pd.Series(np.select(conditions, choices, default="temperate"), index=rainfall.index)


def load_and_prepare(path: Path) -> pd.DataFrame:
    LOG.info("Loading dataset from %s", path.resolve())
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "label" in df.columns and "crop_type" not in df.columns:
        df = df.rename(columns={"label": "crop_type"})

    before = len(df)
    df = df.dropna(subset=["temperature", "rainfall", "humidity", "crop_type"])
    LOG.info("Dropped %d null rows, %d remaining", before - len(df), len(df))

    df["region_type"] = derive_region_type(df["rainfall"])

    n_crops = df["crop_type"].nunique()
    LOG.info("Unique crop types (%d): %s", n_crops, sorted(df["crop_type"].unique()))
    if n_crops < 5:
        raise ValueError(f"Too few crop types ({n_crops}). Need at least 5.")

    LOG.info("Region distribution:\n%s", df["region_type"].value_counts().to_string())
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["temperature", "rainfall", "humidity"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["region_type"]),
        ],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def train_and_save(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> Path:
    df = load_and_prepare(data_path)

    feature_cols = ["temperature", "rainfall", "humidity", "region_type"]
    X = df[feature_cols]
    y = df["crop_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    LOG.info("Train: %d | Test: %d", len(X_train), len(X_test))

    pipeline = build_pipeline()
    LOG.info("Fitting RandomForest(n_estimators=300, class_weight='balanced') ...")
    pipeline.fit(X_train, y_train)

    y_pred_test = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
    train_acc = accuracy_score(y_train, y_pred_train)
    train_f1 = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)

    LOG.info("Train acc=%.4f F1(w)=%.4f", train_acc, train_f1)
    LOG.info("Test  acc=%.4f F1(w)=%.4f", test_acc, test_f1)
    LOG.info("Report:\n%s", classification_report(y_test, y_pred_test, zero_division=0))

    metrics = {
        "train_accuracy": float(train_acc),
        "train_f1_weighted": float(train_f1),
        "test_accuracy": float(test_acc),
        "test_f1_weighted": float(test_f1),
        "holdout_accuracy": float(test_acc),
        "holdout_f1_weighted": float(test_f1),
        "n_classes": int(y.nunique()),
    }

    artifact = {
        "model": pipeline,
        "model_name": "RandomForestClassifier",
        "feature_columns": feature_cols,
        "class_labels": sorted(y.unique().tolist()),
        "metrics": metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    LOG.info("Saved to %s (%.1f MB)", model_path.resolve(), model_path.stat().st_size / 1024 / 1024)
    return model_path


if __name__ == "__main__":
    _configure_logging()
    try:
        train_and_save()
        sys.exit(0)
    except Exception as exc:
        LOG.exception("Training failed: %s", exc)
        sys.exit(1)
