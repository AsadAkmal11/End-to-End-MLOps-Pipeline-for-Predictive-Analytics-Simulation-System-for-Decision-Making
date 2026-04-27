"""
Crop recommendation system (rule-based + trained model).

Inputs:
    - Soil nutrients: N, P, K
    - Rainfall
    - Temperature

Outputs (JSON/dict):
    - best_crop
    - fertilizer_suggestion
    - diagnostics: model prediction/probability and rule-based candidate

Model artifact:
    models/crop_recommender.pkl (created by src/train_recommender.py)

Usage:
    python src/train_recommender.py
    python src/recommend.py --N 90 --P 42 --K 43 --rainfall 200 --temperature 21
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

LOG = logging.getLogger("recommend")

DEFAULT_MODEL = Path("models") / "crop_recommender.pkl"
CORE_FEATURES = ("N", "P", "K", "rainfall", "temperature")


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


def _load_artifact(model_path: Path) -> dict[str, Any]:
    LOG.info("Loading recommender artifact from %s", model_path.resolve())
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. Run src/train_recommender.py first."
        )
    artifact = joblib.load(model_path)
    for k in ("model", "feature_columns", "rule_stats"):
        if k not in artifact:
            raise ValueError(f"Invalid artifact: missing key {k!r}")
    return artifact


def _as_feature_frame(inputs: dict[str, float], feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in inputs]
    if missing:
        raise ValueError(f"Missing required inputs: {missing}")
    row = {c: float(inputs[c]) for c in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)


def _rule_best_crop(inputs: dict[str, float], centroids: dict[str, dict[str, float]], feature_cols: list[str]) -> tuple[str, float]:
    """
    Pick the closest crop centroid by Euclidean distance in feature space.
    Returns (crop, distance). Lower distance is better.
    """
    x = np.array([float(inputs[c]) for c in feature_cols], dtype=float)
    best_crop = None
    best_dist = math.inf
    for crop, cvec in centroids.items():
        mu = np.array([float(cvec[c]) for c in feature_cols], dtype=float)
        dist = float(np.linalg.norm(x - mu))
        if dist < best_dist:
            best_dist = dist
            best_crop = crop
    if best_crop is None:
        raise ValueError("Rule-based centroids unavailable.")
    return best_crop, best_dist


def _fertilizer_suggestion(inputs: dict[str, float], thresholds: dict[str, dict[str, float]]) -> str:
    """
    Simple, explainable rule set:
      - if a nutrient is below 25th percentile → suggest boosting that nutrient
      - if above 75th percentile → suggest reducing / avoiding that nutrient
    Produces a single short recommendation string.
    """
    actions: list[str] = []
    for nutrient in ("N", "P", "K"):
        if nutrient not in thresholds:
            continue
        val = float(inputs[nutrient])
        low = float(thresholds[nutrient]["low"])
        high = float(thresholds[nutrient]["high"])
        if val < low:
            actions.append(f"increase {nutrient}")
        elif val > high:
            actions.append(f"reduce {nutrient}")

    # Context hints based on rainfall/temperature extremes (very simple)
    rain = float(inputs.get("rainfall", 0.0))
    temp = float(inputs.get("temperature", 0.0))
    if "rainfall" in thresholds:
        if rain < float(thresholds["rainfall"]["low"]):
            actions.append("consider irrigation / moisture retention")
        elif rain > float(thresholds["rainfall"]["high"]):
            actions.append("consider drainage / split fertilizer applications")
    if "temperature" in thresholds:
        if temp < float(thresholds["temperature"]["low"]):
            actions.append("use cold-tolerant practices")
        elif temp > float(thresholds["temperature"]["high"]):
            actions.append("use heat-stress mitigation practices")

    if not actions:
        return "Maintain balanced NPK; apply fertilizer based on soil test and local guidelines."
    return "Fertilizer suggestion: " + "; ".join(actions) + "."


def recommend(
    inputs: dict[str, float],
    *,
    model_path: Path = DEFAULT_MODEL,
    min_model_confidence: float = 0.40,
) -> dict[str, Any]:
    artifact = _load_artifact(model_path)
    model = artifact["model"]
    feature_cols = list(artifact["feature_columns"])
    rule_stats = artifact["rule_stats"]
    centroids = rule_stats.get("centroids", {})
    thresholds = rule_stats.get("thresholds", {})

    X = _as_feature_frame(inputs, feature_cols)

    # Model prediction + probability (if available)
    model_pred = model.predict(X)[0]
    model_conf = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = []
        if hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps", {}):
            classes = list(getattr(model.named_steps["clf"], "classes_", []))
        if not classes:
            classes = list(getattr(model, "classes_", []))
        if classes:
            idx = int(np.argmax(proba))
            model_conf = float(proba[idx])
            model_pred = str(classes[idx])
        else:
            model_conf = float(np.max(proba))

    rule_crop, rule_distance = _rule_best_crop(inputs, centroids, feature_cols)

    # Combine: prefer model when confident; otherwise rule-based fallback
    use_rule = model_conf is None or model_conf < float(min_model_confidence)
    best_crop = rule_crop if use_rule else str(model_pred)

    fertilizer = _fertilizer_suggestion(inputs, thresholds)

    response = {
        "best_crop": best_crop,
        "fertilizer_suggestion": fertilizer,
        "inputs": {k: float(inputs[k]) for k in feature_cols},
        "diagnostics": {
            "model_prediction": str(model_pred),
            "model_confidence": model_conf,
            "min_model_confidence": float(min_model_confidence),
            "rule_prediction": str(rule_crop),
            "rule_distance": float(rule_distance),
            "selection": "rule_based" if use_rule else "trained_model",
        },
    }
    return response


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crop recommendation (JSON output)")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model artifact path")
    p.add_argument("--N", type=float, required=True, help="Nitrogen (N)")
    p.add_argument("--P", type=float, required=True, help="Phosphorus (P)")
    p.add_argument("--K", type=float, required=True, help="Potassium (K)")
    p.add_argument("--rainfall", type=float, required=True, help="Rainfall")
    p.add_argument("--temperature", type=float, required=True, help="Temperature")
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.40,
        help="Min model confidence to trust trained model (default: 0.40)",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    _configure_logging()
    args = parse_args(argv)
    payload = {
        "N": args.N,
        "P": args.P,
        "K": args.K,
        "rainfall": args.rainfall,
        "temperature": args.temperature,
    }
    out = recommend(payload, model_path=args.model, min_model_confidence=args.min_confidence)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

