"""
Train rainfall forecasting artifact from Pakistan rainfall history.

Input:
    data/Rain_fall_in_Pakistan.csv

Output:
    models/forecast_model.pkl

The artifact stores:
    - historical monthly rainfall series
    - model type (ARIMA or rolling baseline)
    - holdout metrics
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

LOG = logging.getLogger("train_rainfall_forecast")

DEFAULT_CSV = Path("data") / "Rain_fall_in_Pakistan.csv"
DEFAULT_MODEL_OUT = Path("models") / "forecast_model.pkl"


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


def _print_evaluation_matrix(rows: list[dict[str, Any]]) -> None:
    header = "| Model | Accuracy | F1-Score | RMSE |"
    sep = "|---|---:|---:|---:|"
    print("\n## Evaluation Matrix")
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| {r['model']} | {r['accuracy']:.4f} | {r['f1_score']:.4f} | {r['rmse']:.4f} |"
        )


def _load_monthly_rainfall(csv_path: Path) -> pd.Series:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Rainfall CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, skiprows=[1])
    if "date" not in df.columns or "rfh" not in df.columns:
        raise ValueError("Rainfall CSV must include 'date' and 'rfh' columns.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rfh"] = pd.to_numeric(df["rfh"], errors="coerce")
    df = df.dropna(subset=["date", "rfh"])
    if df.empty:
        raise ValueError("Rainfall CSV has no valid rows after cleaning.")

    # Aggregate all regions into one national monthly signal.
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


def _forecast_with_arima(train: pd.Series, steps: int, order: tuple[int, int, int]) -> tuple[pd.Series, dict[str, Any]]:
    fitted = ARIMA(train, order=order).fit()
    pred = fitted.forecast(steps=steps)
    return pred, {"model_type": "ARIMA", "order": order, "model": fitted}


def _forecast_with_rolling_baseline(train: pd.Series, steps: int, window: int = 6) -> tuple[pd.Series, dict[str, Any]]:
    base = float(train.tail(window).mean())
    idx = pd.date_range(start=train.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    pred = pd.Series([base] * steps, index=idx)
    return pred, {"model_type": "rolling_mean", "window": window, "model": None}


def train_and_save(
    csv_path: Path,
    output_path: Path,
    *,
    order: tuple[int, int, int] = (2, 1, 2),
) -> Path:
    series = _load_monthly_rainfall(csv_path)
    split_idx = int(len(series) * 0.8)
    if split_idx < 24 or len(series) - split_idx < 6:
        raise ValueError("Not enough rainfall history for train/test forecasting split.")

    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    steps = len(test)

    try:
        pred, model_info = _forecast_with_arima(train, steps=steps, order=order)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("ARIMA training failed, using rolling baseline. Error=%s", exc)
        pred, model_info = _forecast_with_rolling_baseline(train, steps=steps, window=6)

    rmse = float(mean_squared_error(test, pred) ** 0.5)
    threshold = float(train.median())
    y_true_bin = (test >= threshold).astype(int)
    y_pred_bin = (pred >= threshold).astype(int)
    acc = float(accuracy_score(y_true_bin, y_pred_bin))
    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

    metrics = {"accuracy": acc, "f1_score": f1, "rmse": rmse}
    _print_evaluation_matrix([{"model": f"rainfall_forecast_{model_info['model_type']}", **metrics}])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model_info["model"],
        "model_type": model_info["model_type"],
        "order": model_info.get("order"),
        "window": model_info.get("window"),
        "history": series,
        "train_history": train,
        "holdout_actual": test,
        "holdout_pred": pred,
        "metrics": metrics,
        "target_column": "rfh",
    }
    joblib.dump(artifact, output_path)
    LOG.info("Saved rainfall forecast artifact to %s", output_path.resolve())
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train rainfall time-series forecast model")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Rainfall CSV path")
    p.add_argument("--output", type=Path, default=DEFAULT_MODEL_OUT, help="Output model artifact path")
    p.add_argument("--order", type=int, nargs=3, default=(2, 1, 2), metavar=("P", "D", "Q"))
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    _configure_logging()
    args = parse_args(argv)
    train_and_save(args.csv, args.output, order=tuple(args.order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
