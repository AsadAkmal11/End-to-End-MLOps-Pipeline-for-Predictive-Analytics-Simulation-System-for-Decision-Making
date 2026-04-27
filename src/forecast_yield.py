"""
Time-series forecasting for crop yield using ARIMA.

This script expects historical data with a date column and a numeric yield
column. It aggregates to monthly average yield, trains ARIMA, forecasts the
next 12 months, saves model artifact, and plots historical + forecast values.

Usage:
    python src/forecast_yield.py
    python src/forecast_yield.py --csv data/crop_yield_dataset.csv --date-col Date --target-col Crop_Yield
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

LOG = logging.getLogger("forecast_yield")

DEFAULT_CSV = Path("data") / "crop_yield_dataset.csv"
DEFAULT_MODEL_PATH = Path("models") / "time_series.pkl"
DEFAULT_PLOT_PATH = Path("models") / "time_series_forecast.png"


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


def load_monthly_series(
    csv_path: Path, date_col: str, target_col: str
) -> pd.Series:
    LOG.info("Loading historical data from %s", csv_path.resolve())
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=[date_col, target_col])
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns {date_col!r} and {target_col!r}."
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[date_col, target_col])
    LOG.info("Dropped %d invalid rows", before - len(df))

    df = df.sort_values(date_col)
    monthly = (
        df.set_index(date_col)[target_col]
        .resample("MS")
        .mean()
        .asfreq("MS")
        .interpolate(method="linear")
    )
    if monthly.empty:
        raise ValueError("No valid monthly series could be generated.")

    LOG.info(
        "Prepared monthly series with %d points (%s -> %s)",
        len(monthly),
        monthly.index.min().date(),
        monthly.index.max().date(),
    )
    return monthly


def fit_forecast(
    series: pd.Series, order: tuple[int, int, int], periods: int
) -> tuple[object, pd.Series]:
    LOG.info("Training ARIMA model with order=%s", order)
    model = ARIMA(series, order=order)
    fitted = model.fit()
    LOG.info("ARIMA fit complete (AIC=%.4f)", float(fitted.aic))

    forecast = fitted.forecast(steps=periods)
    LOG.info("Generated forecast for next %d months", periods)
    return fitted, forecast


def save_artifact(
    fitted_model: object,
    series: pd.Series,
    forecast: pd.Series,
    model_path: Path,
    order: tuple[int, int, int],
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": fitted_model,
        "model_name": "ARIMA",
        "order": order,
        "history": series,
        "forecast": forecast,
    }
    joblib.dump(artifact, model_path)
    LOG.info("Saved model artifact to %s", model_path.resolve())


def plot_forecast(series: pd.Series, forecast: pd.Series, plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, label="Historical Yield", color="tab:blue")
    plt.plot(
        forecast.index,
        forecast.values,
        label="12-Month Forecast",
        color="tab:orange",
        linestyle="--",
    )
    plt.title("Crop Yield Forecast (ARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Yield")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    LOG.info("Saved forecast plot to %s", plot_path.resolve())


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARIMA time-series yield forecast")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Input CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="Date",
        help="Date column name (default: Date)",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="Crop_Yield",
        help="Yield target column name (default: Crop_Yield)",
    )
    parser.add_argument(
        "--order",
        type=int,
        nargs=3,
        default=(1, 1, 1),
        metavar=("P", "D", "Q"),
        help="ARIMA order p d q (default: 1 1 1)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=12,
        help="Forecast horizon in months (default: 12)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Output model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help=f"Output plot path (default: {DEFAULT_PLOT_PATH})",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    _configure_logging()
    args = parse_args(argv)

    order = tuple(args.order)
    series = load_monthly_series(args.csv, args.date_col, args.target_col)
    fitted, forecast = fit_forecast(series, order=order, periods=args.periods)
    save_artifact(fitted, series, forecast, args.model_out, order=order)
    plot_forecast(series, forecast, args.plot_out)

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
