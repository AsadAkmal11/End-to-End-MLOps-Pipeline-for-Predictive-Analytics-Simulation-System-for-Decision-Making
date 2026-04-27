"""
Data preprocessing: load CSV, impute, encode categoricals, scale numerics,
train/test split, and write a single processed dataset CSV.

Usage:
    python src/preprocess.py
    python src/preprocess.py --csv data/Crop_recommendation.csv --target label
    python src/preprocess.py --csv data/crop_yield_dataset.csv --target Crop_Yield --regression --drop-cols Date
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

LOG = logging.getLogger("preprocess")

DEFAULT_CSV = Path("data") / "Crop_recommendation.csv"
OUTPUT_PATH = Path("data") / "processed_data.csv"


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
    root.setLevel(level)


def load_csv(path: Path) -> pd.DataFrame:
    LOG.info("Loading CSV from %s", path.resolve())
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    LOG.info("Loaded shape: %s rows, %s columns", df.shape[0], df.shape[1])
    LOG.info("Columns: %s", list(df.columns))
    return df


def infer_feature_columns(
    df: pd.DataFrame, target_column: str
) -> tuple[list[str], list[str]]:
    if target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column!r} not in columns: {list(df.columns)}"
        )
    feature_df = df.drop(columns=[target_column])
    categorical: list[str] = []
    numerical: list[str] = []
    for col in feature_df.columns:
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            numerical.append(col)
        else:
            categorical.append(col)
    LOG.info(
        "Inferred %d numerical and %d categorical feature columns",
        len(numerical),
        len(categorical),
    )
    LOG.debug("Numerical: %s", numerical)
    LOG.debug("Categorical: %s", categorical)
    return numerical, categorical


def build_feature_preprocessor(
    numerical: Sequence[str], categorical: Sequence[str]
) -> ColumnTransformer:
    transformers: list[tuple] = []
    if numerical:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, list(numerical)))
    if categorical:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, list(categorical)))

    if not transformers:
        raise ValueError("No feature columns left after excluding target.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _transform_to_frame(
    preprocessor: ColumnTransformer, X: pd.DataFrame
) -> pd.DataFrame:
    Xt = preprocessor.transform(X)
    names = preprocessor.get_feature_names_out()
    return pd.DataFrame(Xt, columns=names, index=X.index)


def preprocess_and_save(
    csv_path: Path,
    output_path: Path,
    target_column: str,
    test_size: float,
    random_state: int,
    *,
    regression: bool = False,
    drop_columns: Sequence[str] | None = None,
    max_rows: int | None = None,
) -> Path:
    df = load_csv(csv_path)
    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
        LOG.info("Subsampled to max_rows=%d (shape=%s)", max_rows, df.shape)

    drop_columns = list(drop_columns or [])
    if drop_columns:
        missing = [c for c in drop_columns if c not in df.columns]
        if missing:
            raise ValueError(f"--drop-cols: unknown columns: {missing}")
        df = df.drop(columns=drop_columns)
        LOG.info("Dropped columns: %s", drop_columns)

    numerical, categorical = infer_feature_columns(df, target_column)

    missing_before = df.isna().sum().sum()
    if missing_before:
        LOG.info("Total missing values in raw data: %d", int(missing_before))
    else:
        LOG.info("No missing values in raw data")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify = None
    if not regression and y.nunique() > 1 and not pd.api.types.is_float_dtype(y):
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    LOG.info(
        "Train/test split: train=%d, test=%d (test_size=%.2f)",
        len(X_train),
        len(X_test),
        test_size,
    )

    preprocessor = build_feature_preprocessor(numerical, categorical)
    LOG.info("Fitting imputers, StandardScaler (numeric), OneHotEncoder (categorical) on train set")
    preprocessor.fit(X_train)

    X_train_t = _transform_to_frame(preprocessor, X_train)
    X_test_t = _transform_to_frame(preprocessor, X_test)
    LOG.info(
        "Transformed feature matrix: %d columns (from %d raw features)",
        X_train_t.shape[1],
        X.shape[1],
    )

    train_part = X_train_t.copy()
    train_part["dataset_split"] = "train"
    train_part[target_column] = y_train.values

    test_part = X_test_t.copy()
    test_part["dataset_split"] = "test"
    test_part[target_column] = y_test.values

    if regression:
        LOG.info("Regression target: keeping %s as numeric (no LabelEncoder)", target_column)
    else:
        le = LabelEncoder()
        y_encoded = le.fit_transform(pd.concat([y_train, y_test], axis=0))
        n_train = len(y_train)
        y_train_enc = y_encoded[:n_train]
        y_test_enc = y_encoded[n_train:]
        LOG.info(
            "Target encoded with LabelEncoder (%d classes): %s",
            len(le.classes_),
            list(le.classes_),
        )
        train_part[f"{target_column}_encoded"] = y_train_enc
        test_part[f"{target_column}_encoded"] = y_test_enc

    processed = pd.concat([train_part, test_part], axis=0, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    LOG.info("Saved processed dataset to %s (shape=%s)", output_path.resolve(), processed.shape)
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess CSV and save processed_data.csv")
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Input CSV path (default: {DEFAULT_CSV})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output CSV path (default: {OUTPUT_PATH})",
    )
    p.add_argument(
        "--target",
        type=str,
        default="label",
        help="Name of target column (default: label)",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction for test split (default: 0.2)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)",
    )
    p.add_argument(
        "--regression",
        action="store_true",
        help="Continuous target: no LabelEncoder; no stratified split",
    )
    p.add_argument(
        "--drop-cols",
        type=str,
        default="",
        help="Comma-separated columns to drop before preprocessing (e.g. Date)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If >0, randomly sample this many rows before preprocessing",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    _configure_logging()
    args = parse_args(argv)
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    preprocess_and_save(
        csv_path=args.csv,
        output_path=args.output,
        target_column=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        regression=args.regression,
        drop_columns=drop_cols,
        max_rows=args.max_rows if args.max_rows > 0 else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
