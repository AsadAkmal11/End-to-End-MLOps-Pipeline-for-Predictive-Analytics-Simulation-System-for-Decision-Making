"""
DeepChecks data validation script for crop/agri dataset integrity.

This script is intended for CI execution before pytest.
"""

from __future__ import annotations

from pathlib import Path
import sys

if sys.version_info >= (3, 13):
    print("DeepChecks validation skipped on Python >= 3.13 (CI runs on 3.10).")
    raise SystemExit(0)

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks.data_integrity import DataIntegrity


def run_data_integrity_check(csv_path: Path) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    dataset = Dataset(df)
    result = DataIntegrity().run(dataset)

    # Fail CI if any check condition fails.
    if not result.passed_conditions():
        details = result.to_json()
        raise RuntimeError(f"DeepChecks DataIntegrity failed conditions: {details}")

    print("DeepChecks DataIntegrity passed.")


if __name__ == "__main__":
    run_data_integrity_check(Path("data") / "Crop_recommendation.csv")
