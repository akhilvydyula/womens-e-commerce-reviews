from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    PROCESSED_DIR,
    RAW_FILE_PATH,
    TARGET_COLUMN,
    TEXT_COLUMNS,
    ensure_dirs,
)


def build_data_quality_report(df: pd.DataFrame) -> Dict[str, object]:
    """Build a simple data-quality report for pipeline gating."""
    required_columns = set(TEXT_COLUMNS + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN])
    present_columns = set(df.columns)
    missing_required = sorted(required_columns - present_columns)

    null_counts = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
    top_null_columns = dict(sorted(null_counts.items(), key=lambda kv: kv[1], reverse=True)[:10])

    report: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_required_columns": missing_required,
        "top_10_null_counts": top_null_columns,
    }

    if TARGET_COLUMN in df.columns:
        target_dist = df[TARGET_COLUMN].value_counts(dropna=False, normalize=True).to_dict()
        report["target_distribution"] = {str(k): float(v) for k, v in target_dist.items()}
    else:
        report["target_distribution"] = {}

    return report


def validate_data_gate(report: Dict[str, object], min_rows: int = 1000) -> List[str]:
    """Return blocking issues for training. Empty list means pass."""
    errors: List[str] = []

    if int(report["row_count"]) < min_rows:
        errors.append(f"Row count too low: {report['row_count']} < {min_rows}")

    missing_required = report.get("missing_required_columns", [])
    if missing_required:
        errors.append(f"Missing required columns: {missing_required}")

    target_distribution = report.get("target_distribution", {})
    if not target_distribution:
        errors.append(f"Target column '{TARGET_COLUMN}' missing or empty.")

    return errors


def run_validation(input_csv: Path = RAW_FILE_PATH, min_rows: int = 1000) -> Path:
    """Read a CSV, produce quality report JSON, and fail fast for gate issues."""
    ensure_dirs()
    df = pd.read_csv(input_csv)
    report = build_data_quality_report(df)
    report["input_csv"] = str(input_csv)
    report["gate_min_rows"] = min_rows
    report["gate_errors"] = validate_data_gate(report, min_rows=min_rows)
    report["gate_passed"] = len(report["gate_errors"]) == 0

    out_dir = PROCESSED_DIR / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=RAW_FILE_PATH, help="Path to input CSV.")
    parser.add_argument("--min-rows", type=int, default=1000, help="Minimum row-count gate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = run_validation(input_csv=args.input_csv, min_rows=args.min_rows)
    print(f"Data quality report written to: {path}")
