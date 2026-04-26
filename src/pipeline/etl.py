"""
Batch ETL: Extract → Validate → Transform → Load (ELT-style landing).

Stages (industry vocabulary for students):
  1. Extract: read raw CSV (source of truth under data/raw/).
  2. Validate: schema/quality gate (reuse validate_data_gate).
  3. Transform: `basic_cleaning` + derived fields used downstream (e.g. text concat in training).
  4. Load: write processed artifact for analytics + modeling (clean_reviews.csv).

Training can keep reading raw, or read `PROCESSED_CLEAN_CSV` via `python -m src.train --data-path ...`.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.config import (
    ETL_MANIFEST_DIR,
    PROCESSED_CLEAN_CSV,
    RAW_FILE_PATH,
    ensure_dirs,
)
from src.data import basic_cleaning, load_raw_data, make_text_feature
from src.pipeline.audit import append_audit_event
from src.pipeline.validate import build_data_quality_report, validate_data_gate


def run_etl(
    raw_path: Path = RAW_FILE_PATH,
    output_path: Path = PROCESSED_CLEAN_CSV,
    *,
    min_rows: int = 1000,
    fail_on_gate: bool = True,
    include_text_column: bool = True,
    run_id: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Run full ETL and write manifest + audit lines.

    Returns:
        (path_to_clean_csv, path_to_manifest_json)
    """
    ensure_dirs()
    rid = run_id or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    append_audit_event(
        {"event": "etl_start", "raw_path": str(raw_path), "output_path": str(output_path)},
        run_id=rid,
        component="etl",
    )

    df_raw = load_raw_data(raw_path)
    report_pre = build_data_quality_report(df_raw)
    gate_errors = validate_data_gate(report_pre, min_rows=min_rows)

    append_audit_event(
        {
            "event": "etl_validate_raw",
            "rows_in": int(len(df_raw)),
            "gate_errors": gate_errors,
            "gate_passed": len(gate_errors) == 0,
        },
        run_id=rid,
        component="etl",
    )

    if fail_on_gate and gate_errors:
        append_audit_event(
            {"event": "etl_aborted", "reason": "validation_gate", "errors": gate_errors},
            run_id=rid,
            component="etl",
        )
        raise RuntimeError(f"ETL validation failed: {gate_errors}")

    df_clean = basic_cleaning(df_raw)
    rows_dropped = int(len(df_raw) - len(df_clean))

    if include_text_column:
        df_clean = df_clean.copy()
        df_clean["text"] = make_text_feature(df_clean)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    elapsed = time.perf_counter() - t0
    report_post = build_data_quality_report(df_clean)

    manifest: Dict[str, Any] = {
        "run_id": rid,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(raw_path.resolve()),
        "output_path": str(output_path.resolve()),
        "rows_raw": int(len(df_raw)),
        "rows_clean": int(len(df_clean)),
        "rows_dropped": rows_dropped,
        "min_rows_gate": min_rows,
        "gate_errors_raw": gate_errors,
        "quality_report_raw": report_pre,
        "quality_report_clean": report_post,
        "elapsed_seconds": round(elapsed, 4),
        "include_text_column": include_text_column,
    }

    ETL_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = ETL_MANIFEST_DIR / f"etl_manifest_{rid}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    append_audit_event(
        {
            "event": "etl_complete",
            "manifest": str(manifest_path),
            "rows_clean": int(len(df_clean)),
            "elapsed_seconds": round(elapsed, 4),
        },
        run_id=rid,
        component="etl",
    )

    return output_path, manifest_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raw → clean ETL batch pipeline.")
    p.add_argument("--raw-path", type=Path, default=RAW_FILE_PATH, help="Source CSV.")
    p.add_argument("--output-path", type=Path, default=PROCESSED_CLEAN_CSV, help="Clean output CSV.")
    p.add_argument("--min-rows", type=int, default=1000, help="Minimum rows for quality gate.")
    p.add_argument(
        "--no-fail-on-gate",
        action="store_true",
        help="Log gate errors but still write output (not for production).",
    )
    p.add_argument(
        "--no-text-column",
        action="store_true",
        help="Omit derived 'text' column from output (smaller file; training adds it anyway).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out, man = run_etl(
        raw_path=args.raw_path,
        output_path=args.output_path,
        min_rows=args.min_rows,
        fail_on_gate=not args.no_fail_on_gate,
        include_text_column=not args.no_text_column,
    )
    print(f"ETL wrote: {out}")
    print(f"Manifest:  {man}")
