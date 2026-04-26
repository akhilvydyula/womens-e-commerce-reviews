"""
Append-only audit log for data/ML pipeline runs (JSON Lines).

Use for debugging ("what path did we read?"), compliance hints, and teaching
production patterns. This is not a full enterprise audit system — it is a
minimal, file-based pattern students can extend (SIEM, cloud logging, etc.).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import AUDIT_LOG_DIR, ensure_dirs


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_audit_event(
    event: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
    component: str = "pipeline",
) -> Path:
    """
    Append one JSON object as a single line to today's audit file.

    Each line is self-contained JSON so you can `grep` / stream-parse in tools.
    """
    ensure_dirs()
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "ts_utc": _utc_now_iso(),
        "component": component,
        "run_id": run_id,
        "env": os.environ.get("DEPLOY_ENV", "local"),
        **event,
    }
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = AUDIT_LOG_DIR / f"audit_{day}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    return path
