"""Environment-backed serving configuration with safe local defaults."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _origins() -> list[str]:
    value = os.getenv("EMORA_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    return [origin.strip() for origin in value.split(",") if origin.strip()]


MARK3_CHECKPOINT_ROOT = Path(
    os.getenv(
        "EMORA_MARK3_CHECKPOINT_ROOT",
        str(PROJECT_ROOT / "checkpoints" / "mark3_step3" / "multiseed_20261027"),
    )
)
MARK3_REPORT_PATH = Path(
    os.getenv(
        "EMORA_MARK3_REPORT_PATH",
        str(PROJECT_ROOT / "reports" / "mark3_step3_final_20261027_151500.json"),
    )
)
CORS_ORIGINS = _origins()
MAX_UPLOAD_BYTES = int(os.getenv("EMORA_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_REQUEST_BYTES = int(
    os.getenv("EMORA_MAX_REQUEST_BYTES", str(110 * 1024 * 1024))
)
