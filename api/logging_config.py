"""Small structured logging helpers for API request telemetry."""

from __future__ import annotations

import logging


logger = logging.getLogger("emora.api")


def request_finished(
    request_id: str, method: str, path: str, status_code: int, elapsed_ms: float
) -> None:
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%s elapsed_ms=%.2f",
        request_id,
        method,
        path,
        status_code,
        elapsed_ms,
    )
