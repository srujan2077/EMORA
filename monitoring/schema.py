"""Privacy-preserving monitoring record contract."""

from __future__ import annotations

from typing import Any

import pandas as pd

EMOTIONS = ("happy", "sad", "anger", "surprise", "disgust", "fear")
REQUIRED_FIELDS = {
    "timestamp",
    "text_available",
    "audio_available",
    "video_available",
    "sequence_length",
    "has_numbering_gap",
    *(f"probability_{emotion}" for emotion in EMOTIONS),
}


def validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("Monitoring record collection must not be empty.")
    for index, record in enumerate(records):
        missing = sorted(REQUIRED_FIELDS - record.keys())
        if missing:
            raise ValueError(f"Record {index} is missing fields: {missing}")
        if not isinstance(record["sequence_length"], int) or not 1 <= record["sequence_length"] <= 8:
            raise ValueError(f"Record {index} has invalid sequence_length.")
        if not isinstance(record["has_numbering_gap"], bool):
            raise ValueError(f"Record {index} has invalid has_numbering_gap.")
        for field in REQUIRED_FIELDS - {"timestamp", "sequence_length", "has_numbering_gap"}:
            if field.endswith("_available") and not isinstance(record[field], bool):
                raise ValueError(f"Record {index} has invalid {field}.")
        for emotion in EMOTIONS:
            value = record[f"probability_{emotion}"]
            if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                raise ValueError(f"Record {index} has invalid {emotion} probability.")


def to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    validate_records(records)
    return pd.DataFrame(records)
