"""Generate Evidently input/prediction drift reports without raw media."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.schema import EMOTIONS, to_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--current-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "monitoring" / "reports")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Use two explicit non-production records to verify report generation.",
    )
    return parser.parse_args()


def smoke_records(offset: float) -> list[dict[str, Any]]:
    base = {
        "timestamp": "synthetic-smoke",
        "text_available": True,
        "audio_available": True,
        "video_available": True,
        "sequence_length": 1,
        "has_numbering_gap": False,
    }
    return [
        {
            **base,
            "probability_happy": 0.5 + offset,
            "probability_sad": 0.2,
            "probability_anger": 0.2,
            "probability_surprise": 0.1,
            "probability_disgust": 0.1,
            "probability_fear": 0.1,
        },
        {
            **base,
            "probability_happy": 0.4 + offset,
            "probability_sad": 0.3,
            "probability_anger": 0.2,
            "probability_surprise": 0.1,
            "probability_disgust": 0.1,
            "probability_fear": 0.1,
        },
    ]


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of records.")
    return payload


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        reference = smoke_records(0.0)
        current = smoke_records(0.1)
        source = "synthetic smoke records; not production monitoring"
    elif args.reference_json and args.current_json:
        reference = load_records(args.reference_json)
        current = load_records(args.current_json)
        source = "caller-provided aggregate records; raw media is not accepted"
    else:
        raise ValueError(
            "Provide --reference-json and --current-json, or explicitly use --synthetic-smoke."
        )
    reference_df = to_dataframe(reference)
    current_df = to_dataframe(current)
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise RuntimeError(
            "Evidently is required for report generation; install requirements-mlops.txt."
        ) from exc

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=current_df, reference_data=reference_df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(args.output_dir / "evidently_drift_report.html"))
    summary = {
        "source": source,
        "reference_records": len(reference_df),
        "current_records": len(current_df),
        "columns": list(reference_df.columns),
        "emotions": list(EMOTIONS),
        "test_split_accessed": False,
        "accuracy_claim": False,
        "report": "evidently_drift_report.html",
    }
    (args.output_dir / "evidently_drift_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
