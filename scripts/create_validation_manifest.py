"""Create a non-sensitive manifest for approved Mark 3 validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_NAMES = {"test_windows.pt", "test_predictions.npz", "test_labels.npz"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "audit-results" / "validation_artifact_manifest.json",
    )
    args = parser.parse_args()
    paths = [
        ROOT / "data" / "processed" / "mark3_temporal" / "val_windows.pt",
        ROOT / "scripts" / "evaluate_mark3_step3_final.py",
        ROOT / "scripts" / "evaluate_phase2_threshold_audit.py",
        ROOT / "reports" / "mark3_step3_final_20261027_151500.json",
    ]
    records = []
    for path in paths:
        if path.name in TEST_NAMES or "test" in path.name.lower():
            raise ValueError(f"Refusing test-named artifact: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "purpose": "Mark 3 validation-only reproducibility manifest",
        "split": "validation",
        "test_split_accessed": False,
        "artifacts": records,
        "excluded_names": sorted(TEST_NAMES),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
