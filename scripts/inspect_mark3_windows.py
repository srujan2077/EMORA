
from pathlib import Path
from collections import Counter

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "mark3_temporal"

EXPECTED = {
    "train": 5004,
    "val": 973,
    "test": 884,
}

for split, expected_count in EXPECTED.items():
    path = DATA_DIR / f"{split}_windows.pt"

    print(f"\n{'=' * 60}")
    print(f"{split.upper()} WINDOWS")
    print("=" * 60)

    if not path.is_file():
        print(f"MISSING: {path}")
        continue

    windows = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    print(f"Window count: {len(windows)} (expected {expected_count})")

    if len(windows) != expected_count:
        print("WARNING: unexpected window count")

    context_counts = Counter()
    gap_count = 0
    invalid = []

    for i, item in enumerate(windows):
        length = item["context_length"]
        context_counts[length] += 1

        if item["has_numbering_gap"]:
            gap_count += 1

        if not 1 <= length <= 8:
            invalid.append((i, "invalid context length"))
            continue

        for key in [
            "text_embedding",
            "audio_embedding",
            "video_embedding",
        ]:
            tensor = item[key]

            if tensor.shape != (length, 768):
                invalid.append((i, f"{key} shape={tuple(tensor.shape)}"))

            if not torch.isfinite(tensor).all():
                invalid.append((i, f"{key} has non-finite values"))

        if item["labels"].shape != (6,):
            invalid.append((i, "labels shape is not (6,)"))

        numbers = item["context_segment_numbers"]

        if len(numbers) != length:
            invalid.append((i, "segment-number count mismatch"))

        if numbers and numbers[-1] != item["target_segment_number"]:
            invalid.append((i, "target is not final context segment"))

        if len(numbers) > 1 and numbers != sorted(numbers):
            invalid.append((i, "context not ordered by segment number"))

        if len(set(item["context_segment_ids"])) != length:
            invalid.append((i, "duplicate ID inside window"))

    print("Context-length distribution:", dict(sorted(context_counts.items())))
    print(f"Windows flagged with numbering gaps: {gap_count}")
    print(f"Invalid windows: {len(invalid)}")

    for problem in invalid[:10]:
        print("  ", problem)

    if not invalid and len(windows) == expected_count:
        print("PASS: basic window integrity checks")

print("\nInspection complete. No files modified.")
