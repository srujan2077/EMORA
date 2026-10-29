
from collections import Counter, defaultdict
from pathlib import Path
import json

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "mark2" / "features"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "mark3_temporal"

MAX_CONTEXT = 8
SPLITS = ("train", "val", "test")

MODALITIES = (
    "text_embedding",
    "audio_embedding",
    "video_embedding",
)


def load_split(split_name):
    """Load cached embeddings and group them by video."""
    grouped = defaultdict(list)
    shard_paths = sorted(FEATURE_DIR.glob(f"{split_name}_shard_*.pt"))

    if not shard_paths:
        raise FileNotFoundError(
            f"No feature shards found for split '{split_name}' "
            f"in {FEATURE_DIR}"
        )

    for shard_path in shard_paths:
        data = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )

        required = [
            "segment_id",
            "video_id",
            "segment_number",
            "labels",
            *MODALITIES,
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                f"{shard_path.name} is missing keys: {missing}"
            )

        n = len(data["video_id"])

        for i in range(n):
            video_id = str(data["video_id"][i])
            segment_id = str(data["segment_id"][i])
            segment_number = int(data["segment_number"][i])

            embeddings = {}
            for modality in MODALITIES:
                tensor = data[modality][i].detach().cpu().float()
                if tensor.shape != (768,):
                    raise ValueError(
                        f"{shard_path.name}: {modality} for "
                        f"{segment_id} has shape {tuple(tensor.shape)}"
                    )
                if not torch.isfinite(tensor).all():
                    raise ValueError(
                        f"Non-finite values in {modality}, "
                        f"segment {segment_id}"
                    )
                embeddings[modality] = tensor

            labels = data["labels"][i].detach().cpu().float()
            if labels.shape != (6,):
                raise ValueError(
                    f"Labels for {segment_id} have shape "
                    f"{tuple(labels.shape)}"
                )

            if not torch.all((labels == 0) | (labels == 1)):
                raise ValueError(
                    f"Expected binary labels for segment {segment_id}"
                )

            grouped[video_id].append({
                "segment_id": segment_id,
                "segment_number": segment_number,
                "embeddings": embeddings,
                "labels": labels,
            })

    # Validate uniqueness, then sort using the original segment numbers.
    for video_id, items in grouped.items():
        ids = [item["segment_id"] for item in items]
        numbers = [item["segment_number"] for item in items]

        if len(ids) != len(set(ids)):
            raise ValueError(
                f"Duplicate segment ID in {split_name}/{video_id}"
            )

        if len(numbers) != len(set(numbers)):
            raise ValueError(
                f"Duplicate segment number in {split_name}/{video_id}"
            )

        items.sort(key=lambda item: item["segment_number"])

    return grouped


def make_windows(grouped):
    """Create target-centered windows without crossing videos."""
    windows = []
    stats = Counter()

    for video_id, items in grouped.items():
        stats["videos"] += 1
        stats["segments"] += len(items)

        for target_idx, target in enumerate(items):
            # Select up to 8 segments ending at the target, including it.
            # This is causal context: no future segments are used.
            start_idx = max(0, target_idx - MAX_CONTEXT + 1)
            context = items[start_idx:target_idx + 1]

            numbers = [item["segment_number"] for item in context]
            has_gap = any(
                current != previous + 1
                for previous, current in zip(numbers, numbers[1:])
            )

            windows.append({
                "video_id": video_id,
                "target_segment_id": target["segment_id"],
                "target_segment_number": target["segment_number"],
                "context_segment_ids": [
                    item["segment_id"] for item in context
                ],
                "context_segment_numbers": numbers,
                "context_length": len(context),
                "has_numbering_gap": has_gap,
                "text_embedding": torch.stack([
                    item["embeddings"]["text_embedding"]
                    for item in context
                ]),
                "audio_embedding": torch.stack([
                    item["embeddings"]["audio_embedding"]
                    for item in context
                ]),
                "video_embedding": torch.stack([
                    item["embeddings"]["video_embedding"]
                    for item in context
                ]),
                "labels": target["labels"].clone(),
            })

            stats["windows"] += 1
            stats[f"context_{len(context)}"] += 1
            stats["windows_with_gaps"] += int(has_gap)

    return windows, stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "max_context": MAX_CONTEXT,
        "context_policy": (
            "Causal: up to 8 available preceding segments, "
            "including the target; no future segments."
        ),
        "gap_policy": (
            "Original segment numbers are preserved. Gaps are recorded; "
            "no missing segments are fabricated."
        ),
        "splits": {},
    }

    for split_name in SPLITS:
        print(f"\nLoading {split_name}...")
        grouped = load_split(split_name)
        windows, stats = make_windows(grouped)

        output_path = OUTPUT_DIR / f"{split_name}_windows.pt"
        temp_path = output_path.with_suffix(".pt.tmp")

        # Write to a temporary file before replacing this script's own
        # output file. This script never touches Mark 2/Mark 3 checkpoints.
        torch.save(windows, temp_path)
        temp_path.replace(output_path)

        split_summary = {
            "videos": len(grouped),
            "segments": sum(len(items) for items in grouped.values()),
            "windows": len(windows),
            "windows_with_numbering_gaps": stats["windows_with_gaps"],
            "context_length_counts": {
                str(length): stats[f"context_{length}"]
                for length in range(1, MAX_CONTEXT + 1)
            },
            "output_file": str(output_path.relative_to(PROJECT_ROOT)),
        }
        summary["splits"][split_name] = split_summary

        print(f"Videos: {split_summary['videos']}")
        print(f"Segments: {split_summary['segments']}")
        print(f"Target windows: {split_summary['windows']}")
        print(
            "Windows containing numbering gaps: "
            f"{split_summary['windows_with_numbering_gaps']}"
        )
        print(f"Saved: {output_path}")

    summary_path = OUTPUT_DIR / "sequence_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nSummary saved: {summary_path}")
    print("Sequence construction completed.")


if __name__ == "__main__":
    main()
