"""Read-only structural audit for the Mark 3 temporal windows."""

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json

import torch


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "processed" / "mark2" / "features"
WINDOW_DIR = ROOT / "data" / "processed" / "mark3_temporal"
SPLITS = ("train", "val", "test")


def load_source_records():
    records = defaultdict(list)
    shard_errors = []

    for path in sorted(FEATURE_DIR.glob("*_shard_*.pt")):
        split = path.name.split("_shard_")[0]
        if split not in SPLITS:
            continue
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            required = {
                "segment_id",
                "video_id",
                "segment_number",
                "labels",
            }
            missing = sorted(required - set(data))
            if missing:
                shard_errors.append(
                    {"file": path.name, "error": f"missing keys: {missing}"}
                )
                continue
            for index in range(len(data["video_id"])):
                labels = tuple(float(value) for value in data["labels"][index])
                records[split].append(
                    {
                        "video_id": str(data["video_id"][index]),
                        "segment_id": str(data["segment_id"][index]),
                        "segment_number": int(data["segment_number"][index]),
                        "labels": labels,
                    }
                )
        except Exception as exc:
            shard_errors.append({"file": path.name, "error": str(exc)})

    return records, shard_errors


def count_source_properties(records):
    result = {}
    video_sets = {}
    for split in SPLITS:
        items = records[split]
        by_video = defaultdict(list)
        by_id = defaultdict(set)
        for item in items:
            by_video[item["video_id"]].append(item)
            by_id[(item["video_id"], item["segment_id"])].add(item["labels"])

        duplicate_ids = []
        duplicate_numbers = []
        conflicting_labels = []
        for video_id, video_items in by_video.items():
            ids = [item["segment_id"] for item in video_items]
            numbers = [item["segment_number"] for item in video_items]
            if len(ids) != len(set(ids)):
                duplicate_ids.append(video_id)
            if len(numbers) != len(set(numbers)):
                duplicate_numbers.append(video_id)
        for key, labels in by_id.items():
            if len(labels) > 1:
                conflicting_labels.append(
                    {"video_id": key[0], "segment_id": key[1]}
                )

        video_sets[split] = set(by_video)
        result[split] = {
            "source_records": len(items),
            "videos": len(by_video),
            "duplicate_segment_id_videos": sorted(duplicate_ids),
            "duplicate_segment_number_videos": sorted(duplicate_numbers),
            "conflicting_label_segment_ids": conflicting_labels,
        }

    overlaps = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlaps[f"{left}_vs_{right}"] = sorted(
                video_sets[left] & video_sets[right]
            )
    result["cross_split_video_overlap"] = overlaps
    return result


def gap_statistics(numbers):
    deltas = [current - previous for previous, current in zip(numbers, numbers[1:])]
    gaps = [delta - 1 for delta in deltas if delta > 1]
    return deltas, gaps


def audit_windows(split, source_records):
    path = WINDOW_DIR / f"{split}_windows.pt"
    windows = torch.load(path, map_location="cpu", weights_only=False)
    source_by_key = defaultdict(list)
    for item in source_records[split]:
        source_by_key[
            (item["video_id"], item["segment_id"], item["segment_number"])
        ].append(item["labels"])

    invalid = Counter()
    gap_size_counts = Counter()
    all_delta_counts = Counter()
    affected_windows = 0
    affected_context_transitions = 0
    total_context_transitions = 0
    label_mismatches = []
    examples = []

    for index, window in enumerate(windows):
        ids = [str(value) for value in window["context_segment_ids"]]
        numbers = [int(value) for value in window["context_segment_numbers"]]
        length = int(window["context_length"])
        video_id = str(window["video_id"])
        target_id = str(window["target_segment_id"])
        target_number = int(window["target_segment_number"])

        if length != len(ids) or length != len(numbers):
            invalid["context_length_mismatch"] += 1
        if not numbers or numbers[-1] != target_number:
            invalid["target_number_not_final"] += 1
        if not ids or ids[-1] != target_id:
            invalid["target_id_not_final"] += 1
        if any(current <= previous for previous, current in zip(numbers, numbers[1:])):
            invalid["context_not_strictly_increasing"] += 1
        if any(number > target_number for number in numbers):
            invalid["future_segment_included"] += 1
        if len(ids) != len(set(ids)):
            invalid["duplicate_context_segment_ids"] += 1
        if len(numbers) != len(set(numbers)):
            invalid["duplicate_context_segment_numbers"] += 1

        deltas, gaps = gap_statistics(numbers)
        for delta in deltas:
            all_delta_counts[str(delta)] += 1
        for gap in gaps:
            gap_size_counts[str(gap)] += 1
        total_context_transitions += len(deltas)
        affected_context_transitions += len(gaps)
        if gaps:
            affected_windows += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "window_index": index,
                        "video_id": video_id,
                        "target_segment_id": target_id,
                        "context_segment_numbers": numbers,
                        "missing_counts_between_observations": gaps,
                    }
                )

        actual_gap = bool(gaps)
        if bool(window["has_numbering_gap"]) != actual_gap:
            invalid["has_numbering_gap_mismatch"] += 1

        labels = tuple(float(value) for value in window["labels"])
        source_labels = source_by_key.get((video_id, target_id, target_number), [])
        if source_labels and any(labels != source for source in source_labels):
            invalid["target_label_mismatch"] += 1
            if len(label_mismatches) < 10:
                label_mismatches.append(
                    {
                        "window_index": index,
                        "video_id": video_id,
                        "target_segment_id": target_id,
                    }
                )
        elif not source_labels:
            invalid["target_not_found_in_source"] += 1

    window_count = len(windows)
    return {
        "window_count": window_count,
        "affected_windows": affected_windows,
        "affected_window_percentage": (
            100 * affected_windows / window_count if window_count else 0
        ),
        "context_transitions": total_context_transitions,
        "transitions_crossing_missing_numbers": affected_context_transitions,
        "gap_size_distribution_missing_segment_count": dict(
            sorted(gap_size_counts.items(), key=lambda item: int(item[0]))
        ),
        "all_consecutive_number_differences": dict(
            sorted(all_delta_counts.items(), key=lambda item: int(item[0]))
        ),
        "stored_gap_flag_count": sum(
            bool(window["has_numbering_gap"]) for window in windows
        ),
        "gap_examples": examples,
        "label_mismatch_examples": label_mismatches,
        "invalid_counts": dict(invalid),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports") / "mark3_temporal_audit_20261026_185644.json",
    )
    args = parser.parse_args()

    records, shard_errors = load_source_records()
    source = count_source_properties(records)
    windows = {
        split: audit_windows(split, records)
        for split in SPLITS
    }

    checks = {
        "source_shards_loaded": not shard_errors,
        "split_video_isolation": not any(
            source["cross_split_video_overlap"].values()
        ),
    }
    for split in SPLITS:
        invalid = windows[split]["invalid_counts"]
        checks[f"{split}_window_integrity"] = not invalid
        checks[f"{split}_gap_flag_agreement"] = (
            invalid.get("has_numbering_gap_mismatch", 0) == 0
        )

    report = {
        "description": "Read-only audit of existing Mark 3 temporal windows",
        "inputs": {
            "feature_directory": str(FEATURE_DIR.relative_to(ROOT)),
            "window_directory": str(WINDOW_DIR.relative_to(ROOT)),
        },
        "source": source,
        "shard_errors": shard_errors,
        "windows": windows,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nReport written to {output}")


if __name__ == "__main__":
    main()
