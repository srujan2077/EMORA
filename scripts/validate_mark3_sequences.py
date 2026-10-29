
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "processed" / "mark2" / "features"

# Records are grouped separately by split and video.
records = defaultdict(list)
duplicate_ids = Counter()
loaded_segments = Counter()
errors = []

for path in sorted(FEATURE_DIR.glob("*.pt")):
    split = path.name.split("_shard_")[0]

    try:
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        n = len(data["segment_id"])
        loaded_segments[split] += n

        for i in range(n):
            video_id = str(data["video_id"][i])
            segment_id = str(data["segment_id"][i])
            segment_number = int(data["segment_number"][i])
            labels = tuple(
                float(x)
                for x in data["labels"][i].tolist()
            )

            key = (split, video_id)
            records[key].append({
                "segment_id": segment_id,
                "segment_number": segment_number,
                "labels": labels,
            })

    except Exception as exc:
        errors.append((path.name, str(exc)))

print("=" * 68)
print("EMORA MARK 3 — SEQUENCE INTEGRITY")
print("=" * 68)

split_stats = defaultdict(Counter)

for (split, video_id), items in records.items():
    numbers = [item["segment_number"] for item in items]
    ids = [item["segment_id"] for item in items]

    split_stats[split]["videos"] += 1

    if len(set(ids)) != len(ids):
        split_stats[split]["duplicate_segment_ids"] += 1

    if len(set(numbers)) != len(numbers):
        split_stats[split]["duplicate_segment_numbers"] += 1

    ordered = sorted(set(numbers))

    if len(ordered) >= 2:
        split_stats[split]["videos_with_2plus"] += 1

    if len(ordered) >= 5:
        split_stats[split]["videos_with_5plus"] += 1

    if ordered and ordered != list(range(ordered[0], ordered[-1] + 1)):
        split_stats[split]["videos_with_number_gaps"] += 1

    # Labels may legitimately vary across segments.
    # This checks only whether each segment ID maps to one label vector.
    labels_by_id = defaultdict(set)

    for item in items:
        labels_by_id[item["segment_id"]].add(item["labels"])

    if any(len(label_set) > 1 for label_set in labels_by_id.values()):
        split_stats[split]["segment_ids_with_conflicting_labels"] += 1

for split in sorted(loaded_segments):
    stats = split_stats[split]

    print(f"\nSplit: {split}")
    print(f"Segments read: {loaded_segments[split]}")
    print(f"Video groups: {stats['videos']}")
    print(f"Videos with 2+ segments: {stats['videos_with_2plus']}")
    print(f"Videos with 5+ segments: {stats['videos_with_5plus']}")
    print(f"Videos with duplicate segment IDs: "
          f"{stats['duplicate_segment_ids']}")
    print(f"Videos with duplicate segment numbers: "
          f"{stats['duplicate_segment_numbers']}")
    print(f"Videos with gaps in segment numbering: "
          f"{stats['videos_with_number_gaps']}")
    print(f"Videos with conflicting labels for a segment ID: "
          f"{stats['segment_ids_with_conflicting_labels']}")

print("\n" + "-" * 68)

if errors:
    print(f"Shard loading errors: {len(errors)}")
    for name, error in errors[:10]:
        print(f"{name}: {error}")
else:
    print("All feature shards loaded successfully.")

print("\nRead-only validation complete.")
