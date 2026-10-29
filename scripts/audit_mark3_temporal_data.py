
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "processed" / "mark2" / "features"

split_counts = Counter()
split_videos = defaultdict(set)
video_segments = defaultdict(set)
bad_shards = []

for path in sorted(FEATURE_DIR.glob("*.pt")):
    split = path.name.split("_shard_")[0]

    try:
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        required = [
            "video_id",
            "segment_number",
            "text_embedding",
            "audio_embedding",
            "video_embedding",
            "labels",
        ]

        missing = [key for key in required if key not in data]
        if missing:
            bad_shards.append((path.name, f"missing keys: {missing}"))
            continue

        count = len(data["video_id"])
        split_counts[split] += count

        for video_id, segment_number in zip(
            data["video_id"],
            data["segment_number"],
        ):
            video_id = str(video_id)
            segment_number = int(segment_number)

            split_videos[split].add(video_id)
            video_segments[(split, video_id)].add(segment_number)

    except Exception as exc:
        bad_shards.append((path.name, str(exc)))

print("=" * 65)
print("EMORA — TEMPORAL DATA AUDIT")
print("=" * 65)

for split in sorted(split_counts):
    videos = split_videos[split]
    lengths = [
        len(video_segments[(split, video_id)])
        for video_id in videos
    ]

    print(f"\nSplit: {split}")
    print(f"Segments: {split_counts[split]}")
    print(f"Unique videos: {len(videos)}")

    if lengths:
        print(f"Segments per video: min={min(lengths)}, "
              f"median={sorted(lengths)[len(lengths)//2]}, "
              f"max={max(lengths)}")

all_lengths = [
    len(segment_set)
    for segment_set in video_segments.values()
]

print("\n" + "-" * 65)
print(f"Video groups across splits: {len(video_segments)}")
print(f"Videos with 2+ distinct segment numbers: "
      f"{sum(length >= 2 for length in all_lengths)}")
print(f"Videos with 5+ distinct segment numbers: "
      f"{sum(length >= 5 for length in all_lengths)}")

if bad_shards:
    print(f"\nProblematic shards: {len(bad_shards)}")
    for name, error in bad_shards[:10]:
        print(f"- {name}: {error}")
else:
    print("\nAll inspected shard files loaded and had required fields.")

print("\nRead-only audit complete. No files were modified.")
