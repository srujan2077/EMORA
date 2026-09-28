import random


def create_video_splits(
    segment_ids,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42
):
    """
    Split CMU-MOSEI by source video ID.

    Segments from the same video always remain
    in the same split.
    """

    # Get unique source video IDs
    video_ids = sorted(
        set(
            segment_id.rsplit("[", 1)[0]
            for segment_id in segment_ids
        )
    )

    # Reproducible shuffle
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    n = len(video_ids)

    train_end = int(n * train_ratio)
    val_end = int(
        n * (train_ratio + val_ratio)
    )

    train_video_ids = set(
        video_ids[:train_end]
    )

    val_video_ids = set(
        video_ids[train_end:val_end]
    )

    test_video_ids = set(
        video_ids[val_end:]
    )

    # Convert video splits back to segment splits
    train_segments = []
    val_segments = []
    test_segments = []

    for segment_id in segment_ids:

        video_id = segment_id.rsplit(
            "[", 1
        )[0]

        if video_id in train_video_ids:
            train_segments.append(segment_id)

        elif video_id in val_video_ids:
            val_segments.append(segment_id)

        elif video_id in test_video_ids:
            test_segments.append(segment_id)

    return (
        train_segments,
        val_segments,
        test_segments
    )