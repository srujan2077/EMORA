"""
MARK 2 - CMU-MOSEI DATASET INDEX BUILDER

Creates a metadata index for the extracted CMU-MOSEI segments.

Sources:
    1. Extracted Raw/*.mp4 files
    2. Mark 1 aligned labels
    3. Timestamped MOSEI words

Output:
    data/processed/mark2/
        mark2_index.csv
        train.csv
        val.csv
        test.csv

Split:
    70% train
    15% validation
    15% test

IMPORTANT:
    Splitting happens at VIDEO level, not segment level.
    This prevents segments from the same video leaking
    across train/validation/test.
"""

import csv
import random
import re
from pathlib import Path

from mmsdk import mmdatasdk


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cmu_mosei"
)

VIDEO_ROOT = RAW_ROOT / "Raw"

LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aligned_mark1"
    / "labels_aligned.csd"
)

WORDS_PATH = (
    RAW_ROOT
    / "CMU_MOSEI_TimestampedWords.csd"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mark2"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EMOTION_NAMES = [
    "happy",
    "sad",
    "anger",
    "surprise",
    "disgust",
    "fear",
]


# ============================================================
# HELPERS
# ============================================================

def parse_segment_id(segment_id):
    """
    Convert:

        -ZgjBOA1Yhw[4]

    into:

        video_id = -ZgjBOA1Yhw
        segment_number = 4
    """

    match = re.match(
        r"^(.*)\[(\d+)\]$",
        segment_id,
    )

    if match is None:
        return None, None

    video_id = match.group(1)
    segment_number = int(match.group(2))

    return video_id, segment_number


def make_segment_id(video_id, segment_number):
    return f"{video_id}[{segment_number}]"


def load_csd(path):
    """
    Load a CMU-MOSEI computational sequence.

    IMPORTANT:
    The installed package exposes the loader through
    mmdatasdk, not directly through mmsdk.
    """

    print()
    print("Loading:")
    print(path)

    dataset = mmdatasdk.mmdataset(
        {
            "data": str(path),
        }
    )

    return dataset


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels():

    print()
    print("=" * 70)
    print("LOADING LABELS")
    print("=" * 70)

    label_dataset = load_csd(
        LABEL_PATH
    )

    # --------------------------------------------------------
    # Find All Labels sequence
    # --------------------------------------------------------

    if (
        "All Labels"
        in label_dataset.computational_sequences
    ):

        sequence = (
            label_dataset
            .computational_sequences[
                "All Labels"
            ]
        )

    else:

        sequence_name = next(
            iter(
                label_dataset
                .computational_sequences
            )
        )

        print(
            "Using label sequence:",
            sequence_name
        )

        sequence = (
            label_dataset
            .computational_sequences[
                sequence_name
            ]
        )

    labels = {}

    keys = sequence.data.keys()

    print(
        "Label segments:",
        len(keys)
    )

    # --------------------------------------------------------
    # Process every labeled segment
    # --------------------------------------------------------

    for segment_id in keys:

        try:

            data = sequence.data[
                segment_id
            ]

            features = data["features"]

            if features.ndim == 1:
                values = features
            else:
                values = features[0]

            # Expected format:
            #
            # [sentiment,
            #  happy,
            #  sad,
            #  anger,
            #  surprise,
            #  disgust,
            #  fear]

            if len(values) < 7:
                continue

            emotion_values = values[1:7]

            emotion_labels = [
                1 if float(value) > 0 else 0
                for value in emotion_values
            ]

            # ------------------------------------------------
            # Segment timing
            # ------------------------------------------------

            intervals = data["intervals"]

            start_time = float(
                intervals[0][0]
            )

            end_time = float(
                intervals[0][1]
            )

            # ------------------------------------------------
            # Segment ID
            # ------------------------------------------------

            video_id, segment_number = (
                parse_segment_id(
                    segment_id
                )
            )

            if video_id is None:
                continue

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            labels[segment_id] = {

                "video_id": video_id,

                "segment_number":
                    segment_number,

                "start_time":
                    start_time,

                "end_time":
                    end_time,

                "happy":
                    emotion_labels[0],

                "sad":
                    emotion_labels[1],

                "anger":
                    emotion_labels[2],

                "surprise":
                    emotion_labels[3],

                "disgust":
                    emotion_labels[4],

                "fear":
                    emotion_labels[5],
            }

        except Exception as exc:

            print(
                "WARNING: Could not process label:",
                segment_id,
                "|",
                exc,
            )

    print(
        "Usable label segments:",
        len(labels)
    )

    return labels


# ============================================================
# LOAD TIMESTAMPED WORDS
# ============================================================

def load_words():

    print()
    print("=" * 70)
    print("LOADING TIMESTAMPED WORDS")
    print("=" * 70)

    words_dataset = load_csd(
        WORDS_PATH
    )

    # --------------------------------------------------------
    # Find word sequence
    # --------------------------------------------------------

    if (
        "words"
        in words_dataset.computational_sequences
    ):

        sequence = (
            words_dataset
            .computational_sequences[
                "words"
            ]
        )

    else:

        sequence_name = next(
            iter(
                words_dataset
                .computational_sequences
            )
        )

        print(
            "Using word sequence:",
            sequence_name
        )

        sequence = (
            words_dataset
            .computational_sequences[
                sequence_name
            ]
        )

    words = {}

    print(
        "Videos containing transcripts:",
        len(sequence.data)
    )

    # --------------------------------------------------------
    # Process every video
    # --------------------------------------------------------

    for video_id in sequence.data.keys():

        try:

            data = sequence.data[
                video_id
            ]

            features = data["features"]

            intervals = data["intervals"]

            video_words = []

            for index in range(
                len(features)
            ):

                word_value = features[
                    index
                ][0]

                if isinstance(
                    word_value,
                    bytes,
                ):

                    word = (
                        word_value
                        .decode(
                            "utf-8",
                            errors="ignore",
                        )
                    )

                else:

                    word = str(
                        word_value
                    )

                word = word.strip()

                # Ignore empty entries
                if not word:
                    continue

                # Ignore speech marker
                if word.lower() == "sp":
                    continue

                start = float(
                    intervals[index][0]
                )

                end = float(
                    intervals[index][1]
                )

                video_words.append(
                    (
                        start,
                        end,
                        word,
                    )
                )

            words[video_id] = video_words

        except Exception as exc:

            print(
                "WARNING: Could not process words for:",
                video_id,
                "|",
                exc,
            )

    print(
        "Usable transcript videos:",
        len(words)
    )

    return words


# ============================================================
# BUILD TRANSCRIPT
# ============================================================

def build_transcript(
    video_id,
    segment_start,
    segment_end,
    words,
):
    """
    Select words whose timestamps overlap
    the segment interval.

    overlap:

        word_end > segment_start
        AND
        word_start < segment_end
    """

    video_words = words.get(
        video_id,
        [],
    )

    selected = []

    for (
        word_start,
        word_end,
        word,
    ) in video_words:

        if (
            word_end > segment_start
            and
            word_start < segment_end
        ):

            selected.append(
                word
            )

    return " ".join(
        selected
    )


# ============================================================
# DISCOVER MP4 FILES
# ============================================================

def discover_videos():

    print()
    print("=" * 70)
    print("DISCOVERING EXTRACTED MP4 FILES")
    print("=" * 70)

    if not VIDEO_ROOT.exists():

        raise FileNotFoundError(
            f"Video directory does not exist:\n"
            f"{VIDEO_ROOT}"
        )

    mp4_files = list(
        VIDEO_ROOT.rglob("*.mp4")
    )

    print(
        "MP4 files found:",
        len(mp4_files)
    )

    return mp4_files


# ============================================================
# BUILD VIDEO MAP
# ============================================================

def build_video_map(
    mp4_files
):

    print()
    print(
        "Building MP4 segment map..."
    )

    video_map = {}

    for path in mp4_files:

        # Expected:
        #
        # Raw/
        #   video_id/
        #       segment_number.mp4

        segment_number_text = (
            path.stem
        )

        video_id = (
            path.parent.name
        )

        if not segment_number_text.isdigit():
            continue

        segment_number = int(
            segment_number_text
        )

        segment_id = make_segment_id(
            video_id,
            segment_number,
        )

        video_map[
            segment_id
        ] = str(
            path.relative_to(
                PROJECT_ROOT
            )
        )

    print(
        "Valid MP4 segments:",
        len(video_map)
    )

    return video_map


# ============================================================
# BUILD COMPLETE INDEX
# ============================================================

def build_index(
    labels,
    words,
    video_map,
):

    print()
    print("=" * 70)
    print("BUILDING MARK 2 INDEX")
    print("=" * 70)

    rows = []

    matched = 0
    missing_video = 0
    missing_transcript = 0

    for (
        segment_id,
        label_info,
    ) in labels.items():

        # ----------------------------------------------------
        # MP4 must exist
        # ----------------------------------------------------

        if segment_id not in video_map:

            missing_video += 1

            continue

        video_id = label_info[
            "video_id"
        ]

        segment_number = (
            label_info[
                "segment_number"
            ]
        )

        start_time = (
            label_info[
                "start_time"
            ]
        )

        end_time = (
            label_info[
                "end_time"
            ]
        )

        # ----------------------------------------------------
        # Transcript
        # ----------------------------------------------------

        transcript = build_transcript(
            video_id,
            start_time,
            end_time,
            words,
        )

        if not transcript:

            missing_transcript += 1

        # ----------------------------------------------------
        # Complete row
        # ----------------------------------------------------

        row = {

            "segment_id":
                segment_id,

            "video_id":
                video_id,

            "segment_number":
                segment_number,

            "media_path":
                video_map[
                    segment_id
                ],

            "start_time":
                start_time,

            "end_time":
                end_time,

            "text":
                transcript,

            "happy":
                label_info["happy"],

            "sad":
                label_info["sad"],

            "anger":
                label_info["anger"],

            "surprise":
                label_info["surprise"],

            "disgust":
                label_info["disgust"],

            "fear":
                label_info["fear"],
        }

        rows.append(
            row
        )

        matched += 1

    print()
    print(
        "Matched segments:",
        matched
    )

    print(
        "Missing MP4:",
        missing_video
    )

    print(
        "Missing transcript:",
        missing_transcript
    )

    return rows


# ============================================================
# VIDEO-LEVEL SPLIT
# ============================================================

def create_video_split(
    rows,
    seed=42,
):

    print()
    print("=" * 70)
    print("CREATING VIDEO-LEVEL SPLIT")
    print("=" * 70)

    video_ids = sorted(
        {
            row["video_id"]
            for row in rows
        }
    )

    print(
        "Unique videos:",
        len(video_ids)
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        video_ids
    )

    total = len(
        video_ids
    )

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    train_videos = set(
        video_ids[
            :train_count
        ]
    )

    val_videos = set(
        video_ids[
            train_count:
            train_count + val_count
        ]
    )

    test_videos = set(
        video_ids[
            train_count + val_count:
        ]
    )

    print(
        "Train videos:",
        len(train_videos)
    )

    print(
        "Validation videos:",
        len(val_videos)
    )

    print(
        "Test videos:",
        len(test_videos)
    )

    train_rows = []
    val_rows = []
    test_rows = []

    for row in rows:

        video_id = row[
            "video_id"
        ]

        if video_id in train_videos:

            new_row = dict(
                row
            )

            new_row["split"] = (
                "train"
            )

            train_rows.append(
                new_row
            )

        elif video_id in val_videos:

            new_row = dict(
                row
            )

            new_row["split"] = (
                "val"
            )

            val_rows.append(
                new_row
            )

        elif video_id in test_videos:

            new_row = dict(
                row
            )

            new_row["split"] = (
                "test"
            )

            test_rows.append(
                new_row
            )

    print()
    print(
        "Train segments:",
        len(train_rows)
    )

    print(
        "Validation segments:",
        len(val_rows)
    )

    print(
        "Test segments:",
        len(test_rows)
    )

    return (
        train_rows,
        val_rows,
        test_rows,
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    rows,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "segment_id",
        "video_id",
        "segment_number",
        "media_path",
        "start_time",
        "end_time",
        "text",
        "happy",
        "sad",
        "anger",
        "surprise",
        "disgust",
        "fear",
        "split",
    ]

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        "Saved:",
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MARK 2 - CMU-MOSEI INDEX BUILDER")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Video root:")
    print(VIDEO_ROOT)

    print()
    print("Label file:")
    print(LABEL_PATH)

    print()
    print("Word file:")
    print(WORDS_PATH)

    # --------------------------------------------------------
    # Verify paths
    # --------------------------------------------------------

    required_paths = [
        LABEL_PATH,
        WORDS_PATH,
        VIDEO_ROOT,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required path missing:\n"
                f"{path}"
            )

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    labels = load_labels()

    # --------------------------------------------------------
    # Load timestamped words
    # --------------------------------------------------------

    words = load_words()

    # --------------------------------------------------------
    # Discover videos
    # --------------------------------------------------------

    mp4_files = discover_videos()

    # --------------------------------------------------------
    # Build video lookup
    # --------------------------------------------------------

    video_map = build_video_map(
        mp4_files
    )

    # --------------------------------------------------------
    # Build index
    # --------------------------------------------------------

    rows = build_index(
        labels,
        words,
        video_map,
    )

    if not rows:

        raise RuntimeError(
            "No matching segments were found."
        )

    # --------------------------------------------------------
    # Video-level split
    # --------------------------------------------------------

    (
        train_rows,
        val_rows,
        test_rows,
    ) = create_video_split(
        rows,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    all_rows = (
        train_rows
        + val_rows
        + test_rows
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save complete index
    # --------------------------------------------------------

    save_csv(
        all_rows,
        OUTPUT_ROOT
        / "mark2_index.csv",
    )

    # --------------------------------------------------------
    # Save train
    # --------------------------------------------------------

    save_csv(
        train_rows,
        OUTPUT_ROOT
        / "train.csv",
    )

    # --------------------------------------------------------
    # Save validation
    # --------------------------------------------------------

    save_csv(
        val_rows,
        OUTPUT_ROOT
        / "val.csv",
    )

    # --------------------------------------------------------
    # Save test
    # --------------------------------------------------------

    save_csv(
        test_rows,
        OUTPUT_ROOT
        / "test.csv",
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MARK 2 INDEX BUILD COMPLETE")
    print("=" * 70)

    print()
    print(
        "Total indexed segments:",
        len(all_rows)
    )

    print(
        "Train:",
        len(train_rows)
    )

    print(
        "Validation:",
        len(val_rows)
    )

    print(
        "Test:",
        len(test_rows)
    )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "Files created:"
    )

    print(
        "  mark2_index.csv"
    )

    print(
        "  train.csv"
    )

    print(
        "  val.csv"
    )

    print(
        "  test.csv"
    )

    print()
    print("=" * 70)
    print("READY FOR MARK 2 TRAINING")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()