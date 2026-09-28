import numpy as np
from mmsdk import mmdatasdk

from src.data.split import create_video_splits


# ============================================================
# Configuration
# ============================================================

BASE_PATH = (
    "C:/SRUJAN/multimodal_ai/"
    "data/processed/aligned_mark1"
)

DATA_FILES = {
    "glove_vectors":
        f"{BASE_PATH}/glove_vectors_aligned.csd",

    "COVAREP":
        f"{BASE_PATH}/COVAREP_aligned.csd",

    "FACET 4.2":
        f"{BASE_PATH}/FACET42_aligned.csd",

    "All Labels":
        f"{BASE_PATH}/labels_aligned.csd",
}


# ============================================================
# Load dataset
# ============================================================

print("Loading dataset...")

dataset = mmdatasdk.mmdataset(
    DATA_FILES,
    BASE_PATH
)

print("Dataset loaded.")


# ============================================================
# Find common segments
# ============================================================

text_ids = set(
    dataset.computational_sequences[
        "glove_vectors"
    ].data.keys()
)

audio_ids = set(
    dataset.computational_sequences[
        "COVAREP"
    ].data.keys()
)

vision_ids = set(
    dataset.computational_sequences[
        "FACET 4.2"
    ].data.keys()
)

label_ids = set(
    dataset.computational_sequences[
        "All Labels"
    ].data.keys()
)

common_ids = sorted(
    text_ids
    & audio_ids
    & vision_ids
    & label_ids
)


# ============================================================
# Check every modality
# ============================================================

modalities = {
    "TEXT": "glove_vectors",
    "AUDIO": "COVAREP",
    "VISION": "FACET 4.2",
    "LABELS": "All Labels"
}


for name, key in modalities.items():

    print("\n==============================")
    print(f"Checking {name}")
    print("==============================")

    data = dataset.computational_sequences[
        key
    ].data

    nan_count = 0
    inf_count = 0

    global_min = np.inf
    global_max = -np.inf

    worst_segment = None

    for i, segment_id in enumerate(common_ids):

        features = np.asarray(
            data[segment_id]["features"],
            dtype=np.float32
        )

        # Check NaN
        if np.isnan(features).any():
            nan_count += 1

            if worst_segment is None:
                worst_segment = segment_id

        # Check Inf
        if np.isinf(features).any():
            inf_count += 1

            if worst_segment is None:
                worst_segment = segment_id

        # Ignore non-finite values when checking range
        finite_values = features[
            np.isfinite(features)
        ]

        if finite_values.size > 0:

            current_min = finite_values.min()
            current_max = finite_values.max()

            global_min = min(
                global_min,
                current_min
            )

            global_max = max(
                global_max,
                current_max
            )

        if (i + 1) % 1000 == 0:

            print(
                f"Checked {i + 1}/{len(common_ids)}"
            )


    print("\nResults:")

    print(
        "Segments checked:",
        len(common_ids)
    )

    print(
        "Segments containing NaN:",
        nan_count
    )

    print(
        "Segments containing Inf:",
        inf_count
    )

    print(
        "Global minimum:",
        global_min
    )

    print(
        "Global maximum:",
        global_max
    )

    if worst_segment is not None:

        print(
            "Example problematic segment:",
            worst_segment
        )

    else:

        print(
            "✓ No NaN/Inf found."
        )


print("\n==============================")
print("Feature scan complete.")
print("==============================")