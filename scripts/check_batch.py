import torch
from mmsdk import mmdatasdk

from src.data.split import create_video_splits
from src.data.dataloaders import create_dataloaders


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

print("Loading aligned Mark 1 dataset...")

dataset = mmdatasdk.mmdataset(
    DATA_FILES,
    BASE_PATH
)

print("Dataset loaded.")


# ============================================================
# Create video-level split
# ============================================================

label_ids = list(
    dataset.computational_sequences[
        "All Labels"
    ].data.keys()
)

train_ids, val_ids, test_ids = create_video_splits(
    label_ids,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42
)


print("\nSplit:")
print("Train segments:", len(train_ids))
print("Validation segments:", len(val_ids))
print("Test segments:", len(test_ids))


# ============================================================
# Check for video leakage
# ============================================================

def get_video_id(segment_id):
    return segment_id.rsplit("[", 1)[0]


train_videos = {
    get_video_id(x)
    for x in train_ids
}

val_videos = {
    get_video_id(x)
    for x in val_ids
}

test_videos = {
    get_video_id(x)
    for x in test_ids
}


print("\nVideo split:")
print("Train videos:", len(train_videos))
print("Validation videos:", len(val_videos))
print("Test videos:", len(test_videos))


print("\nChecking for leakage...")

print(
    "Train ∩ Validation:",
    len(train_videos & val_videos)
)

print(
    "Train ∩ Test:",
    len(train_videos & test_videos)
)

print(
    "Validation ∩ Test:",
    len(val_videos & test_videos)
)


# ============================================================
# Create DataLoaders
# ============================================================

print("\nCreating DataLoaders...")

train_loader, val_loader, test_loader = (
    create_dataloaders(
        dataset,
        train_ids,
        val_ids,
        test_ids,
        batch_size=8,
        num_workers=0
    )
)

print("DataLoaders created.")


# ============================================================
# Get one batch
# ============================================================

print("\nLoading one training batch...")

batch = next(iter(train_loader))


# ============================================================
# Print batch information
# ============================================================

print("\n==============================")
print("BATCH CHECK")
print("==============================")

print(
    "Text shape:",
    batch["text"].shape
)

print(
    "Audio shape:",
    batch["audio"].shape
)

print(
    "Vision shape:",
    batch["vision"].shape
)

print(
    "Labels shape:",
    batch["labels"].shape
)

print(
    "Batch size:",
    len(batch["segment_ids"])
)


# ============================================================
# Check for NaN / Inf
# ============================================================

print("\nChecking data validity...")

print(
    "Text NaN:",
    torch.isnan(batch["text"]).any().item()
)

print(
    "Audio NaN:",
    torch.isnan(batch["audio"]).any().item()
)

print(
    "Vision NaN:",
    torch.isnan(batch["vision"]).any().item()
)

print(
    "Labels NaN:",
    torch.isnan(batch["labels"]).any().item()
)


print(
    "Text Inf:",
    torch.isinf(batch["text"]).any().item()
)

print(
    "Audio Inf:",
    torch.isinf(batch["audio"]).any().item()
)

print(
    "Vision Inf:",
    torch.isinf(batch["vision"]).any().item()
)


# ============================================================
# Final result
# ============================================================

print("\n==============================")

if (
    len(train_videos & val_videos) == 0
    and len(train_videos & test_videos) == 0
    and len(val_videos & test_videos) == 0
):
    print("✓ No video leakage.")

else:
    print("✗ VIDEO LEAKAGE DETECTED!")


print("==============================")
print("✓ Batch check completed.")