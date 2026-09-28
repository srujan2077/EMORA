import random
import numpy as np
import torch

from mmsdk import mmdatasdk

from src.models.multimodal_model import MultimodalEmotionModel
from src.data.split import create_video_splits


# ============================================================
# Configuration
# ============================================================

BASE_PATH = "C:/SRUJAN/multimodal_ai/data/processed/aligned_mark1"

DATA_FILES = {
    "glove_vectors": f"{BASE_PATH}/glove_vectors_aligned.csd",
    "COVAREP": f"{BASE_PATH}/COVAREP_aligned.csd",
    "FACET 4.2": f"{BASE_PATH}/FACET42_aligned.csd",
    "All Labels": f"{BASE_PATH}/labels_aligned.csd",
}

CHECKPOINT_PATH = (
    "C:/SRUJAN/multimodal_ai/"
    "checkpoints/mark1_best.pt"
)

EMOTIONS = [
    "Happy",
    "Sad",
    "Anger",
    "Surprise",
    "Disgust",
    "Fear"
]

# Original CMU-MOSEI label order:
# [sentiment, happy, sad, anger, surprise, disgust, fear]
EMOTION_INDICES = [1, 2, 3, 4, 5, 6]

SEED = 42


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Load dataset
# ============================================================

print("\nLoading aligned CMU-MOSEI dataset...")

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

print(
    "Common multimodal segments:",
    len(common_ids)
)


# ============================================================
# Recreate exact Mark 1 video-level split
# ============================================================

train_ids, val_ids, test_ids = create_video_splits(
    common_ids,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=SEED
)

print("Train segments:", len(train_ids))
print("Validation segments:", len(val_ids))
print("Test segments:", len(test_ids))


# ============================================================
# Select test sample
# ============================================================

random.seed(SEED)

segment_id = random.choice(test_ids)

print("\nSelected TEST segment:")
print(segment_id)


# ============================================================
# Load features
# ============================================================

text = np.asarray(
    dataset.computational_sequences[
        "glove_vectors"
    ].data[segment_id]["features"],
    dtype=np.float32
)

audio = np.asarray(
    dataset.computational_sequences[
        "COVAREP"
    ].data[segment_id]["features"],
    dtype=np.float32
)

vision = np.asarray(
    dataset.computational_sequences[
        "FACET 4.2"
    ].data[segment_id]["features"],
    dtype=np.float32
)


# ============================================================
# Clean invalid values
# ============================================================

text = np.nan_to_num(
    text,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

audio = np.nan_to_num(
    audio,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

vision = np.nan_to_num(
    vision,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ============================================================
# Same temporal downsampling used during training
# ============================================================

audio = audio[::10]
vision = vision[::5]


# ============================================================
# Convert to tensors
# ============================================================

text = torch.from_numpy(
    text
).float().unsqueeze(0)

audio = torch.from_numpy(
    audio
).float().unsqueeze(0)

vision = torch.from_numpy(
    vision
).float().unsqueeze(0)


text = text.to(device)
audio = audio.to(device)
vision = vision.to(device)


# ============================================================
# Load Mark 1 model
# ============================================================

model = MultimodalEmotionModel()

model = model.to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "\nLoaded Mark 1 checkpoint from epoch:",
    checkpoint["epoch"]
)


# ============================================================
# Prediction
# ============================================================

with torch.no_grad():

    logits = model(
        text,
        audio,
        vision
    )

    probabilities = torch.sigmoid(
        logits
    )

probabilities = (
    probabilities[0]
    .cpu()
    .numpy()
)


# ============================================================
# Ground-truth labels
# ============================================================

raw_labels = np.asarray(
    dataset.computational_sequences[
        "All Labels"
    ].data[segment_id]["features"],
    dtype=np.float32
)[0]

raw_emotion_labels = raw_labels[
    EMOTION_INDICES
]

ground_truth = (
    raw_emotion_labels > 0
).astype(int)


# ============================================================
# Determine ground-truth emotions
# ============================================================

actual_emotions = [
    emotion
    for emotion, value
    in zip(EMOTIONS, ground_truth)
    if value == 1
]


# ============================================================
# Determine predicted emotion
# ============================================================

best_index = int(
    np.argmax(probabilities)
)

predicted_emotion = EMOTIONS[
    best_index
]

best_score = probabilities[
    best_index
]


# ============================================================
# Determine whether top prediction is present
# in the ground-truth labels
# ============================================================

prediction_correct = (
    ground_truth[best_index] == 1
)


# ============================================================
# Display result
# ============================================================

print("\n")
print("========================================")
print("          MARK 1 RESULT")
print("========================================")

print("\nSegment:")
print(segment_id)

print("\nActual Emotion(s):")

if actual_emotions:
    for emotion in actual_emotions:
        print(f"  ✓ {emotion}")
else:
    print("  No positive emotion label")

print("\nPredicted Emotion:")
print(
    f"  {predicted_emotion.upper()}"
)

print(
    f"  Model score: "
    f"{best_score * 100:.2f}%"
)


# ============================================================
# Display all scores
# ============================================================

print("\nModel Scores:")

for emotion, probability in zip(
    EMOTIONS,
    probabilities
):

    actual = (
        " ← ACTUAL"
        if ground_truth[
            EMOTIONS.index(emotion)
        ] == 1
        else ""
    )

    print(
        f"  {emotion:10s}: "
        f"{probability * 100:6.2f}%"
        f"{actual}"
    )


# ============================================================
# Comparison
# ============================================================

print("\n----------------------------------------")

if prediction_correct:

    print(
        "Top prediction matches "
        "an actual emotion label: YES"
    )

else:

    print(
        "Top prediction matches "
        "an actual emotion label: NO"
    )

print("----------------------------------------")

print("\nMark 1 ground-truth comparison complete.")
print("========================================")