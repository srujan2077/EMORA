import random

import numpy as np
import torch
from mmsdk import mmdatasdk

from src.data.mosei_dataset import MOSEIDataset
from src.data.split import create_video_splits
from src.models.multimodal_model import MultimodalEmotionModel


BASE_PATH = "C:/SRUJAN/multimodal_ai/data/processed/aligned_mark1"

DATA_FILES = {
    "glove_vectors": f"{BASE_PATH}/glove_vectors_aligned.csd",
    "COVAREP": f"{BASE_PATH}/COVAREP_aligned.csd",
    "FACET 4.2": f"{BASE_PATH}/FACET42_aligned.csd",
    "All Labels": f"{BASE_PATH}/labels_aligned.csd",
}

CHECKPOINT_PATH = (
    "C:/SRUJAN/multimodal_ai/checkpoints/mark1_best.pt"
)

EMOTIONS = ["Happy", "Sad", "Anger", "Surprise", "Disgust", "Fear"]
THRESHOLD = 0.5
NUM_SAMPLES = 10
SEED = 42


def emotion_names(binary_values):
    return [
        name
        for name, value in zip(EMOTIONS, binary_values)
        if value == 1
    ]


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Loading aligned dataset...")

    dataset = mmdatasdk.mmdataset(DATA_FILES, BASE_PATH)

    sequences = dataset.computational_sequences

    common_ids = sorted(
        set(sequences["glove_vectors"].data)
        & set(sequences["COVAREP"].data)
        & set(sequences["FACET 4.2"].data)
        & set(sequences["All Labels"].data)
    )

    _, _, test_ids = create_video_splits(
        common_ids,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=SEED
    )

    print("Test segments:", len(test_ids))

    test_dataset = MOSEIDataset(dataset, test_ids)

    model = MultimodalEmotionModel().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint epoch:", checkpoint["epoch"])

    rng = random.Random(SEED)
    selected_indices = rng.sample(
        range(len(test_dataset)),
        k=min(NUM_SAMPLES, len(test_dataset))
    )

    top_matches = 0
    exact_matches = 0

    print("\n" + "=" * 65)
    print("MARK 1 — MULTIPLE TEST-SET PREDICTIONS")
    print("=" * 65)

    for number, index in enumerate(selected_indices, start=1):
        sample = test_dataset[index]

        text = sample["text"].unsqueeze(0).to(device)
        audio = sample["audio"].unsqueeze(0).to(device)
        vision = sample["vision"].unsqueeze(0).to(device)

        actual = sample["labels"].numpy().astype(int)

        with torch.no_grad():
            logits = model(text, audio, vision)
            scores = torch.sigmoid(logits)[0].cpu().numpy()

        predicted = (scores >= THRESHOLD).astype(int)

        top_index = int(np.argmax(scores))
        top_emotion = EMOTIONS[top_index]

        top_match = bool(actual[top_index] == 1)
        exact_match = bool(np.array_equal(actual, predicted))

        top_matches += int(top_match)
        exact_matches += int(exact_match)

        print(f"\nSample {number}: {sample['segment_id']}")
        print("Actual:    ", ", ".join(emotion_names(actual)) or "None")
        print("Predicted: ", ", ".join(emotion_names(predicted)) or "None")
        print(
            f"Top emotion: {top_emotion} "
            f"({scores[top_index] * 100:.2f}%)"
        )
        print("Top-label match:", "YES" if top_match else "NO")
        print("Exact multi-label match:", "YES" if exact_match else "NO")

    total = len(selected_indices)

    print("\n" + "=" * 65)
    print("10-SAMPLE DEMO SUMMARY")
    print("=" * 65)
    print(f"Top-label matches: {top_matches}/{total}")
    print(f"Exact multi-label matches: {exact_matches}/{total}")
    print("=" * 65)


if __name__ == "__main__":
    main()