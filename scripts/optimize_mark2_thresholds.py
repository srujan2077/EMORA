import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

# Reuse the exact Mark 2 architecture and dataset
from train_mark2 import (
    CachedMark2Dataset,
    Mark2FusionModel,
    DEVICE,
    INPUT_DIM,
    PROJECTION_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    FF_DIM,
    DROPOUT,
    NUM_EMOTIONS,
    EMOTION_NAMES,
    CHECKPOINT_DIR,
)
from torch.utils.data import DataLoader


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 64

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "mark2_best.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "mark2_thresholds.json"
)


# ============================================================
# COLLECT PROBABILITIES
# ============================================================

@torch.no_grad()
def collect_predictions(model, loader):

    model.eval()

    all_probabilities = []
    all_labels = []

    for batch in loader:

        text = batch["text"].to(DEVICE)
        audio = batch["audio"].to(DEVICE)
        video = batch["video"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits, _ = model(
            text,
            audio,
            video,
        )

        probabilities = torch.sigmoid(
            logits
        )

        all_probabilities.append(
            probabilities.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

    probabilities = torch.cat(
        all_probabilities
    ).numpy()

    labels = torch.cat(
        all_labels
    ).numpy()

    return probabilities, labels


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

def find_best_threshold(
    probabilities,
    labels,
):

    thresholds = np.arange(
        0.05,
        0.96,
        0.01,
    )

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = float(
                threshold
            )

    return best_threshold, best_f1


# ============================================================
# EVALUATE WITH THRESHOLDS
# ============================================================

def evaluate_thresholds(
    probabilities,
    labels,
    thresholds,
):

    predictions = np.zeros_like(
        probabilities,
        dtype=int,
    )

    for index, threshold in enumerate(
        thresholds
    ):

        predictions[:, index] = (
            probabilities[:, index]
            >= threshold
        ).astype(int)

    micro_f1 = f1_score(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    subset_accuracy = np.mean(
        np.all(
            labels == predictions,
            axis=1,
        )
    )

    emotion_f1 = f1_score(
        labels,
        predictions,
        average=None,
        zero_division=0,
    )

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "subset_accuracy": float(
            subset_accuracy
        ),
        "emotion_f1": [
            float(x)
            for x in emotion_f1
        ],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MARK 2 - THRESHOLD OPTIMIZATION")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print()
    print("Loading validation dataset...")

    val_dataset = CachedMark2Dataset(
        "val"
    )

    print()
    print("Loading test dataset...")

    test_dataset = CachedMark2Dataset(
        "test"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading Mark 2 model...")

    model = Mark2FusionModel(
        input_dim=INPUT_DIM,
        projection_dim=PROJECTION_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        num_emotions=NUM_EMOTIONS,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Best checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Original validation Micro-F1: "
        f"{checkpoint['val_micro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Collect validation predictions
    # --------------------------------------------------------

    print()
    print(
        "Collecting validation predictions..."
    )

    val_probabilities, val_labels = (
        collect_predictions(
            model,
            val_loader,
        )
    )

    # --------------------------------------------------------
    # Optimize thresholds
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OPTIMIZED EMOTION THRESHOLDS")
    print("=" * 70)

    thresholds = []

    for index, emotion in enumerate(
        EMOTION_NAMES
    ):

        threshold, best_f1 = (
            find_best_threshold(
                val_probabilities[:, index],
                val_labels[:, index],
            )
        )

        thresholds.append(
            threshold
        )

        print(
            f"{emotion:10s} -> "
            f"threshold = {threshold:.2f} "
            f"| validation F1 = {best_f1:.4f}"
        )

    # --------------------------------------------------------
    # Compare validation performance
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION COMPARISON")
    print("=" * 70)

    default_thresholds = [
        0.50
    ] * NUM_EMOTIONS

    default_results = evaluate_thresholds(
        val_probabilities,
        val_labels,
        default_thresholds,
    )

    optimized_results = evaluate_thresholds(
        val_probabilities,
        val_labels,
        thresholds,
    )

    print()
    print("Default threshold = 0.50")

    print(
        f"  Micro-F1: "
        f"{default_results['micro_f1']:.4f}"
    )

    print(
        f"  Macro-F1: "
        f"{default_results['macro_f1']:.4f}"
    )

    print(
        f"  Subset Accuracy: "
        f"{default_results['subset_accuracy']:.4f}"
    )

    print()
    print("Optimized thresholds")

    print(
        f"  Micro-F1: "
        f"{optimized_results['micro_f1']:.4f}"
    )

    print(
        f"  Macro-F1: "
        f"{optimized_results['macro_f1']:.4f}"
    )

    print(
        f"  Subset Accuracy: "
        f"{optimized_results['subset_accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Collect TEST predictions
    # --------------------------------------------------------

    print()
    print(
        "Collecting TEST predictions..."
    )

    test_probabilities, test_labels = (
        collect_predictions(
            model,
            test_loader,
        )
    )

    # --------------------------------------------------------
    # Test with default threshold
    # --------------------------------------------------------

    test_default = evaluate_thresholds(
        test_probabilities,
        test_labels,
        default_thresholds,
    )

    # --------------------------------------------------------
    # Test with validation-derived thresholds
    # --------------------------------------------------------

    test_optimized = evaluate_thresholds(
        test_probabilities,
        test_labels,
        thresholds,
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL MARK 2 TEST COMPARISON")
    print("=" * 70)

    print()
    print("TEST — Default 0.50 thresholds")

    print(
        f"  Micro-F1: "
        f"{test_default['micro_f1']:.4f}"
    )

    print(
        f"  Macro-F1: "
        f"{test_default['macro_f1']:.4f}"
    )

    print(
        f"  Subset Accuracy: "
        f"{test_default['subset_accuracy']:.4f}"
    )

    print()
    print(
        "TEST — Validation-optimized thresholds"
    )

    print(
        f"  Micro-F1: "
        f"{test_optimized['micro_f1']:.4f}"
    )

    print(
        f"  Macro-F1: "
        f"{test_optimized['macro_f1']:.4f}"
    )

    print(
        f"  Subset Accuracy: "
        f"{test_optimized['subset_accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Per-emotion comparison
    # --------------------------------------------------------

    print()
    print("Per-emotion TEST F1:")

    for index, emotion in enumerate(
        EMOTION_NAMES
    ):

        print(
            f"  {emotion:10s} "
            f"default={test_default['emotion_f1'][index]:.4f} "
            f"optimized={test_optimized['emotion_f1'][index]:.4f}"
        )

    # --------------------------------------------------------
    # Save thresholds
    # --------------------------------------------------------

    result = {

        "checkpoint": str(
            CHECKPOINT_PATH
        ),

        "thresholds": {
            emotion: threshold
            for emotion, threshold in zip(
                EMOTION_NAMES,
                thresholds,
            )
        },

        "validation": {

            "default": default_results,

            "optimized": optimized_results,
        },

        "test": {

            "default": test_default,

            "optimized": test_optimized,
        },
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(
        f"Threshold configuration saved:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 70)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()