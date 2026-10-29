
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.temporal_fusion_mark3 import TemporalFusionMark3


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 7
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 0

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mark3_temporal"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "mark3_temporal"

CHECKPOINT_PATH = OUTPUT_DIR / "temporal_fusion_best.pt"
RESULTS_PATH = OUTPUT_DIR / "temporal_fusion_results.json"
THRESHOLDS_PATH = OUTPUT_DIR / "temporal_fusion_thresholds.json"

EMOTION_NAMES = [
    "happy",
    "sad",
    "anger",
    "surprise",
    "disgust",
    "fear",
]


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Dataset and batching
# ------------------------------------------------------------

class TemporalWindowDataset(Dataset):
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")

        # These are locally generated feature tensors.
        self.samples = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(self.samples, list) or not self.samples:
            raise ValueError(f"Expected a non-empty list in {path}")

        required = {
            "context_segment_numbers",
            "text_embedding",
            "audio_embedding",
            "video_embedding",
            "labels",
        }

        for i, sample in enumerate(self.samples):
            missing = required - sample.keys()
            if missing:
                raise ValueError(
                    f"Sample {i} is missing keys: {sorted(missing)}"
                )

            for key in (
                "text_embedding",
                "audio_embedding",
                "video_embedding",
            ):
                if sample[key].ndim != 2 or sample[key].shape[-1] != 768:
                    raise ValueError(
                        f"Sample {i}: {key} must have shape [T, 768], "
                        f"got {tuple(sample[key].shape)}"
                    )

            lengths = [
                sample[key].shape[0]
                for key in (
                    "text_embedding",
                    "audio_embedding",
                    "video_embedding",
                )
            ]

            if len(set(lengths)) != 1:
                raise ValueError(
                    f"Sample {i} has inconsistent modality lengths: {lengths}"
                )

            numbers = torch.as_tensor(
                sample["context_segment_numbers"]
            ).flatten()

            if len(numbers) != lengths[0]:
                raise ValueError(
                    f"Sample {i}: segment-number count does not match "
                    f"embedding length."
                )

            if sample["labels"].numel() != 6:
                raise ValueError(f"Sample {i}: expected six labels.")

        print(f"Loaded {len(self.samples):,} windows from {path.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_windows(samples):
    """Right-pad sequences so each target is at length - 1."""

    batch_size = len(samples)
    lengths = torch.tensor(
        [sample["text_embedding"].shape[0] for sample in samples],
        dtype=torch.long,
    )
    max_length = int(lengths.max().item())

    text = torch.zeros(batch_size, max_length, 768)
    audio = torch.zeros_like(text)
    video = torch.zeros_like(text)

    # -1 is reserved for padding, not a valid segment number.
    segment_numbers = torch.full(
        (batch_size, max_length),
        fill_value=-1,
        dtype=torch.long,
    )
    labels = torch.zeros(batch_size, 6, dtype=torch.float32)

    for i, sample in enumerate(samples):
        length = int(lengths[i].item())

        text[i, :length] = sample["text_embedding"].float()
        audio[i, :length] = sample["audio_embedding"].float()
        video[i, :length] = sample["video_embedding"].float()

        numbers = torch.as_tensor(
            sample["context_segment_numbers"],
            dtype=torch.long,
        ).flatten()

        segment_numbers[i, :length] = numbers
        labels[i] = sample["labels"].float().reshape(6)

    return {
        "text": text,
        "audio": audio,
        "video": video,
        "lengths": lengths,
        "segment_numbers": segment_numbers,
        "labels": labels,
    }


# ------------------------------------------------------------
# Metrics and threshold calibration
# ------------------------------------------------------------

def calculate_metrics(y_true, probabilities, thresholds):
    predictions = (probabilities >= thresholds).astype(np.int32)

    return {
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "micro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="micro",
                zero_division=0,
            )
        ),
        "subset_accuracy": float(
            np.all(y_true == predictions, axis=1).mean()
        ),
        "per_emotion_f1": {
            name: float(
                f1_score(
                    y_true[:, i],
                    predictions[:, i],
                    zero_division=0,
                )
            )
            for i, name in enumerate(EMOTION_NAMES)
        },
    }


def calibrate_thresholds(y_true, probabilities):
    """
    Calibrate each label's threshold using validation data only.
    Ties prefer thresholds closer to 0.5.
    """

    candidates = np.arange(0.05, 0.951, 0.01)
    thresholds = np.full(len(EMOTION_NAMES), 0.5, dtype=np.float32)

    for i in range(len(EMOTION_NAMES)):
        best_f1 = -1.0
        best_threshold = 0.5

        for threshold in candidates:
            predictions = (
                probabilities[:, i] >= threshold
            ).astype(np.int32)

            score = f1_score(
                y_true[:, i],
                predictions,
                zero_division=0,
            )

            if (
                score > best_f1 + 1e-12
                or (
                    abs(score - best_f1) <= 1e-12
                    and abs(threshold - 0.5)
                    < abs(best_threshold - 0.5)
                )
            ):
                best_f1 = float(score)
                best_threshold = float(threshold)

        thresholds[i] = best_threshold

    return thresholds


# ------------------------------------------------------------
# Training / evaluation
# ------------------------------------------------------------

def forward_batch(model, batch, device):
    return model(
        text=batch["text"].to(device),
        audio=batch["audio"].to(device),
        video=batch["video"].to(device),
        lengths=batch["lengths"].to(device),
        segment_numbers=batch["segment_numbers"].to(device),
    )


def collect_predictions(model, loader, device, criterion=None, optimizer=None):
    training = optimizer is not None
    model.train(training)

    all_labels = []
    all_probabilities = []
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        labels = batch["labels"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits, _ = forward_batch(model, batch, device)

            loss = criterion(logits, labels) if criterion is not None else None

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP_NORM,
                )
                optimizer.step()

        batch_size = labels.shape[0]
        total_samples += batch_size

        if loss is not None:
            total_loss += float(loss.detach().item()) * batch_size

        all_labels.append(labels.detach().cpu().numpy())
        all_probabilities.append(
            torch.sigmoid(logits.detach()).cpu().numpy()
        )

    y_true = np.concatenate(all_labels, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)

    mean_loss = total_loss / total_samples if criterion is not None else None

    return mean_loss, y_true, probabilities


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Refuse to overwrite any previous run.
    existing = [
        path for path in (
            CHECKPOINT_PATH,
            RESULTS_PATH,
            THRESHOLDS_PATH,
        )
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Output files already exist. Nothing was overwritten:\n"
            + "\n".join(str(path) for path in existing)
            + "\nChoose a new output directory or explicitly archive "
              "the previous run before retrying."
        )

    train_dataset = TemporalWindowDataset(DATA_DIR / "train_windows.pt")
    val_dataset = TemporalWindowDataset(DATA_DIR / "val_windows.pt")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_windows,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_windows,
        pin_memory=torch.cuda.is_available(),
    )

    # Compute positive-class weights from training labels only.
    train_labels = np.stack(
        [
            sample["labels"].detach().cpu().numpy().reshape(6)
            for sample in train_dataset.samples
        ]
    )

    positive_counts = train_labels.sum(axis=0)
    negative_counts = len(train_labels) - positive_counts

    if np.any(positive_counts == 0):
        raise ValueError(
            "At least one emotion has no positive training examples."
        )

    pos_weight = torch.tensor(
        negative_counts / positive_counts,
        dtype=torch.float32,
        device=device,
    ).clamp(min=1.0, max=10.0)

    print(
        "Positive-class weights:",
        {
            name: round(float(pos_weight[i].item()), 3)
            for i, name in enumerate(EMOTION_NAMES)
        },
    )

    model = TemporalFusionMark3().to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print("\nStarting temporal training. Test data is not loaded.")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _, _ = collect_predictions(
            model,
            train_loader,
            device,
            criterion=criterion,
            optimizer=optimizer,
        )

        val_loss, val_y, val_probabilities = collect_predictions(
            model,
            val_loader,
            device,
            criterion=criterion,
        )

        # Checkpoint selection uses a fixed threshold for consistency.
        val_metrics = calculate_metrics(
            val_y,
            val_probabilities,
            np.full(6, 0.5, dtype=np.float32),
        )

        val_macro_f1 = val_metrics["macro_f1"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1_at_0_5": val_macro_f1,
            "val_micro_f1_at_0_5": val_metrics["micro_f1"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_f1@0.5={val_macro_f1:.4f} | "
            f"val_micro_f1@0.5={val_metrics['micro_f1']:.4f}"
        )

        if val_macro_f1 > best_val_macro_f1 + 1e-6:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1_at_0_5": val_macro_f1,
                    "config": {
                        "input_dim": 768,
                        "hidden_dim": 256,
                        "num_heads": 8,
                        "num_layers": 2,
                        "num_emotions": 6,
                        "max_sequence_length": 8,
                        "dropout": 0.15,
                    },
                    "emotion_names": EMOTION_NAMES,
                    "seed": SEED,
                },
                CHECKPOINT_PATH,
            )

            print("  -> Saved new best validation checkpoint.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation Macro-F1 @0.5: {best_val_macro_f1:.4f}")

    # Reload the best checkpoint, then calibrate thresholds on validation.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    _, val_y, val_probabilities = collect_predictions(
        model,
        val_loader,
        device,
        criterion=None,
    )

    calibrated_thresholds = calibrate_thresholds(val_y, val_probabilities)

    val_calibrated_metrics = calculate_metrics(
        val_y,
        val_probabilities,
        calibrated_thresholds,
    )

    # Test data is loaded only after training and threshold calibration finish.
    test_dataset = TemporalWindowDataset(DATA_DIR / "test_windows.pt")
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_windows,
        pin_memory=torch.cuda.is_available(),
    )

    _, test_y, test_probabilities = collect_predictions(
        model,
        test_loader,
        device,
        criterion=None,
    )

    test_metrics_at_05 = calculate_metrics(
        test_y,
        test_probabilities,
        np.full(6, 0.5, dtype=np.float32),
    )

    test_metrics_calibrated = calculate_metrics(
        test_y,
        test_probabilities,
        calibrated_thresholds,
    )

    thresholds_dict = {
        name: float(calibrated_thresholds[i])
        for i, name in enumerate(EMOTION_NAMES)
    }

    results = {
        "model": "TemporalFusionMark3",
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_val_macro_f1_at_0_5": best_val_macro_f1,
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "test_windows": len(test_dataset),
        "checkpoint_selection": "validation_macro_f1_at_threshold_0.5",
        "threshold_calibration": "validation_only_per_emotion_f1",
        "test_evaluated_after_model_selection": True,
        "history": history,
        "validation_calibrated_thresholds": thresholds_dict,
        "validation_metrics_at_0_5": calculate_metrics(
            val_y,
            val_probabilities,
            np.full(6, 0.5, dtype=np.float32),
        ),
        "validation_metrics_calibrated": val_calibrated_metrics,
        "test_metrics_at_0_5": test_metrics_at_05,
        "test_metrics_calibrated": test_metrics_calibrated,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with THRESHOLDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=2)

    print("\nFINAL TEST RESULTS — threshold 0.5")
    print(f"Macro-F1:       {test_metrics_at_05['macro_f1']:.4f}")
    print(f"Micro-F1:       {test_metrics_at_05['micro_f1']:.4f}")
    print(f"Subset accuracy:{test_metrics_at_05['subset_accuracy']:.4f}")

    print("\nFINAL TEST RESULTS — validation-calibrated thresholds")
    print(f"Macro-F1:       {test_metrics_calibrated['macro_f1']:.4f}")
    print(f"Micro-F1:       {test_metrics_calibrated['micro_f1']:.4f}")
    print(f"Subset accuracy:{test_metrics_calibrated['subset_accuracy']:.4f}")

    print("\nPer-emotion test F1 — calibrated thresholds")
    for name, score in test_metrics_calibrated["per_emotion_f1"].items():
        print(f"  {name:10s}: {score:.4f}")

    print("\nCalibrated thresholds:")
    for name, threshold in thresholds_dict.items():
        print(f"  {name:10s}: {threshold:.2f}")

    print("\nSaved artifacts:")
    print(CHECKPOINT_PATH)
    print(RESULTS_PATH)
    print(THRESHOLDS_PATH)


if __name__ == "__main__":
    main()
