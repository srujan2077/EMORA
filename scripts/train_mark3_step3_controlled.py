"""Controlled, candidate-parameterized Mark 3 temporal training experiment."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.temporal_fusion_mark3 import TemporalFusionMark3

EMOTION_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 7
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 0
MAX_CONTEXT = 8
DATA_DIR = ROOT / "data" / "processed" / "mark3_temporal"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def contiguous_suffix_length(numbers):
    length = 1
    for previous, current in zip(reversed(numbers[:-1]), reversed(numbers[1:])):
        if current != previous + 1:
            break
        length += 1
    return length


def transform_sample(sample, candidate):
    if candidate == "metadata_aware":
        return sample

    numbers = [int(value) for value in sample["context_segment_numbers"]]
    keep = contiguous_suffix_length(numbers)
    transformed = dict(sample)
    start = len(numbers) - keep
    for key in ("text_embedding", "audio_embedding", "video_embedding"):
        transformed[key] = sample[key][start:].clone()
    transformed["context_segment_ids"] = list(
        sample["context_segment_ids"][start:]
    )
    transformed["context_segment_numbers"] = numbers[start:]
    transformed["context_length"] = keep
    transformed["has_numbering_gap"] = False
    return transformed


class CandidateDataset(Dataset):
    def __init__(self, path, candidate):
        samples = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Expected a non-empty list in {path}")
        self.samples = [transform_sample(sample, candidate) for sample in samples]
        self.candidate = candidate
        self._validate()

    def _validate(self):
        for index, sample in enumerate(self.samples):
            lengths = [
                sample[key].shape[0]
                for key in ("text_embedding", "audio_embedding", "video_embedding")
            ]
            if len(set(lengths)) != 1 or not 1 <= lengths[0] <= MAX_CONTEXT:
                raise ValueError(f"Invalid modality lengths at sample {index}")
            numbers = [int(value) for value in sample["context_segment_numbers"]]
            if len(numbers) != lengths[0]:
                raise ValueError(f"Number/embedding mismatch at sample {index}")
            if numbers[-1] != int(sample["target_segment_number"]):
                raise ValueError(f"Target is not final at sample {index}")
            if any(current <= previous for previous, current in zip(numbers, numbers[1:])):
                raise ValueError(f"Context is not increasing at sample {index}")
            if any(number > numbers[-1] for number in numbers):
                raise ValueError(f"Future segment at sample {index}")
            if sample["labels"].numel() != 6:
                raise ValueError(f"Invalid labels at sample {index}")
            if self.candidate == "split_at_gap" and any(
                current != previous + 1
                for previous, current in zip(numbers, numbers[1:])
            ):
                raise ValueError(f"Split-at-gap context is not contiguous at {index}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_windows(samples):
    lengths = torch.tensor(
        [sample["text_embedding"].shape[0] for sample in samples],
        dtype=torch.long,
    )
    batch_size = len(samples)
    max_length = int(lengths.max().item())
    text = torch.zeros(batch_size, max_length, 768)
    audio = torch.zeros_like(text)
    video = torch.zeros_like(text)
    segment_numbers = torch.full((batch_size, max_length), -1, dtype=torch.long)
    labels = torch.zeros(batch_size, 6, dtype=torch.float32)
    for index, sample in enumerate(samples):
        length = int(lengths[index].item())
        text[index, :length] = sample["text_embedding"].float()
        audio[index, :length] = sample["audio_embedding"].float()
        video[index, :length] = sample["video_embedding"].float()
        segment_numbers[index, :length] = torch.as_tensor(
            sample["context_segment_numbers"], dtype=torch.long
        )
        labels[index] = sample["labels"].float().reshape(6)
    return {
        "text": text,
        "audio": audio,
        "video": video,
        "lengths": lengths,
        "segment_numbers": segment_numbers,
        "labels": labels,
    }


def metrics(labels, probabilities, thresholds):
    predictions = (probabilities >= thresholds).astype(np.int32)
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "subset_accuracy": float(np.all(labels == predictions, axis=1).mean()),
        "per_emotion_f1": {
            name: float(f1_score(labels[:, index], predictions[:, index], zero_division=0))
            for index, name in enumerate(EMOTION_NAMES)
        },
    }


def calibrate_thresholds(labels, probabilities):
    thresholds = np.full(6, 0.5, dtype=np.float32)
    for index in range(6):
        best_score = -1.0
        best_threshold = 0.5
        for threshold in np.arange(0.05, 0.951, 0.01):
            score = f1_score(
                labels[:, index],
                (probabilities[:, index] >= threshold).astype(np.int32),
                zero_division=0,
            )
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12
                and abs(threshold - 0.5) < abs(best_threshold - 0.5)
            ):
                best_score = float(score)
                best_threshold = float(threshold)
        thresholds[index] = best_threshold
    return thresholds


def collect(model, loader, device, criterion=None, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    labels_all, probabilities_all = [], []
    total_loss = 0.0
    total_samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        labels = batch["labels"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits, _ = model(
                text=batch["text"].to(device),
                audio=batch["audio"].to(device),
                video=batch["video"].to(device),
                lengths=batch["lengths"].to(device),
                segment_numbers=batch["segment_numbers"].to(device),
            )
            loss = criterion(logits, labels) if criterion is not None else None
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
        size = labels.shape[0]
        total_samples += size
        if loss is not None:
            total_loss += float(loss.detach().item()) * size
        labels_all.append(labels.detach().cpu().numpy())
        probabilities_all.append(torch.sigmoid(logits.detach()).cpu().numpy())
    if not labels_all:
        raise ValueError("No batches were processed.")
    return (
        total_loss / total_samples if criterion is not None else None,
        np.concatenate(labels_all),
        np.concatenate(probabilities_all),
    )


def distribution(dataset):
    counts = Counter(str(sample["context_length"]) for sample in dataset.samples)
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def validate_split_isolation(train_dataset, val_dataset):
    train_videos = {str(sample["video_id"]) for sample in train_dataset.samples}
    val_videos = {str(sample["video_id"]) for sample in val_dataset.samples}
    overlap = sorted(train_videos & val_videos)
    if overlap:
        raise ValueError(f"Train/validation video overlap: {overlap[:10]}")
    return len(train_videos), len(val_videos), overlap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("metadata_aware", "split_at_gap"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic PyTorch algorithms")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_epochs < 1 or args.patience < 1:
        raise ValueError("Epoch and patience values must be positive.")
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("results.json", "thresholds.json", "best.pt"):
        path = args.output_dir / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    set_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = CandidateDataset(DATA_DIR / "train_windows.pt", args.candidate)
    val_dataset = CandidateDataset(DATA_DIR / "val_windows.pt", args.candidate)
    train_videos, val_videos, overlap = validate_split_isolation(train_dataset, val_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=collate_windows, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_windows, pin_memory=torch.cuda.is_available(),
    )

    train_labels = np.stack([
        sample["labels"].detach().cpu().numpy().reshape(6)
        for sample in train_dataset.samples
    ])
    positive_counts = train_labels.sum(axis=0)
    negative_counts = len(train_labels) - positive_counts
    if np.any(positive_counts == 0):
        raise ValueError("A training emotion has no positive examples.")
    pos_weight = np.clip(negative_counts / positive_counts, 1.0, 10.0)
    model_config = {
        "input_dim": 768, "hidden_dim": 256, "num_heads": 8,
        "num_layers": 2, "num_emotions": 6, "max_sequence_length": 8,
        "dropout": 0.15,
    }
    model = TemporalFusionMark3(**model_config).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    fixed_thresholds = np.full(6, 0.5, dtype=np.float32)
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        train_loss, _, _ = collect(
            model, train_loader, device, criterion, optimizer, args.max_train_batches
        )
        val_loss, val_labels, val_probabilities = collect(
            model, val_loader, device, criterion, None, args.max_val_batches
        )
        val_metrics = metrics(val_labels, val_probabilities, fixed_thresholds)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1_at_0_5": val_metrics["macro_f1"],
            "val_micro_f1_at_0_5": val_metrics["micro_f1"],
        })
        if val_metrics["macro_f1"] > best_score + 1e-6:
            best_score = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1_at_0_5": best_score,
                "config": model_config,
                "candidate": args.candidate,
                "seed": args.seed,
                "deterministic_algorithms": args.deterministic,
            }, args.output_dir / "best.pt")
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, final_labels, final_probabilities = collect(
        model, val_loader, device, None, None, None
    )
    calibrated = calibrate_thresholds(final_labels, final_probabilities)
    results = {
        "candidate": args.candidate,
        "policy": (
            "Existing metadata-aware windows"
            if args.candidate == "metadata_aware"
            else "Longest contiguous suffix ending at each target"
        ),
        "test_data_loaded": False,
        "device": str(device),
        "seed": args.seed,
        "deterministic_algorithms": args.deterministic,
        "config": {
            "batch_size": BATCH_SIZE, "max_epochs": args.max_epochs,
            "patience": args.patience, "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "grad_clip_norm": GRAD_CLIP_NORM,
            "num_workers": NUM_WORKERS, "model": model_config,
        },
        "inputs": {
            "train": str((DATA_DIR / "train_windows.pt").relative_to(ROOT)),
            "validation": str((DATA_DIR / "val_windows.pt").relative_to(ROOT)),
        },
        "split_integrity": {
            "train_videos": train_videos, "validation_videos": val_videos,
            "overlap_count": len(overlap), "overlap_examples": overlap[:10],
        },
        "class_weight_formula": (
            "clip(negative_counts / positive_counts, 1.0, 10.0), where "
            "negative_counts = N - positive_counts"
        ),
        "class_weights": {
            name: float(pos_weight[index])
            for index, name in enumerate(EMOTION_NAMES)
        },
        "context_length_distributions": {
            "train": distribution(train_dataset), "validation": distribution(val_dataset),
        },
        "history": history,
        "best_epoch": int(checkpoint["epoch"]),
        "best_checkpoint": str((args.output_dir / "best.pt").relative_to(ROOT)),
        "validation_primary_at_0_5": metrics(
            final_labels, final_probabilities, fixed_thresholds
        ),
        "validation_calibrated_thresholds": {
            name: float(calibrated[index])
            for index, name in enumerate(EMOTION_NAMES)
        },
        "validation_calibrated_metrics": metrics(
            final_labels, final_probabilities, calibrated
        ),
        "target_only_note": (
            "Step 2 target-only evaluation was diagnostic only and was not "
            "an independently trained baseline."
        ),
    }
    (args.output_dir / "thresholds.json").write_text(
        json.dumps(results["validation_calibrated_thresholds"], indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
