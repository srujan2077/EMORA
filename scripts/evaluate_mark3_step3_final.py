"""Locked final evaluation for the Mark 3 Step 3 split-at-gap ensemble.

The default invocation is validation-only preflight. Test data is loaded only
when --evaluate-test is supplied, after validation thresholds are frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing the controlled implementation keeps inference preprocessing
# identical to training, including the split-at-gap suffix transform.
from scripts.train_mark3_step3_controlled import (  # noqa: E402
    CandidateDataset,
    DATA_DIR,
    EMOTION_NAMES,
    MAX_CONTEXT,
    collate_windows,
    calibrate_thresholds,
)
from src.models.temporal_fusion_mark3 import TemporalFusionMark3  # noqa: E402


SEEDS = (42, 123, 456, 789, 2026)
CANDIDATE = "split_at_gap"
BATCH_SIZE = 32
NUM_WORKERS = 0
EXPECTED_MODEL_CONFIG = {
    "input_dim": 768,
    "hidden_dim": 256,
    "num_heads": 8,
    "num_layers": 2,
    "num_emotions": 6,
    "max_sequence_length": MAX_CONTEXT,
    "dropout": 0.15,
}
EXPERIMENT_ROOT = (
    ROOT / "checkpoints" / "mark3_step3" / "multiseed_20261027"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="Root containing seed_<seed>/split_at_gap checkpoints.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing train/val/test_windows.pt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON report path; required with --evaluate-test.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Load and evaluate test data after validation calibration.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. Defaults to CUDA when available.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def checkpoint_paths(root: Path) -> list[Path]:
    paths = [
        root / f"seed_{seed}" / CANDIDATE / "best.pt"
        for seed in SEEDS
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ensemble checkpoints: {missing}")
    return paths


def load_models(
    paths: list[Path], device: torch.device
) -> tuple[list[torch.nn.Module], list[dict]]:
    models = []
    headers = []
    for path, seed in zip(paths, SEEDS):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        required = {"model_state_dict", "config", "candidate", "seed"}
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError(f"{path} is missing checkpoint keys: {missing}")
        if checkpoint["candidate"] != CANDIDATE or checkpoint["seed"] != seed:
            raise ValueError(f"Checkpoint identity mismatch: {path}")
        if checkpoint["config"] != EXPECTED_MODEL_CONFIG:
            raise ValueError(f"Model configuration mismatch: {path}")
        if checkpoint.get("deterministic_algorithms") is not True:
            raise ValueError(
                f"Checkpoint {path} does not have deterministic_algorithms=True."
            )
        model = TemporalFusionMark3(**checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        models.append(model)
        headers.append(
            {
                "path": str(path.resolve().relative_to(ROOT)),
                "seed": seed,
                "candidate": checkpoint["candidate"],
                "epoch": int(checkpoint["epoch"]),
                "val_macro_f1_at_0_5": float(
                    checkpoint["val_macro_f1_at_0_5"]
                ),
                "config": checkpoint["config"],
                "deterministic_algorithms": checkpoint.get(
                    "deterministic_algorithms"
                ),
            }
        )
    return models, headers


def validate_split_isolation(
    train_dataset: CandidateDataset, validation_dataset: CandidateDataset
) -> dict:
    train_videos = {str(sample["video_id"]) for sample in train_dataset.samples}
    validation_videos = {
        str(sample["video_id"]) for sample in validation_dataset.samples
    }
    overlap = sorted(train_videos & validation_videos)
    if overlap:
        raise ValueError(f"Train/validation video overlap: {overlap[:10]}")
    return {
        "train_videos": len(train_videos),
        "validation_videos": len(validation_videos),
        "train_validation_overlap_count": len(overlap),
    }


def loader_for(dataset: CandidateDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_windows,
        pin_memory=torch.cuda.is_available(),
    )


def validate_input_embeddings(dataset: CandidateDataset) -> None:
    for index, sample in enumerate(dataset.samples):
        lengths = []
        for modality in ("text_embedding", "audio_embedding", "video_embedding"):
            tensor = sample[modality]
            if tensor.ndim != 2 or tensor.shape[1] != 768:
                raise ValueError(
                    f"Invalid {modality} shape at sample {index}: "
                    f"expected [T, 768], got {tuple(tensor.shape)}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"Non-finite values in {modality} at sample {index}."
                )
            lengths.append(tensor.shape[0])
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Inconsistent modality sequence lengths at sample {index}: "
                f"{lengths}"
            )


@torch.no_grad()
def ensemble_predictions(
    models: list[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    labels_all = []
    ensemble_probability_batches = []
    for batch in loader:
        labels = batch["labels"].cpu().numpy()
        model_probabilities = []
        for model in models:
            logits, _ = model(
                text=batch["text"].to(device),
                audio=batch["audio"].to(device),
                video=batch["video"].to(device),
                lengths=batch["lengths"].to(device),
                segment_numbers=batch["segment_numbers"].to(device),
            )
            probabilities = torch.sigmoid(logits).cpu().numpy()
            if probabilities.shape != (len(labels), len(EMOTION_NAMES)):
                raise ValueError(
                    f"Unexpected prediction shape: {probabilities.shape}"
                )
            if not np.isfinite(probabilities).all():
                raise ValueError("Non-finite ensemble member predictions.")
            model_probabilities.append(probabilities)
        averaged = np.mean(np.stack(model_probabilities, axis=0), axis=0)
        if not np.isfinite(averaged).all() or np.any(
            (averaged < 0) | (averaged > 1)
        ):
            raise ValueError("Invalid averaged probabilities.")
        labels_all.append(labels)
        ensemble_probability_batches.append(averaged)
    if not labels_all:
        raise ValueError("No batches were processed.")
    labels = np.concatenate(labels_all, axis=0)
    probabilities = np.concatenate(ensemble_probability_batches, axis=0)
    if labels.shape != probabilities.shape or labels.shape[1] != 6:
        raise ValueError(
            f"Label/prediction shape mismatch: {labels.shape} vs "
            f"{probabilities.shape}"
        )
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be binary.")
    return labels.astype(np.int32), probabilities


def metric_report(
    labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> dict:
    predictions = (probabilities >= thresholds).astype(np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        average=None,
        zero_division=0,
    )
    per_emotion = {}
    confusion = {}
    for index, name in enumerate(EMOTION_NAMES):
        tn, fp, fn, tp = confusion_matrix(
            labels[:, index], predictions[:, index], labels=(0, 1)
        ).ravel()
        per_emotion[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        confusion[name] = {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    return {
        "thresholds": {
            name: float(thresholds[index])
            for index, name in enumerate(EMOTION_NAMES)
        },
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(labels, predictions, average="micro", zero_division=0)
        ),
        "subset_accuracy": float(np.all(labels == predictions, axis=1).mean()),
        "per_emotion": per_emotion,
        "multilabel_confusion": confusion,
        "metric_definitions": {
            "macro_f1": "Unweighted mean of six emotion-level F1 scores.",
            "micro_f1": "Global F1 after pooling all emotion decisions.",
            "subset_accuracy": "Fraction of windows with all six labels correct.",
            "threshold_rule": "Predict positive when probability >= threshold.",
        },
    }


def finite_thresholds(thresholds: np.ndarray) -> None:
    if thresholds.shape != (6,) or not np.isfinite(thresholds).all():
        raise ValueError("Thresholds must be six finite values.")
    if np.any((thresholds < 0.05) | (thresholds > 0.95)):
        raise ValueError("Thresholds are outside the calibration search range.")


def main() -> None:
    args = parse_args()
    if not args.evaluate_test and args.output is not None:
        raise ValueError("--output is only valid with --evaluate-test.")
    if args.evaluate_test and args.output is None:
        raise ValueError("--output is required with --evaluate-test.")
    output = args.output.resolve() if args.output is not None else None
    if output is not None and output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    experiment_root = args.experiment_root.resolve()
    data_dir = args.data_dir.resolve()
    device = resolve_device(args.device)
    paths = checkpoint_paths(experiment_root)
    models, checkpoint_metadata = load_models(paths, device)

    mode = "final test evaluation" if args.evaluate_test else "validation preflight"
    print(f"LOCKED PROTOCOL: five-model split_at_gap ensemble ({mode})")
    print(f"Device: {device}")

    train_dataset = CandidateDataset(data_dir / "train_windows.pt", CANDIDATE)
    validation_dataset = CandidateDataset(
        data_dir / "val_windows.pt", CANDIDATE
    )
    validate_input_embeddings(train_dataset)
    validate_input_embeddings(validation_dataset)
    split_integrity = validate_split_isolation(train_dataset, validation_dataset)
    labels, probabilities = ensemble_predictions(
        models, loader_for(validation_dataset), device
    )
    thresholds = calibrate_thresholds(labels, probabilities).astype(np.float32)
    finite_thresholds(thresholds)
    validation_metrics = {
        "fixed_0_5": metric_report(
            labels, probabilities, np.full(6, 0.5, dtype=np.float32)
        ),
        "validation_calibrated": metric_report(labels, probabilities, thresholds),
    }

    if not args.evaluate_test:
        print(
            f"Validation windows: {len(validation_dataset)}; "
            "thresholds calibrated from averaged validation probabilities."
        )
        print("Preflight passed. Test data was not loaded.")
        return

    test_dataset = CandidateDataset(data_dir / "test_windows.pt", CANDIDATE)
    validate_input_embeddings(test_dataset)
    train_videos = {str(sample["video_id"]) for sample in train_dataset.samples}
    validation_videos = {
        str(sample["video_id"]) for sample in validation_dataset.samples
    }
    test_videos = {str(sample["video_id"]) for sample in test_dataset.samples}
    if train_videos & test_videos or validation_videos & test_videos:
        raise ValueError("Video leakage detected in test split.")
    test_labels, test_probabilities = ensemble_predictions(
        models, loader_for(test_dataset), device
    )
    report = {
        "protocol": {
            "name": "Mark 3 Step 3 locked final test evaluation",
            "candidate": CANDIDATE,
            "ensemble": "Arithmetic mean of five checkpoint probabilities.",
            "threshold_calibration": (
                "Per-emotion F1 grid search on averaged validation probabilities "
                "only; thresholds are frozen before test inference."
            ),
            "test_evaluated": True,
            "test_used_for_selection_or_tuning": False,
        },
        "checkpoint_metadata": checkpoint_metadata,
        "model_config": EXPECTED_MODEL_CONFIG,
        "emotion_order": EMOTION_NAMES,
        "dataset": {
            "data_dir": str(data_dir),
            "train_windows": len(train_dataset),
            "validation_windows": len(validation_dataset),
            "test_windows": len(test_dataset),
            "split_integrity": {
                **split_integrity,
                "test_videos": len(test_videos),
                "validation_test_overlap_count": len(
                    validation_videos & test_videos
                ),
                "train_test_overlap_count": len(train_videos & test_videos),
            },
        },
        "validation": validation_metrics,
        "test": {
            "fixed_0_5": metric_report(
                test_labels,
                test_probabilities,
                np.full(6, 0.5, dtype=np.float32),
            ),
            "validation_calibrated": metric_report(
                test_labels, test_probabilities, thresholds
            ),
        },
    }
    if output is None:
        raise RuntimeError("Internal error: test output path was not resolved.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Test evaluation executed: yes")
    print(f"Report written to: {output}")


if __name__ == "__main__":
    main()
