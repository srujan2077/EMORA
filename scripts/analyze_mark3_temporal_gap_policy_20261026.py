"""Validation-only Mark 3 temporal evaluation and gap-policy analysis."""

from collections import Counter
from pathlib import Path
import json
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data" / "processed" / "mark3_temporal"
CHECKPOINT_DIR = ROOT / "checkpoints" / "mark3_temporal"
EMOTION_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
SPLITS_FOR_POLICY = ("train", "val")

from scripts.train_mark3_temporal import (  # noqa: E402
    TemporalWindowDataset,
    collate_windows,
)
from src.models.temporal_fusion_mark3 import TemporalFusionMark3  # noqa: E402


def calculate_metrics(labels, probabilities, thresholds):
    predictions = (probabilities >= thresholds).astype(np.int32)
    return {
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(labels, predictions, average="micro", zero_division=0)
        ),
        "per_emotion_f1": {
            name: float(
                f1_score(
                    labels[:, index],
                    predictions[:, index],
                    zero_division=0,
                )
            )
            for index, name in enumerate(EMOTION_NAMES)
        },
    }


@torch.no_grad()
def collect_predictions(model, loader, device, target_only=False):
    model.eval()
    labels_all = []
    probabilities_all = []

    for batch in loader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        lengths = batch["lengths"].to(device)
        segment_numbers = batch["segment_numbers"].to(device)

        if target_only:
            last = lengths - 1
            rows = torch.arange(len(lengths), device=device)
            text = text[rows, last].unsqueeze(1)
            audio = audio[rows, last].unsqueeze(1)
            video = video[rows, last].unsqueeze(1)
            segment_numbers = segment_numbers[rows, last].unsqueeze(1)
            lengths = torch.ones_like(lengths)

        logits, _ = model(
            text=text,
            audio=audio,
            video=video,
            lengths=lengths,
            segment_numbers=segment_numbers,
        )
        labels_all.append(batch["labels"].numpy())
        probabilities_all.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(labels_all), np.concatenate(probabilities_all)


def gap_sizes(numbers):
    return [
        current - previous - 1
        for previous, current in zip(numbers, numbers[1:])
        if current - previous > 1
    ]


def contiguous_suffix_length(numbers):
    """Return the available contiguous suffix length ending at the target."""
    length = 1
    for previous, current in zip(reversed(numbers[:-1]), reversed(numbers[1:])):
        if current != previous + 1:
            break
        length += 1
    return length


def policy_distribution(path):
    samples = torch.load(path, map_location="cpu", weights_only=False)
    original_lengths = Counter()
    suffix_lengths = Counter()
    gap_status_counts = Counter()

    for sample in samples:
        numbers = [int(value) for value in sample["context_segment_numbers"]]
        original_lengths[str(len(numbers))] += 1
        suffix_lengths[str(contiguous_suffix_length(numbers))] += 1
        gap_status_counts["gap" if gap_sizes(numbers) else "no_gap"] += 1

    return {
        "samples": len(samples),
        "existing_context_length_distribution": dict(
            sorted(original_lengths.items(), key=lambda item: int(item[0]))
        ),
        "simulated_split_at_gap_length_distribution": dict(
            sorted(suffix_lengths.items(), key=lambda item: int(item[0]))
        ),
        "original_gap_status_counts": dict(sorted(gap_status_counts.items())),
    }


def subgroup_metrics(labels, probabilities, thresholds, indices):
    indices = np.asarray(indices)
    if len(indices) == 0:
        return {"sample_count": 0, "unstable_small_group": True}
    result = calculate_metrics(
        labels[indices],
        probabilities[indices],
        thresholds,
    )
    result["sample_count"] = int(len(indices))
    result["unstable_small_group"] = bool(len(indices) < 30)
    return result


def grouped_results(dataset, labels, probabilities, thresholds):
    lengths = np.array(
        [sample["context_length"] for sample in dataset.samples],
        dtype=np.int64,
    )
    has_gaps = np.array(
        [bool(sample["has_numbering_gap"]) for sample in dataset.samples],
        dtype=bool,
    )

    by_length = {}
    for length in range(1, 9):
        by_length[str(length)] = subgroup_metrics(
            labels,
            probabilities,
            thresholds,
            np.flatnonzero(lengths == length),
        )

    by_gap = {}
    for name, value in (("no_gap", False), ("gap", True)):
        by_gap[name] = subgroup_metrics(
            labels,
            probabilities,
            thresholds,
            np.flatnonzero(has_gaps == value),
        )
    return {
        "by_context_length": by_length,
        "by_gap_status": by_gap,
        "small_group_threshold": 30,
    }


def load_model(device):
    checkpoint_path = CHECKPOINT_DIR / "temporal_fusion_best.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = TemporalFusionMark3(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def main():
    checkpoint_path = CHECKPOINT_DIR / "temporal_fusion_best.pt"
    thresholds_path = CHECKPOINT_DIR / "temporal_fusion_thresholds.json"
    val_path = DATA_DIR / "val_windows.pt"
    for path in (checkpoint_path, thresholds_path, val_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required analysis input is missing: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(device)
    threshold_dict = json.loads(thresholds_path.read_text(encoding="utf-8"))
    thresholds = np.array(
        [threshold_dict[name] for name in EMOTION_NAMES],
        dtype=np.float32,
    )

    val_dataset = TemporalWindowDataset(val_path)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_windows,
    )
    labels_full, probabilities_full = collect_predictions(
        model, val_loader, device, target_only=False
    )
    labels_target, probabilities_target = collect_predictions(
        model, val_loader, device, target_only=True
    )
    if not np.array_equal(labels_full, labels_target):
        raise RuntimeError("Labels changed between evaluation conditions.")

    fixed_thresholds = np.full(6, 0.5, dtype=np.float32)
    conditions = {}
    for name, labels, probabilities in (
        ("full_context", labels_full, probabilities_full),
        ("target_segment_only_diagnostic", labels_target, probabilities_target),
    ):
        conditions[name] = {
            "threshold_0_5": calculate_metrics(
                labels, probabilities, fixed_thresholds
            ),
            "validation_calibrated_thresholds": calculate_metrics(
                labels, probabilities, thresholds
            ),
            "subgroups_at_threshold_0_5": grouped_results(
                val_dataset, labels, probabilities, fixed_thresholds
            ),
            "subgroups_at_validation_calibrated_thresholds": grouped_results(
                val_dataset, labels, probabilities, thresholds
            ),
        }

    report = {
        "description": (
            "Read-only validation evaluation and simulated split-at-gap "
            "context analysis"
        ),
        "evaluation_split": "validation",
        "test_data_loaded": False,
        "device": str(device),
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "epoch": checkpoint.get("epoch"),
            "checkpoint_val_macro_f1_at_0_5": checkpoint.get(
                "val_macro_f1_at_0_5"
            ),
        },
        "thresholds": {
            "path": str(thresholds_path.relative_to(ROOT)),
            "values": threshold_dict,
            "calibration_scope": (
                "Full validation set; not recalibrated by context length "
                "or gap-status bucket."
            ),
        },
        "conditions": conditions,
        "target_only_interpretation": (
            "Diagnostic only: this input ablation uses a model trained with "
            "full context and is not an independently trained baseline."
        ),
        "gap_policy_simulation": {
            "definition": (
                "For each target, retain the longest contiguous suffix "
                "ending at that target, using original segment numbers. "
                "No renumbering or fabricated embeddings."
            ),
            "note": (
                "This is a distribution-only simulation. No model was "
                "trained or evaluated on simulated contexts."
            ),
            "splits": {
                split: policy_distribution(
                    DATA_DIR / f"{split}_windows.pt"
                )
                for split in SPLITS_FOR_POLICY
            },
        },
        "recommendation": (
            "In Step 3, compare the existing metadata-aware policy against "
            "split-at-gap contexts using identical training seeds, model "
            "configuration, and validation-only checkpoint selection. "
            "Do not treat the simulated policy as superior before that "
            "controlled experiment."
        ),
    }

    output = ROOT / "reports" / "mark3_temporal_gap_policy_analysis_20261026_190208.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nReport written to {output}")


if __name__ == "__main__":
    main()
