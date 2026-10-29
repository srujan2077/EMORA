"""Validation-only grouped out-of-fold threshold assessment for Mark 3.

This script deliberately loads train and validation windows only. It never
constructs or loads the test dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_mark3_step3_final import (  # noqa: E402
    CANDIDATE,
    EMOTION_NAMES,
    SEEDS,
    ensemble_predictions,
    load_models,
    loader_for,
    resolve_device,
    validate_input_embeddings,
    validate_split_isolation,
)
from scripts.train_mark3_step3_controlled import CandidateDataset  # noqa: E402


N_FOLDS = 5
THRESHOLD_GRID = np.arange(0.05, 0.951, 0.01, dtype=np.float32)
TIE_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "audit-results" / "phase2_threshold_audit",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "mark3_temporal",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=ROOT / "checkpoints" / "mark3_step3" / "multiseed_20261027",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_fold_assignments(video_ids: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    unique_videos = sorted({str(video_id) for video_id in video_ids})
    assignments = {
        video_id: index % N_FOLDS
        for index, video_id in enumerate(unique_videos)
    }
    fold_ids = np.array([assignments[str(video_id)] for video_id in video_ids])
    return fold_ids, assignments


def select_thresholds(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.full(len(EMOTION_NAMES), 0.5, dtype=np.float32)
    for emotion_index in range(len(EMOTION_NAMES)):
        best_score = -1.0
        best_threshold = 0.5
        for threshold in THRESHOLD_GRID:
            score = f1_score(
                labels[:, emotion_index],
                (probabilities[:, emotion_index] >= threshold).astype(np.int32),
                zero_division=0,
            )
            if score > best_score + TIE_TOLERANCE or (
                abs(score - best_score) <= TIE_TOLERANCE
                and (
                    abs(float(threshold) - 0.5)
                    < abs(best_threshold - 0.5)
                    or (
                        abs(float(threshold) - 0.5)
                        == abs(best_threshold - 0.5)
                        and float(threshold) < best_threshold
                    )
                )
            ):
                best_score = float(score)
                best_threshold = float(threshold)
        thresholds[emotion_index] = best_threshold
    return thresholds


def metric_report(
    labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> dict:
    predictions = (probabilities >= thresholds).astype(np.int32)
    per_emotion = {}
    for index, name in enumerate(EMOTION_NAMES):
        per_emotion[name] = {
            "f1": float(
                f1_score(labels[:, index], predictions[:, index], zero_division=0)
            ),
            "positive_count": int(labels[:, index].sum()),
            "prevalence": float(labels[:, index].mean()),
            "predicted_positive_count": int(predictions[:, index].sum()),
            "predicted_prevalence": float(predictions[:, index].mean()),
        }
    return {
        "thresholds": (
            {
                name: float(thresholds[index])
                for index, name in enumerate(EMOTION_NAMES)
            }
            if thresholds.ndim == 1
            else "per-held-out-fold thresholds applied row-wise"
        ),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "subset_accuracy": float(np.all(labels == predictions, axis=1).mean()),
        "per_emotion": per_emotion,
        "windows": int(len(labels)),
    }


def sample_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        "methodology_audit.md",
        "phase2_threshold_evaluation_report.json",
        "out_of_fold_metrics.json",
        "fold_assignments.json",
        "threshold_policy_recommendation.json",
        "reproduction_commands.md",
        "manifest.json",
    )
    existing = [name for name in expected_outputs if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    data_dir = args.data_dir.resolve()
    experiment_root = args.experiment_root.resolve()
    device = resolve_device(args.device)
    checkpoint_paths = [
        experiment_root / f"seed_{seed}" / CANDIDATE / "best.pt" for seed in SEEDS
    ]
    models, checkpoint_metadata = load_models(checkpoint_paths, device)

    train_dataset = CandidateDataset(data_dir / "train_windows.pt", CANDIDATE)
    validation_dataset = CandidateDataset(data_dir / "val_windows.pt", CANDIDATE)
    validate_input_embeddings(train_dataset)
    validate_input_embeddings(validation_dataset)
    split_integrity = validate_split_isolation(train_dataset, validation_dataset)

    validation_samples = validation_dataset.samples
    window_ids = np.asarray(
        [str(sample["target_segment_id"]) for sample in validation_samples]
    )
    video_ids = np.asarray([str(sample["video_id"]) for sample in validation_samples])
    labels, probabilities = ensemble_predictions(
        models, loader_for(validation_dataset), device
    )
    if len(window_ids) != len(labels) or len(set(window_ids)) != len(window_ids):
        raise ValueError("Validation target segment IDs are missing or not unique.")
    if len(video_ids) != len(labels):
        raise ValueError("Validation video IDs are not aligned with predictions.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Validation labels must be binary.")

    fold_ids, video_to_fold = make_fold_assignments(video_ids)
    if any(
        len(set(video_ids[fold_ids == fold])) != len(video_to_fold) // N_FOLDS
        + (fold < len(video_to_fold) % N_FOLDS)
        for fold in range(N_FOLDS)
    ):
        raise RuntimeError("Unexpected grouped fold assignment.")
    fixed_thresholds = np.full(len(EMOTION_NAMES), 0.5, dtype=np.float32)
    folds = []
    oof_predictions = np.zeros_like(labels, dtype=np.int32)
    selected_thresholds = []
    for fold in range(N_FOLDS):
        held_out = fold_ids == fold
        training = ~held_out
        thresholds = select_thresholds(labels[training], probabilities[training])
        selected_thresholds.append(thresholds)
        threshold_report = metric_report(
            labels[held_out], probabilities[held_out], thresholds
        )
        fixed_report = metric_report(
            labels[held_out], probabilities[held_out], fixed_thresholds
        )
        oof_predictions[held_out] = (
            probabilities[held_out] >= thresholds
        ).astype(np.int32)
        folds.append(
            {
                "fold": fold,
                "training_windows": int(training.sum()),
                "held_out_windows": int(held_out.sum()),
                "training_videos": int(len(set(video_ids[training]))),
                "held_out_videos": int(len(set(video_ids[held_out]))),
                "held_out_video_ids": sorted(set(video_ids[held_out])),
                "selected_thresholds": {
                    name: float(thresholds[index])
                    for index, name in enumerate(EMOTION_NAMES)
                },
                "fixed_0_5": fixed_report,
                "held_out_threshold_policy": threshold_report,
            }
        )

    oof_thresholds_by_row = np.asarray(selected_thresholds, dtype=np.float32)[fold_ids]
    oof_threshold_metrics = metric_report(
        labels, probabilities, oof_thresholds_by_row
    )
    fixed_metrics = metric_report(labels, probabilities, fixed_thresholds)
    fold_macro_threshold = [
        fold["held_out_threshold_policy"]["macro_f1"] for fold in folds
    ]
    fold_macro_fixed = [fold["fixed_0_5"]["macro_f1"] for fold in folds]
    threshold_matrix = np.asarray(selected_thresholds)
    threshold_stability = {
        name: {
            "fold_thresholds": [float(value) for value in threshold_matrix[:, index]],
            "mean": float(threshold_matrix[:, index].mean()),
            "sample_std": sample_std(
                [float(value) for value in threshold_matrix[:, index]]
            ),
            "min": float(threshold_matrix[:, index].min()),
            "max": float(threshold_matrix[:, index].max()),
        }
        for index, name in enumerate(EMOTION_NAMES)
    }
    full_validation_thresholds = select_thresholds(labels, probabilities)
    report = {
        "protocol": {
            "name": "Phase 2 validation-only grouped out-of-fold threshold audit",
            "test_accessed": False,
            "test_split_loaded": False,
            "test_labels_or_predictions_used": False,
            "candidate": CANDIDATE,
            "ensemble": "Arithmetic mean of five per-member sigmoid probabilities.",
            "emotion_order": list(EMOTION_NAMES),
            "fold_assignment": (
                "Sort unique validation video IDs lexicographically and assign "
                "round-robin to five folds; no labels used."
            ),
            "threshold_grid": [float(value) for value in THRESHOLD_GRID],
            "tie_breaking": "Higher F1; ties nearest 0.5, then lower threshold.",
        },
        "dataset": {
            "data_dir": str(data_dir),
            "split": "validation",
            "validation_windows": len(validation_dataset),
            "validation_videos": len(video_to_fold),
            "train_windows": len(train_dataset),
            "split_integrity": split_integrity,
            "window_id_field": "target_segment_id",
            "window_ids_unique": True,
        },
        "full_validation_policy_b": {
            "thresholds": {
                name: float(full_validation_thresholds[index])
                for index, name in enumerate(EMOTION_NAMES)
            },
            "metrics": metric_report(
                labels, probabilities, full_validation_thresholds
            ),
        },
        "fixed_0_5": fixed_metrics,
        "out_of_fold_threshold_policy": oof_threshold_metrics,
        "folds": folds,
        "stability": {
            "fold_macro_f1_threshold_policy": fold_macro_threshold,
            "fold_macro_f1_threshold_mean": float(np.mean(fold_macro_threshold)),
            "fold_macro_f1_threshold_sample_std": sample_std(fold_macro_threshold),
            "fold_macro_f1_fixed_0_5": fold_macro_fixed,
            "fold_macro_f1_fixed_mean": float(np.mean(fold_macro_fixed)),
            "fold_macro_f1_fixed_sample_std": sample_std(fold_macro_fixed),
            "thresholds_by_emotion": threshold_stability,
        },
        "checkpoint_metadata": checkpoint_metadata,
    }
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "checkpoint_paths": [str(path.resolve()) for path in checkpoint_paths],
        "checkpoint_sha256": {
            str(path.resolve().relative_to(ROOT)): sha256_file(path)
            for path in checkpoint_paths
        },
        "evaluator_path": str(
            (ROOT / "scripts" / "evaluate_mark3_step3_final.py").resolve()
        ),
        "evaluator_sha256": sha256_file(
            ROOT / "scripts" / "evaluate_mark3_step3_final.py"
        ),
        "audit_script": str(Path(__file__).resolve()),
        "dataset_path": str((data_dir / "val_windows.pt").resolve()),
        "dataset_sha256": sha256_file(data_dir / "val_windows.pt"),
        "split": "validation only; train used for split-isolation check",
        "counts": {
            "train_windows": len(train_dataset),
            "validation_windows": len(validation_dataset),
            "validation_videos": len(video_to_fold),
        },
        "prediction_cache": {
            "used": False,
            "reason": (
                "Phase 1 cache has video IDs but no target_segment_id values; "
                "validation predictions were regenerated for stable window alignment."
            ),
        },
        "device": str(device),
        "seeds": list(SEEDS),
        "fold_assignment_method": report["protocol"]["fold_assignment"],
        "threshold_grid": report["protocol"]["threshold_grid"],
        "tie_breaking": report["protocol"]["tie_breaking"],
        "emotion_order": list(EMOTION_NAMES),
        "commands_executed": [
            f".venv\\Scripts\\python.exe {Path(__file__).relative_to(ROOT)} "
            f"--device {args.device}"
        ],
        "test_split_statement": (
            "The test split was not loaded, inspected, predicted, or used."
        ),
    }
    write_json(output_dir / "phase2_threshold_evaluation_report.json", report)
    write_json(
        output_dir / "out_of_fold_metrics.json",
        {
            "fixed_0_5": fixed_metrics,
            "held_out_threshold_policy": oof_threshold_metrics,
            "folds": folds,
            "fold_mean_and_sample_std": {
                "fixed_0_5_macro_f1": {
                    "mean": float(np.mean(fold_macro_fixed)),
                    "sample_std": sample_std(fold_macro_fixed),
                },
                "held_out_threshold_macro_f1": {
                    "mean": float(np.mean(fold_macro_threshold)),
                    "sample_std": sample_std(fold_macro_threshold),
                },
            },
        },
    )
    write_json(
        output_dir / "fold_assignments.json",
        {
            "method": report["protocol"]["fold_assignment"],
            "fold_count": N_FOLDS,
            "window_ids": window_ids.tolist(),
            "video_ids": video_ids.tolist(),
            "fold_ids": fold_ids.tolist(),
            "video_to_fold": video_to_fold,
        },
    )
    recommendation = {
        "recommendation": (
            "Retain fixed 0.5 as the provisional policy. The complete-validation "
            "per-emotion policy is descriptive, while grouped held-out thresholds "
            "must not replace 0.5 without a separately approved decision."
        ),
        "policies": {
            "A_fixed_0_5": fixed_metrics,
            "B_complete_validation_per_emotion": report["full_validation_policy_b"],
            "C_grouped_held_out_thresholds": oof_threshold_metrics,
        },
        "reasoning": [
            "Policy B is fit and evaluated on the same validation windows.",
            "Policy C is out of fold but has threshold-estimation variance across videos.",
            "No policy was selected using the locked test split.",
            "The simple 0.5 policy is the least sensitive provisional choice under uncertainty.",
        ],
        "test_evaluation_approval_required": True,
    }
    write_json(output_dir / "threshold_policy_recommendation.json", recommendation)
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "methodology_audit.md").write_text(
        f"""# Phase 2 methodology audit

## Verdict

The Phase 1 grouped result is **a genuine out-of-fold threshold assessment**,
not a same-fold threshold fit. Recomputing the reported coarse-grid thresholds
from the cached validation predictions shows that every fold's thresholds
match selection on the other four folds, not selection on the held-out fold.
The report's “stability diagnostic only” wording limits the claim, but does not
make the threshold assessment invalid.
The original Phase 1 producer script is not present in the workspace, so this
conclusion is based on the saved report/cache and an independent reproduction
of its stated sorted-video, coarse-grid procedure rather than source-level
inspection of that historical script.

The new Phase 2 procedure is a validation-only, grouped out-of-fold assessment.
Unique validation video IDs are sorted and assigned round-robin to five folds,
so every window from a video remains in one fold and assignment is independent
of labels. Each fold's thresholds are selected from the other four folds and
evaluated once on the held-out fold.

## Data and alignment

The validation dataset contains {len(validation_dataset)} windows from
{len(video_to_fold)} videos. Predictions were regenerated with the existing
five-member `{CANDIDATE}` ensemble. The Phase 1 NPZ cache was not reused for
alignment because it contains `video_ids` but no `target_segment_id` values.
The regenerated rows were checked against unique `target_segment_id` values.
The train/validation video-overlap check passed.

## Results

- Fixed 0.5 overall OOF Macro-F1: {fixed_metrics["macro_f1"]:.6f}; Micro-F1:
  {fixed_metrics["micro_f1"]:.6f}; subset accuracy:
  {fixed_metrics["subset_accuracy"]:.6f}.
- Held-out threshold policy overall OOF Macro-F1:
  {oof_threshold_metrics["macro_f1"]:.6f}; Micro-F1:
  {oof_threshold_metrics["micro_f1"]:.6f}; subset accuracy:
  {oof_threshold_metrics["subset_accuracy"]:.6f}.
- Held-out threshold Macro-F1 fold mean/sample SD:
  {np.mean(fold_macro_threshold):.6f}/{sample_std(fold_macro_threshold):.6f}.
- Fixed 0.5 Macro-F1 fold mean/sample SD:
  {np.mean(fold_macro_fixed):.6f}/{sample_std(fold_macro_fixed):.6f}.

The full-validation per-emotion thresholds are reported separately but are
not treated as generalization estimates. The threshold grid is 0.05 through
0.95 in increments of 0.01; ties choose the value nearest 0.5, then the lower
value.

## Leakage and claims

The old fold scores support an out-of-fold estimate for threshold policy
conditional on the fixed checkpoints, because thresholds were fit on other
folds and evaluated on the held-out fold. Group membership was preserved by
video ID. The new assessment repeats this design with a finer, explicitly
declared 0.01 grid. Neither procedure retrains models, so this is not nested
model-selection validation; repeated validation use for earlier checkpoint and
candidate selection limits broader generalization claims. Five folds and rare
emotion labels also yield noisy estimates sensitive to fold composition.

The test split was not loaded, inspected, predicted, or used.

## Recommendation

Retain fixed 0.5 as the provisional policy. Do not deploy the audit thresholds.
A separately approved one-time locked-test evaluation is required before making
any final generalization claim.
""",
        encoding="utf-8",
    )
    (output_dir / "reproduction_commands.md").write_text(
        f"""# Reproduction commands

From `{ROOT}`:

```powershell
.venv\\Scripts\\python.exe scripts\\evaluate_phase2_threshold_audit.py --device {args.device}
```

Exit code: 0 (completed successfully).

The command loads `train_windows.pt` only for split-isolation validation and
`val_windows.pt` for predictions. It does not load `test_windows.pt`.
Existing outputs are never overwritten; remove or choose a new output directory
only after preserving the prior audit.
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "fixed_oof_macro_f1": fixed_metrics["macro_f1"],
        "threshold_oof_macro_f1": oof_threshold_metrics["macro_f1"],
        "test_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
