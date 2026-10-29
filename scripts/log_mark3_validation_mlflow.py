"""Log the existing Mark 3 validation preflight to the local MLflow store.

This script is deliberately validation-only. It rejects test flags and paths,
never constructs a test dataset, and does not alter model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_mark3_step3_final import (  # noqa: E402
    CANDIDATE,
    EMOTION_NAMES,
    EXPECTED_MODEL_CONFIG,
    EXPERIMENT_ROOT,
    SEEDS,
    ensemble_predictions,
    finite_thresholds,
    load_models,
    loader_for,
    metric_report,
    resolve_device,
    validate_input_embeddings,
    validate_split_isolation,
)
from scripts.train_mark3_step3_controlled import CandidateDataset  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--tracking-uri",
        default=f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}",
    )
    parser.add_argument("--experiment", default="EMORA-Mark3-Validation")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed" / "mark3_temporal")
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--evaluate-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def reject_test_access(args: argparse.Namespace) -> None:
    if args.evaluate_test:
        raise ValueError("This MLflow wrapper is validation-only; --evaluate-test is forbidden.")
    for path in (args.data_dir, args.experiment_root):
        if "test" in str(path).lower():
            raise ValueError(f"Test-named path is forbidden: {path}")


def main() -> None:
    args = parse_args()
    reject_test_access(args)
    data_dir = args.data_dir.resolve()
    experiment_root = args.experiment_root.resolve()
    device = resolve_device(args.device)
    val_path = data_dir / "val_windows.pt"
    if not val_path.is_file():
        raise FileNotFoundError(val_path)
    checkpoint_paths = [
        experiment_root / f"seed_{seed}" / CANDIDATE / "best.pt"
        for seed in SEEDS
    ]
    models, checkpoint_metadata = load_models(checkpoint_paths, device)
    train_dataset = CandidateDataset(data_dir / "train_windows.pt", CANDIDATE)
    validation_dataset = CandidateDataset(val_path, CANDIDATE)
    validate_input_embeddings(train_dataset)
    validate_input_embeddings(validation_dataset)
    split_integrity = validate_split_isolation(train_dataset, validation_dataset)
    labels, probabilities = ensemble_predictions(
        models, loader_for(validation_dataset), device
    )
    thresholds = np.full(len(EMOTION_NAMES), 0.5, dtype=np.float32)
    finite_thresholds(thresholds)
    metrics = metric_report(labels, probabilities, thresholds)
    report = {
        "protocol": "Mark 3 validation-only MLflow tracking",
        "test_split_accessed": False,
        "test_split_loaded": False,
        "test_metrics_logged": False,
        "emotion_order": list(EMOTION_NAMES),
        "metrics": metrics,
        "dataset": {
            "validation_windows": len(validation_dataset),
            "train_windows": len(train_dataset),
            "data_dir": str(data_dir),
            "split_integrity": split_integrity,
        },
        "checkpoint_metadata": checkpoint_metadata,
    }
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="mark3-validation-fixed-0.5") as run:
        mlflow.log_params(
            {
                "candidate": CANDIDATE,
                "ensemble_seeds": ",".join(map(str, SEEDS)),
                "ensemble_size": len(SEEDS),
                "threshold_policy": "fixed_0.5",
                "device": str(device),
                "emotion_order": ",".join(EMOTION_NAMES),
                "validation_windows": len(validation_dataset),
                "git_commit": git_value("rev-parse", "HEAD"),
                "evaluator_sha256": sha256(ROOT / "scripts" / "evaluate_mark3_step3_final.py"),
                "validation_artifact_sha256": sha256(val_path),
                "checkpoint_hashes": json.dumps(
                    {
                        str(path.relative_to(ROOT)): sha256(path)
                        for path in checkpoint_paths
                    },
                    sort_keys=True,
                ),
            }
        )
        mlflow.log_metrics(
            {
                "validation_macro_f1": metrics["macro_f1"],
                "validation_micro_f1": metrics["micro_f1"],
                "validation_subset_accuracy": metrics["subset_accuracy"],
            }
        )
        for name, result in metrics["per_emotion"].items():
            mlflow.log_metric(f"validation_{name}_f1", result["f1"])
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "mark3_validation_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="validation")
        print(
            json.dumps(
                {
                    "run_id": run.info.run_id,
                    "experiment": args.experiment,
                    "metrics": metrics,
                    "test_split_accessed": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
