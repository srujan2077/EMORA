
"""EMORA Mark 3: controlled experiments on cached pooled features.

Models:
    mark3a_concat: concatenation MLP baseline
    mark3b_gated: sample-dependent gated multimodal fusion

The test set is used only after checkpoint selection and threshold tuning.
Existing Mark 2 artifacts are never overwritten.
"""

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fusion_mark3 import (
    Mark3ConcatFusion,
    Mark3GatedFusion,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "mark2" / "features"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "mark3"

EMOTION_NAMES = [
    "happy", "sad", "anger", "surprise", "disgust", "fear"
]

SEED = 42
BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 7
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Cached feature loading
# ------------------------------------------------------------

class CachedShardDataset(Dataset):
    """Load every cached shard for one split into CPU memory."""

    def __init__(self, split: str):
        self.split = split
        paths = sorted(FEATURE_DIR.glob(f"{split}_shard_*.pt"))

        if not paths:
            raise FileNotFoundError(
                f"No {split} shards found in {FEATURE_DIR}"
            )

        parts = {
            "text": [],
            "audio": [],
            "video": [],
            "labels": [],
        }

        expected_keys = {
            "text_embedding": "text",
            "audio_embedding": "audio",
            "video_embedding": "video",
            "labels": "labels",
        }

        for path in paths:
            shard = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            for source_key, target_key in expected_keys.items():
                if source_key not in shard:
                    raise KeyError(
                        f"{path.name} is missing {source_key}"
                    )

                tensor = shard[source_key].float()

                if not torch.isfinite(tensor).all():
                    raise ValueError(
                        f"Non-finite values found in "
                        f"{path.name}:{source_key}"
                    )

                parts[target_key].append(tensor)

        self.text = torch.cat(parts["text"], dim=0)
        self.audio = torch.cat(parts["audio"], dim=0)
        self.video = torch.cat(parts["video"], dim=0)
        self.labels = torch.cat(parts["labels"], dim=0)

        n = self.labels.shape[0]
        if not (
            self.text.shape[0]
            == self.audio.shape[0]
            == self.video.shape[0]
            == n
        ):
            raise ValueError(
                f"Inconsistent tensor lengths in {split} split"
            )

        if self.labels.ndim != 2 or self.labels.shape[1] != 6:
            raise ValueError(
                f"Expected labels shaped (N, 6), got "
                f"{tuple(self.labels.shape)}"
            )

        if not torch.all((self.labels == 0) | (self.labels == 1)):
            raise ValueError(f"{split} labels must be binary")

        print(
            f"Loaded {split}: {n:,} samples from "
            f"{len(paths)} shards"
        )

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return {
            "text": self.text[index],
            "audio": self.audio[index],
            "video": self.video[index],
            "labels": self.labels[index],
        }


# ------------------------------------------------------------
# Metrics and threshold calibration
# ------------------------------------------------------------

def metrics_at_thresholds(logits, labels, thresholds):
    probabilities = torch.sigmoid(logits).cpu().numpy()
    y_true = labels.cpu().numpy().astype(np.int32)
    y_pred = (
        probabilities >= np.asarray(thresholds)[None, :]
    ).astype(np.int32)

    return {
        "micro_f1": float(
            f1_score(
                y_true, y_pred, average="micro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        ),
        "subset_accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "emotion_f1": f1_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist(),
    }


def tune_thresholds(logits, labels):
    """Tune each emotion threshold using validation data only."""
    probabilities = torch.sigmoid(logits).cpu().numpy()
    y_true = labels.cpu().numpy().astype(np.int32)

    candidates = np.arange(0.20, 0.801, 0.01)
    thresholds = []

    for emotion_idx in range(y_true.shape[1]):
        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in candidates:
            predictions = (
                probabilities[:, emotion_idx] >= threshold
            ).astype(np.int32)

            score = f1_score(
                y_true[:, emotion_idx],
                predictions,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        thresholds.append(best_threshold)

    return thresholds


# ------------------------------------------------------------
# Model output adapter
# ------------------------------------------------------------

def forward_model(model, batch):
    output = model(
        batch["text"],
        batch["audio"],
        batch["video"],
    )

    if isinstance(output, tuple):
        logits, weights = output
    else:
        logits, weights = output, None

    return logits, weights


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    logits_list = []
    labels_list = []
    weights_list = []

    for batch in loader:
        batch = {
            key: value.to(DEVICE, non_blocking=True)
            for key, value in batch.items()
        }

        logits, weights = forward_model(model, batch)
        logits_list.append(logits.detach().cpu())
        labels_list.append(batch["labels"].detach().cpu())

        if weights is not None:
            weights_list.append(weights.detach().cpu())

    return (
        torch.cat(logits_list),
        torch.cat(labels_list),
        torch.cat(weights_list) if weights_list else None,
    )


def run_smoke_test(model, train_dataset):
    """Verify shapes, finite loss, and gradient flow on one batch."""
    loader = DataLoader(
        train_dataset,
        batch_size=min(4, len(train_dataset)),
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))
    batch = {
        key: value.to(DEVICE)
        for key, value in batch.items()
    }

    model.train()
    model.zero_grad(set_to_none=True)

    logits, weights = forward_model(model, batch)

    assert logits.shape == (len(batch["labels"]), 6), (
        f"Unexpected logits shape: {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all(), "Non-finite model logits"

    loss = nn.BCEWithLogitsLoss()(logits, batch["labels"])
    assert torch.isfinite(loss), "Non-finite smoke-test loss"

    loss.backward()

    grad_norms = [
        p.grad.detach().norm()
        for p in model.parameters()
        if p.grad is not None
    ]
    assert grad_norms, "No gradients were produced"
    assert all(torch.isfinite(g) for g in grad_norms), (
        "Non-finite gradients"
    )

    if weights is not None:
        assert weights.shape == (len(batch["labels"]), 3)
        assert torch.allclose(
            weights.sum(dim=1),
            torch.ones(len(batch["labels"]), device=DEVICE),
            atol=1e-5,
        ), "Gating weights do not sum to one"

    model.zero_grad(set_to_none=True)
    print(
        f"Smoke test passed: logits={tuple(logits.shape)}, "
        f"loss={loss.item():.4f}"
    )


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_one_model(model_name, model_class, train_ds, val_ds):
    checkpoint_path = OUTPUT_DIR / f"{model_name}_best.pt"
    results_path = OUTPUT_DIR / f"{model_name}_results.json"

    # Never overwrite an existing experiment's artifacts.
    if checkpoint_path.exists() or results_path.exists():
        raise FileExistsError(
            f"Output already exists for {model_name}. "
            "Inspect it first or choose a new run name. "
            f"Checkpoint: {checkpoint_path}"
        )

    set_seed(SEED)

    model = model_class().to(DEVICE)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\n{'=' * 68}")
    print(f"Starting {model_name}")
    print(f"Device: {DEVICE}")
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"{'=' * 68}")

    run_smoke_test(model, train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    positives = train_ds.labels.sum(dim=0)
    negatives = train_ds.labels.shape[0] - positives
    pos_weight = (negatives / positives.clamp(min=1)).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {
                key: value.to(DEVICE, non_blocking=True)
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)
            logits, _ = forward_model(model, batch)
            loss = criterion(logits, batch["labels"])

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            total_loss += loss.item() * len(batch["labels"])

        train_loss = total_loss / len(train_ds)

        val_logits, val_labels, val_weights = collect_predictions(
            model, val_loader
        )
        val_metrics = metrics_at_thresholds(
            val_logits, val_labels, [0.5] * 6
        )

        # Scheduler and checkpoint selection use the same fixed
        # validation metric. Threshold tuning is done after selection.
        scheduler.step(val_metrics["macro_f1"])

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1_at_0.5": val_metrics["macro_f1"],
            "val_micro_f1_at_0.5": val_metrics["micro_f1"],
            "val_subset_accuracy_at_0.5": val_metrics["subset_accuracy"],
            "val_emotion_f1_at_0.5": val_metrics["emotion_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - start,
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss={train_loss:.4f} | "
            f"val macro-F1={val_metrics['macro_f1']:.4f} | "
            f"val micro-F1={val_metrics['micro_f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{epoch_result['seconds']:.1f}s"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())

            # Preserve an independently loadable best checkpoint.
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": best_state,
                    "epoch": best_epoch,
                    "best_val_macro_f1_at_0.5": best_macro_f1,
                    "config": {
                        "input_dim": 768,
                        "projection_dim": 256,
                        "hidden_dim": 256,
                        "dropout": DROPOUT,
                        "num_emotions": 6,
                        "emotion_names": EMOTION_NAMES,
                        "seed": SEED,
                    },
                },
                checkpoint_path,
            )
            print(f"  Saved new best checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  No improvement: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint")

    model.load_state_dict(best_state)

    # Calibrate thresholds on validation data only.
    val_logits, val_labels, val_weights = collect_predictions(
        model, val_loader
    )
    thresholds = tune_thresholds(val_logits, val_labels)

    val_default = metrics_at_thresholds(
        val_logits, val_labels, [0.5] * 6
    )
    val_calibrated = metrics_at_thresholds(
        val_logits, val_labels, thresholds
    )

    # Evaluate the test set once, after model selection and calibration.
    test_loader = DataLoader(
        CachedShardDataset("test"),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_logits, test_labels, test_weights = collect_predictions(
        model, test_loader
    )

    test_default = metrics_at_thresholds(
        test_logits, test_labels, [0.5] * 6
    )
    test_calibrated = metrics_at_thresholds(
        test_logits, test_labels, thresholds
    )

    result = {
        "model_name": model_name,
        "seed": SEED,
        "selected_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
        "selection_metric": "validation_macro_f1_at_threshold_0.5",
        "best_validation_macro_f1_at_0.5": best_macro_f1,
        "thresholds_from_validation": dict(
            zip(EMOTION_NAMES, thresholds)
        ),
        "validation_default_thresholds": val_default,
        "validation_calibrated_thresholds": val_calibrated,
        "test_default_thresholds": test_default,
        "test_calibrated_thresholds": test_calibrated,
        "test_evaluated_once_after_selection": True,
        "history": history,
    }

    if test_weights is not None:
        result["mean_gating_weights_on_test"] = {
            name: float(value)
            for name, value in zip(
                ["text", "audio", "video"],
                test_weights.mean(dim=0).tolist(),
            )
        }

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"\nFinished {model_name}")
    print(f"Selected epoch: {best_epoch}")
    print(f"Validation Macro-F1 @ 0.5: {best_macro_f1:.4f}")
    print(f"Test Macro-F1 @ 0.5: {test_default['macro_f1']:.4f}")
    print(f"Test Micro-F1 @ 0.5: {test_default['micro_f1']:.4f}")
    print(
        "Test Macro-F1 @ validation-calibrated thresholds: "
        f"{test_calibrated['macro_f1']:.4f}"
    )
    print("Validation-calibrated thresholds:", thresholds)

    if "mean_gating_weights_on_test" in result:
        print("Mean test gating weights:", result["mean_gating_weights_on_test"])

    return result


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    set_seed(SEED)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Feature directory: {FEATURE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")

    if not FEATURE_DIR.exists():
        raise FileNotFoundError(FEATURE_DIR)

    # Check ALL final artifact paths before loading data or training.
    expected_outputs = [
        OUTPUT_DIR / "mark3a_concat_best.pt",
        OUTPUT_DIR / "mark3a_concat_results.json",
        OUTPUT_DIR / "mark3b_gated_best.pt",
        OUTPUT_DIR / "mark3b_gated_results.json",
        OUTPUT_DIR / "mark3_pooled_summary.json",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing:
        paths = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Final Mark 3 output files already exist; refusing to overwrite:\n"
            + paths
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = CachedShardDataset("train")
    val_ds = CachedShardDataset("val")

    results = [
        train_one_model("mark3a_concat", Mark3ConcatFusion, train_ds, val_ds),
        train_one_model("mark3b_gated", Mark3GatedFusion, train_ds, val_ds),
    ]

    summary = {
        result["model_name"]: {
            "selected_epoch": result["selected_epoch"],
            "validation_macro_f1_at_0.5":
                result["best_validation_macro_f1_at_0.5"],
            "test_macro_f1_at_0.5":
                result["test_default_thresholds"]["macro_f1"],
            "test_micro_f1_at_0.5":
                result["test_default_thresholds"]["micro_f1"],
            "test_macro_f1_calibrated":
                result["test_calibrated_thresholds"]["macro_f1"],
            "test_micro_f1_calibrated":
                result["test_calibrated_thresholds"]["micro_f1"],
        }
        for result in results
    }

    summary_path = OUTPUT_DIR / "mark3_pooled_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n=== MARK 3 POOLED EXPERIMENT SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
