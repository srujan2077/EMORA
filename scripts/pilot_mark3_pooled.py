"""EMORA Mark 3 pilot: small training subset, full validation, no test access."""
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_mark3_pooled import (
    CachedShardDataset,
    DEVICE,
    EMOTION_NAMES,
    Mark3ConcatFusion,
    Mark3GatedFusion,
    collect_predictions,
    forward_model,
    metrics_at_thresholds,
    run_smoke_test,
    set_seed,
)

PILOT_DIR = PROJECT_ROOT / "checkpoints" / "mark3" / "pilot"
SEED = 42
PILOT_SAMPLES = 512
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25


def train_pilot_model(name, model_class, train_subset, full_train_ds, val_ds):
    checkpoint_path = PILOT_DIR / f"{name}_pilot_best.pt"
    results_path = PILOT_DIR / f"{name}_pilot_results.json"

    for path in (checkpoint_path, results_path):
        if path.exists():
            raise FileExistsError(
                f"Pilot output already exists: {path}\n"
                "Do not overwrite it. Inspect it or choose a new pilot folder."
            )

    set_seed(SEED)
    model = model_class(dropout=DROPOUT).to(DEVICE)
    run_smoke_test(model, train_subset)

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # Use class weights from the complete training split, not the small pilot subset.
    positives = full_train_ds.labels.sum(dim=0)
    negatives = full_train_ds.labels.shape[0] - positives
    pos_weight = (negatives / positives.clamp(min=1)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_val_macro = -1.0
    best_epoch = 0
    best_state = None
    history = []

    print(f"\n{'=' * 60}\nPILOT: {name}\nDevice: {DEVICE}")
    print(f"Training samples: {len(train_subset)} | Validation samples: {len(val_ds)}")
    print("Test set: NOT LOADED OR EVALUATED")
    print(f"{'=' * 60}")

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
                raise RuntimeError(f"Non-finite loss for {name}, epoch {epoch}")
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm for {name}, epoch {epoch}")
            optimizer.step()
            total_loss += loss.item() * len(batch["labels"])

        train_loss = total_loss / len(train_subset)
        val_logits, val_labels, _ = collect_predictions(model, val_loader)
        val_metrics = metrics_at_thresholds(val_logits, val_labels, [0.5] * 6)

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_macro_f1_at_0.5": val_metrics["macro_f1"],
            "validation_micro_f1_at_0.5": val_metrics["micro_f1"],
            "validation_subset_accuracy_at_0.5": val_metrics["subset_accuracy"],
            "seconds": time.time() - start,
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{EPOCHS} | train loss={train_loss:.4f} | "
            f"val macro-F1={val_metrics['macro_f1']:.4f} | "
            f"val micro-F1={val_metrics['micro_f1']:.4f} | "
            f"{record['seconds']:.1f}s"
        )

        if val_metrics["macro_f1"] > best_val_macro:
            best_val_macro = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_name": name,
                    "model_state_dict": best_state,
                    "epoch": best_epoch,
                    "best_validation_macro_f1_at_0.5": best_val_macro,
                    "pilot_only": True,
                    "config": {
                        "input_dim": 768,
                        "projection_dim": 256,
                        "hidden_dim": 256,
                        "dropout": DROPOUT,
                        "num_emotions": 6,
                        "emotion_names": EMOTION_NAMES,
                        "seed": SEED,
                        "pilot_samples": len(train_subset),
                    },
                },
                checkpoint_path,
            )
            print(f"  Saved pilot checkpoint: {checkpoint_path}")

    result = {
        "model_name": name,
        "pilot_only": True,
        "test_set_loaded_or_evaluated": False,
        "seed": SEED,
        "training_samples": len(train_subset),
        "validation_samples": len(val_ds),
        "epochs_run": EPOCHS,
        "selected_epoch": best_epoch,
        "best_validation_macro_f1_at_0.5": best_val_macro,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved pilot results: {results_path}")
    return result


def main():
    set_seed(SEED)
    PILOT_DIR.mkdir(parents=True, exist_ok=True)

    # Preflight all destinations before doing any training.
    summary_path = PILOT_DIR / "pilot_summary.json"
    expected = [
        PILOT_DIR / "mark3a_concat_pilot_best.pt",
        PILOT_DIR / "mark3a_concat_pilot_results.json",
        PILOT_DIR / "mark3b_gated_pilot_best.pt",
        PILOT_DIR / "mark3b_gated_pilot_results.json",
        summary_path,
    ]
    existing = [str(p) for p in expected if p.exists()]
    if existing:
        raise FileExistsError(
            "Pilot artifacts already exist; refusing to overwrite:\n" + "\n".join(existing)
        )

    train_ds = CachedShardDataset("train")
    val_ds = CachedShardDataset("val")
    if len(train_ds) < PILOT_SAMPLES:
        raise ValueError(f"Need at least {PILOT_SAMPLES} train samples, found {len(train_ds)}")

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(train_ds), generator=generator)[:PILOT_SAMPLES].tolist()
    train_subset = Subset(train_ds, indices)

    results = [
        train_pilot_model(
            "mark3a_concat", Mark3ConcatFusion, train_subset, train_ds, val_ds
        ),
        train_pilot_model(
            "mark3b_gated", Mark3GatedFusion, train_subset, train_ds, val_ds
        ),
    ]

    summary = {
        "pilot_only": True,
        "test_set_loaded_or_evaluated": False,
        "training_samples": PILOT_SAMPLES,
        "validation_samples": len(val_ds),
        "models": {
            result["model_name"]: {
                "selected_epoch": result["selected_epoch"],
                "best_validation_macro_f1_at_0.5":
                    result["best_validation_macro_f1_at_0.5"],
            }
            for result in results
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nPILOT COMPLETE — test set was not loaded or evaluated.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
