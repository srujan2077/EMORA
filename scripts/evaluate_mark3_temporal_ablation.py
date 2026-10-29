
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_mark3_temporal import (
    TemporalWindowDataset,
    collate_windows,
    EMOTION_NAMES,
)
from src.models.temporal_fusion_mark3 import TemporalFusionMark3


def metrics(y_true, probabilities, thresholds):
    predictions = (probabilities >= thresholds).astype(np.int32)
    return {
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y_true, predictions, average="micro", zero_division=0)
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


@torch.no_grad()
def predict(model, loader, device, target_only=False):
    model.eval()
    labels_all, probabilities_all = [], []

    for batch in loader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        lengths = batch["lengths"].to(device)
        segment_numbers = batch["segment_numbers"].to(device)

        if target_only:
            # Extract each sample's final valid segment, not its padded end.
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


def main():
    output = ROOT / "checkpoints" / "mark3_temporal"
    checkpoint_path = output / "temporal_fusion_best.pt"
    thresholds_path = output / "temporal_fusion_thresholds.json"

    if not checkpoint_path.exists() or not thresholds_path.exists():
        raise FileNotFoundError(
            "Required temporal checkpoint or thresholds file is missing."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = TemporalFusionMark3(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    with thresholds_path.open("r", encoding="utf-8") as f:
        threshold_dict = json.load(f)

    thresholds = np.array(
        [threshold_dict[name] for name in EMOTION_NAMES],
        dtype=np.float32,
    )

    dataset = TemporalWindowDataset(
        ROOT / "data" / "processed" / "mark3_temporal" / "val_windows.pt"
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_windows,
    )

    # Validation only. This script never loads the test split.
    y_full, p_full = predict(model, loader, device, target_only=False)
    y_target, p_target = predict(model, loader, device, target_only=True)

    if not np.array_equal(y_full, y_target):
        raise RuntimeError("Labels differ between ablation conditions.")

    results = {
        "split": "validation",
        "thresholds": threshold_dict,
        "full_context": {
            "threshold_0_5": metrics(
                y_full, p_full, np.full(6, 0.5, dtype=np.float32)
            ),
            "validation_calibrated_thresholds": metrics(
                y_full, p_full, thresholds
            ),
        },
        "target_segment_only": {
            "threshold_0_5": metrics(
                y_target, p_target, np.full(6, 0.5, dtype=np.float32)
            ),
            "validation_calibrated_thresholds": metrics(
                y_target, p_target, thresholds
            ),
        },
    }

    results_path = output / "temporal_validation_ablation.json"
    if results_path.exists():
        raise FileExistsError(
            f"{results_path} already exists. Refusing to overwrite it."
        )

    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nVALIDATION-ONLY ABLATION")
    for condition, condition_results in (
        ("FULL CONTEXT", results["full_context"]),
        ("TARGET ONLY", results["target_segment_only"]),
    ):
        print(f"\n{condition}")
        for threshold_name, scores in condition_results.items():
            print(
                f"  {threshold_name}: "
                f"Macro-F1={scores['macro_f1']:.4f}, "
                f"Micro-F1={scores['micro_f1']:.4f}"
            )

    print("\nSaved:", results_path)


if __name__ == "__main__":
    main()
