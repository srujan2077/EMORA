
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.temporal_fusion_mark3 import TemporalFusionMark3


def main():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalFusionMark3().to(device)
    model.train()

    # Different sequence lengths test right-padding and final-position selection.
    batch_size, sequence_length, embedding_dim = 3, 8, 768

    text = torch.randn(batch_size, sequence_length, embedding_dim, device=device)
    audio = torch.randn_like(text)
    video = torch.randn_like(text)

    lengths = torch.tensor([8, 5, 1], device=device)
    segment_numbers = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15, 16, 17],
            [20, 21, 24, 25, 26, -1, -1, -1],
            [40, -1, -1, -1, -1, -1, -1, -1],
        ],
        device=device,
    )

    logits, modality_weights = model(
        text=text,
        audio=audio,
        video=video,
        lengths=lengths,
        segment_numbers=segment_numbers,
    )

    assert logits.shape == (batch_size, 6), logits.shape
    assert modality_weights.shape == (batch_size, sequence_length, 3)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(modality_weights).all()

    # Every time step's modality weights must sum to one.
    assert torch.allclose(
        modality_weights.sum(dim=-1),
        torch.ones(batch_size, sequence_length, device=device),
        atol=1e-5,
    )

    labels = torch.randint(0, 2, (batch_size, 6), device=device).float()
    loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()

    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert all(parameter.grad is not None for parameter in trainable_parameters)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in trainable_parameters
    )

    print("DEVICE:", device)
    print("LOGITS:", tuple(logits.shape))
    print("MODALITY WEIGHTS:", tuple(modality_weights.shape))
    print("SMOKE-TEST LOSS:", round(loss.item(), 6))
    print("GRADIENTS: PASS")
    print("MARK 3 TEMPORAL MODEL: PASS")


if __name__ == "__main__":
    main()
