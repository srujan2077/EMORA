import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch

from src.models.fusion_mark3 import (
    Mark3ConcatFusion,
    Mark3GatedFusion,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

batch_size = 4
text = torch.randn(batch_size, 768, device=device)
audio = torch.randn(batch_size, 768, device=device)
video = torch.randn(batch_size, 768, device=device)
labels = torch.randint(0, 2, (batch_size, 6)).float().to(device)

for name, model_class in [
    ("Mark3A Concatenation", Mark3ConcatFusion),
    ("Mark3B Gated Fusion", Mark3GatedFusion),
]:
    model = model_class().to(device)
    model.train()
    model.zero_grad(set_to_none=True)

    output = model(text, audio, video)

    if isinstance(output, tuple):
        logits, weights = output
        print(f"{name} gating weights:\n", weights.detach().cpu())
        assert weights.shape == (batch_size, 3)
        assert torch.allclose(
            weights.sum(dim=1),
            torch.ones(batch_size, device=device),
            atol=1e-5,
        )
    else:
        logits = output

    assert logits.shape == (batch_size, 6)
    assert torch.isfinite(logits).all()

    loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()

    gradients = [
        p.grad for p in model.parameters() if p.grad is not None
    ]
    assert gradients, "No gradients produced"
    assert all(torch.isfinite(g).all() for g in gradients)

    print(f"{name}: PASS | loss={loss.item():.4f}")
    del model

print("\nBoth Mark 3 smoke tests passed.")
