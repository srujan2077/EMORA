from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from src.models.temporal_fusion_mark3 import TemporalFusionMark3


ROOT = Path(__file__).resolve().parents[2]
EMOTION_NAMES = (
    "happy",
    "sad",
    "anger",
    "surprise",
    "disgust",
    "fear",
)
SEEDS = (42, 123, 456, 789, 2026)
CANDIDATE = "split_at_gap"
EXPECTED_MODEL_CONFIG = {
    "input_dim": 768,
    "hidden_dim": 256,
    "num_heads": 8,
    "num_layers": 2,
    "num_emotions": 6,
    "max_sequence_length": 8,
    "dropout": 0.15,
}
DEFAULT_CHECKPOINT_ROOT = (
    ROOT / "checkpoints" / "mark3_step3" / "multiseed_20261027"
)
DEFAULT_REPORT_PATH = (
    ROOT / "reports" / "mark3_step3_final_20261027_151500.json"
)


@dataclass(frozen=True)
class Mark3Prediction:
    probabilities: dict[str, float]
    thresholds: dict[str, float]
    predicted_emotions: tuple[str, ...]
    metadata: dict[str, object]


def _validate_window(
    text: torch.Tensor,
    audio: torch.Tensor,
    video: torch.Tensor,
    segment_numbers: Sequence[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    modalities = {"text": text, "audio": audio, "video": video}
    for name, tensor in modalities.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.ndim != 2 or tensor.shape[1] != 768:
            raise ValueError(
                f"{name} must have shape [T, 768], got {tuple(tensor.shape)}."
            )
        if not torch.is_floating_point(tensor):
            raise ValueError(f"{name} must use a floating-point dtype.")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values.")

    lengths = {tensor.shape[0] for tensor in modalities.values()}
    if len(lengths) != 1:
        raise ValueError("Text, audio, and video must have matching lengths.")

    sequence_length = next(iter(lengths))
    if not 1 <= sequence_length <= 8:
        raise ValueError("The window must contain between 1 and 8 segments.")

    segment_numbers_tensor = torch.as_tensor(segment_numbers)
    if segment_numbers_tensor.ndim != 1:
        raise ValueError("segment_numbers must have shape [T].")
    if segment_numbers_tensor.shape[0] != sequence_length:
        raise ValueError("segment_numbers length must match the modality length.")
    if not torch.is_floating_point(segment_numbers_tensor) and (
        segment_numbers_tensor.dtype != torch.bool
    ):
        numeric_segment_numbers = segment_numbers_tensor
    else:
        if not torch.isfinite(segment_numbers_tensor).all():
            raise ValueError("segment_numbers contains non-finite values.")
        if not torch.equal(
            segment_numbers_tensor,
            segment_numbers_tensor.to(dtype=torch.long),
        ):
            raise ValueError("segment_numbers must contain integers.")
        numeric_segment_numbers = segment_numbers_tensor.to(dtype=torch.long)

    numeric_segment_numbers = numeric_segment_numbers.to(dtype=torch.long)
    if sequence_length > 1 and not torch.all(
        numeric_segment_numbers[1:] > numeric_segment_numbers[:-1]
    ):
        raise ValueError("segment_numbers must be strictly increasing.")

    return text, audio, video, numeric_segment_numbers


def _split_at_gap_suffix(
    text: torch.Tensor,
    audio: torch.Tensor,
    video: torch.Tensor,
    segment_numbers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keep = 1
    for previous, current in zip(
        reversed(segment_numbers[:-1]),
        reversed(segment_numbers[1:]),
    ):
        if int(current) != int(previous) + 1:
            break
        keep += 1

    start = segment_numbers.shape[0] - keep
    return (
        text[start:],
        audio[start:],
        video[start:],
        segment_numbers[start:],
    )


def prepare_window(
    text: torch.Tensor,
    audio: torch.Tensor,
    video: torch.Tensor,
    segment_numbers: Sequence[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and apply the controlled split-at-gap suffix transform."""
    validated = _validate_window(text, audio, video, segment_numbers)
    return _split_at_gap_suffix(*validated)


class Mark3Inference:
    """Inference over one preprocessed causal Mark 3 feature window."""

    def __init__(
        self,
        checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
        report_path: Path = DEFAULT_REPORT_PATH,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.checkpoint_root = Path(checkpoint_root)
        self.report_path = Path(report_path)
        self.thresholds = self._load_thresholds()
        self.models, self.checkpoint_metadata = self._load_models()

    def _checkpoint_paths(self) -> list[Path]:
        paths = [
            self.checkpoint_root
            / f"seed_{seed}"
            / CANDIDATE
            / "best.pt"
            for seed in SEEDS
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing ensemble checkpoints: {missing}")
        return paths

    def _load_thresholds(self) -> dict[str, float]:
        if not self.report_path.is_file():
            raise FileNotFoundError(f"Missing Mark 3 report: {self.report_path}")
        with self.report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)
        try:
            threshold_values = report["validation"]["fixed_0_5"]["thresholds"]
        except (KeyError, TypeError):
            # Keep compatibility with minimal test fixtures while requiring
            # the production report to carry the explicit fixed policy.
            try:
                threshold_values = report["validation"]["validation_calibrated"][
                    "thresholds"
                ]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "Mark 3 report is missing fixed-0.5 validation thresholds."
                ) from exc

        if set(threshold_values) != set(EMOTION_NAMES):
            raise ValueError("Mark 3 thresholds do not match the six emotions.")
        thresholds = {
            name: float(threshold_values[name]) for name in EMOTION_NAMES
        }
        if any(value != 0.5 for value in thresholds.values()):
            raise ValueError("Mark 3 production policy requires fixed 0.5 thresholds.")
        return thresholds

    def _load_models(self) -> tuple[list[TemporalFusionMark3], list[dict]]:
        models = []
        metadata = []
        for path, seed in zip(self._checkpoint_paths(), SEEDS):
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Invalid checkpoint payload: {path}")
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
                    f"Checkpoint {path} does not have "
                    "deterministic_algorithms=True."
                )

            model = TemporalFusionMark3(**checkpoint["config"]).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            models.append(model)
            metadata.append(
                {
                    "path": str(path.resolve()),
                    "seed": seed,
                    "candidate": checkpoint["candidate"],
                    "config": checkpoint["config"],
                }
            )
        return models, metadata

    def predict(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        segment_numbers: Sequence[int] | torch.Tensor,
    ) -> Mark3Prediction:
        text, audio, video, segment_numbers = prepare_window(
            text, audio, video, segment_numbers
        )
        text = text.to(self.device).unsqueeze(0)
        audio = audio.to(self.device).unsqueeze(0)
        video = video.to(self.device).unsqueeze(0)
        lengths = torch.tensor([text.shape[1]], device=self.device)
        segment_numbers = segment_numbers.to(self.device).unsqueeze(0)

        probabilities = []
        with torch.inference_mode():
            for model in self.models:
                logits, _ = model(
                    text=text,
                    audio=audio,
                    video=video,
                    lengths=lengths,
                    segment_numbers=segment_numbers,
                )
                member_probabilities = torch.sigmoid(logits)
                if member_probabilities.shape != (1, 6):
                    raise ValueError("Model returned an unexpected output shape.")
                if not torch.isfinite(member_probabilities).all():
                    raise ValueError("Model returned non-finite probabilities.")
                probabilities.append(member_probabilities[0])

            averaged = torch.stack(probabilities).mean(dim=0).cpu()

        probability_values = {
            name: float(averaged[index])
            for index, name in enumerate(EMOTION_NAMES)
        }
        predicted_emotions = tuple(
            name
            for name in EMOTION_NAMES
            if probability_values[name] >= self.thresholds[name]
        )
        return Mark3Prediction(
            probabilities=probability_values,
            thresholds=dict(self.thresholds),
            predicted_emotions=predicted_emotions,
            metadata={
                "candidate": CANDIDATE,
                "ensemble_size": len(self.models),
                "seeds": list(SEEDS),
                "model_config": dict(EXPECTED_MODEL_CONFIG),
                "segment_count_after_split_at_gap": int(text.shape[1]),
                "threshold_source": str(self.report_path.resolve()),
                "threshold_policy": "fixed_0_5",
                "raw_media_supported": False,
            },
        )
