import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.inference.mark3 import (
    CANDIDATE,
    EMOTION_NAMES,
    EXPECTED_MODEL_CONFIG,
    Mark3Inference,
    prepare_window,
)


def valid_inputs(length=4):
    features = torch.zeros(length, 768)
    return features, features.clone(), features.clone(), list(range(10, 10 + length))


class Mark3WindowValidationTests(unittest.TestCase):
    def test_split_at_gap_keeps_contiguous_suffix(self):
        text, audio, video, _ = valid_inputs(5)
        prepared = prepare_window(text, audio, video, [10, 11, 14, 15, 16])

        self.assertEqual(prepared[0].shape, (3, 768))
        self.assertEqual(prepared[3].tolist(), [14, 15, 16])

    def test_rejects_malformed_shape(self):
        text, audio, video, numbers = valid_inputs()
        with self.assertRaisesRegex(ValueError, "shape"):
            prepare_window(text[:, :767], audio, video, numbers)

    def test_rejects_non_finite_values(self):
        text, audio, video, numbers = valid_inputs()
        text[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            prepare_window(text, audio, video, numbers)

    def test_rejects_invalid_segment_order(self):
        text, audio, video, _ = valid_inputs()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            prepare_window(text, audio, video, [10, 12, 11, 13])

    def test_rejects_mismatched_lengths(self):
        text, audio, video, numbers = valid_inputs()
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            prepare_window(text, audio[:-1], video, numbers)


class Mark3InferenceTests(unittest.TestCase):
    def test_loads_checkpoint_contract_and_returns_six_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "validation": {
                            "validation_calibrated": {
                                "thresholds": {
                                    name: 0.5 for name in EMOTION_NAMES
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_paths = []
            checkpoint = {
                "model_state_dict": {},
                "config": EXPECTED_MODEL_CONFIG,
                "candidate": CANDIDATE,
                "seed": 42,
                "deterministic_algorithms": True,
            }
            for seed in (42, 123, 456, 789, 2026):
                path = root / f"seed_{seed}" / CANDIDATE / "best.pt"
                path.parent.mkdir(parents=True)
                payload = dict(checkpoint)
                payload["seed"] = seed
                torch.save(payload, path)
                checkpoint_paths.append(path)

            class FakeModel:
                def to(self, _device):
                    return self

                def load_state_dict(self, _state_dict, strict=True):
                    self.strict = strict

                def eval(self):
                    return self

                def __call__(self, **_kwargs):
                    return torch.zeros(1, 6), torch.zeros(1, 1, 3)

            with patch(
                "src.inference.mark3.TemporalFusionMark3",
                return_value=FakeModel(),
            ):
                inference = Mark3Inference(
                    checkpoint_root=root,
                    report_path=report,
                    device=torch.device("cpu"),
                )
                text, audio, video, numbers = valid_inputs()
                result = inference.predict(text, audio, video, numbers)

        self.assertEqual(len(inference.models), 5)
        self.assertEqual(set(result.probabilities), set(EMOTION_NAMES))
        self.assertEqual(set(result.thresholds), set(EMOTION_NAMES))
        self.assertEqual(
            result.probabilities,
            {name: 0.5 for name in EMOTION_NAMES},
        )
        self.assertEqual(result.predicted_emotions, EMOTION_NAMES)
        self.assertEqual(result.metadata["ensemble_size"], 5)
        self.assertEqual(result.metadata["seeds"], [42, 123, 456, 789, 2026])


if __name__ == "__main__":
    unittest.main()
