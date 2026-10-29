"""Integration tests for raw-input Mark 3 inference pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from src.encoders.audio_encoder import AudioEncoder
from src.encoders.common import EMBEDDING_DIM
from src.encoders.text_encoder import TextEncoder
from src.encoders.video_encoder import VideoEncoder
from src.inference.mark3 import DEFAULT_CHECKPOINT_ROOT, DEFAULT_REPORT_PATH, Mark3Inference
from src.inference.mark3_raw import Mark3RawInference
from src.inference.segment_manifest import SegmentInput, SegmentManifest


class MockTextEncoder(TextEncoder):
    def __init__(self):
        super().__init__(device="cpu")

    def encode(self, text: str | None) -> torch.Tensor:
        if not text:
            return torch.zeros(EMBEDDING_DIM, dtype=torch.float32)
        return torch.full((EMBEDDING_DIM,), 0.5, dtype=torch.float32)


class MockAudioEncoder(AudioEncoder):
    def __init__(self):
        super().__init__(device="cpu")

    def encode_waveform(self, audio) -> torch.Tensor:
        if len(audio) == 0:
            return torch.zeros(EMBEDDING_DIM, dtype=torch.float32)
        return torch.full((EMBEDDING_DIM,), 0.2, dtype=torch.float32)

    def encode_file(self, *args, **kwargs) -> torch.Tensor:
        return torch.full((EMBEDDING_DIM,), 0.2, dtype=torch.float32)


class MockVideoEncoder(VideoEncoder):
    def __init__(self):
        super().__init__(device="cpu")

    def encode_frames(self, frames) -> torch.Tensor:
        return torch.full((EMBEDDING_DIM,), 0.8, dtype=torch.float32)

    def encode_file(self, *args, **kwargs) -> torch.Tensor:
        return torch.full((EMBEDDING_DIM,), 0.8, dtype=torch.float32)


class Mark3RawInferenceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.has_checkpoints = (
            DEFAULT_CHECKPOINT_ROOT.exists() and DEFAULT_REPORT_PATH.exists()
        )
        if cls.has_checkpoints:
            cls.mark3_inference = Mark3Inference(
                checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
                report_path=DEFAULT_REPORT_PATH,
                device=torch.device("cpu"),
            )

    def setUp(self):
        if not self.has_checkpoints:
            self.skipTest("Checkpoints or report not found")

        self.raw_pipeline = Mark3RawInference(
            mark3_inference=self.mark3_inference,
            text_encoder=MockTextEncoder(),
            audio_encoder=MockAudioEncoder(),
            video_encoder=MockVideoEncoder(),
            device=torch.device("cpu"),
        )

    def test_predict_single_segment_convenience(self):
        pred = self.raw_pipeline.predict_single_segment(
            text="I am absolutely thrilled with this result!",
            audio_waveform=np.ones(16000, dtype=np.float32),
            video_frames=[np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(8)],
        )

        self.assertIsInstance(pred.probabilities, dict)
        self.assertEqual(len(pred.probabilities), 6)
        for emotion, prob in pred.probabilities.items():
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            self.assertIn(emotion, pred.thresholds)

        # Ensure raw media execution metadata
        self.assertTrue(pred.metadata["raw_media_supported"])
        self.assertEqual(pred.metadata["raw_sequence_length"], 1)
        self.assertEqual(pred.metadata["split_sequence_length"], 1)
        self.assertFalse(pred.metadata["has_numbering_gap"])

    def test_predict_manifest_multi_segment_with_gap(self):
        # Create manifest with segments 0, 1, 4, 5
        # split_at_gap should prune 0, 1 and retain [4, 5]
        dummy_audio = np.ones(8000, dtype=np.float32)
        dummy_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(8)]

        segments = [
            SegmentInput(
                segment_number=0,
                text="Beginning utterance",
                audio_waveform=dummy_audio,
                video_frames=dummy_frames,
            ),
            SegmentInput(
                segment_number=1,
                text="Second utterance",
                audio_waveform=dummy_audio,
                video_frames=dummy_frames,
            ),
            SegmentInput(
                segment_number=4,
                text="After gap utterance",
                audio_waveform=dummy_audio,
                video_frames=dummy_frames,
            ),
            SegmentInput(
                segment_number=5,
                text="Final target utterance",
                audio_waveform=dummy_audio,
                video_frames=dummy_frames,
            ),
        ]
        manifest = SegmentManifest(video_id="conv_session_101", segments=segments)

        pred = self.raw_pipeline.predict_manifest(manifest)

        self.assertEqual(pred.metadata["video_id"], "conv_session_101")
        self.assertEqual(pred.metadata["raw_sequence_length"], 4)
        self.assertEqual(pred.metadata["split_sequence_length"], 2)
        self.assertTrue(pred.metadata["has_numbering_gap"])
        self.assertEqual(pred.metadata["used_segment_numbers"], [4, 5])
        self.assertEqual(pred.metadata["target_segment_number"], 5)
        self.assertTrue(pred.metadata["raw_media_supported"])


if __name__ == "__main__":
    unittest.main()
