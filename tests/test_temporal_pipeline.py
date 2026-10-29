"""Unit tests for temporal sequence building and window extraction."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from src.encoders.common import EMBEDDING_DIM
from src.inference.segment_manifest import SegmentInput, SegmentManifest
from src.inference.temporal_pipeline import (
    build_causal_window,
    extract_segment_features,
    prepare_raw_window,
)


class MockEncoder:
    def __init__(self, fill_value: float):
        self.fill_value = fill_value

    def encode(self, *args, **kwargs):
        return torch.full((EMBEDDING_DIM,), self.fill_value, dtype=torch.float32)

    def encode_waveform(self, *args, **kwargs):
        return torch.full((EMBEDDING_DIM,), self.fill_value, dtype=torch.float32)

    def encode_file(self, *args, **kwargs):
        return torch.full((EMBEDDING_DIM,), self.fill_value, dtype=torch.float32)

    def encode_frames(self, *args, **kwargs):
        return torch.full((EMBEDDING_DIM,), self.fill_value, dtype=torch.float32)


class CausalWindowTests(unittest.TestCase):
    def test_window_capped_at_max_window_size(self):
        # 12 consecutive segments 0..11
        segments = [
            SegmentInput(segment_number=i, media_path=f"clip{i}.mp4")
            for i in range(12)
        ]
        manifest = SegmentManifest(video_id="vid_long", segments=segments)

        window = build_causal_window(manifest, max_window_size=8)
        self.assertEqual(len(window), 8)
        self.assertEqual([s.segment_number for s in window], list(range(4, 12)))
        self.assertEqual(window[-1].segment_number, 11)

    def test_short_sequence_retains_all(self):
        segments = [
            SegmentInput(segment_number=i, media_path=f"clip{i}.mp4")
            for i in range(3)
        ]
        manifest = SegmentManifest(video_id="vid_short", segments=segments)
        window = build_causal_window(manifest, max_window_size=8)
        self.assertEqual(len(window), 3)
        self.assertEqual([s.segment_number for s in window], [0, 1, 2])

    def test_intermediate_target_segment(self):
        segments = [
            SegmentInput(segment_number=i, media_path=f"clip{i}.mp4")
            for i in range(10)
        ]
        manifest = SegmentManifest(video_id="vid_mid", segments=segments)
        window = build_causal_window(manifest, target_segment_number=5, max_window_size=8)
        self.assertEqual(len(window), 6)  # segments 0, 1, 2, 3, 4, 5
        self.assertEqual(window[-1].segment_number, 5)

    def test_nonexistent_target_raises_error(self):
        segments = [
            SegmentInput(segment_number=0, media_path="c0.mp4"),
            SegmentInput(segment_number=2, media_path="c2.mp4"),
        ]
        manifest = SegmentManifest(video_id="vid_gap", segments=segments)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            build_causal_window(manifest, target_segment_number=1)


class PrepareRawWindowTests(unittest.TestCase):
    def setUp(self):
        self.text_enc = MockEncoder(fill_value=1.0)
        self.audio_enc = MockEncoder(fill_value=2.0)
        self.video_enc = MockEncoder(fill_value=3.0)

    def test_prepare_raw_window_contiguous(self):
        segments = [
            SegmentInput(
                segment_number=i,
                text=f"Utterance {i}",
                media_path=f"seg_{i}.mp4",
            )
            for i in range(4)
        ]
        text_t, audio_t, video_t, seg_t, meta = prepare_raw_window(
            segments, self.text_enc, self.audio_enc, self.video_enc
        )

        self.assertEqual(text_t.shape, (4, EMBEDDING_DIM))
        self.assertEqual(audio_t.shape, (4, EMBEDDING_DIM))
        self.assertEqual(video_t.shape, (4, EMBEDDING_DIM))
        self.assertEqual(seg_t.tolist(), [0, 1, 2, 3])
        self.assertFalse(meta["has_numbering_gap"])
        self.assertEqual(meta["raw_sequence_length"], 4)
        self.assertEqual(meta["split_sequence_length"], 4)
        self.assertEqual(meta["target_segment_number"], 3)

    def test_prepare_raw_window_splits_at_gap(self):
        # Window with gap: 0, 1, 4, 5. Target is 5.
        # Suffix from gap is [4, 5].
        segments = [
            SegmentInput(segment_number=0, media_path="c0.mp4"),
            SegmentInput(segment_number=1, media_path="c1.mp4"),
            SegmentInput(segment_number=4, media_path="c4.mp4"),
            SegmentInput(segment_number=5, media_path="c5.mp4"),
        ]
        text_t, audio_t, video_t, seg_t, meta = prepare_raw_window(
            segments, self.text_enc, self.audio_enc, self.video_enc
        )

        self.assertEqual(text_t.shape, (2, EMBEDDING_DIM))
        self.assertEqual(seg_t.tolist(), [4, 5])
        self.assertTrue(meta["has_numbering_gap"])
        self.assertEqual(meta["raw_sequence_length"], 4)
        self.assertEqual(meta["split_sequence_length"], 2)
        self.assertEqual(meta["used_segment_numbers"], [4, 5])


if __name__ == "__main__":
    unittest.main()
