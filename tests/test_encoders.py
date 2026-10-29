"""Unit tests for EMORA raw-media encoder wrappers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.encoders.audio_encoder import AudioEncoder
from src.encoders.common import (
    AUDIO_SAMPLE_RATE,
    EMBEDDING_DIM,
    NUM_FRAMES,
    extract_audio_waveform,
    find_ffmpeg,
    resolve_device,
)
from src.encoders.text_encoder import TextEncoder
from src.encoders.video_encoder import VideoEncoder, extract_video_frames


# ============================================================
# Mock Helpers
# ============================================================

class MockTokenizer:
    def __call__(self, text, return_tensors="pt", max_length=256, truncation=True, padding=True):
        # Return 2 tokens with mask [1, 1]
        return {
            "input_ids": torch.tensor([[101, 102]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }


class MockTextModel:
    def __init__(self, device=torch.device("cpu")):
        self.device = device

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        # hidden states: [batch=1, seq_len=2, hidden_dim=768]
        # token 0 = 2.0, token 1 = 4.0 -> mean should be 3.0
        hidden = torch.ones((1, 2, EMBEDDING_DIM), dtype=torch.float32, device=self.device)
        hidden[:, 0, :] = 2.0
        hidden[:, 1, :] = 4.0
        return SimpleNamespace(last_hidden_state=hidden)


class MockAudioProcessor:
    def __call__(self, audio, sampling_rate=16000, return_tensors="pt", padding=True):
        return {"input_values": torch.ones((1, len(audio)), dtype=torch.float32)}


class MockAudioModel:
    def __init__(self, device=torch.device("cpu")):
        self.device = device

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        # hidden states: [batch=1, seq_len=4, hidden_dim=768]
        # frames = 1, 2, 3, 4 -> mean = 2.5
        hidden = torch.zeros((1, 4, EMBEDDING_DIM), dtype=torch.float32, device=self.device)
        hidden[:, 0, :] = 1.0
        hidden[:, 1, :] = 2.0
        hidden[:, 2, :] = 3.0
        hidden[:, 3, :] = 4.0
        return SimpleNamespace(last_hidden_state=hidden)


class MockImageProcessor:
    def __call__(self, images=None, return_tensors="pt"):
        num_images = len(images) if images is not None else 8
        return {"pixel_values": torch.ones((num_images, 3, 224, 224), dtype=torch.float32)}


class MockVisionModel:
    def __init__(self, device=torch.device("cpu")):
        self.device = device

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        num_frames = kwargs["pixel_values"].shape[0]
        # hidden states: [num_frames, num_patches=197, hidden_dim=768]
        # CLS is index 0. Set frame i's CLS to (i + 1)
        hidden = torch.zeros((num_frames, 197, EMBEDDING_DIM), dtype=torch.float32, device=self.device)
        for i in range(num_frames):
            hidden[i, 0, :] = float(i + 1)
        return SimpleNamespace(last_hidden_state=hidden)


# ============================================================
# Test Cases
# ============================================================

class TextEncoderTests(unittest.TestCase):
    def setUp(self):
        self.encoder = TextEncoder(
            device="cpu",
            tokenizer=MockTokenizer(),
            model=MockTextModel(),
        )

    def test_encode_normal_text(self):
        embedding = self.encoder.encode("I am feeling great today.")
        self.assertEqual(embedding.shape, (EMBEDDING_DIM,))
        self.assertEqual(embedding.dtype, torch.float32)
        self.assertEqual(embedding.device.type, "cpu")
        # Token 0 is 2.0, token 1 is 4.0, attention mask is [1, 1] -> mean is 3.0
        self.assertTrue(torch.allclose(embedding, torch.full((EMBEDDING_DIM,), 3.0)))

    def test_empty_and_whitespace_produce_zeros(self):
        for empty_val in ["", "   ", "\n\t", None, 123]:
            embedding = self.encoder.encode(empty_val)
            self.assertEqual(embedding.shape, (EMBEDDING_DIM,))
            self.assertTrue(torch.all(embedding == 0.0))

    def test_attention_mask_aware_pooling(self):
        class MaskTokenizer:
            def __call__(self, text, **kwargs):
                return {
                    "input_ids": torch.tensor([[101, 102, 0]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 0, 0]], dtype=torch.long),
                }

        class VaryingHiddenModel:
            def to(self, device):
                return self
            def eval(self):
                return self
            def __call__(self, **kwargs):
                hidden = torch.zeros((1, 3, EMBEDDING_DIM), dtype=torch.float32)
                hidden[:, 0, :] = 7.0  # valid token
                hidden[:, 1, :] = 999.0  # masked token
                hidden[:, 2, :] = 999.0  # masked token
                return SimpleNamespace(last_hidden_state=hidden)

        enc = TextEncoder(device="cpu", tokenizer=MaskTokenizer(), model=VaryingHiddenModel())
        emb = enc.encode("test")
        self.assertTrue(torch.allclose(emb, torch.full((EMBEDDING_DIM,), 7.0)))


class AudioEncoderTests(unittest.TestCase):
    def setUp(self):
        self.encoder = AudioEncoder(
            device="cpu",
            processor=MockAudioProcessor(),
            model=MockAudioModel(),
        )

    def test_encode_waveform(self):
        waveform = np.ones(16000, dtype=np.float32)
        embedding = self.encoder.encode_waveform(waveform)
        self.assertEqual(embedding.shape, (EMBEDDING_DIM,))
        self.assertEqual(embedding.dtype, torch.float32)
        # 1.0, 2.0, 3.0, 4.0 -> mean 2.5
        self.assertTrue(torch.allclose(embedding, torch.full((EMBEDDING_DIM,), 2.5)))

    def test_encode_empty_waveform_returns_zeros(self):
        embedding = self.encoder.encode_waveform(np.array([], dtype=np.float32))
        self.assertEqual(embedding.shape, (EMBEDDING_DIM,))
        self.assertTrue(torch.all(embedding == 0.0))

    def test_torch_tensor_input_accepted(self):
        waveform = torch.ones(8000, dtype=torch.float32)
        embedding = self.encoder.encode(waveform)
        self.assertEqual(embedding.shape, (EMBEDDING_DIM,))
        self.assertTrue(torch.allclose(embedding, torch.full((EMBEDDING_DIM,), 2.5)))


class VideoEncoderTests(unittest.TestCase):
    def setUp(self):
        self.encoder = VideoEncoder(
            device="cpu",
            processor=MockImageProcessor(),
            model=MockVisionModel(),
            num_frames=8,
        )

    def test_encode_frames(self):
        # 8 RGB frames of 224x224
        frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(8)]
        embedding = self.encoder.encode_frames(frames)
        self.assertEqual(embedding.shape, (EMBEDDING_DIM,))
        self.assertEqual(embedding.dtype, torch.float32)
        # Frame i's CLS is (i + 1) for i in 0..7 -> mean(1..8) = 4.5
        self.assertTrue(torch.allclose(embedding, torch.full((EMBEDDING_DIM,), 4.5)))

    def test_empty_frames_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.encoder.encode_frames([])


class CommonUtilsTests(unittest.TestCase):
    def test_resolve_device(self):
        dev = resolve_device("cpu")
        self.assertEqual(dev.type, "cpu")

    def test_nonexistent_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            extract_audio_waveform("nonexistent_path_xyz.mp4")
        with self.assertRaises(FileNotFoundError):
            extract_video_frames("nonexistent_path_xyz.mp4")


if __name__ == "__main__":
    unittest.main()
