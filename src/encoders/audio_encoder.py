"""EMORA audio encoder wrapper using Wav2Vec2-base-960h."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from src.encoders.common import (
    AUDIO_MODEL_NAME,
    AUDIO_SAMPLE_RATE,
    EMBEDDING_DIM,
    extract_audio_waveform,
    resolve_device,
)

logger = logging.getLogger(__name__)


class AudioEncoder:
    """Wraps facebook/wav2vec2-base-960h to generate 768-d audio embeddings.

    Replicates the exact training-time extraction from
    scripts/extract_mark2_features.py:
    - Audio extracted via FFmpeg to 16 kHz mono float32 PCM
    - No artificial audio normalization / clipping
    - AutoProcessor with sampling_rate=16000, padding=True
    - Mean pooling over hidden states: hidden.mean(dim=1)
    - Empty audio produces zeros(768)
    """

    def __init__(
        self,
        model_name: str = AUDIO_MODEL_NAME,
        device: torch.device | str | None = None,
        processor: Any = None,
        model: Any = None,
        ffmpeg_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.processor = processor
        self.model = model
        self.ffmpeg_path = ffmpeg_path

    def load(self) -> AudioEncoder:
        """Load processor and model if not already provided."""
        if self.processor is None:
            logger.info("Loading audio processor: %s", self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        if self.model is None:
            logger.info("Loading audio model: %s on %s", self.model_name, self.device)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        return self

    @torch.inference_mode()
    def encode_waveform(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode a 16 kHz mono float32 waveform into a 768-d embedding tensor.

        Parameters
        ----------
        audio : 1D numpy array or torch Tensor of float32 samples.

        Returns
        -------
        torch.Tensor of shape (768,), dtype float32, on CPU.
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        if audio is None or len(audio) == 0:
            return torch.zeros(EMBEDDING_DIM, dtype=torch.float32)

        if self.processor is None or self.model is None:
            self.load()

        inputs = self.processor(
            audio,
            sampling_rate=AUDIO_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # [1, seq_len, 768]
        embedding = hidden.mean(dim=1)

        return embedding.squeeze(0).float().cpu()

    def encode_file(
        self,
        media_path: str | Path,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> torch.Tensor:
        """Extract audio using FFmpeg and encode it into a 768-d embedding tensor.

        Parameters
        ----------
        media_path : path to video or audio file.
        start_time : optional start offset in seconds.
        end_time : optional end offset in seconds.

        Returns
        -------
        torch.Tensor of shape (768,), dtype float32, on CPU.
        """
        waveform = extract_audio_waveform(
            media_path,
            ffmpeg_path=self.ffmpeg_path,
            sample_rate=AUDIO_SAMPLE_RATE,
            start_time=start_time,
            end_time=end_time,
        )
        return self.encode_waveform(waveform)

    def encode(
        self,
        source: str | Path | np.ndarray | torch.Tensor,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> torch.Tensor:
        """Universal entry point: encode either a waveform or a media file path."""
        if isinstance(source, (np.ndarray, torch.Tensor)):
            return self.encode_waveform(source)
        return self.encode_file(source, start_time=start_time, end_time=end_time)
