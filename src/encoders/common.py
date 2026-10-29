"""Shared utilities for EMORA raw-media encoders."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ============================================================
# Constants matching training-time extraction
# ============================================================

TEXT_MODEL_NAME = "microsoft/deberta-v3-base"
AUDIO_MODEL_NAME = "facebook/wav2vec2-base-960h"
VISION_MODEL_NAME = "google/vit-base-patch16-224-in21k"

EMBEDDING_DIM = 768
NUM_FRAMES = 8
AUDIO_SAMPLE_RATE = 16000
MAX_TOKEN_LENGTH = 256

EMOTION_NAMES = ("happy", "sad", "anger", "surprise", "disgust", "fear")


# ============================================================
# Device helpers
# ============================================================

def resolve_device(device: torch.device | str | None = None) -> torch.device:
    """Return an explicit torch.device, defaulting to CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# FFmpeg helpers
# ============================================================

def find_ffmpeg() -> str:
    """Locate the FFmpeg binary, preferring PATH discovery."""
    which_path = shutil.which("ffmpeg")
    if which_path:
        return which_path

    # Fallback candidates used by the project
    candidates = [
        r"C:\Users\sruja\AppData\Local\Microsoft\WinGet\Packages"
        r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        r"\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for path in candidates:
        if Path(path).exists():
            return path

    # Last resort: assume it's on PATH and let subprocess fail clearly
    return "ffmpeg"


def extract_audio_waveform(
    media_path: str | Path,
    *,
    ffmpeg_path: str | None = None,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    timeout_seconds: int = 60,
    start_time: float | None = None,
    end_time: float | None = None,
) -> np.ndarray:
    """Extract mono float32 PCM audio from a media file using FFmpeg.

    Matches the training-time extraction:
        ffmpeg -hide_banner -loglevel error -i <path> -vn -ac 1
               -ar 16000 -f f32le -

    Parameters
    ----------
    media_path : path to video or audio file
    ffmpeg_path : explicit FFmpeg binary path (auto-discovered if None)
    sample_rate : target sample rate (default 16000)
    timeout_seconds : subprocess timeout
    start_time : optional start offset in seconds (-ss)
    end_time : optional end offset in seconds (-to)

    Returns
    -------
    np.ndarray of shape [N] with dtype float32
    """
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()

    media_path = str(media_path)
    if not Path(media_path).exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    command = [ffmpeg_path, "-hide_banner", "-loglevel", "error"]

    # Timestamp filtering (for segment extraction from full videos)
    if start_time is not None:
        command.extend(["-ss", str(start_time)])
    if end_time is not None:
        command.extend(["-to", str(end_time)])

    command.extend([
        "-i", media_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-",
    ])

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"FFmpeg timed out after {timeout_seconds}s while processing "
            f"{media_path}"
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="ignore").strip()
        raise RuntimeError(
            f"FFmpeg audio extraction failed (exit {result.returncode}): "
            f"{stderr or 'unknown error'}"
        )

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio
