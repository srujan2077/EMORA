"""Temporal sequence building and window feature extraction for EMORA Mark 3 raw-input inference."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch

from src.encoders.audio_encoder import AudioEncoder
from src.encoders.text_encoder import TextEncoder
from src.encoders.video_encoder import VideoEncoder
from src.inference.mark3 import prepare_window
from src.inference.segment_manifest import SegmentInput, SegmentManifest

logger = logging.getLogger(__name__)

MAX_WINDOW_SIZE = 8


def build_causal_window(
    manifest: SegmentManifest,
    target_segment_number: int | None = None,
    max_window_size: int = MAX_WINDOW_SIZE,
) -> list[SegmentInput]:
    """Extract a causal context window ending at target_segment_number.

    Replicates the window construction logic from scripts/build_mark3_sequences.py:
    - Context consists of strictly causal preceding segments (including target)
    - Window size is capped at max_window_size (default 8)
    - If target_segment_number is None, the final segment is targeted

    Parameters
    ----------
    manifest : validated SegmentManifest
    target_segment_number : segment_number to predict (default: last segment)
    max_window_size : maximum sequence length (default: 8)

    Returns
    -------
    list of SegmentInput objects in chronological order, length in [1, max_window_size]
    """
    if target_segment_number is None:
        target_segment_number = manifest.segments[-1].segment_number

    # Select causal segments up to target
    preceding = [
        seg for seg in manifest.segments
        if seg.segment_number <= target_segment_number
    ]
    if not preceding:
        raise ValueError(
            f"No segments found with segment_number <= {target_segment_number} "
            f"in manifest for video '{manifest.video_id}'"
        )

    # Ensure the target itself is in the window
    if preceding[-1].segment_number != target_segment_number:
        raise ValueError(
            f"Target segment_number {target_segment_number} does not exist in manifest"
        )

    # Take the most recent causal history up to max_window_size
    window = preceding[-max_window_size:]
    return window


def extract_segment_features(
    segment: SegmentInput,
    text_encoder: TextEncoder,
    audio_encoder: AudioEncoder,
    video_encoder: VideoEncoder,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract 768-d embeddings for a single segment across all 3 modalities.

    Returns
    -------
    tuple of (text_emb, audio_emb, video_emb), each of shape (768,) float32
    """
    # 1. Text
    text_emb = text_encoder.encode(segment.text)

    # 2. Audio
    if segment.audio_waveform is not None:
        audio_emb = audio_encoder.encode_waveform(segment.audio_waveform)
    elif segment.resolved_audio_path is not None:
        audio_emb = audio_encoder.encode_file(
            segment.resolved_audio_path,
            start_time=segment.start_time,
            end_time=segment.end_time,
        )
    else:
        raise ValueError(f"No audio source for segment {segment.segment_number}")

    # 3. Video
    if segment.video_frames is not None:
        video_emb = video_encoder.encode_frames(segment.video_frames)
    elif segment.resolved_video_path is not None:
        video_emb = video_encoder.encode_file(
            segment.resolved_video_path,
            start_time=segment.start_time,
            end_time=segment.end_time,
        )
    else:
        raise ValueError(f"No video source for segment {segment.segment_number}")

    return text_emb, audio_emb, video_emb


def prepare_raw_window(
    window_segments: list[SegmentInput],
    text_encoder: TextEncoder,
    audio_encoder: AudioEncoder,
    video_encoder: VideoEncoder,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Extract raw embeddings for a causal window and apply the Mark 3 split-at-gap transform.

    Parameters
    ----------
    window_segments : list of SegmentInput objects in causal order
    text_encoder : TextEncoder instance
    audio_encoder : AudioEncoder instance
    video_encoder : VideoEncoder instance

    Returns
    -------
    text_tensor : shape [T, 768]
    audio_tensor : shape [T, 768]
    video_tensor : shape [T, 768]
    segment_numbers_tensor : shape [T]
    metadata : dict with details on raw window length, post-split length, and gap detection
    """
    if not window_segments:
        raise ValueError("window_segments must not be empty")

    text_list = []
    audio_list = []
    video_list = []
    segment_numbers = []

    for seg in window_segments:
        t_emb, a_emb, v_emb = extract_segment_features(
            seg, text_encoder, audio_encoder, video_encoder
        )
        text_list.append(t_emb)
        audio_list.append(a_emb)
        video_list.append(v_emb)
        segment_numbers.append(seg.segment_number)

    raw_text = torch.stack(text_list, dim=0)    # [N, 768]
    raw_audio = torch.stack(audio_list, dim=0)  # [N, 768]
    raw_video = torch.stack(video_list, dim=0)  # [N, 768]

    raw_length = len(segment_numbers)
    has_gap = any(
        segment_numbers[i] - segment_numbers[i - 1] > 1
        for i in range(1, len(segment_numbers))
    )

    # Use the existing Mark 3 prepare_window (which runs validation + split_at_gap)
    text_t, audio_t, video_t, seg_t = prepare_window(
        raw_text, raw_audio, raw_video, segment_numbers
    )

    metadata = {
        "raw_sequence_length": raw_length,
        "split_sequence_length": int(seg_t.shape[0]),
        "has_numbering_gap": has_gap,
        "input_segment_numbers": segment_numbers,
        "used_segment_numbers": seg_t.tolist(),
        "target_segment_number": segment_numbers[-1],
    }

    return text_t, audio_t, video_t, seg_t, metadata
