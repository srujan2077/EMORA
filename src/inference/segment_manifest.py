"""Segment manifest data structures and validation for EMORA Mark 3 raw-input inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


@dataclass
class SegmentInput:
    """Input specification for a single multimodal conversation segment.

    Parameters
    ----------
    segment_number : non-negative integer indicating temporal order.
    text : spoken utterance text (or None/empty; empty yields zero embedding).
    media_path : optional unified media path (video containing audio/visual).
    video_path : optional explicit path to video file.
    audio_path : optional explicit path to audio file.
    start_time : optional start offset in seconds.
    end_time : optional end offset in seconds.
    audio_waveform : optional in-memory 16 kHz mono float32 waveform.
    video_frames : optional in-memory sequence of 8 RGB uint8 frames.
    segment_id : optional segment identifier string.
    """

    segment_number: int
    text: str | None = None
    media_path: str | None = None
    video_path: str | None = None
    audio_path: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    audio_waveform: np.ndarray | torch.Tensor | None = field(
        default=None, repr=False
    )
    video_frames: Sequence[np.ndarray] | None = field(
        default=None, repr=False
    )
    segment_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.segment_number, int) or self.segment_number < 0:
            raise ValueError(
                f"segment_number must be a non-negative integer, got {self.segment_number!r}"
            )
        if self.start_time is not None and self.start_time < 0:
            raise ValueError(
                f"start_time cannot be negative, got {self.start_time}"
            )
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time <= self.start_time
        ):
            raise ValueError(
                f"end_time ({self.end_time}) must be greater than start_time ({self.start_time})"
            )

    @property
    def resolved_video_path(self) -> str | None:
        """Return video_path if present, else fallback to media_path."""
        return self.video_path or self.media_path

    @property
    def resolved_audio_path(self) -> str | None:
        """Return audio_path if present, else fallback to media_path."""
        return self.audio_path or self.media_path

    def has_audio_source(self) -> bool:
        """Check if any audio input source is configured."""
        return (
            self.audio_waveform is not None
            or self.resolved_audio_path is not None
        )

    def has_video_source(self) -> bool:
        """Check if any video input source is configured."""
        return (
            (self.video_frames is not None and len(self.video_frames) > 0)
            or self.resolved_video_path is not None
        )


@dataclass
class SegmentManifest:
    """A collection of temporally ordered segments for a single video/session.

    Enforces:
    - video_id is non-empty string
    - At least one segment is provided
    - segment_number values are strictly monotonically increasing
    - Each segment has accessible media sources
    """

    video_id: str
    segments: list[SegmentInput]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate manifest invariants."""
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string")
        if not self.segments:
            raise ValueError("SegmentManifest must contain at least one segment")

        numbers = [seg.segment_number for seg in self.segments]
        for i in range(len(numbers) - 1):
            if numbers[i + 1] <= numbers[i]:
                raise ValueError(
                    f"Segments must be strictly increasing in segment_number. "
                    f"Got {numbers[i]} followed by {numbers[i + 1]} at indices {i}, {i + 1}."
                )

        # Check media source availability
        for seg in self.segments:
            if not seg.has_audio_source():
                raise ValueError(
                    f"Segment {seg.segment_number} has no audio source "
                    "(must provide audio_waveform, audio_path, or media_path)"
                )
            if not seg.has_video_source():
                raise ValueError(
                    f"Segment {seg.segment_number} has no video source "
                    "(must provide video_frames, video_path, or media_path)"
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to serializable dictionary (omits raw arrays)."""
        return {
            "video_id": self.video_id,
            "segments": [
                {
                    "segment_number": s.segment_number,
                    "segment_id": s.segment_id or f"{self.video_id}[{s.segment_number}]",
                    "text": s.text,
                    "media_path": s.media_path,
                    "video_path": s.video_path,
                    "audio_path": s.audio_path,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                }
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentManifest:
        """Construct a SegmentManifest from a dict representation."""
        video_id = data.get("video_id")
        raw_segments = data.get("segments", [])
        if not isinstance(raw_segments, list):
            raise ValueError("'segments' field must be a list")

        segment_objects = []
        for item in raw_segments:
            segment_objects.append(
                SegmentInput(
                    segment_number=int(item["segment_number"]),
                    text=item.get("text"),
                    media_path=item.get("media_path"),
                    video_path=item.get("video_path"),
                    audio_path=item.get("audio_path"),
                    start_time=item.get("start_time"),
                    end_time=item.get("end_time"),
                    segment_id=item.get("segment_id"),
                )
            )
        return cls(video_id=video_id, segments=segment_objects)

    @classmethod
    def from_json(cls, json_str: str) -> SegmentManifest:
        """Parse a JSON string into a SegmentManifest."""
        data = json.loads(json_str)
        return cls.from_dict(data)
