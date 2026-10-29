"""EMORA video encoder wrapper using ViT-base-patch16-224-in21k."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from src.encoders.common import (
    EMBEDDING_DIM,
    NUM_FRAMES,
    VISION_MODEL_NAME,
    resolve_device,
)

logger = logging.getLogger(__name__)


def extract_video_frames(
    video_path: str | Path,
    *,
    num_frames: int = NUM_FRAMES,
    start_time: float | None = None,
    end_time: float | None = None,
) -> list[np.ndarray]:
    """Read evenly spaced RGB frames from a video file using OpenCV.

    Replicates the exact frame sampling logic from scripts/extract_mark2_features.py:
    - 8 uniformly sampled frames across the segment or time window
    - Frames converted from BGR to RGB
    - OpenCV VideoCapture seeking via CAP_PROP_POS_FRAMES

    Parameters
    ----------
    video_path : path to video file
    num_frames : number of frames to sample (default 8)
    start_time : optional start time in seconds
    end_time : optional end time in seconds

    Returns
    -------
    List of RGB numpy arrays of shape [H, W, 3] (uint8).
    """
    video_path = str(video_path)
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"No frames found: {video_path}")

        # Compute index range: either from timestamp offsets or full video
        if start_time is not None or end_time is not None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0

            first_frame = (
                max(0, int(start_time * fps)) if start_time is not None else 0
            )
            last_frame = (
                min(total_frames - 1, int(end_time * fps))
                if end_time is not None
                else total_frames - 1
            )
            if last_frame < first_frame:
                first_frame, last_frame = last_frame, first_frame
            if first_frame >= total_frames:
                first_frame = max(0, total_frames - 1)
                last_frame = first_frame
            indices = np.linspace(first_frame, last_frame, num_frames).astype(int)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

        frames: list[np.ndarray] = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = cap.read()
            if not success or frame is None:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        if len(frames) == 0:
            raise RuntimeError(f"Could not decode frames: {video_path}")

        if len(frames) < num_frames:
            logger.warning(
                "Decoded %d frames (expected %d) from %s",
                len(frames),
                num_frames,
                video_path,
            )

        return frames
    finally:
        cap.release()


class VideoEncoder:
    """Wraps google/vit-base-patch16-224-in21k to generate 768-d video embeddings.

    Replicates the exact training-time extraction from
    scripts/extract_mark2_features.py:
    - 8 frames uniformly sampled per segment
    - AutoImageProcessor for resizing / normalization
    - ViT CLS token extraction per frame (hidden[:, 0, :])
    - Temporal mean pooling across frames: cls_embeddings.mean(dim=0)
    """

    def __init__(
        self,
        model_name: str = VISION_MODEL_NAME,
        device: torch.device | str | None = None,
        processor: Any = None,
        model: Any = None,
        num_frames: int = NUM_FRAMES,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.processor = processor
        self.model = model
        self.num_frames = num_frames

    def load(self) -> VideoEncoder:
        """Load image processor and model if not already provided."""
        if self.processor is None:
            logger.info("Loading vision image processor: %s", self.model_name)
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        if self.model is None:
            logger.info("Loading vision model: %s on %s", self.model_name, self.device)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        return self

    @torch.inference_mode()
    def encode_frames(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """Encode a sequence of RGB frames into a 768-d embedding tensor.

        Parameters
        ----------
        frames : sequence of RGB numpy arrays [H, W, 3] (uint8).

        Returns
        -------
        torch.Tensor of shape (768,), dtype float32, on CPU.
        """
        if frames is None or len(frames) == 0:
            raise ValueError("frames must be a non-empty sequence of RGB images")

        if self.processor is None or self.model is None:
            self.load()

        inputs = self.processor(images=list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        outputs = self.model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state  # [num_frames, patch_tokens, 768]

        # Extract CLS token for each frame
        cls_embeddings = hidden[:, 0, :]  # [num_frames, 768]

        # Temporal mean across frames
        embedding = cls_embeddings.mean(dim=0)
        return embedding.float().cpu()

    def encode_file(
        self,
        video_path: str | Path,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> torch.Tensor:
        """Extract frames from video and encode into a 768-d embedding tensor."""
        frames = extract_video_frames(
            video_path,
            num_frames=self.num_frames,
            start_time=start_time,
            end_time=end_time,
        )
        return self.encode_frames(frames)

    def encode(
        self,
        source: str | Path | Sequence[np.ndarray],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> torch.Tensor:
        """Universal entry point: encode either decoded frames or a video path."""
        if isinstance(source, (list, tuple)) and len(source) > 0 and isinstance(source[0], np.ndarray):
            return self.encode_frames(source)
        return self.encode_file(source, start_time=start_time, end_time=end_time)
