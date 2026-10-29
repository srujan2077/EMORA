"""Raw-media inference pipeline for EMORA Mark 3.

Connects raw media inputs (text, audio, video) through modality encoders
and temporal window aggregation into the frozen 5-seed Mark 3 ensemble.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.encoders.audio_encoder import AudioEncoder
from src.encoders.text_encoder import TextEncoder
from src.encoders.video_encoder import VideoEncoder
from src.inference.mark3 import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_REPORT_PATH,
    Mark3Inference,
    Mark3Prediction,
)
from src.inference.segment_manifest import SegmentInput, SegmentManifest
from src.inference.temporal_pipeline import (
    MAX_WINDOW_SIZE,
    build_causal_window,
    prepare_raw_window,
)

logger = logging.getLogger(__name__)


class Mark3RawInference:
    """Full-pipeline inference for Mark 3 accepting raw multimodal inputs.

    Parameters
    ----------
    mark3_inference : Optional pre-initialized Mark3Inference instance.
    text_encoder : Optional pre-initialized TextEncoder instance.
    audio_encoder : Optional pre-initialized AudioEncoder instance.
    video_encoder : Optional pre-initialized VideoEncoder instance.
    device : Target torch device (cuda if available, else cpu).
    checkpoint_root : Path to 5-seed checkpoint folder.
    report_path : Path to frozen evaluation report with calibrated thresholds.
    """

    def __init__(
        self,
        mark3_inference: Mark3Inference | None = None,
        text_encoder: TextEncoder | None = None,
        audio_encoder: AudioEncoder | None = None,
        video_encoder: VideoEncoder | None = None,
        device: torch.device | str | None = None,
        checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
        report_path: Path = DEFAULT_REPORT_PATH,
    ) -> None:
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_root = Path(checkpoint_root)
        self.report_path = Path(report_path)

        # Encoders (lazy-loaded or supplied)
        self.text_encoder = text_encoder or TextEncoder(device=self.device)
        self.audio_encoder = audio_encoder or AudioEncoder(device=self.device)
        self.video_encoder = video_encoder or VideoEncoder(device=self.device)

        # Mark 3 temporal fusion ensemble
        self.mark3_inference = mark3_inference or Mark3Inference(
            checkpoint_root=self.checkpoint_root,
            report_path=self.report_path,
            device=self.device,
        )

    def load_encoders(self) -> None:
        """Explicitly preload all three modality backbones."""
        self.text_encoder.load()
        self.audio_encoder.load()
        self.video_encoder.load()

    def predict_manifest(
        self,
        manifest: SegmentManifest,
        target_segment_number: int | None = None,
        max_window_size: int = MAX_WINDOW_SIZE,
    ) -> Mark3Prediction:
        """Run raw inference on a conversational session manifest.

        Parameters
        ----------
        manifest : validated SegmentManifest containing chronological segments.
        target_segment_number : target segment number to classify (defaults to final).
        max_window_size : maximum causal history window size (default 8).

        Returns
        -------
        Mark3Prediction with predicted emotions, probabilities, and enriched pipeline metadata.
        """
        # 1. Causal window selection (up to max_window_size ending at target)
        window = build_causal_window(
            manifest,
            target_segment_number=target_segment_number,
            max_window_size=max_window_size,
        )

        # 2. Extract features and apply split_at_gap transform
        text_t, audio_t, video_t, seg_t, pipeline_meta = prepare_raw_window(
            window,
            text_encoder=self.text_encoder,
            audio_encoder=self.audio_encoder,
            video_encoder=self.video_encoder,
        )

        # 3. Model forward pass through the 5-seed ensemble
        prediction = self.mark3_inference.predict(
            text=text_t,
            audio=audio_t,
            video=video_t,
            segment_numbers=seg_t,
        )

        # 4. Enrich metadata to indicate raw media pipeline execution
        enriched_metadata = dict(prediction.metadata)
        enriched_metadata.update(pipeline_meta)
        enriched_metadata["video_id"] = manifest.video_id
        enriched_metadata["raw_media_supported"] = True

        return Mark3Prediction(
            probabilities=prediction.probabilities,
            thresholds=prediction.thresholds,
            predicted_emotions=prediction.predicted_emotions,
            metadata=enriched_metadata,
        )

    def predict_single_segment(
        self,
        *,
        text: str | None = None,
        media_path: str | Path | None = None,
        video_path: str | Path | None = None,
        audio_path: str | Path | None = None,
        audio_waveform: np.ndarray | torch.Tensor | None = None,
        video_frames: Sequence[np.ndarray] | None = None,
        segment_number: int = 0,
        video_id: str = "raw_single_segment",
    ) -> Mark3Prediction:
        """Convenience entry point for a single standalone segment (window length = 1).

        Ideal for single video/audio clips uploaded via the API or UI.
        """
        segment = SegmentInput(
            segment_number=segment_number,
            text=text,
            media_path=str(media_path) if media_path is not None else None,
            video_path=str(video_path) if video_path is not None else None,
            audio_path=str(audio_path) if audio_path is not None else None,
            audio_waveform=audio_waveform,
            video_frames=video_frames,
        )
        manifest = SegmentManifest(video_id=video_id, segments=[segment])
        return self.predict_manifest(manifest)
