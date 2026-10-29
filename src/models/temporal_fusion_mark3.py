
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TemporalFusionMark3(nn.Module):
    """
    EMORA Mark 3: Causal Temporal Multimodal Fusion.

    Inputs:
        text:           [B, T, 768]
        audio:          [B, T, 768]
        video:          [B, T, 768]
        lengths:        [B] valid sequence lengths
        segment_numbers: [B, T] original segment numbers; padding may be -1

    Output:
        logits:         [B, 6]
        modality_weights: [B, T, 3]

    Each window contains only past/current segments. The classifier uses
    the representation of the final valid segment in each window.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        num_emotions: int = 6,
        max_sequence_length: int = 8,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length

        # Independent modality-specific representations.
        self.text_projection = self._make_projection(input_dim, hidden_dim, dropout)
        self.audio_projection = self._make_projection(input_dim, hidden_dim, dropout)
        self.video_projection = self._make_projection(input_dim, hidden_dim, dropout)

        # Learn how much each modality contributes at every time step.
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

        # Fuse concatenated features and gated modality information.
        self.fusion_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # Learned relative positions: 0 is oldest, later positions are newer.
        self.position_embedding = nn.Embedding(
            max_sequence_length, hidden_dim
        )

        # Encode the number of missing segment numbers between observations.
        # The input is log1p(gap), keeping large gaps numerically manageable.
        self.gap_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions),
        )

    @staticmethod
    def _make_projection(
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        lengths: torch.Tensor,
        segment_numbers: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text.ndim != 3 or audio.ndim != 3 or video.ndim != 3:
            raise ValueError("Modality inputs must have shape [B, T, D].")

        if text.shape != audio.shape or text.shape != video.shape:
            raise ValueError("Text, audio, and video shapes must match.")

        batch_size, sequence_length, input_dim = text.shape

        if input_dim != self.text_projection[0].in_features:
            raise ValueError(
                f"Expected input dimension "
                f"{self.text_projection[0].in_features}, got {input_dim}."
            )

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds maximum "
                f"{self.max_sequence_length}."
            )

        lengths = lengths.to(device=text.device, dtype=torch.long)

        if lengths.shape != (batch_size,):
            raise ValueError("lengths must have shape [B].")

        if torch.any(lengths < 1) or torch.any(lengths > sequence_length):
            raise ValueError("Every length must be between 1 and T.")

        # True means padding; the Transformer ignores these positions.
        positions = torch.arange(sequence_length, device=text.device)
        padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)

        text_features = self.text_projection(text)
        audio_features = self.audio_projection(audio)
        video_features = self.video_projection(video)

        # [B, T, 3H]
        concatenated = torch.cat(
            [text_features, audio_features, video_features],
            dim=-1,
        )

        # [B, T, 3]; each set of weights sums to 1.
        modality_weights = torch.softmax(
            self.modality_gate(concatenated),
            dim=-1,
        )

        stacked = torch.stack(
            [text_features, audio_features, video_features],
            dim=2,
        )  # [B, T, 3, H]

        gated_features = (
            stacked * modality_weights.unsqueeze(-1)
        ).sum(dim=2)  # [B, T, H]

        fused_features = (
            self.fusion_projection(concatenated) + gated_features
        )

        # Relative position within the available context.
        position_ids = positions.unsqueeze(0).expand(batch_size, -1)
        position_features = self.position_embedding(position_ids)

        # Explicitly encode gaps in the original segment numbering.
        if segment_numbers is None:
            gap_values = torch.zeros(
                batch_size,
                sequence_length,
                device=text.device,
                dtype=text.dtype,
            )
        else:
            if segment_numbers.shape != (batch_size, sequence_length):
                raise ValueError(
                    "segment_numbers must have shape [B, T]."
                )

            segment_numbers = segment_numbers.to(device=text.device)

            differences = (
                segment_numbers[:, 1:] - segment_numbers[:, :-1]
            ).to(dtype=text.dtype)

            # Only positive differences represent valid forward progress.
            gaps = torch.log1p((differences - 1).clamp(min=0))

            gap_values = torch.zeros(
                batch_size,
                sequence_length,
                device=text.device,
                dtype=text.dtype,
            )

            if sequence_length > 1:
                gap_values[:, 1:] = gaps

            # Padded entries must not inject arbitrary gap information.
            gap_values = gap_values.masked_fill(padding_mask, 0.0)

        gap_features = self.gap_projection(gap_values.unsqueeze(-1))

        fused_features = self.fusion_norm(
            fused_features + position_features + gap_features
        )

        # These windows already contain only past/current segments, so
        # padding masks are sufficient; no future segment is included.
        temporal_features = self.temporal_encoder(
            fused_features,
            src_key_padding_mask=padding_mask,
        )

        # Select each sample's final VALID position, not the padded endpoint.
        final_indices = lengths - 1
        batch_indices = torch.arange(batch_size, device=text.device)
        target_features = temporal_features[batch_indices, final_indices]

        logits = self.classifier(target_features)

        return logits, modality_weights
