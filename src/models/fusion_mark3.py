
"""EMORA Mark 3: pooled multimodal fusion models."""

import torch
from torch import nn


class Mark3ConcatFusion(nn.Module):
    """Control model: concatenate projected modality embeddings."""

    def __init__(
        self,
        input_dim: int = 768,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.25,
        num_emotions: int = 6,
    ):
        super().__init__()

        self.text_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.audio_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.video_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions),
        )

    def forward(self, text, audio, video):
        text = self.text_projection(text)
        audio = self.audio_projection(audio)
        video = self.video_projection(video)

        fused = torch.cat((text, audio, video), dim=-1)
        logits = self.classifier(fused)
        return logits


class Mark3GatedFusion(nn.Module):
    """Learn sample-dependent weights for text, audio, and video."""

    def __init__(
        self,
        input_dim: int = 768,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.25,
        num_emotions: int = 6,
    ):
        super().__init__()

        self.text_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.audio_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.video_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(projection_dim * 3, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, 3),
        )

        self.fusion_norm = nn.LayerNorm(projection_dim)

        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions),
        )

    def forward(self, text, audio, video):
        text = self.text_projection(text)
        audio = self.audio_projection(audio)
        video = self.video_projection(video)

        stacked = torch.stack((text, audio, video), dim=1)
        concatenated = torch.cat((text, audio, video), dim=-1)

        # One normalized modality weight per sample.
        weights = torch.softmax(self.gate(concatenated), dim=-1)

        # Weighted sum across modalities.
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        fused = self.fusion_norm(fused)

        logits = self.classifier(fused)
        return logits, weights
