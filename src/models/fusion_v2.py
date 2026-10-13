"""
Mark 2 - Cross-Modal Transformer Fusion

Text   ──┐
Audio  ──┼──> 256-D modality tokens
Video  ──┘
              ↓
       Modality embeddings
              ↓
    Cross-Modal Transformer
              ↓
          [CLS] token
              ↓
        Fusion embedding
              ↓
       6-emotion logits
"""

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
        num_modalities=3,
        num_layers=2,
        num_heads=8,
        feedforward_dim=1024,
        dropout=0.1,
        num_emotions=6,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_modalities = num_modalities

        # ----------------------------------------------------
        # Learnable CLS token
        # ----------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.randn(
                1,
                1,
                embedding_dim
            )
        )

        # ----------------------------------------------------
        # Learnable modality embeddings
        #
        # 0 = Text
        # 1 = Audio
        # 2 = Vision
        # ----------------------------------------------------

        self.modality_embeddings = nn.Parameter(
            torch.randn(
                1,
                num_modalities,
                embedding_dim
            )
        )

        # ----------------------------------------------------
        # Transformer encoder
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ----------------------------------------------------
        # Fusion normalization
        # ----------------------------------------------------

        self.fusion_norm = nn.LayerNorm(
            embedding_dim
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                embedding_dim,
                256
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                256,
                num_emotions
            )
        )

    def forward(
        self,
        text_embedding,
        audio_embedding,
        vision_embedding,
    ):
        """
        Parameters
        ----------
        text_embedding:
            [B, 256]

        audio_embedding:
            [B, 256]

        vision_embedding:
            [B, 256]

        Returns
        -------
        logits:
            [B, 6]

        fusion_embedding:
            [B, 256]
        """

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if text_embedding.dim() != 2:
            raise ValueError(
                f"Expected text embedding [B,D], "
                f"got {text_embedding.shape}"
            )

        if audio_embedding.dim() != 2:
            raise ValueError(
                f"Expected audio embedding [B,D], "
                f"got {audio_embedding.shape}"
            )

        if vision_embedding.dim() != 2:
            raise ValueError(
                f"Expected vision embedding [B,D], "
                f"got {vision_embedding.shape}"
            )

        # ----------------------------------------------------
        # Convert modality embeddings into tokens
        #
        # [B,256]
        #    ↓
        # [B,1,256]
        # ----------------------------------------------------

        text_token = text_embedding.unsqueeze(1)

        audio_token = audio_embedding.unsqueeze(1)

        vision_token = vision_embedding.unsqueeze(1)

        # ----------------------------------------------------
        # Stack modalities
        #
        # [B,3,256]
        # ----------------------------------------------------

        modality_tokens = torch.cat(
            [
                text_token,
                audio_token,
                vision_token,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Add modality identity
        # ----------------------------------------------------

        modality_tokens = (
            modality_tokens
            + self.modality_embeddings
        )

        batch_size = modality_tokens.size(0)

        # ----------------------------------------------------
        # Expand CLS token for batch
        #
        # [1,1,256]
        #       ↓
        # [B,1,256]
        # ----------------------------------------------------

        cls_token = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        # ----------------------------------------------------
        # Complete transformer sequence
        #
        # [CLS]
        # [TEXT]
        # [AUDIO]
        # [VISION]
        #
        # [B,4,256]
        # ----------------------------------------------------

        tokens = torch.cat(
            [
                cls_token,
                modality_tokens,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Cross-modal self-attention
        # ----------------------------------------------------

        transformed = self.transformer(
            tokens
        )

        # ----------------------------------------------------
        # CLS token becomes global fusion representation
        # ----------------------------------------------------

        fusion_embedding = transformed[:, 0]

        fusion_embedding = self.fusion_norm(
            fusion_embedding
        )

        # ----------------------------------------------------
        # Emotion classification
        # ----------------------------------------------------

        logits = self.classifier(
            fusion_embedding
        )

        return logits, fusion_embedding


# ============================================================
# MARK 2 FUSION MODEL (Matches checkpoints/mark2_best.pt)
# ============================================================

class Mark2FusionModel(nn.Module):
    """
    Trained Mark 2 Cross-Modal Transformer Fusion Model.
    Architecture matches checkpoints/mark2_best.pt exactly.
    """

    def __init__(
        self,
        input_dim=768,
        projection_dim=256,
        num_heads=8,
        num_layers=2,
        ff_dim=1024,
        dropout=0.1,
        num_emotions=6,
    ):
        super().__init__()

        self.text_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.audio_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.video_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, projection_dim)
        )

        self.modality_embeddings = nn.Parameter(
            torch.randn(1, 4, projection_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projection_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fusion_norm = nn.LayerNorm(projection_dim)

        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_emotions),
        )

    def forward(self, text, audio, video):
        text = self.text_projection(text)
        audio = self.audio_projection(audio)
        video = self.video_projection(video)

        batch_size = text.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)

        sequence = torch.cat(
            [
                cls,
                text.unsqueeze(1),
                audio.unsqueeze(1),
                video.unsqueeze(1),
            ],
            dim=1,
        )

        sequence = sequence + self.modality_embeddings
        encoded = self.transformer(sequence)
        fusion = encoded[:, 0, :]
        fusion = self.fusion_norm(fusion)
        logits = self.classifier(fusion)
        return logits, fusion


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MARK 2 - CROSS-MODAL FUSION TEST")
    print("=" * 60)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Create fusion model
    # --------------------------------------------------------

    model = CrossModalFusion(
        embedding_dim=256,
        num_modalities=3,
        num_layers=2,
        num_heads=8,
        feedforward_dim=1024,
        dropout=0.1,
        num_emotions=6,
    ).to(device)

    print()
    print("Fusion model created.")

    # --------------------------------------------------------
    # Fake encoder outputs
    # --------------------------------------------------------

    batch_size = 2

    text = torch.randn(
        batch_size,
        256,
        device=device,
    )

    audio = torch.randn(
        batch_size,
        256,
        device=device,
    )

    vision = torch.randn(
        batch_size,
        256,
        device=device,
    )

    print()
    print("Text embedding:", text.shape)
    print("Audio embedding:", audio.shape)
    print("Vision embedding:", vision.shape)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    print()
    print("Running cross-modal Transformer...")

    logits, fusion_embedding = model(
        text,
        audio,
        vision,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("Fusion embedding:")
    print("Shape:", fusion_embedding.shape)

    print()
    print("Emotion logits:")
    print("Shape:", logits.shape)

    print()
    print("First sample logits:")
    print(logits[0])

    print()
    print("=" * 60)
    print("CROSS-MODAL FUSION TEST COMPLETE")
    print("=" * 60)