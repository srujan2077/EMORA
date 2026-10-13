"""
Mark 2 - ViT Vision Encoder

MOSEI video frames
        ↓
ViT-base-patch16-224
        ↓
768-D frame embeddings
        ↓
Temporal mean pooling
        ↓
Projection
        ↓
256-D video embedding
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class ViTVisionEncoder(nn.Module):

    MODEL_NAME = "google/vit-base-patch16-224-in21k"

    def __init__(
        self,
        output_dim=256,
        num_frames=8,
        freeze=True,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.freeze = freeze

        print("Loading ViT image processor...")
        self.processor = AutoImageProcessor.from_pretrained(
            self.MODEL_NAME
        )

        print("Loading ViT model...")
        self.encoder = AutoModel.from_pretrained(
            self.MODEL_NAME
        )

        hidden_size = self.encoder.config.hidden_size

        print("ViT hidden size:", hidden_size)

        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

            self.encoder.eval()

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, video):
        """
        Parameters
        ----------
        video:
            Tensor of shape:

                [B, T, 3, 224, 224]

            where:
                B = batch size
                T = number of frames

        Returns
        -------
        Tensor:
            [B, 256]
        """

        if video.dim() != 5:
            raise ValueError(
                f"Expected video tensor [B,T,3,H,W], "
                f"got {video.shape}"
            )

        batch_size, num_frames, channels, height, width = video.shape

        if channels != 3:
            raise ValueError(
                f"Expected 3 RGB channels, got {channels}"
            )

        # ----------------------------------------------------
        # Convert [0,1] tensors to uint8 images [0,255]
        # ----------------------------------------------------

        frames = (
            video.detach()
            .clamp(0, 1)
            .mul(255)
            .to(torch.uint8)
        )

        # Flatten:
        #
        # [B,T,3,H,W]
        #       ↓
        # [B*T,3,H,W]
        #
        frames = frames.reshape(
            batch_size * num_frames,
            channels,
            height,
            width,
        )

        # Processor expects images in H,W,C format.
        images = [
            frame.permute(1, 2, 0)
            .cpu()
            .numpy()
            for frame in frames
        ]

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        device = next(self.parameters()).device

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # ViT forward
        # ----------------------------------------------------

        if self.freeze:

            with torch.no_grad():
                outputs = self.encoder(**inputs)

        else:

            outputs = self.encoder(**inputs)

        # ViT output:
        #
        # [B*T, sequence_length, 768]
        #
        # CLS token = first token

        frame_embeddings = outputs.last_hidden_state[:, 0]

        # [B*T,768]
        #
        # reshape back to:
        #
        # [B,T,768]

        frame_embeddings = frame_embeddings.reshape(
            batch_size,
            num_frames,
            -1,
        )

        # ----------------------------------------------------
        # Temporal pooling
        # ----------------------------------------------------

        video_embedding = frame_embeddings.mean(
            dim=1
        )

        # [B,768] → [B,256]

        video_embedding = self.projection(
            video_embedding
        )

        return video_embedding


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MARK 2 - VIT VISION ENCODER TEST")
    print("=" * 60)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Initialize model
    # --------------------------------------------------------

    print()
    print("Initializing ViT...")

    model = ViTVisionEncoder(
        output_dim=256,
        num_frames=8,
        freeze=True,
    ).to(device)

    # --------------------------------------------------------
    # Synthetic video
    # --------------------------------------------------------

    print()
    print("Creating test video...")

    # Same format produced by MOSEIMark2Dataset:
    #
    # [batch, frames, channels, height, width]

    test_video = torch.rand(
        2,
        8,
        3,
        224,
        224,
        dtype=torch.float32,
    )

    print(
        "Input video shape:",
        test_video.shape
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    print()
    print("Running ViT...")

    with torch.inference_mode():

        embeddings = model(
            test_video.to(device)
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print(
        "Embedding shape:",
        embeddings.shape
    )

    print(
        "Embedding dtype:",
        embeddings.dtype
    )

    print(
        "Embedding device:",
        embeddings.device
    )

    print(
        "First embedding:",
        embeddings[0][:10]
    )

    print()
    print("=" * 60)
    print("VIT VISION ENCODER TEST COMPLETE")
    print("=" * 60)