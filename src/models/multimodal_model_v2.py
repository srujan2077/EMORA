"""
MARK 2 - FULL MULTIMODAL EMOTION MODEL

Text
  -> DeBERTa-v3
  -> 256-D

Audio
  -> Wav2Vec2
  -> 256-D

Video
  -> ViT
  -> 256-D

          ↓

Cross-Modal Transformer

          ↓

256-D Fusion Representation

          ↓

6 Emotion Logits
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# MARK 2 ENCODERS
# ============================================================

from src.models.text_encoder_v2 import DeBERTaTextEncoder
from src.models.audio_encoder_v2 import Wav2Vec2AudioEncoder
from src.models.vision_encoder_v2 import ViTVisionEncoder
from src.models.fusion_v2 import CrossModalFusion


# ============================================================
# FULL MARK 2 MODEL
# ============================================================

class MultimodalEmotionModelV2(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
        num_emotions=6,
        freeze_encoders=True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_emotions = num_emotions

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        self.text_encoder = DeBERTaTextEncoder(
            output_dim=embedding_dim,
            max_length=128,
            freeze=freeze_encoders,
        )

        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        self.audio_encoder = Wav2Vec2AudioEncoder(
            output_dim=embedding_dim,
            sampling_rate=16000,
            freeze=freeze_encoders,
        )

        # ----------------------------------------------------
        # VISION
        # ----------------------------------------------------

        self.vision_encoder = ViTVisionEncoder(
            output_dim=embedding_dim,
            num_frames=8,
            freeze=freeze_encoders,
        )

        # ----------------------------------------------------
        # CROSS-MODAL FUSION
        # ----------------------------------------------------

        self.fusion = CrossModalFusion(
            embedding_dim=embedding_dim,
            num_modalities=3,
            num_layers=2,
            num_heads=8,
            feedforward_dim=1024,
            dropout=0.1,
            num_emotions=num_emotions,
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        texts,
        audio,
        video,
    ):
        """
        Parameters
        ----------
        texts:
            list[str]

        audio:
            list[Tensor]
            or Tensor [B,T]

        video:
            Tensor [B,8,3,224,224]

        Returns
        -------
        logits:
            [B,6]

        fusion_embedding:
            [B,256]
        """

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        text_embedding = self.text_encoder(
            texts
        )

        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        audio_embedding = self.audio_encoder(
            audio
        )

        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        vision_embedding = self.vision_encoder(
            video
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        logits, fusion_embedding = self.fusion(
            text_embedding,
            audio_embedding,
            vision_embedding,
        )

        return logits, fusion_embedding


# ============================================================
# TEST WITH REAL MOSEI SAMPLE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MARK 2 - FULL MULTIMODAL MODEL TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print()
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Import dataset
    # --------------------------------------------------------

    from src.data.mosei_mark2 import MOSEIMark2Dataset

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("Loading Mark 2 dataset...")

    dataset = MOSEIMark2Dataset(
        segment_ids=[
            "-ZgjBOA1Yhw[4]"
        ]
    )

    print(
        "Dataset size:",
        len(dataset)
    )

    # --------------------------------------------------------
    # Load real sample
    # --------------------------------------------------------

    print()
    print("Loading real MOSEI sample...")

    sample = dataset[0]

    text = sample["text"]
    audio = sample["audio"]
    video = sample["video"]
    labels = sample["labels"]

    print()
    print("Segment:", sample["segment_id"])

    print()
    print("Transcript:")
    print(text)

    print()
    print("Audio shape:")
    print(audio.shape)

    print()
    print("Video shape:")
    print(video.shape)

    print()
    print("Labels:")
    print(labels)

    # --------------------------------------------------------
    # Add batch dimension to video
    # --------------------------------------------------------

    video = video.unsqueeze(0)

    # Audio stays as list because lengths can vary
    audio = [audio]

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Initializing complete Mark 2 model...")

    model = MultimodalEmotionModelV2(
        embedding_dim=256,
        num_emotions=6,
        freeze_encoders=True,
    ).to(device)

    model.eval()

    # --------------------------------------------------------
    # Full forward pass
    # --------------------------------------------------------

    print()
    print("Running FULL MULTIMODAL FORWARD PASS...")

    with torch.inference_mode():

        logits, fusion_embedding = model(
            texts=[text],
            audio=audio,
            video=video.to(device),
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MARK 2 RESULTS")
    print("=" * 70)

    print()
    print("Text embedding:")
    print(
        model.text_encoder([text]).shape
    )

    print()
    print("Audio embedding:")
    print(
        model.audio_encoder(audio).shape
    )

    print()
    print("Vision embedding:")
    print(
        model.vision_encoder(video.to(device)).shape
    )

    print()
    print("Fusion embedding:")
    print(
        "Shape:",
        fusion_embedding.shape
    )

    print()
    print("Emotion logits:")
    print(
        "Shape:",
        logits.shape
    )

    print()
    print("Raw logits:")
    print(logits)

    # --------------------------------------------------------
    # Convert logits to probabilities
    # --------------------------------------------------------

    probabilities = torch.sigmoid(
        logits
    )

    print()
    print("Emotion probabilities:")

    emotion_names = [
        "Happy",
        "Sad",
        "Anger",
        "Surprise",
        "Disgust",
        "Fear",
    ]

    for name, probability in zip(
        emotion_names,
        probabilities[0],
    ):
        print(
            f"{name:10s}: "
            f"{probability.item():.4f}"
        )

    # --------------------------------------------------------
    # Binary predictions
    # --------------------------------------------------------

    predictions = (
        probabilities >= 0.5
    ).float()

    print()
    print("Binary predictions:")

    for name, prediction in zip(
        emotion_names,
        predictions[0],
    ):
        print(
            f"{name:10s}: "
            f"{int(prediction.item())}"
        )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print()
    print("Ground truth:")

    for name, label in zip(
        emotion_names,
        labels,
    ):
        print(
            f"{name:10s}: "
            f"{int(label.item())}"
        )

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    if device.type == "cuda":

        allocated = (
            torch.cuda.memory_allocated(device)
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved(device)
            / (1024 ** 3)
        )

        print()
        print("GPU memory allocated:")
        print(f"{allocated:.2f} GB")

        print()
        print("GPU memory reserved:")
        print(f"{reserved:.2f} GB")

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FULL MARK 2 MULTIMODAL TEST COMPLETE")
    print("=" * 70)