import torch
import torch.nn as nn

from .text_encoder import TextEncoder
from .audio_encoder import AudioEncoder
from .vision_encoder import VisionEncoder


class MultimodalEmotionModel(nn.Module):
    """
    Mark 1 multimodal emotion recognition model.

    Inputs:
        Text   -> GloVe word vectors
        Audio  -> COVAREP features
        Vision -> FACET 4.2 features

    Outputs:
        Six emotion logits:
        happy, sad, anger, surprise, disgust, fear
    """

    def __init__(
        self,
        embedding_dim=128,
        fusion_dim=256,
        num_emotions=6,
        dropout=0.2
    ):
        super().__init__()

        # Individual modality encoders
        self.text_encoder = TextEncoder(
            input_dim=300,
            output_dim=embedding_dim
        )

        self.audio_encoder = AudioEncoder(
            input_dim=74,
            output_dim=embedding_dim
        )

        self.vision_encoder = VisionEncoder(
            input_dim=35,
            output_dim=embedding_dim
        )

        # 128 + 128 + 128 = 384
        fusion_input_dim = embedding_dim * 3

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Emotion classifier
        self.classifier = nn.Linear(
            fusion_dim,
            num_emotions
        )

    def forward(self, text, audio, vision):
        """
        Inputs:
            text:
                [batch, text_sequence, 300]

            audio:
                [batch, audio_sequence, 74]

            vision:
                [batch, vision_sequence, 35]

        Output:
            logits:
                [batch, 6]
        """

        # Encode each modality
        text_embedding = self.text_encoder(text)
        audio_embedding = self.audio_encoder(audio)
        vision_embedding = self.vision_encoder(vision)

        # Concatenate modality representations
        fused = torch.cat(
            [
                text_embedding,
                audio_embedding,
                vision_embedding
            ],
            dim=1
        )

        # Learn multimodal representation
        fused = self.fusion(fused)

        # Predict six emotions
        logits = self.classifier(fused)

        return logits