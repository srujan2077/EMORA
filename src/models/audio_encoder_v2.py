"""
Mark 2 - Wav2Vec2 Audio Encoder

Raw MOSEI audio waveform
        ↓
Wav2Vec2-base
        ↓
Mean pooling
        ↓
Projection
        ↓
256-D audio embedding
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel


class Wav2Vec2AudioEncoder(nn.Module):

    MODEL_NAME = "facebook/wav2vec2-base-960h"

    def __init__(
        self,
        output_dim=256,
        sampling_rate=16000,
        freeze=True,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.freeze = freeze

        print("Loading Wav2Vec2 processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_NAME
        )

        print("Loading Wav2Vec2 model...")
        self.encoder = AutoModel.from_pretrained(
            self.MODEL_NAME
        )

        hidden_size = self.encoder.config.hidden_size

        print("Wav2Vec2 hidden size:", hidden_size)

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

    @staticmethod
    def mean_pooling(hidden_states, attention_mask=None):
        """
        Mean-pool Wav2Vec2 frame representations.

        hidden_states:
            [batch, time, hidden]

        attention_mask:
            [batch, time]
        """

        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).float()

        masked_embeddings = hidden_states * mask

        summed = masked_embeddings.sum(dim=1)

        counts = mask.sum(dim=1).clamp(min=1e-9)

        return summed / counts

    def forward(self, audio):
        """
        Parameters
        ----------
        audio:
            Either:

            Tensor:
                [batch, samples]

            OR

            list of tensors:
                [samples]

        Returns
        -------
        Tensor:
            [batch, output_dim]
        """

        # Convert tensor batch into list of individual waveforms
        if isinstance(audio, torch.Tensor):

            if audio.dim() == 1:
                audio = [audio]

            elif audio.dim() == 2:
                audio = [
                    audio[i]
                    for i in range(audio.size(0))
                ]

            else:
                raise ValueError(
                    f"Expected audio tensor with 1 or 2 dimensions, "
                    f"got {audio.shape}"
                )

        # Convert everything to CPU numpy arrays for processor
        audio_arrays = [
            waveform.detach()
            .cpu()
            .float()
            .numpy()
            for waveform in audio
        ]

        # Wav2Vec2 processor handles variable-length audio
        inputs = self.processor(
            audio_arrays,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        device = next(self.parameters()).device

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # Frozen encoder = no gradient computation
        if self.freeze:

            with torch.no_grad():

                outputs = self.encoder(
                    **inputs
                )

        else:

            outputs = self.encoder(
                **inputs
            )

        # [B, time, hidden]
        hidden_states = outputs.last_hidden_state

        # Wav2Vec2 attention mask
        attention_mask = inputs.get(
            "attention_mask",
            None
        )

        # Mean pooling
        pooled = self.mean_pooling(
            hidden_states,
            attention_mask
        )

        # Project to common 256-D multimodal space
        embedding = self.projection(
            pooled
        )

        return embedding


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MARK 2 - WAV2VEC2 AUDIO ENCODER TEST")
    print("=" * 60)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    print()
    print("Initializing Wav2Vec2...")

    model = Wav2Vec2AudioEncoder(
        output_dim=256,
        sampling_rate=16000,
        freeze=True,
    ).to(device)

    # --------------------------------------------------------
    # Synthetic test audio
    # --------------------------------------------------------

    print()
    print("Creating test audio...")

    duration = 3.0

    sample_rate = 16000

    num_samples = int(
        duration * sample_rate
    )

    # Simple 440 Hz sine wave
    time = torch.arange(
        num_samples,
        dtype=torch.float32
    ) / sample_rate

    waveform = torch.sin(
        2 * torch.pi * 440 * time
    )

    # Two examples
    test_audio = [
        waveform,
        waveform[: int(2.5 * sample_rate)]
    ]

    print(
        "Audio 1:",
        test_audio[0].shape
    )

    print(
        "Audio 2:",
        test_audio[1].shape
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    print()
    print("Running Wav2Vec2...")

    with torch.inference_mode():

        embeddings = model(
            test_audio
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("Input batch:", len(test_audio))

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
    print("WAV2VEC2 AUDIO ENCODER TEST COMPLETE")
    print("=" * 60)