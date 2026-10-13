"""
Mark 2 - DeBERTa-v3 Text Encoder

Raw transcript
      ↓
DeBERTa-v3
      ↓
Mean pooling
      ↓
Projection
      ↓
Text embedding
"""

import torch
import torch.nn as nn

from transformers import (
    AutoModel,
    AutoTokenizer,
)


class DeBERTaTextEncoder(nn.Module):
    """
    Frozen DeBERTa-v3 encoder for Mark 2.

    Input:
        List[str] of transcript segments

    Output:
        [batch_size, output_dim]
    """

    MODEL_NAME = "microsoft/deberta-v3-base"

    def __init__(
        self,
        output_dim=256,
        max_length=128,
        freeze=True,
    ):
        super().__init__()

        self.max_length = max_length

        # ---------------------------------------------------------
        # Tokenizer
        # ---------------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

        # ---------------------------------------------------------
        # Pretrained DeBERTa
        # ---------------------------------------------------------

        self.encoder = AutoModel.from_pretrained(
            self.MODEL_NAME
        )

        hidden_size = (
            self.encoder.config.hidden_size
        )

        # ---------------------------------------------------------
        # Freeze pretrained model initially
        # ---------------------------------------------------------

        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

            self.encoder.eval()

        self.freeze = freeze

        # ---------------------------------------------------------
        # Projection
        # ---------------------------------------------------------

        self.projection = nn.Sequential(
            nn.Linear(
                hidden_size,
                output_dim,
            ),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    # =============================================================
    # Mean pooling
    # =============================================================

    @staticmethod
    def mean_pooling(
        hidden_states,
        attention_mask,
    ):
        """
        Mean-pool token embeddings while ignoring padding.
        """

        mask = attention_mask.unsqueeze(-1).float()

        masked_embeddings = (
            hidden_states * mask
        )

        summed = masked_embeddings.sum(
            dim=1
        )

        counts = mask.sum(
            dim=1
        ).clamp(
            min=1e-9
        )

        return summed / counts

    # =============================================================
    # Forward
    # =============================================================

    def forward(self, texts):

        if isinstance(texts, str):
            texts = [texts]

        # ---------------------------------------------------------
        # Tokenization
        # ---------------------------------------------------------

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        device = next(
            self.parameters()
        ).device

        tokens = {
            key: value.to(device)
            for key, value in tokens.items()
        }

        # ---------------------------------------------------------
        # DeBERTa forward
        # ---------------------------------------------------------

        if self.freeze:

            with torch.no_grad():
                outputs = self.encoder(
                    **tokens
                )

        else:

            outputs = self.encoder(
                **tokens
            )

        # ---------------------------------------------------------
        # Mean pooling
        # ---------------------------------------------------------

        pooled = self.mean_pooling(
            outputs.last_hidden_state,
            tokens["attention_mask"],
        )

        # ---------------------------------------------------------
        # Projection
        # ---------------------------------------------------------

        embedding = self.projection(
            pooled
        )

        return embedding


# =============================================================
# Standalone test
# =============================================================

if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("MARK 2 - DEBERTA TEXT ENCODER TEST")
    print("=" * 60)

    print()
    print("Device:")
    print(device)

    print()
    print("Loading DeBERTa-v3...")

    model = DeBERTaTextEncoder(
        output_dim=256,
        max_length=128,
        freeze=True,
    )

    model = model.to(device)

    test_texts = [
        (
            "impose them if it is profitable for them "
            "to pay at the expense of consumers"
        ),
        (
            "this is another example transcript "
            "for testing the text encoder"
        ),
    ]

    with torch.inference_mode():

        embeddings = model(
            test_texts
        )

    print()
    print("Input batch:")
    print(len(test_texts))

    print()
    print("Embedding shape:")
    print(embeddings.shape)

    print()
    print("Embedding dtype:")
    print(embeddings.dtype)

    print()
    print("Embedding device:")
    print(embeddings.device)

    print()
    print("First embedding:")
    print(
        embeddings[0][:10]
    )

    print()
    print("=" * 60)
    print("DEBERTA TEXT ENCODER TEST COMPLETE")
    print("=" * 60)