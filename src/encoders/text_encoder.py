"""EMORA text encoder wrapper using DeBERTa-v3-base."""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from src.encoders.common import (
    EMBEDDING_DIM,
    MAX_TOKEN_LENGTH,
    TEXT_MODEL_NAME,
    resolve_device,
)

logger = logging.getLogger(__name__)


class TextEncoder:
    """Wraps microsoft/deberta-v3-base to generate 768-d text embeddings.

    Replicates the exact training-time tokenization and pooling from
    scripts/extract_mark2_features.py:
    - max_length: 256
    - truncation: True
    - padding: True
    - attention-mask-aware mean pooling with clamp(min=1)
    - empty/non-string input produces zeros(768)
    """

    def __init__(
        self,
        model_name: str = TEXT_MODEL_NAME,
        device: torch.device | str | None = None,
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.tokenizer = tokenizer
        self.model = model

    def load(self) -> TextEncoder:
        """Load tokenizer and model if not already provided."""
        if self.tokenizer is None:
            logger.info("Loading text tokenizer: %s", self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            logger.info("Loading text model: %s on %s", self.model_name, self.device)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        return self

    @torch.inference_mode()
    def encode(self, text: str | None) -> torch.Tensor:
        """Encode a single text utterance into a 768-d embedding tensor.

        Parameters
        ----------
        text : input string or None

        Returns
        -------
        torch.Tensor of shape (768,), dtype float32, on CPU.
        """
        if not isinstance(text, str) or not text.strip():
            return torch.zeros(EMBEDDING_DIM, dtype=torch.float32)

        if self.tokenizer is None or self.model is None:
            self.load()

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            padding=True,
        )

        encoded = {key: val.to(self.device) for key, val in encoded.items()}

        outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state  # [1, seq_len, 768]

        attention_mask = encoded["attention_mask"].unsqueeze(-1)  # [1, seq_len, 1]
        masked_hidden = hidden * attention_mask
        summed = masked_hidden.sum(dim=1)
        count = attention_mask.sum(dim=1).clamp(min=1)

        embedding = summed / count
        return embedding.squeeze(0).float().cpu()
