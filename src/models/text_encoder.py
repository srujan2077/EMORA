import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Encodes a sequence of word-level embeddings into
    a fixed-size text representation.
    """

    def __init__(
        self,
        input_dim=300,
        hidden_dim=128,
        output_dim=128,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.projection = nn.Linear(
            hidden_dim * 2,
            output_dim
        )

    def forward(self, x):
        """
        Input:
            x: [batch_size, sequence_length, input_dim]

        Output:
            embedding: [batch_size, output_dim]
        """

        _, (hidden, _) = self.lstm(x)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1
        )

        combined = self.dropout(combined)

        embedding = self.projection(combined)

        return embedding