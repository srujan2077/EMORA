import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    Encodes a sequence of acoustic features into
    a fixed-size audio representation.
    """

    def __init__(
        self,
        input_dim=74,
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