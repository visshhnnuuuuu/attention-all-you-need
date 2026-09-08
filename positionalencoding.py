import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        self.d_model = d_model

        # ----------------------------------------
        # Create positional encoding matrix
        # Shape: (max_len, d_model)
        # ----------------------------------------

        pe = torch.zeros(
            max_len,
            d_model
        )

        # ----------------------------------------
        # Position indices
        # Shape: (max_len, 1)
        # ----------------------------------------

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        # ----------------------------------------
        # Frequency terms
        # ----------------------------------------

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (-math.log(10000.0) / d_model)
        )

        # ----------------------------------------
        # Sin for even dimensions
        # ----------------------------------------

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        # ----------------------------------------
        # Cos for odd dimensions
        # ----------------------------------------

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # Add batch dimension
        # (1, max_len, d_model)

        pe = pe.unsqueeze(0)

        # Store as non-trainable buffer

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        # x:
        # (batch_size, sequence_length, d_model)

        seq_len = x.size(1)

        # Scale embeddings

        x = x * math.sqrt(
            self.d_model
        )

        # Add positional information

        x = x + self.pe[:, :seq_len, :]

        return x