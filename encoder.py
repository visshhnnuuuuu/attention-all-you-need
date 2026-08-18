import torch
import torch.nn as nn

from encoder_block import EncoderBlock


class Encoder(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, num_layers=6):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):

        for layer in self.layers:
            x = layer(x, mask)

        return x