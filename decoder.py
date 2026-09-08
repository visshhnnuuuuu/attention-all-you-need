import torch
import torch.nn as nn

from decoder_block import DecoderBlock


class Decoder(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        num_layers=6,
        dropout=0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x,
        encoder_output,
        self_mask=None,
        cross_mask=None
    ):

        for layer in self.layers:

            x = layer(
                x,
                encoder_output,
                self_mask,
                cross_mask
            )

        return x