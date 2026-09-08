import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from generator import Generator


class Transformer(nn.Module):

    def __init__(
        self,
        src_embedding,
        tgt_embedding,
        src_positional_encoding,
        tgt_positional_encoding,
        d_model,
        num_heads,
        d_ff,
        src_vocab_size,
        tgt_vocab_size,
        num_layers=6,
        dropout=0.1
    ):
        super().__init__()

        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding

        self.src_positional_encoding = src_positional_encoding
        self.tgt_positional_encoding = tgt_positional_encoding

        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout
        )

        self.generator = Generator(
            d_model=d_model,
            vocab_size=tgt_vocab_size
        )

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        cross_mask=None
    ):

        src = self.src_embedding(src)

        src = self.src_positional_encoding(src)

        encoder_output = self.encoder(
            src,
            src_mask
        )

        tgt = self.tgt_embedding(tgt)

        tgt = self.tgt_positional_encoding(tgt)

        decoder_output = self.decoder(
            tgt,
            encoder_output,
            self_mask=tgt_mask,
            cross_mask=cross_mask
        )

        logits = self.generator(
            decoder_output
        )

        return logits