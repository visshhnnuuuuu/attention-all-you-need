import torch
import torch.nn as nn

from layer_norm import LayerNorm
from multihead_attention import MultiHeadAttention
from feed_forward import FeedForward


class DecoderBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):
        super().__init__()

        # 1. Masked Self-Attention
        self.norm1 = LayerNorm(d_model)

        self.self_attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.dropout1 = nn.Dropout(dropout)

        # 2. Cross-Attention
        self.norm2 = LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.dropout2 = nn.Dropout(dropout)

        # 3. Feed Forward
        self.norm3 = LayerNorm(d_model)

        self.feed_forward = FeedForward(
            d_model,
            d_ff
        )

        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x,
        encoder_output,
        self_mask=None,
        cross_mask=None
    ):

        # Self-Attention
        attention_output = self.self_attention(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            mask=self_mask
        )

        attention_output = self.dropout1(
            attention_output
        )

        x = x + attention_output

        # Cross-Attention
        cross_attention_output = self.cross_attention(
            self.norm2(x),
            encoder_output,
            encoder_output,
            mask=cross_mask
        )

        cross_attention_output = self.dropout2(
            cross_attention_output
        )

        x = x + cross_attention_output

        # Feed Forward
        feed_forward_output = self.feed_forward(
            self.norm3(x)
        )

        feed_forward_output = self.dropout3(
            feed_forward_output
        )

        x = x + feed_forward_output

        return x