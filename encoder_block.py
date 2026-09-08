import torch.nn as nn

from layer_norm import LayerNorm
from multihead_attention import MultiHeadAttention
from feed_forward import FeedForward


class EncoderBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = LayerNorm(d_model)

        self.attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(d_model)

        self.feed_forward = FeedForward(
            d_model,
            d_ff
        )

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        normalized_x = self.norm1(x)

        attention_output = self.attention(
            normalized_x,
            normalized_x,
            normalized_x,
            mask=mask
        )

        attention_output = self.dropout1(
            attention_output
        )

        x = x + attention_output

        normalized_x = self.norm2(x)

        feed_forward_output = self.feed_forward(
            normalized_x
        )

        feed_forward_output = self.dropout2(
            feed_forward_output
        )

        x = x + feed_forward_output

        return x