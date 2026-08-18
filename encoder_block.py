import torch
import torch.nn as nn

from layer_norm import LayerNorm
from multihead_attantion import MultiHeadAttention
from feed_forward import FeedForward


class EncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.norm1 = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)

        self.norm2 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x, mask=None):

        # Self-Attention + Residual Connection
        x = x + self.attention(self.norm1(x), mask)

        # Feed Forward + Residual Connection
        x = x + self.feed_forward(self.norm2(x))

        return x