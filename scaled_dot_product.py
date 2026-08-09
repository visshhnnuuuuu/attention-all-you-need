import math
import torch
import torch.nn as nn
class ScaleDotProductAttention(nn.Module):

    def __init__(self, d_model):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x):

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1))

        scores = scores / math.sqrt(Q.size(-1))

        attention = torch.softmax(scores, dim=-1)

        output = torch.matmul(attention, V)

        return output