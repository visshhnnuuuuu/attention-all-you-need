import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

        self.eps = eps

    def forward(self, x):

        # Mean across embedding/features
        mean = x.mean(dim=-1, keepdim=True)

        # Variance across embedding/features
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(variance + self.eps)

        # Scale and shift
        output = self.gamma * x_norm + self.beta

        return output