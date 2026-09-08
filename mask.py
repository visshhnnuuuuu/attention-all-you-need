import torch


def create_causal_mask(seq_len, device=None):

    mask = torch.tril(
        torch.ones(seq_len, seq_len, device=device)
    )

    return mask 