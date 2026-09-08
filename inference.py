import torch


def create_padding_mask(
    tokens,
    pad_token_id
):

    mask = (
        tokens != pad_token_id
    ).unsqueeze(1).unsqueeze(2)

    return mask


def create_causal_mask(
    seq_len,
    device
):

    return torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            device=device
        )
    )


def generate(
    model,
    src_tokenizer,
    tgt_tokenizer,
    src_text,
    max_len=48,
    device="cpu"
):

    model = model.to(device)

    model.eval()

    # =========================================
    # Encode source
    # =========================================

    src_tokens = src_tokenizer.encode_text(
        src_text
    )

    src_tokens = (
        [src_tokenizer.sos_token]
        + src_tokens
        + [src_tokenizer.eos_token]
    )

    src_tokens = src_tokens[:max_len]

    src = torch.tensor(
        [src_tokens],
        dtype=torch.long,
        device=device
    )

    # =========================================
    # Source padding mask
    # =========================================

    src_mask = create_padding_mask(
        src,
        src_tokenizer.pad_token
    )

    # =========================================
    # Start decoder with <SOS>
    # =========================================

    tgt = torch.tensor(
        [[tgt_tokenizer.sos_token]],
        dtype=torch.long,
        device=device
    )

    # =========================================
    # Autoregressive generation
    # =========================================

    for _ in range(max_len - 1):

        tgt_seq_len = tgt.size(1)

        tgt_mask = create_causal_mask(
            tgt_seq_len,
            device
        )

        with torch.no_grad():

            logits = model(
                src=src,
                tgt=tgt,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                cross_mask=src_mask
            )

        # Last position
        next_token_logits = (
            logits[:, -1, :]
        )

        # Greedy decoding
        next_token = torch.argmax(
            next_token_logits,
            dim=-1
        ).item()

        # Stop at EOS
        if (
            next_token
            == tgt_tokenizer.eos_token
        ):

            break

        next_token_tensor = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=device
        )

        tgt = torch.cat(
            [
                tgt,
                next_token_tensor
            ],
            dim=1
        )

    # =========================================
    # Decode
    # =========================================

    generated_tokens = tgt[0].tolist()

    generated_tokens = generated_tokens[1:]

    generated_text = (
        tgt_tokenizer.decode_tokens(
            generated_tokens
        )
    )

    return generated_text