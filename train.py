import torch
import torch.nn as nn
import torch.optim as optim


def create_padding_mask(tokens, pad_token_id):

    mask = (
        tokens != pad_token_id
    ).unsqueeze(1).unsqueeze(2)

    return mask


def create_causal_mask(seq_len, device):

    mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            device=device
        )
    )

    return mask


def create_target_mask(
    decoder_input,
    pad_token_id,
    device
):

    seq_len = decoder_input.size(1)

    # Causal mask
    causal_mask = create_causal_mask(
        seq_len,
        device
    )

    causal_mask = (
        causal_mask
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # Padding mask
    padding_mask = create_padding_mask(
        decoder_input,
        pad_token_id
    )

    # Combine causal + padding mask
    target_mask = (
        causal_mask
        * padding_mask
    )

    return target_mask


def evaluate(
    model,
    dataloader,
    pad_token_id,
    src_pad_token_id,
    device="cpu"
):

    model.eval()

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_token_id,
        label_smoothing=0.1
    )

    total_loss = 0

    with torch.no_grad():

        for src, tgt in dataloader:

            src = src.to(device)
            tgt = tgt.to(device)

            # Teacher forcing
            decoder_input = tgt[:, :-1]
            target = tgt[:, 1:]

            # Source padding mask
            src_mask = create_padding_mask(
                src,
                src_pad_token_id
            )

            # Target causal + padding mask
            tgt_mask = create_target_mask(
                decoder_input,
                pad_token_id,
                device
            )

            logits = model(
                src=src,
                tgt=decoder_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                cross_mask=src_mask
            )

            loss = criterion(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                target.reshape(-1)
            )

            total_loss += loss.item()

    average_loss = (
        total_loss / len(dataloader)
    )

    return average_loss


def train(
    model,
    dataloader,
    pad_token_id,
    src_pad_token_id,
    validation_loader=None,
    num_epochs=10,
    learning_rate=1e-3,
    device="cpu"
):

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_token_id,
        label_smoothing=0.1
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0

        for src, tgt in dataloader:

            src = src.to(device)
            tgt = tgt.to(device)

            # Teacher forcing
            decoder_input = tgt[:, :-1]
            target = tgt[:, 1:]

            # Source padding mask
            src_mask = create_padding_mask(
                src,
                src_pad_token_id
            )

            # Target causal + padding mask
            tgt_mask = create_target_mask(
                decoder_input,
                pad_token_id,
                device
            )

            # Forward pass
            logits = model(
                src=src,
                tgt=decoder_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                cross_mask=src_mask
            )

            # Calculate loss
            loss = criterion(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                target.reshape(-1)
            )

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        train_loss = (
            total_loss / len(dataloader)
        )

        if validation_loader is not None:

            validation_loss = evaluate(
                model=model,
                dataloader=validation_loader,
                pad_token_id=pad_token_id,
                src_pad_token_id=src_pad_token_id,
                device=device
            )

            print(
                f"Epoch {epoch + 1}/{num_epochs} "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {validation_loss:.4f}"
            )

        else:

            print(
                f"Epoch {epoch + 1}/{num_epochs} "
                f"Train Loss: {train_loss:.4f}"
            )