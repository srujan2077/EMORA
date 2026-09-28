import torch


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device
):
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:

        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(
            text,
            audio,
            vision
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:

        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            text,
            audio,
            vision
        )

        loss = criterion(
            logits,
            labels
        )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches