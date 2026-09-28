import torch
from torch.nn.utils.rnn import pad_sequence


def multimodal_collate(batch):
    """
    Pads variable-length multimodal sequences
    so they can be processed as a batch.
    """

    text = [item["text"] for item in batch]
    audio = [item["audio"] for item in batch]
    vision = [item["vision"] for item in batch]

    labels = torch.stack(
        [item["labels"] for item in batch]
    )

    segment_ids = [
        item["segment_id"]
        for item in batch
    ]

    # Pad sequences to the longest sequence
    # in the current batch.
    text = pad_sequence(
        text,
        batch_first=True
    )

    audio = pad_sequence(
        audio,
        batch_first=True
    )

    vision = pad_sequence(
        vision,
        batch_first=True
    )

    return {
        "text": text,
        "audio": audio,
        "vision": vision,
        "labels": labels,
        "segment_ids": segment_ids
    }