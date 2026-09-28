import torch
from torch.utils.data import DataLoader

from .mosei_dataset import MOSEIDataset
from .collate import multimodal_collate


def create_dataloaders(
    dataset,
    train_ids,
    val_ids,
    test_ids,
    batch_size=8,
    num_workers=0
):
    """
    Create train, validation, and test DataLoaders.
    """

    train_dataset = MOSEIDataset(
        dataset,
        train_ids
    )

    val_dataset = MOSEIDataset(
        dataset,
        val_ids
    )

    test_dataset = MOSEIDataset(
        dataset,
        test_ids
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multimodal_collate,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multimodal_collate,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multimodal_collate,
        num_workers=num_workers
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )