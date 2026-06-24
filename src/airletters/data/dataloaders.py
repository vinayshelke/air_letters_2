"""Dataloader builders for AirLetters experiments."""

from __future__ import annotations

from typing import Any, Mapping

from torch.utils.data import DataLoader

from airletters.data.dataset import AirLettersDataset


def create_dataloaders(config: Mapping[str, Any]) -> tuple[dict[str, DataLoader], dict[str, int]]:
    """Create train, validation, and test dataloaders with shared labels."""
    train_dataset = AirLettersDataset.from_config(config, split="train", load_video=True)
    label_to_index = train_dataset.label_to_index

    val_dataset = AirLettersDataset.from_config(
        config,
        split="val",
        label_to_index=label_to_index,
        load_video=True,
    )
    test_dataset = AirLettersDataset.from_config(
        config,
        split="test",
        label_to_index=label_to_index,
        load_video=True,
    )

    training_config = config["training"]
    evaluation_config = config["evaluation"]

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=int(training_config["batch_size"]),
            shuffle=True,
            num_workers=int(training_config["num_workers"]),
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=int(evaluation_config["batch_size"]),
            shuffle=False,
            num_workers=int(evaluation_config["num_workers"]),
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=int(evaluation_config["batch_size"]),
            shuffle=False,
            num_workers=int(evaluation_config["num_workers"]),
            pin_memory=True,
        ),
    }

    return dataloaders, label_to_index

