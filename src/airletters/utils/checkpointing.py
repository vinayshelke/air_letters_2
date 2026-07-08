"""Checkpoint helpers for AirLetters experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: Mapping[str, float],
    config: Mapping[str, Any],
    label_to_index: Mapping[str, int],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """Save model state and experiment metadata."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": dict(metrics),
            "config": dict(config),
            "label_to_index": dict(label_to_index),
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_only: bool = False,
) -> dict[str, Any]:
    """Load model weights and return checkpoint metadata.

    Args:
        path: Path to the ``.pt`` checkpoint file.
        model: Model to load weights into. If *None* or *map_only* is True,
            weights are not loaded (only metadata is returned).
        device: Device to map tensors to.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional learning rate scheduler to restore state.
        map_only: If True, only load the checkpoint dict without applying
            weights to the model.
    """
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if not map_only:
        if model is not None and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint
