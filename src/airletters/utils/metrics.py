"""Metric helpers."""

from __future__ import annotations

import torch


def batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    """Return number of correct predictions and total samples."""
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return int(correct), int(targets.numel())

