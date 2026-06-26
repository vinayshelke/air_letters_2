"""Training and evaluation loops for the MediaPipe Landmark Transformer.

These functions mirror ``airletters.utils.train_eval`` but read
``batch["landmarks"]`` instead of ``batch["video"]``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from airletters.utils.metrics import batch_accuracy


def train_one_epoch_landmark(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    mixed_precision: bool,
    grad_clip_norm: float | None,
    epoch: int,
    log_every_n_steps: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Train the landmark transformer for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc=f"train epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        if max_batches is not None and step > max_batches:
            break

        landmarks = batch["landmarks"].to(device, non_blocking=True)   # (B, T, 63)
        targets = batch["label_id"].to(device, non_blocking=True)       # (B,)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=mixed_precision):
            logits = model(landmarks)
            loss = criterion(logits, targets)

        if scaler is not None and mixed_precision:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_correct, batch_total = batch_accuracy(logits.detach(), targets)
        running_loss += float(loss.item()) * batch_total
        correct += batch_correct
        total += batch_total

        if step % log_every_n_steps == 0:
            progress.set_postfix(loss=running_loss / total, accuracy=correct / total)

    return {"loss": running_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate_landmark(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    max_batches: int | None = None,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    """Evaluate the landmark transformer on one split."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for step, batch in enumerate(
        tqdm(dataloader, desc=f"evaluate {split_name}", leave=False), start=1
    ):
        if max_batches is not None and step > max_batches:
            break

        landmarks = batch["landmarks"].to(device, non_blocking=True)
        targets = batch["label_id"].to(device, non_blocking=True)

        logits = model(landmarks)
        loss = criterion(logits, targets)

        batch_correct, batch_total = batch_accuracy(logits, targets)
        predictions = logits.argmax(dim=1)
        running_loss += float(loss.item()) * batch_total
        correct += batch_correct
        total += batch_total

        if collect_predictions:
            all_targets.extend(targets.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    metrics: dict[str, Any] = {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "num_samples": total,
    }
    if collect_predictions:
        metrics["targets"] = all_targets
        metrics["predictions"] = all_predictions
    return metrics
