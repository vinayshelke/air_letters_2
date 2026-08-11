"""Training pipeline for the CNN+BiLSTM AirLetters baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch import nn

from airletters.config import get_artifact_dir, load_config
from airletters.data import create_dataloaders
from airletters.models import create_model
from airletters.utils.checkpointing import load_checkpoint, save_checkpoint
from airletters.utils.plots import save_training_curves
from airletters.utils.reproducibility import seed_everything
from airletters.utils.train_eval import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN+BiLSTM on AirLetters")
    parser.add_argument("--config", default="configs/digits.yaml", help="Path to a YAML config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches for smoke tests.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit validation batches for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["project"]["seed"]))

    training_config = config["training"]
    requested_device = training_config["device"]
    device = torch.device(requested_device if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    mixed_precision = bool(training_config["mixed_precision"]) and device.type == "cuda"

    dataloaders, label_to_index = create_dataloaders(config)
    model = create_model(config, num_classes=len(label_to_index)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training_config["label_smoothing"]))
    optimizer = _create_optimizer(model, training_config)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    # Learning rate scheduler — cosine annealing decays LR smoothly to near zero
    epochs = args.epochs or int(training_config["epochs"])
    scheduler_name = str(training_config.get("scheduler", "none")).lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        print(f"Using CosineAnnealingLR scheduler (T_max={epochs})")
    else:
        scheduler = None

    # Resume checkpoint state if requested
    start_epoch = 0
    best_metric = float("-inf")
    if args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            checkpoint = load_checkpoint(
                resume_path,
                model=model,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            start_epoch = int(checkpoint["epoch"])
            best_metric = float(checkpoint["metrics"].get(training_config["save_best_metric"], float("-inf")))
            print(f"Resumed training from {resume_path} at epoch {start_epoch} (best metric so far: {best_metric:.4f})")
        else:
            print(f"Warning: resume checkpoint not found at: {args.resume}. Starting from scratch.")

    checkpoints_dir = get_artifact_dir(config, "checkpoints_dir")
    logs_dir = get_artifact_dir(config, "logs_dir")
    outputs_dir = get_artifact_dir(config, "outputs_dir")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / "metrics.jsonl"
    
    # Only clear metrics log if starting a new run from scratch
    if args.resume is None:
        metrics_path.write_text("", encoding="utf-8")

    # epochs already computed above for the scheduler
    best_metric_name = str(training_config["save_best_metric"])

    print(f"Device: {device}")
    print(f"Train samples: {len(dataloaders['train'].dataset)}")
    print(f"Val samples: {len(dataloaders['val'].dataset)}")
    print(f"Classes: {len(label_to_index)}")
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Model parameters: {total_params / 1_000_000:.2f}M total, {trainable_params / 1_000_000:.2f}M trainable")

    for epoch in range(start_epoch + 1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            mixed_precision=mixed_precision,
            grad_clip_norm=float(training_config["grad_clip_norm"]),
            epoch=epoch,
            log_every_n_steps=int(training_config["log_every_n_steps"]),
            max_batches=args.max_train_batches,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            split_name="val",
            max_batches=args.max_val_batches,
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        _append_jsonl(metrics_path, metrics)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {metrics['train_loss']:.4f} acc {metrics['train_accuracy']:.4f} | "
            f"val loss {metrics['val_loss']:.4f} acc {metrics['val_accuracy']:.4f} | "
            f"lr {current_lr:.2e}"
        )

        # Step the scheduler after each epoch
        if scheduler is not None:
            scheduler.step()

        save_checkpoint(
            checkpoints_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
            label_to_index=label_to_index,
            scheduler=scheduler,
        )

        current_metric = float(metrics[best_metric_name])
        if current_metric > best_metric:
            best_metric = current_metric
            save_checkpoint(
                checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
                label_to_index=label_to_index,
                scheduler=scheduler,
            )

    # Save training curves to outputs/
    save_training_curves(metrics_path, outputs_dir)


def _append_jsonl(path: Path, row: dict[str, float | int]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n")


def _create_optimizer(model: nn.Module, training_config: dict) -> torch.optim.Optimizer:
    optimizer_name = str(training_config["optimizer"]).lower()
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config["weight_decay"])

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


if __name__ == "__main__":
    main()
