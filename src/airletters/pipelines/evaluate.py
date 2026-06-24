"""Evaluation pipeline for trained AirLetters checkpoints."""

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
from airletters.utils.checkpointing import load_checkpoint
from airletters.utils.plots import save_evaluation_plots
from airletters.utils.train_eval import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CNN+BiLSTM on AirLetters")
    parser.add_argument("--config", default="configs/cnn_bilstm_digits.yaml", help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to a trained checkpoint.")
    parser.add_argument("--split", default="val", choices=("val", "test"), help="Split to evaluate.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for smoke tests.")
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save confusion matrix and per-class accuracy plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    requested_device = config["training"]["device"]
    device = torch.device(requested_device if requested_device == "cuda" and torch.cuda.is_available() else "cpu")

    dataloaders, _ = create_dataloaders(config)
    # Load checkpoint first so we can read the label mapping (and thus num_classes)
    # even before building the model; we pass a temporary model to load_checkpoint.
    checkpoint = load_checkpoint(args.checkpoint, None, device, map_only=True)
    num_classes = len(checkpoint["label_to_index"])
    model = create_model(config, num_classes=num_classes).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["training"]["label_smoothing"]))

    metrics = evaluate(
        model=model,
        dataloader=dataloaders[args.split],
        criterion=criterion,
        device=device,
        split_name=args.split,
        max_batches=args.max_batches,
        collect_predictions=args.save_plots,
    )

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"{args.split} samples: {metrics['num_samples']}")
    print(f"{args.split} loss: {metrics['loss']:.4f}")
    print(f"{args.split} accuracy: {metrics['accuracy']:.4f}")

    if args.save_plots:
        outputs_dir = get_artifact_dir(config, "outputs_dir") / args.split
        index_to_label = {index: label for label, index in checkpoint["label_to_index"].items()}
        save_evaluation_plots(
            targets=metrics["targets"],
            predictions=metrics["predictions"],
            index_to_label=index_to_label,
            output_dir=outputs_dir,
            split_name=args.split,
        )
        metrics_path = outputs_dir / f"{args.split}_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "num_samples": metrics["num_samples"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved evaluation outputs to: {outputs_dir}")


if __name__ == "__main__":
    main()
