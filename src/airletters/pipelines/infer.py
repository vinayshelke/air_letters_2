"""Single-video inference for trained AirLetters checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from airletters.config import load_config
from airletters.data.video import load_video_frames
from airletters.models import create_model
from airletters.utils.checkpointing import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CNN+BiLSTM inference on one video")
    parser.add_argument("--config", default="configs/cnn_bilstm_digits.yaml", help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to a trained checkpoint.")
    parser.add_argument("--video", default="videos/00000000.mp4", help="Path to one AirLetters video.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions to print.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Inference video does not exist: {video_path}")

    requested_device = config["training"]["device"]
    device = torch.device(requested_device if requested_device == "cuda" and torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(args.checkpoint, None, device, map_only=True)
    num_classes = len(checkpoint["label_to_index"])
    model = create_model(config, num_classes=num_classes).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    index_to_label = {index: label for label, index in checkpoint["label_to_index"].items()}

    video_config = config["data"].get("video", {})
    video = load_video_frames(
        video_path,
        num_frames=int(video_config.get("num_frames", 16)),
        image_size=int(video_config.get("image_size", 112)),
        resize_short_edge=int(video_config.get("resize_short_edge", 128)),
        mean=list(video_config.get("mean", [0.485, 0.456, 0.406])),
        std=list(video_config.get("std", [0.229, 0.224, 0.225])),
        is_training=False,
        train_crop_scale=list(video_config.get("train_crop_scale", [0.7, 1.0])),
    ).unsqueeze(0)

    model.eval()
    logits = model(video.to(device))
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    top_k = min(args.top_k, probabilities.numel())
    scores, indices = probabilities.topk(top_k)

    print(f"Video: {video_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    for rank, (score, index) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        print(f"{rank}. {index_to_label[index]} ({score:.4f})")


if __name__ == "__main__":
    main()
