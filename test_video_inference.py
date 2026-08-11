#!/usr/bin/env python3
"""Test video inference script - loads a video and predicts the digit/class.

Usage:
    python test_video_inference.py <video_path>
    python test_video_inference.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from airletters.config import load_config
from airletters.data.video import load_video_frames
from airletters.models import create_model
from airletters.utils.checkpointing import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test video inference for trained AirLetters digit recognition model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict class for a single video
  python test_video_inference.py videos/00000001.mp4
  
  # Show top 3 predictions
  python test_video_inference.py videos/00000001.mp4 --top-k 3
  
  # Use CPU instead of GPU
  python test_video_inference.py videos/00000001.mp4 --device cpu
        """,
    )
    parser.add_argument("video", help="Path to a test video (.mp4)")
    parser.add_argument("--config", default="configs/digits.yaml", help="Path to config file (default: digits.yaml)")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to trained checkpoint (default: best.pt)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Device (default: auto-detect)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show (default: 3)")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    # Verify video exists
    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)
    
    # Verify checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    # Select device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load checkpoint to get model config
    print("\nLoading checkpoint...")
    checkpoint = load_checkpoint(args.checkpoint, None, device, map_only=True)
    num_classes = len(checkpoint["label_to_index"])
    index_to_label = {index: label for label, index in checkpoint["label_to_index"].items()}
    checkpoint_epoch = checkpoint.get("epoch", "unknown")
    
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Number of classes: {num_classes}")
    
    # Create and load model
    print("Creating model...")
    model = create_model(config, num_classes=num_classes).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()
    
    # Load and preprocess video
    print(f"\nProcessing video: {video_path}")
    video_config = config["data"]["video"]
    
    try:
        video_tensor = load_video_frames(
            video_path,
            num_frames=int(video_config["num_frames"]),
            image_size=int(video_config["image_size"]),
            resize_short_edge=int(video_config["resize_short_edge"]),
            mean=list(video_config["mean"]),
            std=list(video_config["std"]),
            is_training=False,
            train_crop_scale=list(video_config["train_crop_scale"]),
        ).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"ERROR: Failed to load video: {e}")
        sys.exit(1)
    
    # Run inference
    print("Running inference...")
    logits = model(video_tensor.to(device))
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    
    # Get top-k predictions
    top_k = min(args.top_k, probabilities.numel())
    scores, indices = probabilities.topk(top_k)
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Video: {video_path}")
    
    # Top prediction with confidence
    top_idx = indices[0].item()
    top_score = scores[0].item()
    top_label = index_to_label[top_idx]
    
    print(f"\nPredicted class: {top_label}")
    print(f"Confidence: {top_score:.4f} ({top_score*100:.2f}%)")
    
    print(f"\nTop {top_k} predictions:")
    print("-" * 70)
    for rank, (score, index) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        label = index_to_label[index]
        print(f"{rank:2d}. {label:40s} {score:.4f} ({score*100:6.2f}%)")
    
    print("="*70)


if __name__ == "__main__":
    main()
