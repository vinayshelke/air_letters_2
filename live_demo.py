#!/usr/bin/env python3
"""Live webcam demo for AirLetters digit recognition.

This script captures a sequence of frames from your webcam and predicts the digit
you are drawing in the air. The model uses EfficientNet-B0 + BiLSTM trained on
the Qualcomm AirLetters dataset.

Usage:
    python live_demo.py
    python live_demo.py --duration 3.0 --camera 0

Controls:
    SPACEBAR  - Start recording a gesture
    Q         - Quit the application

Tips:
    - Make sure your webcam is working and visible
    - Draw slowly and smoothly for best results
    - Keep your hand within the frame for the full duration
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airletters.config import load_config
from airletters.data.video import load_video_frames
from airletters.models import create_model
from airletters.utils.checkpointing import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live webcam demo for AirLetters digit recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="configs/digits.yaml",
        help="Path to config file (default: digits.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
        help="Path to trained checkpoint (default: best.pt)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.5,
        help="Recording duration in seconds (default: 2.5)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device (default: auto-detect)",
    )
    return parser.parse_args()


def _simplify_label(label: str) -> str:
    """Simplify verbose class labels to short readable forms."""
    return (
        label
        .replace("Drawing the digit ", "")
        .replace("Drawing the letter ", "")
        .replace(" in the air", "")
        .replace("Doing nothing", "Nothing")
        .replace("Doing Other Things", "Other")
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {args.config}")
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
    
    print(f"Loading checkpoint from: {args.checkpoint} on device: {device}")
    
    # Load checkpoint and model
    checkpoint = load_checkpoint(args.checkpoint, None, device, map_only=True)
    num_classes = len(checkpoint["label_to_index"])
    model = create_model(config, num_classes=num_classes).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    model.eval()
    
    index_to_label = {index: label for label, index in checkpoint["label_to_index"].items()}
    video_config = config["data"]["video"]

    # Open Camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}")
        sys.exit(1)

    # Frame parameters
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print("\n" + "="*70)
    print("AirLetters Live Demo - Digit Recognition")
    print("="*70)
    print("Controls:")
    print("  SPACEBAR - Start drawing a digit")
    print("  Q        - Quit")
    print("="*70 + "\n")

    state = "IDLE"  # IDLE, RECORDING, PREDICTING, SHOW_RESULT
    frames_buffer = []
    start_time = 0.0
    prediction = ""
    confidence = 0.0
    temp_video_path = Path("temp_live.mp4")

    frame_count = 0
    fps_counter = 0
    fps_timer = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame")
            break

        frame_count += 1
        fps_counter += 1

        # Mirror frame for natural writing feel
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # Calculate FPS
        if time.time() - fps_timer > 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        current_time = time.time()

        # Draw FPS in corner
        cv2.putText(
            display_frame,
            f"FPS: {current_fps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if state == "IDLE":
            cv2.putText(
                display_frame,
                "Press SPACEBAR to draw a digit",
                (30, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                "Press Q to quit",
                (30, height // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                1,
            )

        elif state == "RECORDING":
            elapsed = current_time - start_time
            remaining = max(0.0, args.duration - elapsed)
            frames_buffer.append(cv2.flip(frame, 1))  # Un-mirror back to correct writing direction

            # Show recording status and countdown
            cv2.putText(
                display_frame,
                "RECORDING! Draw now...",
                (30, height // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                display_frame,
                f"{remaining:.1f}s",
                (30, height // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
            )
            
            # Progress bar
            progress = int((elapsed / args.duration) * (width - 60))
            cv2.rectangle(display_frame, (30, height - 30), (30 + progress, height - 20), (0, 0, 255), -1)
            cv2.rectangle(display_frame, (30, height - 30), (width - 30, height - 20), (200, 200, 200), 2)

            if elapsed >= args.duration:
                state = "PREDICTING"

        elif state == "PREDICTING":
            cv2.putText(
                display_frame,
                "Analyzing...",
                (30, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 0),
                2,
            )
            cv2.imshow("AirLetters Live Demo", display_frame)
            cv2.waitKey(1)

            # Write buffer to temporary mp4 file
            try:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
                for f in frames_buffer:
                    out.write(f)
                out.release()

                # Load and preprocess
                video_tensor = load_video_frames(
                    temp_video_path,
                    num_frames=int(video_config["num_frames"]),
                    image_size=int(video_config["image_size"]),
                    resize_short_edge=int(video_config["resize_short_edge"]),
                    mean=list(video_config["mean"]),
                    std=list(video_config["std"]),
                    is_training=False,
                    train_crop_scale=list(video_config["train_crop_scale"]),
                ).unsqueeze(0).to(device)

                # Inference
                logits = model(video_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                score, idx = probs.topk(1)
                
                prediction = index_to_label[idx.item()]
                prediction = _simplify_label(prediction)
                confidence = score.item() * 100
            except Exception as e:
                print(f"Error during prediction: {e}")
                prediction = "Error"
                confidence = 0.0

            # Delete temp file
            if temp_video_path.exists():
                temp_video_path.unlink()

            state = "SHOW_RESULT"
            start_time = time.time()

        elif state == "SHOW_RESULT":
            # Display Prediction
            cv2.putText(
                display_frame,
                f"You drew: {prediction}",
                (30, height // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 0),
                3,
            )
            cv2.putText(
                display_frame,
                f"Confidence: {confidence:.1f}%",
                (30, height // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                "Press SPACEBAR to try again",
                (30, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                1,
            )

        cv2.imshow("AirLetters Live Demo", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nQuitting...")
            break
        elif key == ord(" ") and (state == "IDLE" or state == "SHOW_RESULT"):
            state = "RECORDING"
            frames_buffer = []
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == "__main__":
    main()
