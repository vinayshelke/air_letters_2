"""Live webcam demo for trained AirLetters models."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airletters.config import load_config
from airletters.data.video import load_video_frames
from airletters.models import create_model
from airletters.utils.checkpointing import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live webcam inference")
    parser.add_argument("--config", default="configs/letters.yaml", help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to a trained checkpoint.")
    parser.add_argument("--duration", type=float, default=2.5, help="Recording duration in seconds.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {args.checkpoint} on device: {device}")
    
    checkpoint = load_checkpoint(args.checkpoint, None, device, map_only=True)
    num_classes = len(checkpoint["label_to_index"])
    model = create_model(config, num_classes=num_classes).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    model.eval()
    
    index_to_label = {index: label for label, index in checkpoint["label_to_index"].items()}
    video_config = config["data"]["video"]

    # Open Camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    # Frame parameters
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print("\n--- Live Demo Controls ---")
    print("Press SPACEBAR to start drawing")
    print("Press Q to quit")
    print("--------------------------")

    state = "IDLE"  # IDLE, RECORDING, PREDICTING, SHOW_RESULT
    frames_buffer = []
    start_time = 0.0
    prediction = ""
    confidence = 0.0
    temp_video_path = Path("temp_live.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Mirror frame for natural writing feel
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        current_time = time.time()

        if state == "IDLE":
            cv2.putText(
                display_frame,
                "Press SPACEBAR to draw",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

        elif state == "RECORDING":
            elapsed = current_time - start_time
            remaining = max(0.0, args.duration - elapsed)
            frames_buffer.append(cv2.flip(frame, 1))  # Un-mirror back to correct writing direction

            # Show recording status and countdown
            cv2.putText(
                display_frame,
                f"RECORDING! Draw now... {remaining:.1f}s",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            # Progress bar
            progress = int((elapsed / args.duration) * (width - 60))
            cv2.rectangle(display_frame, (30, height - 30), (30 + progress, height - 20), (0, 0, 255), -1)

            if elapsed >= args.duration:
                state = "PREDICTING"

        elif state == "PREDICTING":
            cv2.putText(
                display_frame,
                "Analyzing...",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2,
            )
            cv2.imshow("AirLetters Live Demo", display_frame)
            cv2.waitKey(1)

            # Write buffer to temporary mp4 file
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
            for f in frames_buffer:
                out.write(f)
            out.release()

            # Load and preprocess
            try:
                video_tensor = load_video_frames(
                    temp_video_path,
                    num_frames=int(video_config["num_frames"]),
                    image_size=int(video_config["image_size"]),
                    resize_short_edge=int(video_config["resize_short_edge"]),
                    mean=list(video_config["mean"]),
                    std=list(video_config["std"]),
                    is_training=False,
                    train_crop_scale=list(video_config["train_crop_scale"]),
                    sampling_strategy=str(video_config.get("sampling_strategy", "uniform")),
                ).unsqueeze(0).to(device)

                # Inference
                logits = model(video_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                score, idx = probs.topk(1)
                
                prediction = index_to_label[idx.item()]
                # Simplify verbose labels to short readable forms:
                #   "Drawing the digit 3 in the air"  -> "3"
                #   "Drawing the letter A in the air" -> "A"
                #   "Doing nothing"                   -> "Nothing"
                #   "Doing Other Things"              -> "Other"
                prediction = (
                    prediction
                    .replace("Drawing the digit ", "")
                    .replace("Drawing the letter ", "")
                    .replace(" in the air", "")
                    .replace("Doing nothing", "Nothing")
                    .replace("Doing Other Things", "Other")
                )
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
                f"You drew: {prediction} ({confidence:.1f}%)",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 0),
                3,
            )
            cv2.putText(
                display_frame,
                "Press SPACEBAR to try again",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                1,
            )

        cv2.imshow("AirLetters Live Demo", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" ") and (state == "IDLE" or state == "SHOW_RESULT"):
            state = "RECORDING"
            frames_buffer = []
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
