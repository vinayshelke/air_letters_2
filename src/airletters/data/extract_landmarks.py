"""Offline landmark pre-extraction script for AirLetters.

Run this script ONCE before training to extract MediaPipe hand landmarks
from every video and save them as ``.npy`` files in a cache directory.
Subsequent training runs read from the cache instead of re-running MediaPipe.

Usage::

    python src/airletters/data/extract_landmarks.py \\
        --config configs/mediapipe_transformer_digits.yaml

The script is idempotent: videos whose cache file already exists are skipped.
Use ``--overwrite`` to force re-extraction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from tqdm import tqdm

from airletters.config import get_split_csv, get_videos_dir, load_config
from airletters.data.dataset import AirLettersDataset
from airletters.data.landmarks import extract_landmarks_from_frames
from airletters.data.video import load_video_frames_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-extract MediaPipe hand landmarks for AirLetters videos."
    )
    parser.add_argument(
        "--config",
        default="configs/mediapipe_transformer_digits.yaml",
        help="Path to the experiment config YAML.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even if a cache file already exists.",
    )
    return parser.parse_args()


def extract_for_split(
    config: dict,
    split: str,
    cache_dir: Path,
    num_frames: int,
    image_size: int,
    extraction_config: dict,
    overwrite: bool,
) -> None:
    """Extract landmarks for all videos in one split."""
    dataset = AirLettersDataset.from_config(config, split=split, load_video=False)
    videos_dir = get_videos_dir(config)

    skipped = 0
    processed = 0

    for row in tqdm(dataset.records.itertuples(), total=len(dataset.records), desc=split):
        filename: str = row.filename  # type: ignore[attr-defined]
        cache_file = cache_dir / (Path(filename).stem + ".npy")

        if cache_file.exists() and not overwrite:
            skipped += 1
            continue

        video_path = videos_dir / filename
        if not video_path.is_file():
            # Write zeros so the dataset does not crash later on
            np.save(cache_file, np.zeros((num_frames, 63), dtype=np.float32))
            continue

        # Load raw uint8 RGB frames — no normalisation, just resize
        frames_rgb = load_video_frames_raw(
            str(video_path),
            num_frames=num_frames,
            image_size=image_size,
        )  # (T, H, W, 3) uint8 RGB

        landmarks = extract_landmarks_from_frames(
            frames_rgb,
            min_hand_detection_confidence=float(
                extraction_config.get("min_hand_detection_confidence", 0.25)
            ),
            min_hand_presence_confidence=float(
                extraction_config.get("min_hand_presence_confidence", 0.25)
            ),
            min_tracking_confidence=float(
                extraction_config.get("min_tracking_confidence", 0.25)
            ),
            use_video_mode=bool(extraction_config.get("use_video_mode", True)),
            frame_timestamp_step_ms=int(
                extraction_config.get("frame_timestamp_step_ms", 33)
            ),
        )  # (T, 63)
        np.save(cache_file, landmarks)
        processed += 1

    print(
        f"  [{split}] processed={processed}, skipped(cached)={skipped}, "
        f"total={len(dataset.records)}"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    lm_config = config["data"].get("landmarks", {})
    num_frames = int(lm_config.get("num_frames", 32))
    image_size = int(lm_config.get("image_size", 224))
    cache_dir = Path(lm_config.get("cache_dir", "landmarks_cache"))
    extraction_config = dict(lm_config.get("extraction", {}))
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cache directory : {cache_dir.resolve()}")
    print(f"Frames per video: {num_frames}")
    print(f"Resize to       : {image_size}px")
    print(
        "Extraction      : "
        f"video_mode={bool(extraction_config.get('use_video_mode', True))}, "
        f"det={float(extraction_config.get('min_hand_detection_confidence', 0.25))}, "
        f"presence={float(extraction_config.get('min_hand_presence_confidence', 0.25))}, "
        f"tracking={float(extraction_config.get('min_tracking_confidence', 0.25))}"
    )

    for split in args.splits:
        extract_for_split(
            config=config,
            split=split,
            cache_dir=cache_dir,
            num_frames=num_frames,
            image_size=image_size,
            extraction_config=extraction_config,
            overwrite=args.overwrite,
        )

    print("Done. You can now run train_landmark.py.")


if __name__ == "__main__":
    main()
