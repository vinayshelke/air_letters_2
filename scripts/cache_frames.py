"""Pre-extract and cache motion-sampled frames to disk as JPEG files.

Run this ONCE before training to avoid re-decoding videos every epoch.

For training videos: extracts ``num_variants`` different stochastic
motion-sampled frame sets per video (preserving temporal augmentation).
For val/test videos: extracts 1 deterministic frame set.

Usage
-----
    python scripts/cache_frames.py --config configs/digits.yaml

After caching, add to the config:

    data:
      video:
        cache_dir: frames_cache       # path relative to dataset_root
        num_cache_variants: 3         # how many temporal draws to cache per train video

The loader will automatically use the cache if it exists.

Disk usage estimate
-------------------
    num_videos * num_variants * num_frames * (224*224*3 bytes in JPEG ~= 15 KB)
    4800 train videos * 3 variants * 32 frames * 15 KB  ≈  6.9 GB
    960  val videos   * 1 variant  * 32 frames * 15 KB  ≈  0.46 GB
    Total ≈ ~7.4 GB
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airletters.config import load_config, get_videos_dir, get_split_csv
from airletters.data.video import (
    _compute_motion_scores,
    _motion_sample_indices,
    _read_frame_at,
    _resize_short_edge,
    _uniform_sample_indices,
)


def video_cache_key(video_path: Path) -> str:
    """Short stable key for a video file (stem + 8-char path hash)."""
    path_hash = hashlib.sha1(str(video_path).encode()).hexdigest()[:8]
    return f"{video_path.stem}_{path_hash}"


def extract_frames(
    video_path: Path,
    num_frames: int,
    resize_short_edge: int,
    sampling_strategy: str,
    is_training: bool,
) -> list[np.ndarray] | None:
    """Return a list of ``num_frames`` BGR uint8 arrays, or None on error."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if sampling_strategy == "motion" and total > num_frames:
            indices = _motion_sample_indices(capture, total, num_frames, is_training)
        else:
            indices = _uniform_sample_indices(total, num_frames)
        frames = [_read_frame_at(capture, idx, resize_short_edge) for idx in indices]
    finally:
        capture.release()
    return frames


def cache_video(
    video_path: Path,
    cache_dir: Path,
    num_frames: int,
    resize_short_edge: int,
    sampling_strategy: str,
    is_training: bool,
    num_variants: int,
) -> None:
    """Extract and save frames for one video under ``cache_dir``."""
    key = video_cache_key(video_path)
    video_cache = cache_dir / key

    # How many variants to produce for this video
    variants = num_variants if is_training else 1

    for v in range(variants):
        variant_dir = video_cache / f"v{v}"
        # Check if already cached (all frames present)
        if variant_dir.exists() and len(list(variant_dir.glob("*.jpg"))) == num_frames:
            continue

        frames = extract_frames(
            video_path, num_frames, resize_short_edge, sampling_strategy, is_training
        )
        if frames is None:
            print(f"  [WARN] Could not open: {video_path}")
            return

        variant_dir.mkdir(parents=True, exist_ok=True)
        for f_idx, frame in enumerate(frames):
            # frame is RGB uint8 from _read_frame_at — convert to BGR for cv2 JPEG write
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out_path = variant_dir / f"f{f_idx:03d}.jpg"
            cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-extract motion-sampled frames to disk.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Splits to cache (default: train val test)."
    )
    parser.add_argument(
        "--num-variants", type=int, default=None,
        help="Override number of train variants (default: from config or 3)."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    video_config = config["data"]["video"]

    num_frames = int(video_config["num_frames"])
    resize_short_edge = int(video_config["resize_short_edge"])
    sampling_strategy = str(video_config.get("sampling_strategy", "uniform"))
    num_variants = args.num_variants or int(video_config.get("num_cache_variants", 3))

    dataset_root = Path(config["paths"]["dataset_root"])
    videos_dir = get_videos_dir(config)

    cache_dir_name = str(video_config.get("cache_dir", "frames_cache"))
    cache_dir = dataset_root / cache_dir_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config          : {args.config}")
    print(f"Videos dir      : {videos_dir}")
    print(f"Cache dir       : {cache_dir}")
    print(f"Num frames      : {num_frames}")
    print(f"Resize short edge: {resize_short_edge}")
    print(f"Sampling        : {sampling_strategy}")
    print(f"Train variants  : {num_variants}")
    print()

    import pandas as pd
    from airletters.data.dataset import _apply_class_filter, _apply_subset

    for split in args.splits:
        csv_path = get_split_csv(config, split)
        records = pd.read_csv(csv_path, skipinitialspace=True)
        records["filename"] = records["filename"].astype(str).str.strip()
        records["label"] = records["label"].astype(str).str.strip()

        # Apply same class filter as training
        class_filter = config["data"].get("class_filter")
        if class_filter:
            records = _apply_class_filter(records, class_filter)

        # Apply subset (same videos the model will actually see)
        subset_config = config["data"].get("subset")
        if subset_config and bool(subset_config.get("enabled", False)):
            records = _apply_subset(records, split, subset_config)

        is_training = (split == "train")
        variants = num_variants if is_training else 1
        print(f"Caching {split} split: {len(records)} videos × {variants} variant(s)")

        for _, row in tqdm(records.iterrows(), total=len(records), desc=f"  {split}"):
            video_path = videos_dir / str(row["filename"])
            if not video_path.exists():
                print(f"  [WARN] Missing: {video_path}")
                continue
            cache_video(
                video_path=video_path,
                cache_dir=cache_dir,
                num_frames=num_frames,
                resize_short_edge=resize_short_edge,
                sampling_strategy=sampling_strategy,
                is_training=is_training,
                num_variants=variants,
            )

    print()
    print("Done! All frames cached to:", cache_dir)
    print("Now add to your config:")
    print(f"  data:")
    print(f"    video:")
    print(f"      cache_dir: {cache_dir_name}")
    print(f"      num_cache_variants: {num_variants}")


if __name__ == "__main__":
    main()
