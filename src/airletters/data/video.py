"""Video decoding utilities for AirLetters."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def load_video_frames(
    video_path: str | Path,
    num_frames: int,
    image_size: int,
    resize_short_edge: int,
    mean: list[float],
    std: list[float],
    is_training: bool,
    train_crop_scale: list[float],
    sampling_strategy: str = "uniform",
) -> torch.Tensor:
    """Load uniformly sampled RGB frames as a normalized float tensor.

    Returns:
        Tensor with shape ``[num_frames, 3, image_size, image_size]``.
    """
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video file: {path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if sampling_strategy == "motion" and total_frames > num_frames:
            frame_indices = _motion_sample_indices(
                capture, total_frames, num_frames, is_training,
            )
        else:
            frame_indices = _uniform_sample_indices(total_frames, num_frames)
        frames = [_read_frame_at(capture, frame_index, resize_short_edge) for frame_index in frame_indices]
    finally:
        capture.release()

    frames = _crop_frames(frames, image_size, is_training, train_crop_scale)
    video = np.stack(frames).astype(np.float32) / 255.0
    video = torch.from_numpy(video).permute(0, 3, 1, 2)

    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (video - mean_tensor) / std_tensor


def _uniform_sample_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros(num_frames, dtype=np.int64)
    return np.linspace(0, total_frames - 1, num_frames).round().astype(np.int64)


def _compute_motion_scores(capture: cv2.VideoCapture, total_frames: int) -> np.ndarray:
    """Compute per-frame motion scores via low-res grayscale frame differencing."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_gray = None
    scores = np.zeros(total_frames)

    for i in range(total_frames):
        ok, frame = capture.read()
        if not ok or frame is None:
            prev_gray = None
            continue
        small = cv2.resize(frame, (32, 32))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None:
            scores[i] = np.abs(gray - prev_gray).mean()
        prev_gray = gray

    return scores


def _motion_sample_indices(
    capture: cv2.VideoCapture,
    total_frames: int,
    num_frames: int,
    is_training: bool,
) -> np.ndarray:
    """Sample frame indices weighted by inter-frame motion.

    The video is divided into ``num_frames`` equal segments.  Within each
    segment the frame with the highest motion score is selected
    (deterministic, used at validation) or a frame is sampled with
    motion-weighted probability (stochastic, used at training).
    """
    scores = _compute_motion_scores(capture, total_frames)

    segment_boundaries = np.linspace(0, total_frames, num_frames + 1).astype(int)
    indices = np.zeros(num_frames, dtype=np.int64)

    for s in range(num_frames):
        start = segment_boundaries[s]
        end = segment_boundaries[s + 1]
        if start >= end:
            indices[s] = start
            continue

        seg_scores = scores[start:end]

        if is_training:
            # Stochastic: sample within segment using motion as probability
            seg_probs = seg_scores + 1e-6
            seg_probs = seg_probs / seg_probs.sum()
            offset = np.random.choice(len(seg_probs), p=seg_probs)
            indices[s] = start + offset
        else:
            # Deterministic: pick highest-motion frame in segment
            indices[s] = start + np.argmax(seg_scores)

    return indices


def _read_frame_at(capture: cv2.VideoCapture, frame_index: int, resize_short_edge: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok or frame is None:
        frame = np.zeros((resize_short_edge, resize_short_edge, 3), dtype=np.uint8)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = _resize_short_edge(frame, resize_short_edge)
    return frame


def _resize_short_edge(frame: np.ndarray, resize_short_edge: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if min(height, width) == resize_short_edge:
        return frame

    scale = resize_short_edge / min(height, width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _crop_frames(
    frames: list[np.ndarray],
    image_size: int,
    is_training: bool,
    train_crop_scale: list[float],
) -> list[np.ndarray]:
    height, width = frames[0].shape[:2]
    crop_size = min(height, width)

    if is_training:
        min_scale, max_scale = train_crop_scale
        crop_size = int(round(crop_size * np.random.uniform(min_scale, max_scale)))
        top = np.random.randint(0, max(height - crop_size + 1, 1))
        left = np.random.randint(0, max(width - crop_size + 1, 1))
    else:
        top = max((height - crop_size) // 2, 0)
        left = max((width - crop_size) // 2, 0)

    cropped = [frame[top : top + crop_size, left : left + crop_size] for frame in frames]
    return [
        cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        for frame in cropped
    ]
