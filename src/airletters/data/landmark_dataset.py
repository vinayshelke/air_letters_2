"""PyTorch dataset that reads pre-cached MediaPipe landmark files.

Before using this dataset, run the offline extraction script once::

    python src/airletters/data/extract_landmarks.py \\
        --config configs/mediapipe_transformer_digits.yaml

Each ``.npy`` file contains a ``float32`` array of shape ``(T, 63)``
(21 landmarks × (x, y, z)) for one video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from airletters.config import get_split_csv, get_videos_dir
from airletters.data.dataset import (
    AirLettersDataset,
    _apply_class_filter,   # noqa: PLC2701
    _apply_subset,          # noqa: PLC2701
    build_label_mapping,
    REQUIRED_COLUMNS,
)

import pandas as pd


class LandmarkDataset(Dataset):
    """Dataset that loads pre-extracted MediaPipe landmarks from ``.npy`` cache.

    Args:
        csv_path: Path to a split CSV (train / val / test).
        cache_dir: Directory containing ``<video_stem>.npy`` landmark files.
        label_to_index: Optional pre-built label mapping; derived from the
            training split if not provided.
        num_frames: Expected sequence length (must match extraction settings).
        split: ``"train"``, ``"val"``, or ``"test"``; used for subsetting.
        subset_config: Subset config block from the YAML (may be ``None``).
        class_filter: Optional class filter string / list.
    """

    def __init__(
        self,
        csv_path: str | Path,
        cache_dir: str | Path,
        label_to_index: Mapping[str, int] | None = None,
        num_frames: int = 32,
        normalize: bool = True,
        fill_missing: bool = True,
        add_velocity: bool = True,
        split: str | None = None,
        subset_config: Mapping[str, Any] | None = None,
        class_filter: str | list[str] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.cache_dir = Path(cache_dir)
        self.num_frames = num_frames
        self.normalize = normalize
        self.fill_missing = fill_missing
        self.add_velocity = add_velocity

        self.records: pd.DataFrame = pd.read_csv(self.csv_path, skipinitialspace=True)
        missing_cols = set(REQUIRED_COLUMNS) - set(self.records.columns)
        if missing_cols:
            raise ValueError(
                f"{self.csv_path} is missing columns: {', '.join(sorted(missing_cols))}"
            )
        self.records["filename"] = self.records["filename"].astype(str).str.strip()
        self.records["label"] = self.records["label"].astype(str).str.strip()

        if class_filter is not None:
            self.records = _apply_class_filter(self.records, class_filter)

        if split is not None and subset_config and bool(subset_config.get("enabled", False)):
            self.records = _apply_subset(self.records, split, subset_config)

        self.label_to_index: dict[str, int] = (
            dict(label_to_index)
            if label_to_index is not None
            else build_label_mapping(self.records["label"].tolist())
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        split: str,
        label_to_index: Mapping[str, int] | None = None,
    ) -> "LandmarkDataset":
        """Construct a dataset from a YAML config dict."""
        lm_config = config["data"].get("landmarks", {})
        cache_dir = Path(lm_config.get("cache_dir", "landmarks_cache"))
        num_frames = int(lm_config.get("num_frames", 32))
        features_config = lm_config.get("features", {})

        # Resolve CSV path using the same helper as AirLettersDataset
        from airletters.config import get_split_csv
        csv_path = get_split_csv(config, split)

        subset_config = config["data"].get("subset")
        class_filter = config["data"].get("class_filter")

        return cls(
            csv_path=csv_path,
            cache_dir=cache_dir,
            label_to_index=label_to_index,
            num_frames=num_frames,
            normalize=bool(features_config.get("normalize", True)),
            fill_missing=bool(features_config.get("fill_missing", True)),
            add_velocity=bool(features_config.get("add_velocity", True)),
            split=split,
            subset_config=subset_config,
            class_filter=class_filter,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records.iloc[index]
        label: str = str(row["label"])
        stem = Path(str(row["filename"])).stem
        cache_file = self.cache_dir / f"{stem}.npy"

        if cache_file.exists():
            landmarks = np.load(cache_file)  # (T, 63) float32
        else:
            # Graceful fallback — training will still work (all zeros)
            landmarks = np.zeros((self.num_frames, 63), dtype=np.float32)

        # Ensure consistent sequence length (pad or truncate)
        if len(landmarks) < self.num_frames:
            pad = np.zeros((self.num_frames - len(landmarks), 63), dtype=np.float32)
            landmarks = np.concatenate([landmarks, pad], axis=0)
        elif len(landmarks) > self.num_frames:
            landmarks = landmarks[: self.num_frames]

        landmarks = prepare_landmark_features(
            landmarks=landmarks,
            normalize=self.normalize,
            fill_missing=self.fill_missing,
            add_velocity=self.add_velocity,
        )

        return {
            "landmarks": torch.from_numpy(landmarks),  # (T, feature_dim)
            "label": label,
            "label_id": int(self.label_to_index[label]),
            "id": int(row["id"]),
        }


def create_landmark_dataloaders(
    config: Mapping[str, Any],
) -> tuple[dict[str, torch.utils.data.DataLoader], dict[str, int]]:
    """Create train / val / test landmark dataloaders with a shared label mapping."""
    from torch.utils.data import DataLoader

    train_ds = LandmarkDataset.from_config(config, split="train")
    label_to_index = train_ds.label_to_index

    val_ds = LandmarkDataset.from_config(config, split="val", label_to_index=label_to_index)
    test_ds = LandmarkDataset.from_config(config, split="test", label_to_index=label_to_index)

    training_cfg = config["training"]
    evaluation_cfg = config["evaluation"]

    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=int(training_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(training_cfg["num_workers"]),
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=int(evaluation_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(evaluation_cfg["num_workers"]),
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=int(evaluation_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(evaluation_cfg["num_workers"]),
            pin_memory=True,
        ),
    }
    return dataloaders, label_to_index


def prepare_landmark_features(
    landmarks: np.ndarray,
    normalize: bool = True,
    fill_missing: bool = True,
    add_velocity: bool = True,
) -> np.ndarray:
    """Convert raw cached MediaPipe landmarks into model-ready features."""
    features = np.asarray(landmarks, dtype=np.float32).copy()
    if features.ndim != 2 or features.shape[1] != 63:
        raise ValueError(
            f"Expected landmark array with shape (T, 63), got {features.shape}"
        )

    valid_mask = features_valid_mask(features)
    if fill_missing:
        features = fill_missing_landmark_frames(features, valid_mask)
        valid_mask = features_valid_mask(features)

    if normalize:
        features = normalize_landmarks(features, valid_mask)

    if add_velocity:
        velocity = np.zeros_like(features, dtype=np.float32)
        velocity[1:] = features[1:] - features[:-1]
        if not fill_missing:
            valid_pairs = valid_mask[1:] & valid_mask[:-1]
            velocity[1:] *= valid_pairs[:, None]
        features = np.concatenate([features, velocity], axis=1)

    return features.astype(np.float32, copy=False)


def features_valid_mask(landmarks: np.ndarray) -> np.ndarray:
    """Return ``True`` for frames with a detected hand."""
    return np.abs(landmarks).sum(axis=1) > 0


def fill_missing_landmark_frames(
    landmarks: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Fill zero landmark frames with the nearest valid frame."""
    if not valid_mask.any():
        return landmarks

    filled = landmarks.copy()
    valid_indices = np.flatnonzero(valid_mask)
    nearest_right_positions = np.searchsorted(valid_indices, np.arange(len(landmarks)))

    for idx in np.flatnonzero(~valid_mask):
        right_pos = nearest_right_positions[idx]
        left_pos = right_pos - 1

        left_idx = valid_indices[left_pos] if left_pos >= 0 else None
        right_idx = (
            valid_indices[right_pos]
            if right_pos < len(valid_indices)
            else None
        )

        if left_idx is None:
            fill_idx = right_idx
        elif right_idx is None:
            fill_idx = left_idx
        else:
            fill_idx = left_idx if idx - left_idx <= right_idx - idx else right_idx
        filled[idx] = landmarks[int(fill_idx)]

    return filled


def normalize_landmarks(
    landmarks: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Center and scale each sequence while preserving hand trajectory."""
    if not valid_mask.any():
        return landmarks

    points = landmarks.reshape(landmarks.shape[0], 21, 3).copy()
    valid_points = points[valid_mask]

    center = valid_points.reshape(-1, 3).mean(axis=0, keepdims=True)
    points = points - center.reshape(1, 1, 3)

    valid_centered = points[valid_mask].reshape(-1, 3)
    scale = np.linalg.norm(valid_centered, axis=1).max()
    scale = max(float(scale), eps)
    points = points / scale

    normalized = points.reshape(landmarks.shape).astype(np.float32)
    normalized[~valid_mask] = 0.0
    return normalized
