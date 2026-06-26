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
        split: str | None = None,
        subset_config: Mapping[str, Any] | None = None,
        class_filter: str | list[str] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.cache_dir = Path(cache_dir)
        self.num_frames = num_frames

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

        return {
            "landmarks": torch.from_numpy(landmarks),  # (T, 63)
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
