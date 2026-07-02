"""Dataset definitions for the official AirLetters CSV splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from torch.utils.data import Dataset

from airletters.config import get_split_csv, get_videos_dir
from airletters.data.video import load_video_frames


REQUIRED_COLUMNS = ("id", "filename", "label", "worker_id", "video_duration")

# Convenience sets for class filtering
DIGIT_LABELS: frozenset[str] = frozenset(
    f"Drawing the digit {d} in the air" for d in range(10)
)
LETTER_LABELS: frozenset[str] = frozenset(
    f"Drawing the letter {c} in the air" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
# Special non-gesture classes present in the AirLetters dataset
SPECIAL_LABELS: frozenset[str] = frozenset(["Doing nothing", "Doing Other Things"])

_FILTER_PRESETS: dict[str, frozenset[str]] = {
    "digits": DIGIT_LABELS,
    "digits_with_special": DIGIT_LABELS | SPECIAL_LABELS,
    "letters": LETTER_LABELS,
    # 26 letters + 2 special classes (28 total) — main experiment
    "letters_with_special": LETTER_LABELS | SPECIAL_LABELS,
    # Everything in the dataset
    "all": LETTER_LABELS | DIGIT_LABELS | SPECIAL_LABELS,
}


def build_label_mapping(labels: Iterable[str]) -> dict[str, int]:
    """Build a stable alphabetical label-to-index mapping."""
    return {label: index for index, label in enumerate(sorted(set(labels)))}


class AirLettersDataset(Dataset):
    """PyTorch dataset wrapper for AirLetters split metadata.

    The dataset can return metadata only or decode a fixed-length frame tensor
    for model training.
    """

    def __init__(
        self,
        csv_path: str | Path,
        videos_dir: str | Path,
        label_to_index: Mapping[str, int] | None = None,
        validate_files: bool = False,
        load_video: bool = False,
        num_frames: int = 16,
        image_size: int = 112,
        resize_short_edge: int = 128,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        split: str | None = None,
        subset_config: Mapping[str, Any] | None = None,
        train_crop_scale: list[float] | None = None,
        class_filter: str | list[str] | None = None,
        sampling_strategy: str = "uniform",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.videos_dir = Path(videos_dir)
        self.load_video = load_video
        self.num_frames = num_frames
        self.image_size = image_size
        self.resize_short_edge = resize_short_edge
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.is_training = split == "train"
        self.train_crop_scale = train_crop_scale or [0.7, 1.0]
        self.sampling_strategy = sampling_strategy
        self.records = pd.read_csv(self.csv_path, skipinitialspace=True)
        self._validate_columns()

        self.records["filename"] = self.records["filename"].astype(str).str.strip()
        self.records["label"] = self.records["label"].astype(str).str.strip()

        # Apply class filter before subsetting so sample counts are accurate
        if class_filter is not None:
            self.records = _apply_class_filter(self.records, class_filter)

        if split is not None and subset_config and bool(subset_config.get("enabled", False)):
            self.records = _apply_subset(self.records, split, subset_config)

        self.label_to_index = dict(label_to_index) if label_to_index else build_label_mapping(
            self.records["label"].tolist()
        )

        if validate_files:
            self.validate_video_files()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        split: str,
        label_to_index: Mapping[str, int] | None = None,
        validate_files: bool = False,
        load_video: bool = False,
    ) -> "AirLettersDataset":
        """Create a dataset from the configured official split paths."""
        video_config = config["data"]["video"]
        subset_config = config["data"].get("subset")
        class_filter = config["data"].get("class_filter")  # e.g. "digits", "letters", or a list
        return cls(
            csv_path=get_split_csv(config, split),
            videos_dir=get_videos_dir(config),
            label_to_index=label_to_index,
            validate_files=validate_files,
            load_video=load_video,
            num_frames=int(video_config["num_frames"]),
            image_size=int(video_config["image_size"]),
            resize_short_edge=int(video_config["resize_short_edge"]),
            mean=list(video_config["mean"]),
            std=list(video_config["std"]),
            split=split,
            subset_config=subset_config,
            train_crop_scale=list(video_config["train_crop_scale"]),
            class_filter=class_filter,
            sampling_strategy=str(video_config.get("sampling_strategy", "uniform")),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records.iloc[index]
        label = row["label"]

        sample = {
            "id": int(row["id"]),
            "video_path": str(self.videos_dir / row["filename"]),
            "label": label,
            "label_id": int(self.label_to_index[label]),
            "worker_id": int(row["worker_id"]),
            "video_duration": int(row["video_duration"]),
        }

        if self.load_video:
            sample["video"] = load_video_frames(
                sample["video_path"],
                num_frames=self.num_frames,
                image_size=self.image_size,
                resize_short_edge=self.resize_short_edge,
                mean=self.mean,
                std=self.std,
                is_training=self.is_training,
                train_crop_scale=self.train_crop_scale,
                sampling_strategy=self.sampling_strategy,
            )

        return sample

    def validate_video_files(self) -> None:
        """Raise a clear error if any CSV row points to a missing video file."""
        missing = [
            filename
            for filename in self.records["filename"].tolist()
            if not (self.videos_dir / filename).is_file()
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} video files referenced by {self.csv_path} were not found. "
                f"First missing files: {preview}"
            )

    def _validate_columns(self) -> None:
        missing_columns = set(REQUIRED_COLUMNS) - set(self.records.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{self.csv_path} is missing required columns: {missing}")


def create_split_datasets(
    config: Mapping[str, Any],
    validate_files: bool = False,
    load_video: bool = False,
) -> dict[str, AirLettersDataset]:
    """Create train, validation, and test datasets with one shared label mapping."""
    train_dataset = AirLettersDataset.from_config(
        config,
        split="train",
        validate_files=validate_files,
        load_video=load_video,
    )
    label_to_index = train_dataset.label_to_index

    return {
        "train": train_dataset,
        "val": AirLettersDataset.from_config(
            config,
            split="val",
            label_to_index=label_to_index,
            validate_files=validate_files,
            load_video=load_video,
        ),
        "test": AirLettersDataset.from_config(
            config,
            split="test",
            label_to_index=label_to_index,
            validate_files=validate_files,
            load_video=load_video,
        ),
    }


def _apply_class_filter(
    records: pd.DataFrame,
    class_filter: str | list[str],
) -> pd.DataFrame:
    """Keep only rows whose label matches the given filter.

    ``class_filter`` can be:
    - ``"digits"``  — keeps all 10 digit classes
    - ``"letters"`` — keeps all 26 letter classes
    - A list of exact label strings
    """
    if isinstance(class_filter, str):
        preset = _FILTER_PRESETS.get(class_filter.lower())
        if preset is None:
            raise ValueError(
                f"Unknown class_filter preset '{class_filter}'. "
                f"Expected one of: {list(_FILTER_PRESETS)} or a list of label strings."
            )
        keep = preset
    else:
        keep = frozenset(class_filter)

    filtered = records[records["label"].isin(keep)].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError(
            f"class_filter '{class_filter}' matched 0 rows. Check label spelling in the CSV."
        )
    return filtered


def _apply_subset(
    records: pd.DataFrame,
    split: str,
    subset_config: Mapping[str, Any],
) -> pd.DataFrame:
    seed = int(subset_config.get("seed", 42))
    split_config = subset_config.get(split, {})
    samples_per_class = split_config.get("samples_per_class")
    max_samples = split_config.get("max_samples")

    if samples_per_class is not None:
        sampled_groups = []
        for _, group in records.groupby("label"):
            sample_size = min(len(group), int(samples_per_class))
            sampled_groups.append(group.sample(n=sample_size, random_state=seed))
        sampled = pd.concat(sampled_groups, ignore_index=True)
    else:
        sampled = records.copy()

    if max_samples is not None and len(sampled) > int(max_samples):
        sampled = sampled.sample(n=int(max_samples), random_state=seed).reset_index(drop=True)

    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
