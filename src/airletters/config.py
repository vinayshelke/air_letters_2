"""Configuration helpers for AirLetters experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
SPLIT_TO_CSV_KEY = {
    "train": "train_csv",
    "val": "val_csv",
    "test": "test_csv",
}


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a YAML config file as a plain dictionary."""
    path = _resolve_path(config_path, PROJECT_ROOT)
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_dataset_root(config: Mapping[str, Any]) -> Path:
    """Return the configured dataset root directory."""
    paths = _paths_section(config)
    return _resolve_path(paths.get("dataset_root", "."), PROJECT_ROOT)


def get_videos_dir(config: Mapping[str, Any]) -> Path:
    """Return the directory containing AirLetters video files."""
    paths = _paths_section(config)
    return _resolve_path(paths["videos_dir"], get_dataset_root(config))


def get_split_csv(config: Mapping[str, Any], split: str) -> Path:
    """Return the CSV path for one official split."""
    if split not in SPLIT_TO_CSV_KEY:
        valid = ", ".join(SPLIT_TO_CSV_KEY)
        raise ValueError(f"Unknown split '{split}'. Expected one of: {valid}.")

    paths = _paths_section(config)
    return _resolve_path(paths[SPLIT_TO_CSV_KEY[split]], get_dataset_root(config))


def get_artifact_dir(config: Mapping[str, Any], key: str) -> Path:
    """Return a project artifact directory such as checkpoints, logs, or outputs."""
    paths = _paths_section(config)
    return _resolve_path(paths[key], PROJECT_ROOT)


def _paths_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "paths" not in config:
        raise KeyError("Config is missing the required 'paths' section.")
    return config["paths"]


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()

