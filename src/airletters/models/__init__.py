"""Model builders for AirLetters."""

from __future__ import annotations
from typing import Any, Mapping

import torch.nn as nn

from airletters.models.cnn_bilstm import CNNBiLSTM
from airletters.models.cnn_bilstm import create_model as _create_cnn_bilstm
from airletters.models.landmark_transformer import LandmarkTransformer
from airletters.models.landmark_transformer import create_model as _create_landmark_transformer


def create_model(
    config: Mapping[str, Any],
    num_classes: int | None = None,
) -> nn.Module:
    """Dispatch to the correct model factory based on ``config["model"]["name"]``."""
    name: str = str(config["model"]["name"])
    if name == "cnn_bilstm":
        return _create_cnn_bilstm(config, num_classes=num_classes)
    if name == "mediapipe_transformer":
        return _create_landmark_transformer(config, num_classes=num_classes)
    raise ValueError(
        f"Unknown model name '{name}'. Expected one of: cnn_bilstm, mediapipe_transformer."
    )


__all__ = ["CNNBiLSTM", "LandmarkTransformer", "create_model"]
