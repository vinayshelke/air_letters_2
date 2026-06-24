"""CNN plus BiLSTM baseline for AirLetters gesture recognition."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torchvision import models


class CNNBiLSTM(nn.Module):
    """Frame-level CNN encoder followed by a temporal LSTM classifier."""

    def __init__(
        self,
        frame_encoder: nn.Module,
        feature_dim: int,
        num_classes: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.frame_encoder = frame_encoder
        self.temporal_encoder = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_multiplier = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size * direction_multiplier, num_classes),
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """Run a batch of videos through the model.

        Args:
            videos: Tensor with shape ``[batch, frames, channels, height, width]``.
        """
        batch_size, num_frames, channels, height, width = videos.shape
        frames = videos.view(batch_size * num_frames, channels, height, width)
        features = self.frame_encoder(frames)
        features = features.view(batch_size, num_frames, -1)

        sequence_output, _ = self.temporal_encoder(features)
        final_timestep = sequence_output[:, -1, :]
        return self.classifier(final_timestep)


def create_model(config: Mapping[str, Any], num_classes: int | None = None) -> CNNBiLSTM:
    """Create the configured CNN+BiLSTM baseline.

    Args:
        config: Full experiment config dict.
        num_classes: Optional override for the number of output classes.
            If *None*, the value from ``config["data"]["num_classes"]`` is used.
            Pass the actual count from ``label_to_index`` to stay consistent
            with class-filtered or subset datasets.
    """
    model_config = config["model"]
    if model_config["name"] != "cnn_bilstm":
        raise ValueError(f"Unsupported model name: {model_config['name']}")

    frame_encoder, feature_dim = _build_frame_encoder(
        name=model_config["frame_encoder"],
        pretrained=bool(model_config["pretrained"]),
    )

    if bool(model_config["freeze_frame_encoder"]):
        for parameter in frame_encoder.parameters():
            parameter.requires_grad = False

    resolved_num_classes = num_classes if num_classes is not None else int(config["data"]["num_classes"])

    return CNNBiLSTM(
        frame_encoder=frame_encoder,
        feature_dim=feature_dim,
        num_classes=resolved_num_classes,
        lstm_hidden_size=int(model_config["lstm_hidden_size"]),
        lstm_num_layers=int(model_config["lstm_num_layers"]),
        dropout=float(model_config["dropout"]),
        bidirectional=bool(model_config["bidirectional"]),
    )


def _build_frame_encoder(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim

    raise ValueError(
        f"Unsupported frame encoder '{name}'. Expected one of: resnet18, resnet50, efficientnet_b0."
    )

