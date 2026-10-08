"""CNN plus Bidirectional GRU baseline for AirLetters gesture recognition.

Architecture
------------
1. Frame Encoder  (EfficientNet-B0, ImageNet pretrained)
   Processes every frame independently:  (B*T, C, H, W) -> (B*T, feature_dim=1280)

2. Bidirectional GRU  (temporal encoder)
   - Receives per-frame features as a sequence  (B, T, 1280)
   - 2 stacked GRU layers, hidden_size=256, bidirectional=True
   - At every timestep the forward and backward hidden states are concatenated:
       output shape  (B, T, 512)
   - Dropout is applied BETWEEN layer 1 and layer 2 (when num_layers > 1),
     exactly like the BiLSTM baseline.
   - GRU uses 1/3 fewer parameters than LSTM (no separate cell-state gate).

3. Temporal Aggregation
   Mean pool over all T timesteps: (B, T, 512) -> (B, 512)
   Same strategy as BiLSTM baseline for a fair comparison.

4. Classification Head
   Dropout(dropout) -> Linear(512, num_classes)

Why GRU over LSTM for comparison?
- GRU merges the forget/input gates into a single update gate -> fewer parameters.
- Often trains faster and generalises comparably to LSTM on small-to-medium datasets.
- Provides a third data point: BiLSTM vs BiGRU vs Transformer.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torchvision import models


class CNNBiGRU(nn.Module):
    """Frame-level CNN encoder followed by a Bidirectional GRU classifier."""

    def __init__(
        self,
        frame_encoder: nn.Module,
        feature_dim: int,
        num_classes: int,
        gru_hidden_size: int,
        gru_num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        """Initialise the CNN + BiGRU model.

        Args:
            frame_encoder: Pretrained CNN backbone (classifier head removed).
            feature_dim: Output dimensionality of ``frame_encoder``
                (1280 for EfficientNet-B0).
            num_classes: Number of output gesture classes.
            gru_hidden_size: Hidden dimension of each GRU direction.
                The concatenated output dim = ``gru_hidden_size * 2`` when
                ``bidirectional=True``.
            gru_num_layers: Number of stacked GRU layers.
            dropout: Dropout probability applied between GRU layers
                (only when ``gru_num_layers > 1``) and before the linear
                classifier.
            bidirectional: Whether to run the GRU in both directions.
        """
        super().__init__()
        self.frame_encoder = frame_encoder
        self.temporal_encoder = nn.GRU(
            input_size=feature_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_multiplier = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_size * direction_multiplier, num_classes),
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """Run a batch of videos through the model.

        Args:
            videos: Float tensor of shape
                ``[batch, frames, channels, height, width]``.

        Returns:
            Class logits of shape ``[batch, num_classes]``.
        """
        batch_size, num_frames, channels, height, width = videos.shape

        # --- 1. Per-frame CNN feature extraction ---
        frames = videos.view(batch_size * num_frames, channels, height, width)
        features = self.frame_encoder(frames)                 # (B*T, feature_dim)
        features = features.view(batch_size, num_frames, -1)  # (B, T, feature_dim)

        # --- 2. Bidirectional GRU temporal encoding ---
        # gru returns (output, h_n); we only need output.
        sequence_output, _ = self.temporal_encoder(features)  # (B, T, hidden*dirs)

        # --- 3. Mean-pool over all timesteps ---
        pooled = sequence_output.mean(dim=1)                  # (B, hidden*dirs)

        # --- 4. Classification ---
        return self.classifier(pooled)                        # (B, num_classes)


def create_bigru_model(
    config: Mapping[str, Any],
    num_classes: int | None = None,
) -> CNNBiGRU:
    """Build a ``CNNBiGRU`` from a config dict.

    Args:
        config: Full experiment config dict (loaded from a YAML file).
        num_classes: Optional override for the number of output classes.
            Falls back to ``config["data"]["num_classes"]`` when ``None``.
    """
    model_config = config["model"]
    if model_config["name"] != "cnn_bigru":
        raise ValueError(
            f"Expected model name 'cnn_bigru', got '{model_config['name']}'"
        )

    frame_encoder, feature_dim = _build_frame_encoder(
        name=model_config["frame_encoder"],
        pretrained=bool(model_config["pretrained"]),
    )

    if bool(model_config["freeze_frame_encoder"]):
        for parameter in frame_encoder.parameters():
            parameter.requires_grad = False

    resolved_num_classes = (
        num_classes if num_classes is not None
        else int(config["data"]["num_classes"])
    )

    return CNNBiGRU(
        frame_encoder=frame_encoder,
        feature_dim=feature_dim,
        num_classes=resolved_num_classes,
        gru_hidden_size=int(model_config["gru_hidden_size"]),
        gru_num_layers=int(model_config["gru_num_layers"]),
        dropout=float(model_config["dropout"]),
        bidirectional=bool(model_config["bidirectional"]),
    )


def _build_frame_encoder(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Build the CNN frame encoder backbone."""
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
        feature_dim = model.classifier[1].in_features   # 1280
        model.classifier = nn.Identity()
        return model, feature_dim

    if name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        feature_dim = model.classifier[6].in_features   # 4096
        model.classifier[6] = nn.Identity()
        return model, feature_dim

    if name == "vgg19":
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        model = models.vgg19(weights=weights)
        feature_dim = model.classifier[6].in_features   # 4096
        model.classifier[6] = nn.Identity()
        return model, feature_dim

    raise ValueError(
        f"Unsupported frame encoder '{name}'. "
        f"Expected one of: resnet18, resnet50, efficientnet_b0, vgg16, vgg19."
    )
