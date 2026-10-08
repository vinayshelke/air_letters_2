"""CNN plus Transformer encoder baseline for AirLetters gesture recognition.

Architecture
------------
1. Frame Encoder (EfficientNet-B0, ImageNet pretrained)
   - Processes every frame independently: (B*T, C, H, W) -> (B*T, feature_dim)
   - EfficientNet-B0 feature_dim = 1280

2. Input Projection
   - Linear(feature_dim, d_model)  -- maps 1280-d CNN features to Transformer d_model
   - Learnable positional embeddings added (one per temporal position, up to max_seq_len)

3. Transformer Encoder
   - N stacked TransformerEncoderLayer blocks
   - Each block: Multi-Head Self-Attention (nhead) -> Dropout -> Add+Norm
                 Feed-Forward (dim_feedforward) -> Dropout -> Add+Norm
   - Operates on sequence shape: (B, T, d_model)

4. Temporal Aggregation
   - Mean pool over all T timesteps: (B, T, d_model) -> (B, d_model)
   - Same strategy as BiLSTM baseline for a fair comparison

5. Classification Head
   - Dropout(dropout) -> Linear(d_model, num_classes)
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from torchvision import models


class AttentionPool(nn.Module):
    """Learned attention pooling over the temporal dimension.

    Instead of uniform mean pooling over all timesteps, learned attention pooling
    computes an importance score for each frame and computes a weighted average.
    This enables the model to focus on the salient stroke segments and ignore idle
    frames at the start and end of the gesture.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_frames, d_model)
        scores = self.score(x)                  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        pooled = (x * weights).sum(dim=1)       # (B, d_model)
        return pooled


class CNNTransformer(nn.Module):
    """Frame-level CNN encoder followed by a Transformer temporal classifier."""

    def __init__(
        self,
        frame_encoder: nn.Module,
        feature_dim: int,
        num_classes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int = 64,
    ) -> None:
        """Initialise the CNN+Transformer model.

        Args:
            frame_encoder: Pretrained CNN backbone (classifier head removed).
            feature_dim: Output dimensionality of ``frame_encoder`` (e.g. 1280 for EfficientNet-B0).
            num_classes: Number of output gesture classes.
            d_model: Internal Transformer embedding dimension. Must be divisible by ``nhead``.
            nhead: Number of attention heads in each Transformer layer.
            num_layers: Number of stacked ``TransformerEncoderLayer`` blocks.
            dim_feedforward: Hidden size of the position-wise feed-forward sub-layer.
            dropout: Dropout probability used inside Transformer layers and before classifier.
            max_seq_len: Maximum temporal sequence length (number of frames). Used to
                         pre-allocate learnable positional embeddings.
        """
        super().__init__()
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )

        self.frame_encoder = frame_encoder

        # Project CNN feature_dim -> d_model
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Learnable positional embeddings: one vector per temporal position
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Transformer Encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",      # GELU outperforms ReLU in attention-based models
            batch_first=True,       # Input shape: (batch, seq, d_model)
            norm_first=True,        # Pre-LN: more stable training than post-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),   # Final layer norm after all encoder layers
            enable_nested_tensor=False,   # Pre-LN (norm_first) is incompatible with nested tensors
        )

        # Learned attention pooling over time
        self.pool = AttentionPool(d_model)

        # Classification head (same pattern as BiLSTM baseline)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """Run a batch of videos through the model.

        Args:
            videos: Float tensor of shape ``[batch, frames, channels, height, width]``.

        Returns:
            Class logits of shape ``[batch, num_classes]``.
        """
        batch_size, num_frames, channels, height, width = videos.shape

        # --- 1. Per-frame CNN feature extraction ---
        frames = videos.view(batch_size * num_frames, channels, height, width)
        features = self.frame_encoder(frames)                    # (B*T, feature_dim)
        features = features.view(batch_size, num_frames, -1)     # (B, T, feature_dim)

        # --- 2. Project to d_model + add positional embeddings ---
        x = self.input_proj(features)                            # (B, T, d_model)
        x = x + self.pos_embedding[:, :num_frames, :]           # broadcast positional embed

        # --- 3. Transformer Encoder ---
        x = self.transformer(x)                                  # (B, T, d_model)

        # --- 4. Temporal attention pooling ---
        pooled = self.pool(x)                                    # (B, d_model)

        # --- 5. Classification ---
        return self.classifier(pooled)                           # (B, num_classes)


def create_transformer_model(
    config: Mapping[str, Any],
    num_classes: int | None = None,
) -> CNNTransformer:
    """Create the configured CNN+Transformer model.

    Args:
        config: Full experiment config dict (loaded from ``digits_transformer.yaml``).
        num_classes: Optional override for number of output classes. If ``None``,
                     falls back to ``config["data"]["num_classes"]``.
    """
    model_config = config["model"]
    if model_config["name"] != "cnn_transformer":
        raise ValueError(
            f"Expected model name 'cnn_transformer', got '{model_config['name']}'"
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

    return CNNTransformer(
        frame_encoder=frame_encoder,
        feature_dim=feature_dim,
        num_classes=resolved_num_classes,
        d_model=int(model_config["d_model"]),
        nhead=int(model_config["nhead"]),
        num_layers=int(model_config["num_layers"]),
        dim_feedforward=int(model_config["dim_feedforward"]),
        dropout=float(model_config["dropout"]),
        max_seq_len=int(config["data"]["video"]["num_frames"]) + 4,  # small buffer
    )


def _build_frame_encoder(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Build the CNN frame encoder backbone (shared with BiLSTM baseline)."""
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
