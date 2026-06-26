"""Transformer model for gesture recognition from hand landmarks.

Architecture::

    LandmarkTransformer
    ├── input_proj     : Linear(63 → d_model)         — project keypoints
    ├── pos_encoding   : LearnedPositionalEncoding     — temporal position
    ├── transformer    : nn.TransformerEncoder          — global attention
    ├── pool           : mean over time                 — aggregate sequence
    └── classifier     : Linear(d_model → num_classes) — output logits
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


class LearnedPositionalEncoding(nn.Module):
    """Learnable positional embedding table (one vector per time step)."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to ``x`` (shape ``[B, T, d_model]``)."""
        t = x.size(1)
        positions = torch.arange(t, device=x.device).unsqueeze(0)  # (1, T)
        return x + self.embedding(positions)  # broadcast over batch


class LandmarkTransformer(nn.Module):
    """Transformer encoder that classifies gesture sequences from hand landmarks.

    Args:
        input_dim: Dimensionality of each time-step feature (63 for 21 landmarks).
        d_model: Internal model width.
        nhead: Number of attention heads (must divide ``d_model`` evenly).
        num_layers: Number of ``TransformerEncoderLayer`` blocks.
        dim_feedforward: Width of the FFN inside each encoder layer.
        dropout: Dropout probability applied in the Transformer and classifier.
        max_seq_len: Maximum sequence length (used to size the positional table).
        num_classes: Number of output gesture classes.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = LearnedPositionalEncoding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, d_model) convention
            norm_first=True,   # Pre-LN — more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Classify a batch of landmark sequences.

        Args:
            landmarks: Tensor of shape ``(batch, time, 63)``.

        Returns:
            Logit tensor of shape ``(batch, num_classes)``.
        """
        # Create a boolean key-padding mask: True where the entire frame is
        # zero (i.e. no hand was detected), so attention ignores those steps.
        # Shape: (B, T)
        padding_mask: torch.Tensor = (landmarks.abs().sum(dim=-1) == 0)

        x = self.input_proj(landmarks)     # (B, T, d_model)
        x = self.pos_encoding(x)           # (B, T, d_model)
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, T, d_model)
        x = self.norm(x)

        # Mean-pool over non-padded time steps
        # Invert mask: True → valid frame
        valid_mask = (~padding_mask).float().unsqueeze(-1)  # (B, T, 1)
        # Avoid division by zero when a whole sample has no hands detected
        num_valid = valid_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        x = (x * valid_mask).sum(dim=1) / num_valid        # (B, d_model)

        return self.classifier(x)


def create_model(
    config: Mapping[str, Any],
    num_classes: int | None = None,
) -> LandmarkTransformer:
    """Build a ``LandmarkTransformer`` from a config dict.

    Args:
        config: Full experiment config dict.
        num_classes: Optional override; if ``None``, reads from
            ``config["data"]["num_classes"]``.
    """
    model_config = config["model"]
    if model_config["name"] != "mediapipe_transformer":
        raise ValueError(
            f"Unsupported model name '{model_config['name']}'. "
            "Expected 'mediapipe_transformer'."
        )

    resolved_num_classes = (
        num_classes if num_classes is not None else int(config["data"]["num_classes"])
    )

    return LandmarkTransformer(
        input_dim=int(model_config.get("input_dim", 63)),
        d_model=int(model_config["d_model"]),
        nhead=int(model_config["nhead"]),
        num_layers=int(model_config["num_layers"]),
        dim_feedforward=int(model_config["dim_feedforward"]),
        dropout=float(model_config["dropout"]),
        max_seq_len=int(model_config["max_seq_len"]),
        num_classes=resolved_num_classes,
    )
