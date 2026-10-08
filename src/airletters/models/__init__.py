"""Model builders for AirLetters."""

from typing import Any, Mapping

from airletters.models.cnn_bilstm import CNNBiLSTM
from airletters.models.cnn_bilstm import create_model as _create_bilstm
from airletters.models.cnn_bigru import CNNBiGRU
from airletters.models.cnn_bigru import create_bigru_model as _create_bigru
from airletters.models.cnn_transformer import CNNTransformer
from airletters.models.cnn_transformer import create_transformer_model as _create_transformer


def create_model(
    config: Mapping[str, Any],
    num_classes: int | None = None,
) -> CNNBiLSTM | CNNBiGRU | CNNTransformer:
    """Unified model factory — routes by ``config["model"]["name"]``.

    Supported names:
        * ``"cnn_bilstm"``      -> CNN + Bidirectional LSTM
        * ``"cnn_bigru"``       -> CNN + Bidirectional GRU
        * ``"cnn_transformer"`` -> CNN + Transformer Encoder
    """
    name = config["model"]["name"]
    if name == "cnn_bilstm":
        return _create_bilstm(config, num_classes=num_classes)
    if name == "cnn_bigru":
        return _create_bigru(config, num_classes=num_classes)
    if name == "cnn_transformer":
        return _create_transformer(config, num_classes=num_classes)
    raise ValueError(
        f"Unknown model name '{name}'. "
        f"Expected one of: 'cnn_bilstm', 'cnn_bigru', 'cnn_transformer'."
    )


__all__ = ["CNNBiLSTM", "CNNBiGRU", "CNNTransformer", "create_model"]

