"""Plotting helpers for evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def save_training_curves(
    metrics_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Read metrics.jsonl and save loss + accuracy training curves.

    Saves ``training_curves.png`` inside *output_dir*.
    """
    path = Path(metrics_path)
    if not path.exists() or path.stat().st_size == 0:
        return

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return

    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    train_acc = [r["train_accuracy"] * 100 for r in rows]
    val_acc = [r["val_accuracy"] * 100 for r in rows]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    # --- Loss ---
    ax_loss.plot(epochs, train_loss, marker="o", linewidth=2, label="Train", color="#4C72B0")
    ax_loss.plot(epochs, val_loss, marker="s", linewidth=2, label="Val", color="#DD8452")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # --- Accuracy ---
    ax_acc.plot(epochs, train_acc, marker="o", linewidth=2, label="Train", color="#4C72B0")
    ax_acc.plot(epochs, val_acc, marker="s", linewidth=2, label="Val", color="#DD8452")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Top-1 Accuracy (%)")
    ax_acc.set_ylim(0, 100)
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "training_curves.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved -> {save_path}")


def save_evaluation_plots(
    targets: list[int],
    predictions: list[int],
    index_to_label: dict[int, str],
    output_dir: str | Path,
    split_name: str,
) -> None:
    """Save confusion matrix and per-class accuracy plots."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    labels = [index_to_label[index] for index in sorted(index_to_label)]
    target_names = [_short_label(label) for label in labels]
    label_indices = list(range(len(labels)))

    matrix = confusion_matrix(targets, predictions, labels=label_indices, normalize="true")
    _save_confusion_matrix(matrix, target_names, path / f"{split_name}_confusion_matrix.png")
    _save_per_class_accuracy(matrix, target_names, path / f"{split_name}_per_class_accuracy.png")


def _save_confusion_matrix(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title("Normalized Confusion Matrix")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_per_class_accuracy(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    per_class_accuracy = np.diag(matrix)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(np.arange(len(labels)), per_class_accuracy)
    ax.set_title("Top-1 Accuracy per Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _short_label(label: str) -> str:
    if label == "Doing nothing":
        return "N"
    if label == "Doing Other Things":
        return "O"
    if "digit" in label:
        return label.split("digit ", maxsplit=1)[1].split(" ", maxsplit=1)[0]
    if "letter" in label:
        return label.split("letter ", maxsplit=1)[1].split(" ", maxsplit=1)[0]
    return label

