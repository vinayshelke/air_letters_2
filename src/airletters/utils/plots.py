"""Plotting helpers for evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


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
    probabilities: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Save normalized/raw confusion matrices with numbers, per-class accuracy with numbers, and ROC curve."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    labels = [index_to_label[index] for index in sorted(index_to_label)]
    target_names = [_short_label(label) for label in labels]
    label_indices = list(range(len(labels)))

    # Raw counts confusion matrix
    cm_raw = confusion_matrix(targets, predictions, labels=label_indices)
    _save_confusion_matrix(
        cm_raw,
        target_names,
        path / f"{split_name}_confusion_matrix_raw.png",
        title=f"Confusion Matrix (Raw Counts) - {split_name.capitalize()}",
        fmt="d",
        cmap="Blues",
    )

    # Normalized confusion matrix
    cm_norm = confusion_matrix(targets, predictions, labels=label_indices, normalize="true")
    # Replace NaN if any class has 0 samples
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)
    _save_confusion_matrix(
        cm_norm,
        target_names,
        path / f"{split_name}_confusion_matrix_normalized.png",
        title=f"Normalized Confusion Matrix - {split_name.capitalize()}",
        fmt=".2f",
        cmap="Blues",
    )

    # Per-class accuracy with numbers
    _save_per_class_accuracy(cm_norm, target_names, path / f"{split_name}_per_class_accuracy.png")

    # ROC Curves (Macro & Micro) if probabilities provided
    roc_results: dict[str, Any] = {}
    if probabilities is not None and len(probabilities) > 0:
        roc_results = _save_roc_curve(
            targets=targets,
            probabilities=np.array(probabilities),
            num_classes=len(labels),
            target_names=target_names,
            path=path / f"{split_name}_roc_curve.png",
        )

    # Detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=label_indices, zero_division=0
    )
    
    per_class_metrics = {}
    for i, name in enumerate(labels):
        short = target_names[i]
        per_class_metrics[name] = {
            "short_name": short,
            "accuracy": float(cm_norm[i, i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
        }

    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        targets, predictions, average="weighted", zero_division=0
    )

    detailed_report = {
        "macro_avg": {
            "precision": float(macro_prec),
            "recall": float(macro_rec),
            "f1_score": float(macro_f1),
        },
        "weighted_avg": {
            "precision": float(weighted_prec),
            "recall": float(weighted_rec),
            "f1_score": float(weighted_f1),
        },
        "per_class": per_class_metrics,
    }
    if "macro_roc_auc" in roc_results:
        detailed_report["roc_auc"] = roc_results

    return detailed_report


def _save_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    path: Path,
    title: str,
    fmt: str = "d",
    cmap: str = "Blues",
) -> None:
    n_classes = len(labels)
    # Scale figure size dynamically based on number of classes
    fig_size = max(12, int(n_classes * 0.42))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    vmax = None if fmt == "d" else 1.0
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(labels, rotation=90, fontsize=8 if n_classes > 20 else 10)
    ax.set_yticklabels(labels, fontsize=8 if n_classes > 20 else 10)

    # Annotate numbers in each cell
    threshold = (matrix.max() / 2.0) if matrix.max() > 0 else 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            val = matrix[i, j]
            if fmt == "d":
                text_str = f"{int(val)}" if val > 0 else "0"
            else:
                text_str = f"{val:.2f}"
            
            # Highlight non-zero or prominent cells
            color = "white" if val > threshold else "black"
            font_size = 6 if n_classes > 30 else (7 if n_classes > 20 else 9)
            ax.text(j, i, text_str, ha="center", va="center", color=color, fontsize=font_size)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"Saved: {path.name}")


def _save_per_class_accuracy(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    per_class_accuracy = np.diag(matrix) * 100.0  # Percentage
    n_classes = len(labels)
    fig_width = max(14, int(n_classes * 0.45))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bars = ax.bar(np.arange(n_classes), per_class_accuracy, color="#2b5c8f", edgecolor="#1a3654", width=0.7)
    ax.set_title("Per-Class Top-1 Accuracy (%)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Class", fontsize=12, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0.0, 108.0)
    ax.set_xticks(np.arange(n_classes))
    ax.set_xticklabels(labels, rotation=90, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate each bar with its exact percentage number
    for bar, val in zip(bars, per_class_accuracy):
        height = bar.get_height()
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7 if n_classes > 25 else 8,
            rotation=90 if n_classes > 25 else 0,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"Saved: {path.name}")


def _save_roc_curve(
    targets: list[int],
    probabilities: np.ndarray,
    num_classes: int,
    target_names: list[str],
    path: Path,
) -> dict[str, float]:
    """Plot multiclass ROC curves (Micro-average, Macro-average, and per-class)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # One-hot encode targets
    y_true_onehot = np.zeros((len(targets), num_classes), dtype=float)
    for i, t in enumerate(targets):
        y_true_onehot[i, t] = 1.0

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), probabilities.ravel())
    roc_auc_micro = float(roc_auc_score(y_true_onehot, probabilities, average="micro"))
    ax.plot(
        fpr_micro,
        tpr_micro,
        label=f"Micro-average ROC (AUC = {roc_auc_micro:.3f})",
        color="deeppink",
        linestyle=":",
        linewidth=3,
    )

    # Macro-average ROC
    try:
        roc_auc_macro = float(roc_auc_score(y_true_onehot, probabilities, average="macro"))
    except ValueError:
        roc_auc_macro = 0.0

    # Compute individual class ROC curves
    for i in range(num_classes):
        if np.sum(y_true_onehot[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], probabilities[:, i])
            auc_val = float(roc_auc_score(y_true_onehot[:, i], probabilities[:, i]))
            # Plot individual lines lightly so figure remains readable
            ax.plot(fpr, tpr, lw=1, alpha=0.35, label=f"ROC class {target_names[i]} (AUC = {auc_val:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"Multi-Class ROC Curve (Macro AUC = {roc_auc_macro:.3f})", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Legend: keep micro, random, and top classes if too many
    if num_classes <= 15:
        ax.legend(loc="lower right", fontsize=8)
    else:
        # For 28 or 38 classes, showing 40 lines in legend is messy: show summary legend
        handles, leg_labels = ax.get_legend_handles_labels()
        ax.legend(
            [handles[0], handles[-1]],
            [leg_labels[0], f"Macro-average (AUC = {roc_auc_macro:.3f})"],
            loc="lower right",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"Saved: {path.name}")
    return {"macro_roc_auc": roc_auc_macro, "micro_roc_auc": roc_auc_micro}


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

