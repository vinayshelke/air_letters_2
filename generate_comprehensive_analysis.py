"""Generate comprehensive evaluation analysis including ROC curves, per-class metrics, and probability distributions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from airletters.config import get_artifact_dir, load_config
from airletters.data import create_dataloaders
from airletters.models import create_model
from airletters.utils.checkpointing import load_checkpoint


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


@torch.no_grad()
def evaluate_with_probabilities(
    model: nn.Module,
    dataloader,
    device: torch.device,
    split_name: str = "test",
) -> dict[str, Any]:
    """Evaluate model and collect targets, predictions, and probabilities."""
    model.eval()
    
    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_logits = []
    
    print(f"Evaluating on {split_name} set...")
    for batch_idx, batch in enumerate(dataloader):
        videos = batch["video"].to(device, non_blocking=True)
        targets = batch["label_id"].to(device, non_blocking=True)
        
        logits = model(videos)
        probs = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        
        all_targets.extend(targets.cpu().numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_probabilities.extend(probs.cpu().numpy())
        all_logits.extend(logits.cpu().numpy())
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {batch_idx + 1} batches...")
    
    all_probabilities = np.array(all_probabilities)
    all_logits = np.array(all_logits)
    
    return {
        "targets": np.array(all_targets),
        "predictions": np.array(all_predictions),
        "probabilities": all_probabilities,
        "logits": all_logits,
    }


def compute_per_class_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    index_to_label: dict[int, str],
    num_classes: int,
) -> dict[str, Any]:
    """Compute per-class precision, recall, F1, and accuracy."""
    
    metrics = {}
    
    # Compute per-class metrics
    precisions = precision_score(targets, predictions, labels=range(num_classes), zero_division=0, average=None)
    recalls = recall_score(targets, predictions, labels=range(num_classes), zero_division=0, average=None)
    f1_scores = f1_score(targets, predictions, labels=range(num_classes), zero_division=0, average=None)
    
    # Per-class accuracy (recall/sensitivity)
    per_class_acc = []
    for class_idx in range(num_classes):
        mask = targets == class_idx
        if mask.sum() > 0:
            acc = (predictions[mask] == class_idx).mean()
        else:
            acc = 0.0
        per_class_acc.append(acc)
    per_class_acc = np.array(per_class_acc)
    
    # Macro and micro averages
    macro_precision = precisions.mean()
    macro_recall = recalls.mean()
    macro_f1 = f1_scores.mean()
    
    # Micro average = overall accuracy
    micro_precision = accuracy_score(targets, predictions)
    micro_recall = accuracy_score(targets, predictions)
    micro_f1 = accuracy_score(targets, predictions)
    
    metrics["per_class"] = {}
    for class_idx in range(num_classes):
        label = index_to_label.get(class_idx, f"Class {class_idx}")
        metrics["per_class"][label] = {
            "precision": float(precisions[class_idx]),
            "recall": float(recalls[class_idx]),
            "f1": float(f1_scores[class_idx]),
            "accuracy": float(per_class_acc[class_idx]),
            "support": int((targets == class_idx).sum()),
        }
    
    metrics["macro"] = {
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1": float(macro_f1),
    }
    
    metrics["micro"] = {
        "precision": float(micro_precision),
        "recall": float(micro_recall),
        "f1": float(micro_f1),
    }
    
    metrics["overall_accuracy"] = float(accuracy_score(targets, predictions))
    
    return metrics


def compute_roc_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    index_to_label: dict[int, str],
    num_classes: int,
) -> dict[str, Any]:
    """Compute ROC curves and AUC metrics for multiclass classification."""
    
    metrics = {}
    metrics["per_class"] = {}
    
    # One-vs-rest ROC curves and AUC for each class
    roc_curves = {}
    aucs = {}
    
    for class_idx in range(num_classes):
        # Binary classification: this class vs all others
        binary_targets = (targets == class_idx).astype(int)
        
        # Get probability of this class
        class_probs = probabilities[:, class_idx]
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(binary_targets, class_probs)
        roc_auc = auc(fpr, tpr)
        
        label = index_to_label.get(class_idx, f"Class {class_idx}")
        roc_curves[class_idx] = (fpr, tpr)
        aucs[class_idx] = roc_auc
        
        metrics["per_class"][label] = {
            "auc": float(roc_auc),
        }
    
    # Compute macro-average AUC
    macro_auc = np.mean(list(aucs.values()))
    
    # Compute micro-average AUC (one-vs-rest, averaged)
    micro_auc = np.mean(list(aucs.values()))  # For multiclass, micro ~= macro
    
    metrics["macro_auc"] = float(macro_auc)
    metrics["micro_auc"] = float(micro_auc)
    metrics["_roc_curves"] = roc_curves
    metrics["_aucs"] = aucs
    
    return metrics


def save_per_class_metrics_table(
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Save per-class metrics as CSV."""
    import csv
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['Class', 'Precision', 'Recall', 'F1', 'Accuracy', 'AUC', 'Support']
        )
        writer.writeheader()
        
        for class_name, class_metrics in metrics["per_class"].items():
            auc_val = class_metrics.get("auc", None)
            writer.writerow({
                'Class': class_name,
                'Precision': f"{class_metrics['precision']:.4f}",
                'Recall': f"{class_metrics['recall']:.4f}",
                'F1': f"{class_metrics['f1']:.4f}",
                'Accuracy': f"{class_metrics['accuracy']:.4f}",
                'AUC': f"{auc_val:.4f}" if auc_val else "N/A",
                'Support': class_metrics.get('support', 'N/A'),
            })
    
    print(f"Saved per-class metrics to: {output_path}")


def save_overall_metrics_json(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_metrics: dict[str, Any],
    roc_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Save overall metrics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "overall_accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": class_metrics["macro"]["precision"],
        "macro_recall": class_metrics["macro"]["recall"],
        "macro_f1": class_metrics["macro"]["f1"],
        "macro_auc": roc_metrics["macro_auc"],
        "micro_precision": class_metrics["micro"]["precision"],
        "micro_recall": class_metrics["micro"]["recall"],
        "micro_f1": class_metrics["micro"]["f1"],
        "micro_auc": roc_metrics["micro_auc"],
        "num_samples": len(targets),
        "num_classes": len(class_metrics["per_class"]),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved overall metrics to: {output_path}")


def plot_roc_curves_multiclass(
    roc_metrics: dict[str, Any],
    index_to_label: dict[int, str],
    output_path: Path,
    num_classes: int,
) -> None:
    """Plot ROC curves for all classes in one figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    roc_curves = roc_metrics["_roc_curves"]
    aucs = roc_metrics["_aucs"]
    
    # Create a grid of subplots (e.g., 4x7 for 28 classes)
    n_cols = 7
    n_rows = (num_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
    axes = axes.flatten()
    
    for class_idx in range(num_classes):
        ax = axes[class_idx]
        
        if class_idx in roc_curves:
            fpr, tpr = roc_curves[class_idx]
            roc_auc = aucs[class_idx]
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            
            label = index_to_label.get(class_idx, f"Class {class_idx}")
            ax.set_title(f"{label}", fontsize=10)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
    
    # Hide unused subplots
    for idx in range(num_classes, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('ROC Curves - One-vs-Rest (All Classes)', fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved ROC curves plot to: {output_path}")


def plot_macro_micro_roc(
    targets: np.ndarray,
    probabilities: np.ndarray,
    roc_metrics: dict[str, Any],
    index_to_label: dict[int, str],
    num_classes: int,
    output_path: Path,
) -> None:
    """Plot macro and micro average ROC curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    roc_curves = roc_metrics["_roc_curves"]
    aucs = roc_metrics["_aucs"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all class ROC curves lightly
    for class_idx in range(num_classes):
        if class_idx in roc_curves:
            fpr, tpr = roc_curves[class_idx]
            ax.plot(fpr, tpr, alpha=0.2, linewidth=1, color='gray')
    
    # Plot macro-average
    macro_auc = roc_metrics["macro_auc"]
    # For macro, average the TPR at each FPR
    all_fpr = np.unique(np.concatenate([roc_curves[i][0] for i in range(num_classes) if i in roc_curves]))
    mean_tpr = np.zeros_like(all_fpr)
    for class_idx in range(num_classes):
        if class_idx in roc_curves:
            fpr, tpr = roc_curves[class_idx]
            mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= num_classes
    
    ax.plot(all_fpr, mean_tpr, color='blue', lw=3, label=f'Macro-average (AUC = {macro_auc:.3f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Macro-average ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f"Saved macro-average ROC curve to: {output_path}")


def plot_probability_distributions(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    index_to_label: dict[int, str],
    num_classes: int,
    output_dir: Path,
    num_samples_per_class: int = 3,
) -> None:
    """Plot probability distributions for representative examples."""
    output_dir = ensure_dir(output_dir)
    
    # Select representative examples: some correct, some incorrect
    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        
        if class_mask.sum() == 0:
            continue
        
        class_probs = probabilities[class_mask]
        class_preds = predictions[class_mask]
        
        # Separate correct and incorrect
        correct_mask = (class_preds == class_idx)
        incorrect_mask = ~correct_mask
        
        label = index_to_label.get(class_idx, f"Class {class_idx}")
        
        # Create figure with correct and incorrect examples
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 5),
            gridspec_kw={'width_ratios': [1, 1]}
        )
        
        # Plot correct predictions
        if correct_mask.sum() > 0:
            # Pick most and least confident correct predictions
            correct_probs = class_probs[correct_mask]
            correct_confidences = correct_probs[:, class_idx]
            
            # Most confident
            best_idx = correct_confidences.argmax()
            ax = axes[0]
            ax.barh(range(num_classes), correct_probs[best_idx])
            ax.set_yticks(range(num_classes))
            ax.set_yticklabels([index_to_label.get(i, f"C{i}") for i in range(num_classes)], fontsize=8)
            ax.set_xlabel('Probability')
            ax.set_title(f"{label} - Correct (High Confidence)\nProb={correct_confidences[best_idx]:.4f}", fontsize=10)
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3, axis='x')
        else:
            axes[0].text(0.5, 0.5, 'No correct\npredictions', ha='center', va='center', fontsize=12)
            axes[0].set_xticks([])
            axes[0].set_yticks([])
        
        # Plot incorrect predictions
        if incorrect_mask.sum() > 0:
            incorrect_probs = class_probs[incorrect_mask]
            incorrect_preds = class_preds[incorrect_mask]
            incorrect_confidences = incorrect_probs[:, class_idx]
            
            # Most confident incorrect
            best_incorrect_idx = incorrect_confidences.argmax()
            ax = axes[1]
            ax.barh(range(num_classes), incorrect_probs[best_incorrect_idx])
            ax.set_yticks(range(num_classes))
            ax.set_yticklabels([index_to_label.get(i, f"C{i}") for i in range(num_classes)], fontsize=8)
            ax.set_xlabel('Probability')
            pred_label = index_to_label.get(incorrect_preds[best_incorrect_idx], "Unknown")
            ax.set_title(f"{label} - Incorrect\nPredicted: {pred_label}, Prob={incorrect_confidences[best_incorrect_idx]:.4f}", fontsize=10)
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3, axis='x')
        else:
            axes[1].text(0.5, 0.5, 'All correct', ha='center', va='center', fontsize=12)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
        
        fig.suptitle(f"Probability Distributions: {label}", fontsize=12, fontweight='bold')
        fig.tight_layout()
        
        safe_label = label.replace(" ", "_").replace("/", "_")
        fig.savefig(output_dir / f"prob_dist_{safe_label}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Saved probability distribution plots to: {output_dir}")


def plot_confusion_matrix_with_labels(
    targets: np.ndarray,
    predictions: np.ndarray,
    index_to_label: dict[int, str],
    num_classes: int,
    output_path: Path,
) -> None:
    """Plot both normalized and raw confusion matrices."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute confusion matrices
    cm_raw = confusion_matrix(targets, predictions, labels=range(num_classes))
    cm_normalized = cm_raw.astype('float') / cm_raw.sum(axis=1, keepdims=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', ax=ax1, cbar_kws={'label': 'Normalized Count'})
    labels = [index_to_label.get(i, f"C{i}") for i in range(num_classes)]
    ax1.set_xticks(np.arange(num_classes) + 0.5)
    ax1.set_yticks(np.arange(num_classes) + 0.5)
    ax1.set_xticklabels(labels, rotation=90, fontsize=7)
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Normalized Confusion Matrix')
    
    # Raw confusion matrix
    sns.heatmap(cm_raw, annot=False, fmt='d', cmap='Greens', ax=ax2, cbar_kws={'label': 'Count'})
    ax2.set_xticks(np.arange(num_classes) + 0.5)
    ax2.set_yticks(np.arange(num_classes) + 0.5)
    ax2.set_xticklabels(labels, rotation=90, fontsize=7)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('Raw Confusion Matrix')
    
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved confusion matrices to: {output_path}")


def plot_f1_scores(
    class_metrics: dict[str, Any],
    index_to_label: dict[int, str],
    output_path: Path,
) -> None:
    """Plot F1 scores per class as bar chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    classes = sorted(class_metrics["per_class"].keys())
    f1_values = [class_metrics["per_class"][c]["f1"] for c in classes]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = plt.cm.RdYlGn(np.array(f1_values))
    ax.bar(range(len(classes)), f1_values, color=colors)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=9)
    ax.set_ylabel('F1 Score')
    ax.set_ylim([0, 1])
    ax.set_title('Per-Class F1 Scores')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved F1 scores plot to: {output_path}")


def main():
    # Configuration - use digits.yaml as the checkpoint is for digits
    config = load_config("configs/digits.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Configuration: {config['data']['class_filter']}")
    
    # Directories
    results_root = Path("results")
    metrics_dir = ensure_dir(results_root / "metrics")
    figures_dir = ensure_dir(results_root / "figures")
    eval_figures_dir = ensure_dir(figures_dir / "evaluation")
    prob_figures_dir = ensure_dir(figures_dir / "probability")
    
    # Load data and model
    print("\nLoading data and model...")
    dataloaders, _ = create_dataloaders(config)
    checkpoint = load_checkpoint("checkpoints/best.pt", None, device, map_only=True)
    num_classes = len(checkpoint["label_to_index"])
    index_to_label = {i: label for label, i in checkpoint["label_to_index"].items()}
    
    model = create_model(config, num_classes=num_classes).to(device)
    load_checkpoint("checkpoints/best.pt", model, device)
    
    # Evaluate with probabilities
    print("\nEvaluating on test set...")
    eval_results = evaluate_with_probabilities(
        model, dataloaders["test"], device, split_name="test"
    )
    
    targets = eval_results["targets"]
    predictions = eval_results["predictions"]
    probabilities = eval_results["probabilities"]
    
    # Compute metrics
    print("\nComputing metrics...")
    class_metrics = compute_per_class_metrics(targets, predictions, index_to_label, num_classes)
    roc_metrics = compute_roc_metrics(targets, probabilities, index_to_label, num_classes)
    
    # Save metrics
    print("\nSaving metrics...")
    # Merge class and ROC metrics for CSV export
    merged_metrics = {"per_class": {}}
    for class_name in class_metrics["per_class"]:
        merged_metrics["per_class"][class_name] = {
            **class_metrics["per_class"][class_name],
            "auc": roc_metrics["per_class"].get(class_name, {}).get("auc", None),
        }
    save_per_class_metrics_table(
        merged_metrics, 
        metrics_dir / "per_class_metrics.csv"
    )
    
    # Merge metrics for JSON
    metrics_for_json = {
        "class_metrics": {k: v for k, v in class_metrics.items() if k != "per_class"},
        "per_class": class_metrics["per_class"],
        "roc": {k: v for k, v in roc_metrics.items() if not k.startswith("_")},
    }
    
    save_overall_metrics_json(
        targets, predictions, probabilities, class_metrics, roc_metrics,
        metrics_dir / "overall_metrics.json"
    )
    
    with open(metrics_dir / "detailed_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics_for_json, f, indent=2)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_roc_curves_multiclass(roc_metrics, index_to_label, eval_figures_dir / "roc_curves_all_classes.png", num_classes)
    plot_macro_micro_roc(targets, probabilities, roc_metrics, index_to_label, num_classes, eval_figures_dir / "roc_curve_macro_average.png")
    plot_confusion_matrix_with_labels(targets, predictions, index_to_label, num_classes, eval_figures_dir / "confusion_matrices.png")
    plot_f1_scores(class_metrics, index_to_label, eval_figures_dir / "f1_scores.png")
    plot_probability_distributions(targets, predictions, probabilities, index_to_label, num_classes, prob_figures_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Overall Test Accuracy: {class_metrics['overall_accuracy']:.4f}")
    print(f"Macro-average Precision: {class_metrics['macro']['precision']:.4f}")
    print(f"Macro-average Recall: {class_metrics['macro']['recall']:.4f}")
    print(f"Macro-average F1: {class_metrics['macro']['f1']:.4f}")
    print(f"Macro-average AUC: {roc_metrics['macro_auc']:.4f}")
    print(f"\nNumber of samples: {len(targets)}")
    print(f"Number of classes: {num_classes}")
    print("="*80)
    
    print(f"\n✓ All results saved to: {results_root}")
    print(f"  - Metrics: {metrics_dir}")
    print(f"  - Figures: {figures_dir}")


if __name__ == "__main__":
    main()
