"""
Metrics computation utilities.
Computes accuracy, precision, recall, F1-score, and confusion matrix.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_metrics(y_true, y_pred, class_names=None):
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels (list or numpy array)
        y_pred: Predicted labels (list or numpy array)
        class_names: List of class names (default: from config)

    Returns:
        dict with accuracy, precision, recall, f1, and per-class report
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Per-class metrics
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report

    # Print summary
    print("\n" + "=" * 60)
    print("  PERFORMANCE METRICS")
    print("=" * 60)
    print(f"  Overall Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (macro):   {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):      {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (macro):    {metrics['f1_macro']:.4f}")
    print("-" * 60)
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (weighted):  {metrics['f1_weighted']:.4f}")
    print("=" * 60)

    # Per-class breakdown
    print("\n  Per-Class Results:")
    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 55)
    for cls_name in class_names:
        if cls_name in report:
            cls = report[cls_name]
            print(f"  {cls_name:<15} {cls['precision']:>10.4f} {cls['recall']:>10.4f} "
                  f"{cls['f1-score']:>10.4f} {cls['support']:>10.0f}")
    print()

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, normalize=False):
    """
    Plot and save confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot (default: results/confusion_matrix.png)
        normalize: Whether to normalize the confusion matrix
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Confusion matrix saved to: {save_path}")
    plt.close()
