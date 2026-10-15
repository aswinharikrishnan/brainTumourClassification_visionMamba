"""
Visualization utilities.
Plots training curves, prediction samples, and Grad-CAM.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: dict with keys:
            - train_acc, val_acc: lists of accuracy per epoch
            - train_loss, val_loss: lists of loss per epoch
        save_path: Path to save the plot
    """
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "training_curves.png")

    epochs = range(1, len(history["train_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Accuracy Curves ----
    ax1.plot(epochs, history["train_acc"], "b-o", label="Training Accuracy", markersize=4, linewidth=2)
    ax1.plot(epochs, history["val_acc"], "r-o", label="Validation Accuracy", markersize=4, linewidth=2)
    ax1.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.0, 1.05])

    # Mark best validation accuracy
    best_epoch = np.argmax(history["val_acc"]) + 1
    best_val_acc = max(history["val_acc"])
    ax1.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
                label=f"Best: {best_val_acc:.4f} @ Epoch {best_epoch}")
    ax1.legend(fontsize=10)

    # ---- Loss Curves ----
    ax2.plot(epochs, history["train_loss"], "b-o", label="Training Loss", markersize=4, linewidth=2)
    ax2.plot(epochs, history["val_loss"], "r-o", label="Validation Loss", markersize=4, linewidth=2)
    ax2.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Training curves saved to: {save_path}")
    plt.close()


def visualize_predictions(model, test_loader, class_names=None, device="cpu",
                          num_samples=12, save_path=None):
    """
    Visualize model predictions on sample test images.

    Args:
        model: Trained model
        test_loader: Test DataLoader
        class_names: List of class names
        device: Device (cpu/cuda)
        num_samples: Number of samples to display
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "predictions.png")

    model.eval()

    # Get a batch of images
    images, labels = next(iter(test_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]

    with torch.no_grad():
        outputs = model(images.to(device))
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        confidences = torch.max(probs, dim=1).values

    # Denormalize images for display
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)

    # Plot grid
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i in range(num_samples):
        img = images[i].cpu() * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()

        true_label = class_names[labels[i].item()]
        pred_label = class_names[preds[i].item()]
        conf = confidences[i].item() * 100

        axes[i].imshow(img)
        axes[i].axis("off")

        # Green if correct, red if wrong
        color = "green" if true_label == pred_label else "red"
        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label} ({conf:.1f}%)",
            fontsize=10,
            color=color,
            fontweight="bold",
        )

    # Hide unused subplots
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Model Predictions on Test Samples",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Predictions visualization saved to: {save_path}")
    plt.close()


def plot_class_distribution(train_loader, test_loader=None, class_names=None, save_path=None):
    """
    Plot class distribution in training and test sets.

    Args:
        train_loader: Training DataLoader
        test_loader: Test DataLoader (optional)
        class_names: List of class names
        save_path: Path to save plot
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "class_distribution.png")

    from collections import Counter

    # Count training labels
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    train_counts = Counter(train_labels)

    fig, axes = plt.subplots(1, 2 if test_loader else 1, figsize=(14, 6))
    if test_loader is None:
        axes = [axes]

    # Training distribution
    train_values = [train_counts.get(i, 0) for i in range(len(class_names))]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    axes[0].bar(class_names, train_values, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Training Set Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Images", fontsize=11)
    for i, v in enumerate(train_values):
        axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")

    # Test distribution
    if test_loader:
        test_labels = []
        for _, labels in test_loader:
            test_labels.extend(labels.numpy())
        test_counts = Counter(test_labels)
        test_values = [test_counts.get(i, 0) for i in range(len(class_names))]
        axes[1].bar(class_names, test_values, color=colors, edgecolor="black", linewidth=0.5)
        axes[1].set_title("Test Set Distribution", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("Number of Images", fontsize=11)
        for i, v in enumerate(test_values):
            axes[1].text(i, v + 5, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Class distribution saved to: {save_path}")
    plt.close()
