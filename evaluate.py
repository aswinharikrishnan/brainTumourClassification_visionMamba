"""
Evaluation Script for Brain Tumor Classification using Vision Mamba.
====================================================================

Loads a trained model checkpoint and evaluates on the test set.

Usage:
    python evaluate.py                                    # Use default best_model.pth
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --normalize                        # Normalized confusion matrix
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm

import config
from models.classifier import BrainTumorVim
from utils.data_loader import get_data_loaders
from utils.metrics import compute_metrics, plot_confusion_matrix
from utils.visualization import visualize_predictions


def evaluate(args):
    """Evaluate a trained model on the test set."""
    print("=" * 70)
    print("  Brain Tumor Classification - Evaluation")
    print("=" * 70)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Load checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"\n  ERROR: Checkpoint not found: {checkpoint_path}")
        print(f"  Train the model first: python train.py")
        return

    print(f"  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model variant from checkpoint
    variant = checkpoint.get("variant", "vim_tiny")
    class_names = checkpoint.get("class_names", config.CLASS_NAMES)
    best_val_acc = checkpoint.get("val_acc", 0.0)

    print(f"  Model variant: {variant}")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")

    # Build model
    model = BrainTumorVim(
        num_classes=config.NUM_CLASSES,
        variant=variant,
        img_size=config.IMG_SIZE,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    model.summary()

    # Load test data
    print("\n  Loading test data...")
    _, _, test_loader, _ = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if test_loader is None:
        print("  ERROR: Test data not found!")
        return

    # Run evaluation
    print("\n" + "-" * 70)
    print("  Running Evaluation on Test Set...")
    print("-" * 70)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, class_names)

    # Confusion matrix
    plot_confusion_matrix(
        all_labels, all_preds, class_names,
        normalize=args.normalize,
    )

    # Visualize predictions
    visualize_predictions(model, test_loader, class_names, device)

    print("\n  Evaluation complete! Results saved in:", config.RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Classifier")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                        help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize confusion matrix")

    args = parser.parse_args()
    evaluate(args)
