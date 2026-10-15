"""
Single Image Prediction Script.
================================

Classify a single MRI image and display the result with confidence scores.

Usage:
    python predict.py --image path/to/mri_image.jpg
    python predict.py --image path/to/mri_image.jpg --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import config
from models.classifier import BrainTumorVim
from utils.augmentation import get_test_transforms


def predict_single_image(image_path, model, device, class_names=None, transform=None):
    """
    Predict the tumor class for a single MRI image.

    Args:
        image_path: Path to the MRI image
        model: Trained model
        device: torch device
        class_names: List of class names
        transform: Image transform

    Returns:
        predicted_class, confidence, all_probabilities
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    if transform is None:
        transform = get_test_transforms()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1).squeeze()

    predicted_idx = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_idx]
    confidence = probabilities[predicted_idx].item() * 100

    return predicted_class, confidence, probabilities.cpu().numpy()


def display_prediction(image_path, predicted_class, confidence, probabilities,
                       class_names=None, save_path=None):
    """Display the prediction result with confidence bar chart."""
    if class_names is None:
        class_names = config.CLASS_NAMES

    image = Image.open(image_path).convert("RGB")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.2]})

    # Show image
    ax1.imshow(image)
    ax1.set_title(f"Prediction: {predicted_class}\nConfidence: {confidence:.1f}%",
                  fontsize=14, fontweight="bold", color="green")
    ax1.axis("off")

    # Confidence bar chart
    colors = ["#FF6B6B" if i != class_names.index(predicted_class) else "#2ECC71"
              for i in range(len(class_names))]
    bars = ax2.barh(class_names, probabilities * 100, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Confidence (%)", fontsize=12)
    ax2.set_title("Class Probabilities", fontsize=14, fontweight="bold")
    ax2.set_xlim([0, 110])

    # Add percentage labels on bars
    for bar, prob in zip(bars, probabilities):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{prob*100:.1f}%", va="center", fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Prediction saved to: {save_path}")

    plt.show()
    plt.close()


def main(args):
    """Main prediction function."""
    print("=" * 60)
    print("  Brain Tumor MRI - Single Image Prediction")
    print("=" * 60)

    # Validate image path
    if not os.path.exists(args.image):
        print(f"\n  ERROR: Image not found: {args.image}")
        return

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"\n  ERROR: Checkpoint not found: {args.checkpoint}")
        print(f"  Train the model first: python train.py")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    variant = checkpoint.get("variant", "vim_tiny")
    class_names = checkpoint.get("class_names", config.CLASS_NAMES)

    # Build and load model
    model = BrainTumorVim(
        num_classes=config.NUM_CLASSES,
        variant=variant,
        img_size=config.IMG_SIZE,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"  Model: Vision Mamba ({variant})")
    print(f"  Image: {args.image}")

    # Predict
    predicted_class, confidence, probabilities = predict_single_image(
        args.image, model, device, class_names
    )

    # Display results
    print(f"\n  ┌──────────────────────────────────────┐")
    print(f"  │  Predicted Class:  {predicted_class:<18} │")
    print(f"  │  Confidence:       {confidence:>6.2f}%            │")
    print(f"  └──────────────────────────────────────┘")
    print(f"\n  All Probabilities:")
    for cls_name, prob in zip(class_names, probabilities):
        bar = "█" * int(prob * 30)
        print(f"    {cls_name:<15} {prob*100:6.2f}%  {bar}")

    # Display visualization
    save_path = os.path.join(config.RESULTS_DIR, "single_prediction.png")
    display_prediction(args.image, predicted_class, confidence, probabilities,
                       class_names, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict brain tumor from MRI image")
    parser.add_argument("--image", type=str, required=True, help="Path to MRI image")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                        help="Path to model checkpoint")

    args = parser.parse_args()
    main(args)
