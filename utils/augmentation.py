"""
Data augmentation transforms for training and testing.
Mirrors the augmentation used in the original EfficientNetB3 project:
  - Resize, normalization, rotation, flipping, scaling
"""

from torchvision import transforms
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def get_train_transforms(img_size=None):
    """
    Training transforms with data augmentation.
    Augmentation strategy:
      1. Resize to img_size
      2. Random rotation (±20°)
      3. Random horizontal flip (50%)
      4. Random vertical flip (30%)
      5. Color jitter (brightness, contrast)
      6. Random erasing (10%)
      7. Normalize with ImageNet stats

    Args:
        img_size: Target image size (default: from config)
    Returns:
        torchvision.transforms.Compose
    """
    if img_size is None:
        img_size = config.IMG_SIZE

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=config.ROTATION_DEGREES),
        transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
        transforms.RandomVerticalFlip(p=config.VERTICAL_FLIP_PROB),
        transforms.ColorJitter(
            brightness=config.COLOR_JITTER_BRIGHTNESS,
            contrast=config.COLOR_JITTER_CONTRAST,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        transforms.RandomErasing(p=config.RANDOM_ERASING_PROB),
    ])


def get_test_transforms(img_size=None):
    """
    Test/Validation transforms (no augmentation, only resize + normalize).

    Args:
        img_size: Target image size (default: from config)
    Returns:
        torchvision.transforms.Compose
    """
    if img_size is None:
        img_size = config.IMG_SIZE

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])
