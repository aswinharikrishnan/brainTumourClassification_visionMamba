"""
Data loading utilities for Brain Tumor MRI Dataset.
Handles dataset download, loading, splitting, and DataLoader creation.

Dataset: Brain Tumor MRI Dataset by Masoud Nickparvar
URL: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Classes: glioma, meningioma, notumor, pituitary (4 classes)
Total Images: ~7023 (Training: ~5712, Testing: ~1311)
"""

import os
import sys
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from utils.augmentation import get_train_transforms, get_test_transforms


def download_dataset(data_dir=None):
    """
    Download the Brain Tumor MRI Dataset from Kaggle.

    Prerequisites:
      - Install opendatasets: pip install opendatasets
      - Have a Kaggle API key (kaggle.json) or enter credentials when prompted

    Args:
        data_dir: Directory to download to (default: config.DATA_DIR)
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    try:
        import opendatasets as od
        print(f"Downloading dataset to: {data_dir}")
        print(f"Dataset: {config.KAGGLE_DATASET}")
        print("You will need your Kaggle credentials (username + API key).")
        print("Get your API key from: https://www.kaggle.com/settings -> API -> Create New Token\n")
        od.download(
            f"https://www.kaggle.com/datasets/{config.KAGGLE_DATASET}",
            data_dir=os.path.dirname(data_dir),
        )
        print("\nDataset downloaded successfully!")

        # The dataset gets downloaded into a subdirectory; move contents up
        downloaded_dir = os.path.join(
            os.path.dirname(data_dir),
            "brain-tumor-mri-dataset"
        )
        if os.path.exists(downloaded_dir) and downloaded_dir != data_dir:
            import shutil
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
            shutil.move(downloaded_dir, data_dir)
            print(f"Dataset moved to: {data_dir}")

    except ImportError:
        print("opendatasets not installed. Install it with: pip install opendatasets")
        print(f"\nAlternatively, download manually from:")
        print(f"  https://www.kaggle.com/datasets/{config.KAGGLE_DATASET}")
        print(f"\nExtract to: {data_dir}")
        print(f"Expected structure:")
        print(f"  {data_dir}/")
        print(f"  ├── Training/")
        print(f"  │   ├── glioma/")
        print(f"  │   ├── meningioma/")
        print(f"  │   ├── notumor/")
        print(f"  │   └── pituitary/")
        print(f"  └── Testing/")
        print(f"      ├── glioma/")
        print(f"      ├── meningioma/")
        print(f"      ├── notumor/")
        print(f"      └── pituitary/")


def get_data_loaders(
    train_dir=None,
    test_dir=None,
    batch_size=None,
    val_split=None,
    num_workers=None,
    img_size=None,
):
    """
    Create train, validation, and test DataLoaders.

    The training set is split into train and validation sets.
    Training data uses augmentation; validation and test use only resize + normalize.

    Args:
        train_dir:   Path to training images (default: config.TRAIN_DIR)
        test_dir:    Path to test images (default: config.TEST_DIR)
        batch_size:  Batch size (default: config.BATCH_SIZE)
        val_split:   Fraction of training data for validation (default: config.VAL_SPLIT)
        num_workers: DataLoader workers (default: config.NUM_WORKERS)
        img_size:    Image size (default: config.IMG_SIZE)

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    if train_dir is None:
        train_dir = config.TRAIN_DIR
    if test_dir is None:
        test_dir = config.TEST_DIR
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if val_split is None:
        val_split = config.VAL_SPLIT
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    if img_size is None:
        img_size = config.IMG_SIZE

    # Verify dataset exists
    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            f"Download the dataset first using: python -c \"from utils.data_loader import download_dataset; download_dataset()\"\n"
            f"Or download manually from: https://www.kaggle.com/datasets/{config.KAGGLE_DATASET}"
        )

    # Get transforms
    train_transforms = get_train_transforms(img_size)
    test_transforms = get_test_transforms(img_size)

    # Load full training dataset (with train transforms initially)
    full_train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    class_names = full_train_dataset.classes

    print(f"\nDataset Summary:")
    print(f"  Classes: {class_names}")
    print(f"  Total training images: {len(full_train_dataset)}")

    # Print per-class counts
    from collections import Counter
    label_counts = Counter(full_train_dataset.targets)
    for cls_idx, cls_name in enumerate(class_names):
        print(f"    {cls_name}: {label_counts[cls_idx]} images")

    # Split training into train + validation
    total_train = len(full_train_dataset)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size

    # Use a fixed seed for reproducible splits
    generator = torch.Generator().manual_seed(config.SEED)
    train_indices, val_indices = random_split(
        range(total_train), [train_size, val_size], generator=generator
    )

    # Create subsets with appropriate transforms
    train_subset = Subset(full_train_dataset, train_indices.indices)

    # For validation, we need the test transforms (no augmentation)
    val_dataset = ImageFolder(root=train_dir, transform=test_transforms)
    val_subset = Subset(val_dataset, val_indices.indices)

    # Test dataset
    if os.path.exists(test_dir):
        test_dataset = ImageFolder(root=test_dir, transform=test_transforms)
        print(f"  Total test images: {len(test_dataset)}")
    else:
        test_dataset = None
        print(f"  Test directory not found: {test_dir}")

    print(f"\n  Train split: {train_size} images")
    print(f"  Validation split: {val_size} images")
    print(f"  Batch size: {batch_size}\n")

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.PIN_MEMORY,
        )

    return train_loader, val_loader, test_loader, class_names
