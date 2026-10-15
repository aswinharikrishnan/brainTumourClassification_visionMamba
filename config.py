"""
Configuration file for Brain Tumor Classification using Vision Mamba.
All hyperparameters and paths are centralized here.
"""

import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# DATASET
# ============================================================
# Kaggle Dataset: "Brain Tumor MRI Dataset" by Masoud Nickparvar
# URL: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"

# Class names (alphabetical order as loaded by ImageFolder)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = 4

# ============================================================
# MODEL - Vision Mamba (Vim)
# ============================================================
IMG_SIZE = 224              # Input image resolution
PATCH_SIZE = 16             # Patch size for tokenization
IN_CHANNELS = 3             # RGB images
EMBED_DIM = 192             # Embedding dimension (Vim-Tiny)
DEPTH = 24                  # Number of Mamba blocks (Vim-Tiny)
SSM_D_STATE = 16            # SSM state dimension
SSM_DT_RANK = "auto"        # SSM dt rank
SSM_CONV_SIZE = 3           # SSM convolution kernel size
SSM_EXPAND = 2              # SSM expansion factor
USE_BIDIRECTIONAL = True    # Bidirectional scanning for images

# Model variant selector
# Options: "vim_tiny", "vim_small"
MODEL_VARIANT = "vim_tiny"

# Vim-Tiny:  depth=24, embed_dim=192, ~7M params
# Vim-Small: depth=24, embed_dim=384, ~26M params

MODEL_CONFIGS = {
    "vim_tiny": {
        "embed_dim": 192,
        "depth": 24,
        "ssm_d_state": 16,
        "ssm_expand": 2,
    },
    "vim_small": {
        "embed_dim": 384,
        "depth": 24,
        "ssm_d_state": 16,
        "ssm_expand": 2,
    },
}

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
MIN_LR = 1e-6

# Early stopping
PATIENCE = 7
MIN_DELTA = 0.001

# Mixed precision
USE_AMP = True

# ============================================================
# DATA AUGMENTATION
# ============================================================
# ImageNet normalization stats (for transfer learning compatibility)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation parameters
ROTATION_DEGREES = 20
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.3
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
RANDOM_ERASING_PROB = 0.1

# ============================================================
# VALIDATION SPLIT
# ============================================================
VAL_SPLIT = 0.15  # 15% of training data for validation

# ============================================================
# LOGGING
# ============================================================
SEED = 42
NUM_WORKERS = 2   # DataLoader workers (set to 0 on Windows)
PIN_MEMORY = True
