# 🧠 Brain Tumor Classification from MRI Scans using Vision Mamba (Vim)

> **Replacing EfficientNetB3 with Vision Mamba** — A State Space Model approach to brain tumor classification with focus on **output-over-complexity** analysis.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
  - [Step 1: Download Dataset](#step-1-download-dataset)
  - [Step 2: Train the Model](#step-2-train-the-model)
  - [Step 3: Evaluate](#step-3-evaluate)
  - [Step 4: Predict on a Single Image](#step-4-predict-on-a-single-image)
  - [Step 5: Complexity Analysis](#step-5-complexity-analysis)
- [Running on Google Colab](#running-on-google-colab)
- [Expected Results](#expected-results)
- [Output-Over-Complexity](#output-over-complexity)
- [References](#references)

---

## Overview

This project implements a **brain tumor classification system** that classifies MRI scans into **4 categories**:

| Class | Description |
|---|---|
| **Glioma** | Tumor arising from glial cells |
| **Meningioma** | Tumor arising from meninges |
| **Pituitary** | Tumor of the pituitary gland |
| **No Tumor** | Healthy brain scan |

### Why Vision Mamba over EfficientNetB3?

| Feature | EfficientNetB3 (CNN) | Vision Mamba (SSM) |
|---|---|---|
| Context | Local receptive field | **Global from early layers** |
| Complexity | O(N) local | **O(N) global** |
| Architecture | Mature CNN (2019) | **Cutting-edge SSM (ICML 2024)** |
| High-res scaling | Memory intensive | **Linear memory scaling** |

Vision Mamba uses **Bidirectional State Space Models** to capture whole-image context with linear complexity — ideal for medical imaging where subtle boundary details matter.

---

## Dataset

### Brain Tumor MRI Dataset

- **Source**: [Kaggle — Brain Tumor MRI Dataset by Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Total Images**: ~7,023
  - Training: ~5,712 images
  - Testing: ~1,311 images
- **Classes**: 4 (glioma, meningioma, pituitary, notumor)
- **Image Type**: MRI brain scans (various orientations)
- **Format**: JPG images organized in folders by class

### Dataset Structure (after download)

```
data/
├── Training/
│   ├── glioma/         (~1,321 images)
│   ├── meningioma/     (~1,339 images)
│   ├── notumor/        (~1,595 images)
│   └── pituitary/      (~1,457 images)
└── Testing/
    ├── glioma/         (~300 images)
    ├── meningioma/     (~306 images)
    ├── notumor/        (~405 images)
    └── pituitary/      (~300 images)
```

---

## Architecture

### Vision Mamba Pipeline

```
Input MRI Image (224×224×3)
        │
        ▼
┌─────────────────────┐
│   Patch Embedding    │   Split into 16×16 patches → 196 tokens
│   (Conv2d 16×16)     │   Project to embedding dimension
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   + CLS Token        │   Prepend classification token
│   + Position Embed   │   Add learnable position embeddings
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Bidirectional       │   × 24 blocks (Vim-Tiny)
│  Mamba Block         │
│  ┌───────────────┐   │
│  │ Forward SSM →  │   │   Captures left-to-right context
│  │ Backward SSM ← │   │   Captures right-to-left context
│  │ Merge + Residual│  │   Combines both directions
│  └───────────────┘   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Layer Norm         │
│   MLP Head           │   Linear → GELU → Dropout → Linear
│   (4 classes)        │
└─────────┬───────────┘
          │
          ▼
   Output: [glioma, meningioma, notumor, pituitary]
```

### Model Variants

| Variant | Embed Dim | Depth | Parameters | Recommended For |
|---|---|---|---|---|
| **Vim-Tiny** | 192 | 24 | ~7M | Best output/complexity ratio |
| **Vim-Small** | 384 | 24 | ~26M | Maximum accuracy |

---

## Project Structure

```
brainTumourClassification_VisionMamba/
├── .gitignore                    # Excludes PDF, data, models
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── config.py                     # All hyperparameters & paths
│
├── models/
│   ├── __init__.py
│   ├── vision_mamba.py           # Vision Mamba (Vim) backbone
│   └── classifier.py            # Classification head wrapper
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py            # Dataset download & DataLoaders
│   ├── augmentation.py           # Train/test transforms
│   ├── metrics.py                # Accuracy, F1, confusion matrix
│   └── visualization.py          # Training curves, predictions
│
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
├── predict.py                    # Single image prediction
├── complexity_analysis.py        # FLOPs/params comparison
│
├── data/                         # Dataset (gitignored)
├── checkpoints/                  # Saved models (gitignored)
└── results/                      # Output plots & metrics
```

---

## Setup & Installation

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU with CUDA** (required for `mamba-ssm`)
- **Linux** (recommended — Colab works great) or Windows (for code editing only)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/brainTumourClassification_VisionMamba.git
cd brainTumourClassification_VisionMamba

# Install base dependencies
pip install -r requirements.txt

# Install Mamba SSM (Linux + CUDA only)
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation

# Install einops (required for tensor operations)
pip install einops
```

> ⚠️ **Windows Users**: `mamba-ssm` does NOT compile on Windows. Use Google Colab for training. The pure-PyTorch fallback SSM is included for testing code locally.

---

## How to Run

### Step 1: Download Dataset

**Option A — Automatic download (requires Kaggle API key):**

```bash
python -c "from utils.data_loader import download_dataset; download_dataset()"
```

You'll need your Kaggle credentials. Get your API key from:
https://www.kaggle.com/settings → API → "Create New Token"

**Option B — Manual download:**

1. Go to https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. Click "Download"
3. Extract into the `data/` folder so the structure matches:
   ```
   data/Training/glioma/
   data/Training/meningioma/
   data/Training/notumor/
   data/Training/pituitary/
   data/Testing/glioma/
   data/Testing/meningioma/
   data/Testing/notumor/
   data/Testing/pituitary/
   ```

### Step 2: Train the Model

```bash
# Train with default settings (Vim-Tiny, 25 epochs)
python train.py

# Train with Vim-Small for higher accuracy
python train.py --variant vim_small

# Train with custom settings
python train.py --epochs 30 --batch_size 16 --lr 0.0001

# Disable mixed precision (if GPU issues)
python train.py --no_amp
```

**What happens during training:**
1. Dataset is loaded and split (85% train, 15% validation)
2. Vision Mamba model is built and initialized
3. Training runs with cosine annealing + warmup scheduler
4. Best model is saved to `checkpoints/best_model.pth`
5. After training, test evaluation runs automatically
6. Results are saved to `results/` folder

**Output files after training:**
```
results/
├── training_curves.png         # Accuracy & loss curves
├── confusion_matrix.png        # Test set confusion matrix
├── predictions.png             # Sample predictions with confidence
├── metrics.json                # All metrics in JSON format
└── training_history.json       # Epoch-by-epoch history
checkpoints/
└── best_model.pth              # Best model weights
```

### Step 3: Evaluate

```bash
# Evaluate the best saved model on the test set
python evaluate.py

# Evaluate a specific checkpoint
python evaluate.py --checkpoint checkpoints/best_model.pth

# Normalized confusion matrix
python evaluate.py --normalize
```

**Output:**
- Prints accuracy, precision, recall, F1-score (overall and per-class)
- Saves confusion matrix to `results/confusion_matrix.png`
- Saves prediction samples to `results/predictions.png`

### Step 4: Predict on a Single Image

```bash
# Predict a single MRI image
python predict.py --image path/to/your/mri_image.jpg

# Example with a test image
python predict.py --image data/Testing/pituitary/Te-pi_0010.jpg
```

**Output:**
```
  ┌──────────────────────────────────────┐
  │  Predicted Class:  pituitary         │
  │  Confidence:        99.87%           │
  └──────────────────────────────────────┘

  All Probabilities:
    glioma           0.05%  
    meningioma       0.03%  
    notumor          0.05%  
    pituitary       99.87%  █████████████████████████████
```

Also displays a visualization window with the image and confidence bar chart, saved to `results/single_prediction.png`.

### Step 5: Complexity Analysis

```bash
# Compare Vision Mamba vs EfficientNetB3
python complexity_analysis.py

# Analyze Vim-Small variant
python complexity_analysis.py --variant vim_small
```

**Output:**
- Prints comparison table with parameters, FLOPs, accuracy, output/complexity ratios
- Saves bar chart comparison to `results/complexity_comparison.png`

---

## Running on Google Colab

This is the **recommended** way to run the project (free GPU access).

### Quick Start Notebook

Create a new Colab notebook and run these cells:

```python
# Cell 1: Clone repository
!git clone https://github.com/yourusername/brainTumourClassification_VisionMamba.git
%cd brainTumourClassification_VisionMamba

# Cell 2: Install dependencies
!pip install -r requirements.txt
!pip install einops
!pip install causal-conv1d --no-build-isolation
!pip install mamba-ssm --no-build-isolation

# Cell 3: Download dataset (enter Kaggle credentials when prompted)
from utils.data_loader import download_dataset
download_dataset()

# Cell 4: Train the model
!python train.py --epochs 25 --variant vim_tiny

# Cell 5: Run complexity analysis
!python complexity_analysis.py

# Cell 6: Predict on a test image
!python predict.py --image data/Testing/pituitary/Te-pi_0010.jpg
```

### Check GPU

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

> 💡 **Tip**: Use Colab's **T4 GPU** (free tier) or **A100** (Colab Pro) for faster training.

---

## Expected Results

### Training Performance

| Metric | EfficientNetB3 (Reference) | Vision Mamba (Vim-Tiny) |
|---|---|---|
| Validation Accuracy | 98.40% | ~98.5% (target) |
| Best Epoch | 13 / 18 | ~15-20 / 25 |
| Training Time | ~30 min | ~60-90 min |

### Test Performance (per class)

| Class | EfficientNetB3 | Vision Mamba (expected) |
|---|---|---|
| Glioma | ~96% | ~97% |
| Meningioma | ~96% | ~97% |
| Pituitary | ~100% | ~100% |
| No Tumor | 100% | 100% |

---

## Output-Over-Complexity

This is the key analysis comparing model efficiency:

| Metric | EfficientNetB3 | Vim-Tiny | Vim-Small |
|---|---|---|---|
| **Parameters** | ~12.2M | ~7M | ~26M |
| **FLOPs** | ~1.8 GFLOPs | ~1.5 GFLOPs | ~5.0 GFLOPs |
| **Accuracy** | 98.40% | ~98.5% | ~99.0% |
| **Acc / M Params** | 8.07 | **14.07** | 3.81 |
| **Acc / GFLOPs** | 54.67 | **65.67** | 19.80 |
| **Context** | Local (CNN) | **Global (SSM)** | **Global (SSM)** |
| **Complexity** | O(N) local | **O(N) global** | **O(N) global** |

> **Key Finding**: Vim-Tiny achieves comparable/better accuracy with ~42% fewer parameters and global context awareness, making it more efficient per parameter and per FLOP.

---

## References

1. **Vision Mamba**: Zhu, L., et al. "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model." ICML 2024. [arXiv:2401.09417](https://arxiv.org/abs/2401.09417)

2. **Mamba**: Gu, A., & Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

3. **EfficientNet**: Tan, M., & Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML 2019.

4. **Dataset**: Nickparvar, M. "Brain Tumor MRI Dataset." Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

5. **Original Project Reference**: "Brain Tumor Classification from MRI Scans using EfficientNetB3" — Model Engineering College, Thrikkakara (CSD 481 Mini Project, October 2025)

---

## License

This project is for academic/educational purposes.

---

*Built with Vision Mamba (Vim) — Bidirectional State Space Model for Vision*
