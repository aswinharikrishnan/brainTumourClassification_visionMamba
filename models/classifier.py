"""
Brain Tumor Classifier using Vision Mamba backbone.
Wraps the VisionMamba model with convenience methods for
training, evaluation, and inference.
"""

import torch
import torch.nn as nn
from .vision_mamba import VisionMamba, vim_tiny, vim_small


class BrainTumorVim(nn.Module):
    """
    Brain Tumor Classification model using Vision Mamba.

    This wrapper provides:
      - Easy model variant selection (vim_tiny, vim_small)
      - Proper weight initialization for fine-tuning
      - Feature extraction mode support
      - Model summary and parameter counting

    Args:
        num_classes: Number of tumor classes (default: 4)
        variant:     Model variant - "vim_tiny" or "vim_small"
        img_size:    Input image resolution (default: 224)
        pretrained:  Whether to load pretrained weights (default: False)
        drop_rate:   Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        num_classes=4,
        variant="vim_tiny",
        img_size=224,
        pretrained=False,
        drop_rate=0.1,
    ):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes

        # Build backbone based on variant
        if variant == "vim_tiny":
            self.backbone = vim_tiny(
                num_classes=num_classes,
                img_size=img_size,
                drop_rate=drop_rate,
            )
        elif variant == "vim_small":
            self.backbone = vim_small(
                num_classes=num_classes,
                img_size=img_size,
                drop_rate=drop_rate,
            )
        else:
            raise ValueError(f"Unknown variant: {variant}. Use 'vim_tiny' or 'vim_small'.")

        # Replace classification head with a more robust one
        embed_dim = self.backbone.embed_dim
        self.backbone.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x):
        """Forward pass through the model."""
        return self.backbone(x)

    def get_features(self, x):
        """Extract features before classification head (for Grad-CAM etc.)."""
        return self.backbone.forward_features(x)

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def freeze_backbone(self, num_layers_to_freeze=None):
        """
        Freeze backbone layers for transfer learning.
        Args:
            num_layers_to_freeze: Number of Mamba blocks to freeze.
                                  If None, freezes all except the head.
        """
        if num_layers_to_freeze is None:
            # Freeze everything except the classification head
            for name, param in self.backbone.named_parameters():
                if "head" not in name:
                    param.requires_grad = False
        else:
            # Freeze first N layers
            for i, layer in enumerate(self.backbone.layers):
                if i < num_layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def summary(self):
        """Print model summary."""
        total, trainable = self.count_parameters()
        print("=" * 60)
        print(f"  Brain Tumor Classifier - Vision Mamba ({self.variant})")
        print("=" * 60)
        print(f"  Variant:            {self.variant}")
        print(f"  Embedding Dim:      {self.backbone.embed_dim}")
        print(f"  Depth (Blocks):     {self.backbone.depth}")
        print(f"  Num Classes:        {self.num_classes}")
        print(f"  Total Parameters:   {total:,}")
        print(f"  Trainable Params:   {trainable:,}")
        print(f"  Total Size:         {total * 4 / (1024**2):.2f} MB (FP32)")
        print("=" * 60)


def build_model(config):
    """
    Factory function to build model from config.

    Args:
        config: Configuration module with model settings.
    Returns:
        model: BrainTumorVim instance.
    """
    model = BrainTumorVim(
        num_classes=config.NUM_CLASSES,
        variant=config.MODEL_VARIANT,
        img_size=config.IMG_SIZE,
        pretrained=False,
        drop_rate=0.1,
    )
    return model
