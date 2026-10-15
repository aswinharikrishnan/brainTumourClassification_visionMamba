"""
Vision Mamba (Vim) - Pure PyTorch Implementation
=================================================
Bidirectional State Space Model backbone for image classification.

Based on: "Vision Mamba: Efficient Visual Representation Learning with
           Bidirectional State Space Model" (ICML 2024)
Paper: https://arxiv.org/abs/2401.09417
Repo:  https://github.com/hustvl/Vim

This implementation provides:
  1. An optimized path using `mamba_ssm` (requires Linux + CUDA)
  2. A pure-PyTorch fallback SSM for environments without CUDA extensions

Architecture:
  Input Image → Patch Embed → Pos Embed → [Bidirectional Mamba Blocks x L] → Norm → Head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

# Try importing the optimized Mamba implementation
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[INFO] mamba_ssm not found. Using pure-PyTorch SSM fallback.")
    print("       For best performance, install on Linux+CUDA:")
    print("       pip install causal-conv1d mamba-ssm --no-build-isolation")


# ============================================================
# Pure PyTorch SSM (Fallback when mamba_ssm is unavailable)
# ============================================================
class PurePyTorchSSM(nn.Module):
    """
    Simplified Selective State Space Model in pure PyTorch.
    This is a fallback for environments without CUDA-compiled mamba_ssm.
    It implements the core S6 (Selective Scan) mechanism.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Input projection: project to 2 * d_inner (for z and x)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Convolution (depthwise)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters projection
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)

        # dt (delta) projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Learnable SSM parameters
        # A is initialized with a special log-space initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # 1D Conv
        x_conv = x_proj.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # trim padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM parameter projections
        ssm_params = self.x_proj(x_conv)  # (B, L, d_state*2 + 1)
        B_param = ssm_params[:, :, :self.d_state]  # (B, L, d_state)
        C_param = ssm_params[:, :, self.d_state:2 * self.d_state]  # (B, L, d_state)
        dt = ssm_params[:, :, -1:]  # (B, L, 1)

        # Delta (dt) - softplus for positivity
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Selective scan (sequential for fallback - not optimized)
        y = self._selective_scan(x_conv, dt, A, B_param, C_param)

        # Apply gate (z) and output projection
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y

    def _selective_scan(self, x, dt, A, B, C):
        """
        Selective scan operation (sequential implementation).
        Args:
            x:  (B, L, d_inner) - input after conv
            dt: (B, L, d_inner) - delta time
            A:  (d_inner, d_state) - state matrix
            B:  (B, L, d_state) - input-dependent B
            C:  (B, L, d_state) - input-dependent C
        Returns:
            y:  (B, L, d_inner) - output
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize: dA = exp(dt * A), dB = dt * B
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B, L, d_inner, d_state)
        dA = torch.exp(dt_A)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)

        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)  # (B, d_inner, d_state)
            y_t = torch.sum(h * C[:, t].unsqueeze(1), dim=-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y


# ============================================================
# Bidirectional Mamba Block
# ============================================================
class BidirectionalMambaBlock(nn.Module):
    """
    A single bidirectional Mamba block.
    Processes tokens in both forward and backward directions
    to capture global context (crucial for 2D image understanding).
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, use_bidirectional=True):
        super().__init__()
        self.use_bidirectional = use_bidirectional

        # Layer normalization
        self.norm = nn.LayerNorm(dim)

        # Forward SSM
        if MAMBA_AVAILABLE:
            self.ssm_forward = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.ssm_forward = PurePyTorchSSM(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

        # Backward SSM (only if bidirectional)
        if use_bidirectional:
            if MAMBA_AVAILABLE:
                self.ssm_backward = Mamba(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            else:
                self.ssm_backward = PurePyTorchSSM(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )

            # Merge projection for bidirectional outputs
            self.merge_proj = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            out: (batch, seq_len, dim)
        """
        residual = x
        x = self.norm(x)

        # Forward scan
        y_forward = self.ssm_forward(x)

        if self.use_bidirectional:
            # Backward scan (reverse → SSM → reverse back)
            x_rev = torch.flip(x, dims=[1])
            y_backward = self.ssm_backward(x_rev)
            y_backward = torch.flip(y_backward, dims=[1])

            # Merge forward and backward
            y = self.merge_proj(torch.cat([y_forward, y_backward], dim=-1))
        else:
            y = y_forward

        # Residual connection
        return y + residual


# ============================================================
# Patch Embedding
# ============================================================
class PatchEmbed(nn.Module):
    """
    Convert 2D image into a sequence of flattened patch embeddings.
    Similar to Vision Transformer (ViT) patch embedding.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Use a convolution to extract and project patches in one step
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            tokens: (batch, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


# ============================================================
# Vision Mamba (Vim) Model
# ============================================================
class VisionMamba(nn.Module):
    """
    Vision Mamba (Vim) - Bidirectional State Space Model for Vision.

    Architecture:
        Input Image → Patch Embed → CLS Token + Pos Embed
        → [Bidirectional Mamba Block × depth] → Layer Norm → CLS Head

    Args:
        img_size:         Input image resolution (default: 224)
        patch_size:       Size of image patches (default: 16)
        in_channels:      Number of input channels (default: 3 for RGB)
        num_classes:      Number of classification classes (default: 4)
        embed_dim:        Embedding dimension (default: 192 for Vim-Tiny)
        depth:            Number of Mamba blocks (default: 24)
        d_state:          SSM state dimension (default: 16)
        d_conv:           SSM conv kernel size (default: 4)
        expand:           SSM expansion factor (default: 2)
        use_bidirectional: Use bidirectional scanning (default: True)
        drop_rate:        Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=4,
        embed_dim=192,
        depth=24,
        d_state=16,
        d_conv=4,
        expand=2,
        use_bidirectional=True,
        drop_rate=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stacked Bidirectional Mamba blocks
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(
                dim=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_bidirectional=use_bidirectional,
            )
            for _ in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Position embedding: truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Linear and LayerNorm layers
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """Extract features before the classification head."""
        batch_size = x.shape[0]

        # Patch embedding: (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.patch_embed(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through Mamba blocks
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.norm(x)

        # Return CLS token representation
        return x[:, 0]

    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width) - input images
        Returns:
            logits: (batch, num_classes) - class logits
        """
        features = self.forward_features(x)
        logits = self.head(features)
        return logits


# ============================================================
# Factory Functions
# ============================================================
def vim_tiny(num_classes=4, img_size=224, **kwargs):
    """Create a Vim-Tiny model (~7M parameters)."""
    return VisionMamba(
        img_size=img_size,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=24,
        d_state=16,
        d_conv=4,
        expand=2,
        **kwargs,
    )


def vim_small(num_classes=4, img_size=224, **kwargs):
    """Create a Vim-Small model (~26M parameters)."""
    return VisionMamba(
        img_size=img_size,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=384,
        depth=24,
        d_state=16,
        d_conv=4,
        expand=2,
        **kwargs,
    )
