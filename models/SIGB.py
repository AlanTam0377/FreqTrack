# FreqTrack: A Frequency-Enhanced Spatiotemporal Network
# Implementation of Spectral-Interactive Gated Block (SIGB)

import torch
import torch.nn as nn

try:
    from timm.models.layers import DropPath
except ImportError:
    class DropPath(nn.Module):
        def __init__(self, drop_prob=None):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  
            return x.div(keep_prob) * random_tensor

from .MPCA import MPCA
from .FDConv import FDConv, FDConv_VisDrone

try:
    from ultralytics.nn.modules.block import C2f
except ImportError:
    class C2f(nn.Module):
        def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
            super().__init__()
            self.c = int(c2 * e)


class GatedFDConvBlock(nn.Module):
    """
    Our implementation of the Gated Frequency-Domain Convolution Block.
    Integrates Frequency Domain Convolution (FDConv) into a gated architecture
    to achieve noise-suppressed spatial-spectral feature fusion.

    Args:
        dim: Input feature channels.
        expansion_ratio: Hidden channel expansion ratio.
        kernel_size: Kernel size for the spatial/frequency convolution.
        conv_ratio: Ratio of channels to conduct depthwise FDConv.
    """
    def __init__(self,
                 dim,
                 expansion_ratio=8 / 3,
                 kernel_size=7,
                 conv_ratio=1.0,
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()

        self.norm = nn.GroupNorm(1, dim)

        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
        self.act = act_layer()

        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)

        # Core Frequency Domain Convolution (FDConv)
        # Using FDConv_VisDrone for enhanced high-frequency small object detection
        self.conv = FDConv_VisDrone(conv_channels, conv_channels, kernel_size=kernel_size)

        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Multi-Path Coordinate Attention (MPCA) for multi-dimensional recalibration
        self.mpca = MPCA(channels=dim)

    def forward(self, x):
        """
        Input x shape: [B, C, H, W]
        """
        shortcut = x

        x = self.norm(x)

        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)

        # Frequency-domain convolution on the content branch
        c = self.conv(c)

        # Gated fusion
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=1))

        x = self.drop_path(x)

        if hasattr(self, 'mpca'):
            x = self.mpca(x)

        return x + shortcut


class SIGB(C2f):
    """
    Spectral-Interactive Gated Block (SIGB).
    Replaces standard Bottlenecks in CSP architectures with GatedFDConvBlocks
    to preserve fine-grained structural details against background noise for UAV tracking.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            GatedFDConvBlock(dim=self.c) for _ in range(n)
        )