# FreqTrack: A Frequency-Enhanced Spatiotemporal Network
# Implementation of Multi-Path Coordinate Attention (MPCA)

import torch
import torch.nn as nn

class BasicConv(nn.Module):
    """
    Standard Convolution Block with Batch Normalization and Activation.
    Replaces the custom 'Conv' to make the module standalone.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MPCA(nn.Module):
    """
    Multi-Path Coordinate Attention (MPCA).
    Recalibrates features along horizontal (H), vertical (W), and channel (C) dimensions
    independently to preserve precise coordinates of highly dynamic targets in UAV scenarios.
    """
    def __init__(self, channels) -> None:
        super().__init__()

        # 1. Channel Attention Path (Global Average Pooling)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv(channels, channels, kernel_size=1)
        )

        # 2. Spatial Coordinate Paths (1D Pooling along H and W)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 3. Shared convolutions for spatial paths
        self.conv_hw = BasicConv(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv_pool_hw = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        x: Input feature map of shape [B, C, H, W]
        """
        _, _, h, w = x.size()

        # --- Step 1: Multi-dimensional Pooling ---
        x_pool_h = self.pool_h(x)
        x_pool_w = self.pool_w(x).permute(0, 1, 3, 2)
        x_pool_ch = self.gap(x)

        # --- Step 2: Spatial Feature Processing ---
        x_pool_hw = torch.cat([x_pool_h, x_pool_w], dim=2)
        x_pool_hw = self.conv_hw(x_pool_hw)

        x_pool_h, x_pool_w = torch.split(x_pool_hw, [h, w], dim=2)

        # --- Step 3: Attention Weight Generation ---
        x_pool_hw_weight = self.conv_pool_hw(x_pool_hw).sigmoid()
        x_pool_h_weight, x_pool_w_weight = torch.split(x_pool_hw_weight, [h, w], dim=2)

        x_pool_h = x_pool_h * x_pool_h_weight
        x_pool_w = x_pool_w * x_pool_w_weight

        x_pool_ch = x_pool_ch * torch.mean(x_pool_hw_weight, dim=2, keepdim=True)

        # --- Step 4: Final Recalibration (Broadcasting) ---
        out = x * x_pool_h.sigmoid() * x_pool_w.permute(0, 1, 3, 2).sigmoid() * x_pool_ch.sigmoid()

        return out