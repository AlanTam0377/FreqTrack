# FreqTrack: A Frequency-Enhanced Spatiotemporal Network
# Implementation of Phase I: Spectral Gating (SG) Mechanism for Efficient Encoder

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralGatingFilter(nn.Module):
    """
    Core Spectral Gating (SG) Filter.
    Achieves O(N log N) global context modeling by replacing standard spatial
    self-attention with element-wise multiplication in the frequency domain.
    """
    def __init__(self, dim, h=64, w=64):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w

        # Define the learnable spectral filter
        self.complex_weight = nn.Parameter(
            torch.randn(dim, h, w // 2 + 1, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x):
        """
        x: Input feature map of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        x_dtype = x.dtype

        # 1. 2D Real Fast Fourier Transform (Spatial -> Frequency)
        x_fft = torch.fft.rfft2(x.float(), norm='ortho')

        freq_h, freq_w = x_fft.shape[2], x_fft.shape[3]

        # 2. Dynamic Resolution Handling
        if freq_h != self.h or freq_w != (self.w // 2 + 1):
            weight = self.complex_weight.permute(3, 0, 1, 2).reshape(1, 2 * C, self.h, self.w // 2 + 1)
            weight_interpolated = F.interpolate(
                weight,
                size=(freq_h, freq_w),
                mode='bilinear',
                align_corners=True
            )
            weight_interpolated = weight_interpolated.view(2, C, freq_h, freq_w).permute(1, 2, 3, 0)
            weight_complex = torch.complex(weight_interpolated[..., 0], weight_interpolated[..., 1])
        else:
            weight_complex = torch.complex(self.complex_weight[..., 0], self.complex_weight[..., 1])

        # 3. Spectral Gating (Element-wise multiplication in frequency domain)
        x_fft_gated = x_fft * weight_complex

        # 4. Inverse 2D Real FFT (Frequency -> Spatial)
        x_filtered = torch.fft.irfft2(x_fft_gated, s=(H, W), norm='ortho')

        return x_filtered.to(x_dtype)


class Mlp(nn.Module):
    """
    Standard Feed-Forward Network (FFN) used in Transformer blocks.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpectralGatingBlock(nn.Module):
    """
    The complete Spectral Gating Block intended to replace the heavy AIFI module.
    Contains the SG filter, Layer Normalization, and FFN with residual connections.
    """
    def __init__(self, dim, h=64, w=64, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()

        self.norm1 = nn.GroupNorm(1, dim)
        self.filter = SpectralGatingFilter(dim, h=h, w=w)

        self.norm2 = nn.GroupNorm(1, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        x = x + self.filter(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x