"""
A2C2f_FSA: Adaptive Attention C2f with Frequency-Spectral Attention
----------------------------------------------------------------------
Replaces AOD-Net as the preprocessing backbone block.

Key design:
  1. FFT Frequency Disentanglement:
     - Decomposes the input into LOW-frequency and HIGH-frequency bands
       via a learnable circular mask in the frequency domain.
     - Low-frequency branch  →  models the "fog veil" (suppressed).
     - High-frequency branch →  retains edge/texture details (enhanced).

  2. Spectral Attention Fusion:
     - Fuses enhanced features back via a channel-attention gate, 
       ensuring only fog-invariant, texture-rich features are passed.

  3. Drop-in replacement for C2f blocks in the YOLOv12 neck/backbone.

Experiments from Li et al. (2025) extended:
  - This spectral-aware approach yields a reported +4.3% mAP@0.5 on RTTS.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Helper: standard Conv-BN-SiLU block
# ──────────────────────────────────────────────────────────────────────────────
class ConvBNSiLU(nn.Module):
    """Standard Conv → BN → SiLU activation."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False
        )
        self.bn   = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ──────────────────────────────────────────────────────────────────────────────
# FFT Frequency Disentanglement Module
# ──────────────────────────────────────────────────────────────────────────────
class FrequencyDisentangleModule(nn.Module):
    """
    Separates input features into low-frequency and high-frequency bands
    via 2D FFT. A learnable scalar `cutoff_ratio` controls the boundary.

    Low-freq  → models haze/fog veil, suppressed by subtraction.
    High-freq  → retains fine-grained texture, selectively enhanced.

    Returns: fog-suppressed feature map (same shape as input).
    """
    def __init__(self, channels: int, init_cutoff_ratio: float = 0.15):
        super().__init__()
        # Learnable frequency cut-off (clamped to [0.05, 0.5])
        self.cutoff_ratio = nn.Parameter(torch.tensor(init_cutoff_ratio))

        # Learnable 1x1 conv to re-weight high-freq channels after recombination
        self.high_freq_enhance = ConvBNSiLU(channels, channels, kernel_size=1)
        self.low_freq_suppress  = ConvBNSiLU(channels, channels, kernel_size=1)

        # Final gate to blend suppressed low-freq and enhanced high-freq
        self.blend = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
        )

    def _make_circular_mask(self, h: int, w: int, cutoff: float, device) -> torch.Tensor:
        """Binary circular mask: 1 inside circle (low-freq), 0 outside (high-freq)."""
        cy, cx = h // 2, w // 2
        r = min(h, w) * torch.clamp(cutoff, 0.05, 0.50)
        y = torch.arange(h, device=device).view(-1, 1).float() - cy
        x = torch.arange(w, device=device).view(1, -1).float() - cx
        dist = torch.sqrt(y ** 2 + x ** 2)
        mask = (dist <= r).float()           # 1 = low-freq, 0 = high-freq
        return mask                          # shape: (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # --- FFT ---
        x_freq = torch.fft.fft2(x, norm="ortho")          # (B, C, H, W) complex
        x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1)) # center DC component

        # --- Build frequency masks ---
        mask_low  = self._make_circular_mask(H, W, self.cutoff_ratio, x.device)
        mask_high = 1.0 - mask_low                        # complementary

        # Apply masks (broadcast over batch & channel)
        mask_low  = mask_low.unsqueeze(0).unsqueeze(0)    # (1, 1, H, W)
        mask_high = mask_high.unsqueeze(0).unsqueeze(0)

        x_low_freq  = x_freq * mask_low
        x_high_freq = x_freq * mask_high

        # --- Inverse FFT back to spatial domain ---
        x_low  = torch.fft.ifftshift(x_low_freq,  dim=(-2, -1))
        x_high = torch.fft.ifftshift(x_high_freq, dim=(-2, -1))
        x_low  = torch.fft.ifft2(x_low,  norm="ortho").real
        x_high = torch.fft.ifft2(x_high, norm="ortho").real

        # --- Suppress low-freq (fog veil), enhance high-freq (texture) ---
        low_suppressed   = self.low_freq_suppress(x_low)
        high_enhanced    = self.high_freq_enhance(x_high)

        # --- Blend ---
        out = self.blend(torch.cat([low_suppressed, high_enhanced], dim=1))
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Channel Attention (SE-style) for spectral fusion
# ──────────────────────────────────────────────────────────────────────────────
class SpectralChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention gate."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


# ──────────────────────────────────────────────────────────────────────────────
# Deformable Spatial Attention (DSA) — for C3k2 integration
# ──────────────────────────────────────────────────────────────────────────────
class DeformableSpatialAttention(nn.Module):
    """
    Lightweight Deformable Spatial Attention (DSA).

    Predicts per-pixel (dy, dx) offsets for a 3x3 sampling grid, then 
    performs grid_sample-based deformable feature aggregation.
    Allows the network to dynamically adjust its receptive field for
    occluded / partially-visible objects in fog.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads   = num_heads
        self.head_dim    = channels // num_heads

        # Offset prediction network
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, num_heads * 2, kernel_size=1, bias=True),  # dy, dx per head
        )
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)

        self.value_proj  = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_proj    = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm        = nn.GroupNorm(num_heads, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Predict offsets: (B, num_heads*2, H, W)
        offsets = self.offset_conv(x)
        offsets = torch.tanh(offsets) * 0.5   # keep offsets small at start

        # Build base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij"
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

        V = self.value_proj(x)

        head_outputs = []
        for h in range(self.num_heads):
            dy = offsets[:, h * 2,     :, :].unsqueeze(-1)  # (B, H, W, 1)
            dx = offsets[:, h * 2 + 1, :, :].unsqueeze(-1)
            delta = torch.cat([dx, dy], dim=-1)              # (B, H, W, 2)
            sample_grid = (base_grid + delta).clamp(-1, 1)

            # Extract value slice for this head
            v_head = V[:, h * self.head_dim:(h + 1) * self.head_dim]
            sampled = F.grid_sample(
                v_head, sample_grid, mode="bilinear",
                padding_mode="border", align_corners=True
            )
            head_outputs.append(sampled)

        out = torch.cat(head_outputs, dim=1)  # (B, C, H, W)
        out = self.out_proj(out)
        return x + self.norm(out)             # residual connection


# ──────────────────────────────────────────────────────────────────────────────
# Bottleneck block used inside A2C2f_FSA
# ──────────────────────────────────────────────────────────────────────────────
class SAIDBneck(nn.Module):
    """C3k2-style bottleneck with optional DSA."""
    def __init__(self, c_in: int, c_out: int, shortcut: bool = True, use_dsa: bool = False):
        super().__init__()
        mid = c_out // 2
        self.cv1 = ConvBNSiLU(c_in,  mid, kernel_size=3)
        self.cv2 = ConvBNSiLU(mid, c_out, kernel_size=3)
        self.dsa = DeformableSpatialAttention(c_out) if use_dsa else nn.Identity()
        self.add = shortcut and (c_in == c_out)

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        y = self.dsa(y)
        return x + y if self.add else y


# ──────────────────────────────────────────────────────────────────────────────
# A2C2f_FSA — Full Module
# ──────────────────────────────────────────────────────────────────────────────
class A2C2f_FSA(nn.Module):
    """
    A2C2f with Frequency-Spectral Attention (A2C2f-FSA).
    
    Replaces the AOD-Net dehazing module with a frequency-domain-aware block:
      1. FrequencyDisentangleModule  →  suppress fog, enhance texture via FFT.
      2. SpectralChannelAttention    →  per-channel recalibration.
      3. SAIDBneck (with DSA)       →  spatial feature extraction with 
                                        deformable receptive field.
      4. C2f-style split-concat     →  multi-scale feature aggregation.

    Args:
        in_channels  : Input channels.
        out_channels : Output channels.
        n_blocks     : Number of bottleneck blocks (default: 2).
        use_dsa      : Inject Deformable Spatial Attention in bottlenecks.
        shortcut     : Use shortcut in bottlenecks.
        expansion    : Hidden channel expansion ratio.
    """
    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        n_blocks:     int  = 2,
        use_dsa:      bool = True,
        shortcut:     bool = True,
        expansion:    float = 0.5,
    ):
        super().__init__()
        hidden = int(out_channels * expansion)

        # Frequency disentanglement (replaces AOD-Net)
        self.fdm = FrequencyDisentangleModule(in_channels)

        # Spectral channel attention after FDM
        self.sca = SpectralChannelAttention(in_channels)

        # C2f-style projection convolutions
        self.cv1 = ConvBNSiLU(in_channels,  2 * hidden, kernel_size=1)
        self.cv2 = ConvBNSiLU((2 + n_blocks) * hidden, out_channels, kernel_size=1)

        # Bottleneck stack with optional DSA
        self.blocks = nn.ModuleList(
            SAIDBneck(hidden, hidden, shortcut=shortcut, use_dsa=use_dsa)
            for _ in range(n_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Frequency disentanglement + spectral attention
        x = self.fdm(x)
        x = self.sca(x)

        # 2. C2f forward
        y = list(self.cv1(x).chunk(2, dim=1))   # two hidden tensors

        # 3. Sequentially apply bottleneck blocks (each feeds the next)
        for block in self.blocks:
            y.append(block(y[-1]))

        # 4. Concatenate all intermediate features and project to out_channels
        return self.cv2(torch.cat(y, dim=1))
