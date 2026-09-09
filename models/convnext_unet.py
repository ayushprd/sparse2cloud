"""ConvNeXt U-Net for dense IWC profile prediction.

IceCloudNet-style architecture adapted for 64x64 VIIRS patches.
Predicts N_out (PCA coefficients or raw levels) at every pixel.

Input:  (B, 10, 64, 64) VIIRS + (B, 104) ERA5 + (B,) era5_valid
Output: (B, N_out, 64, 64) per-pixel predictions

Architecture:
    3-level encoder-decoder with ConvNeXt blocks + LinearAttention + skip connections.
    ERA5 injected at bottleneck via additive conditioning.
    ~14M params with base_dim=96, dim_mults=[1, 2, 4].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial


# Building blocks
class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm (B, C, H, W)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class StochasticDepth(nn.Module):
    """Drop entire residual branch with probability p during training."""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device))
        return x * mask / keep


class ConvNextBlock(nn.Module):
    """ConvNeXt block: 7x7 depthwise → LayerNorm → expand → GELU → compress.

    Follows the ConvNeXt V1 design (Liu et al., CVPR 2022).
    """
    def __init__(self, dim, dim_out=None, mult=4, drop_path=0.0, dropout=0.0):
        super().__init__()
        dim_out = dim_out or dim
        hidden = dim * mult

        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.pwconv2 = nn.Conv2d(hidden, dim_out, 1)
        self.drop_path = StochasticDepth(drop_path)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        residual = self.res_conv(x)
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.pwconv2(h)
        return residual + self.drop_path(h)


class LinearAttention(nn.Module):
    """Efficient O(N) linear attention via kernel trick.

    Uses ELU+1 kernel approximation: k(x) = ELU(x) + 1
    Computes: O = normalize(softmax(Q) @ (softmax(K)^T @ V))
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.norm = LayerNorm2d(dim)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            LayerNorm2d(dim),
        )

    def forward(self, x):
        h = self.norm(x)
        B, C, H, W = h.shape
        qkv = self.to_qkv(h).chunk(3, dim=1)  # 3 x (B, inner, H, W)
        q, k, v = map(
            lambda t: t.view(B, self.heads, self.dim_head, H * W),
            qkv
        )

        # ELU+1 kernel for linear attention
        q = F.elu(q, alpha=1.0) + 1.0
        k = F.elu(k, alpha=1.0) + 1.0

        # Linear attention: O(N) instead of O(N^2)
        k_sum = k.sum(dim=-1, keepdim=True)  # (B, heads, dim_head, 1)
        kv = torch.einsum("bhdn,bhen->bhde", k, v)  # (B, heads, dim_head, dim_head)
        qkv = torch.einsum("bhdn,bhde->bhen", q, kv)  # (B, heads, dim_head, N)
        normalizer = torch.einsum("bhdn,bhdn->bhn", q, k_sum.expand_as(q))  # (B, heads, N)
        normalizer = normalizer.unsqueeze(2) + 1e-6  # (B, heads, 1, N)

        out = qkv / normalizer  # (B, heads, dim_head, N)
        out = out.reshape(B, self.heads * self.dim_head, H, W)
        return x + self.to_out(out)


class Downsample(nn.Module):
    """Strided convolution for 2x downsampling."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Transposed convolution for 2x upsampling."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim_in, dim_out, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# Encoder / Decoder levels
class EncoderLevel(nn.Module):
    """Two ConvNextBlocks + LinearAttention + Downsample.

    Blocks operate at `dim`. Downsample transitions from `dim` to `dim_next`.
    Skip connection is taken BEFORE downsampling (at full resolution).
    """
    def __init__(self, dim, dim_next, n_blocks=2, mult=4, drop_path=0.0, heads=4,
                 dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(ConvNextBlock(dim, dim, mult=mult, drop_path=drop_path,
                                             dropout=dropout))
        self.attn = LinearAttention(dim, heads=heads)
        self.down = Downsample(dim, dim_next)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.attn(x)
        skip = x
        x = self.down(x)
        return x, skip


class DecoderLevel(nn.Module):
    """Upsample + skip concat + Conv1x1 + two ConvNextBlocks + LinearAttention."""
    def __init__(self, dim_in, dim_skip, dim_out, n_blocks=2, mult=4, drop_path=0.0,
                 heads=4, dropout=0.0):
        super().__init__()
        self.up = Upsample(dim_in, dim_out)
        self.skip_proj = nn.Conv2d(dim_out + dim_skip, dim_out, 1)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(ConvNextBlock(dim_out, dim_out, mult=mult, drop_path=drop_path,
                                             dropout=dropout))
        self.attn = LinearAttention(dim_out, heads=heads)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch (shouldn't happen with 64→32→16→8)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.skip_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.attn(x)
        return x


class Bottleneck(nn.Module):
    """Mid block: ConvNextBlock + Attention + ConvNextBlock + ERA5 injection."""
    def __init__(self, dim, era5_dim=104, mult=4, heads=4):
        super().__init__()
        self.block1 = ConvNextBlock(dim, dim, mult=mult)
        self.attn = LinearAttention(dim, heads=heads)
        self.block2 = ConvNextBlock(dim, dim, mult=mult)

        # ERA5 conditioning
        self.use_era5 = era5_dim > 0
        if self.use_era5:
            self.era5_mlp = nn.Sequential(
                nn.Linear(era5_dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

    def forward(self, x, era5=None, era5_valid=None):
        x = self.block1(x)
        x = self.attn(x)

        # Add ERA5 conditioning
        if self.use_era5 and era5 is not None:
            era5_emb = self.era5_mlp(era5)  # (B, dim)
            if era5_valid is not None:
                era5_emb = era5_emb * era5_valid.unsqueeze(1)
            x = x + era5_emb[:, :, None, None]  # broadcast spatially

        x = self.block2(x)
        return x


# Main model
class ConvNextUNet(nn.Module):
    """ConvNeXt U-Net for dense IWC profile prediction.

    Args:
        in_channels: Number of input channels (default: 10 for VIIRS)
        out_channels: Number of output channels per pixel (default: 30 for PCA)
        base_dim: Base channel width (default: 96)
        dim_mults: Channel multipliers per level (default: [1, 2, 4])
        n_blocks: ConvNextBlocks per level (default: 2)
        mult: ConvNextBlock expansion ratio (default: 4)
        heads: Attention heads (default: 4)
        drop_path_rate: Max stochastic depth rate (default: 0.25)
        era5_dim: ERA5 input dimension (default: 104)
    """
    def __init__(
        self,
        in_channels=10,
        out_channels=30,
        base_dim=96,
        dim_mults=(1, 2, 4),
        n_blocks=2,
        mult=4,
        heads=4,
        drop_path_rate=0.25,
        era5_dim=104,
        dropout=0.0,
        quantile_mode=False,
        quantiles=(0.1, 0.5, 0.9),
        physics_head=False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.quantile_mode = quantile_mode
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles) if quantile_mode else 0
        self.physics_head = physics_head
        dims = [base_dim * m for m in dim_mults]
        n_levels = len(dims)

        # Initial projection
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 3, padding=1),
            LayerNorm2d(dims[0]),
        )

        # Stochastic depth rates linearly increase
        total_blocks = n_levels * n_blocks * 2 + 2  # encoder + decoder + bottleneck
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        dp_idx = 0

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(n_levels):
            dim_next = dims[i + 1] if i < n_levels - 1 else dims[i]
            dp = dp_rates[dp_idx:dp_idx + n_blocks]
            dp_idx += n_blocks
            self.encoders.append(EncoderLevel(
                dim=dims[i], dim_next=dim_next,
                n_blocks=n_blocks, mult=mult,
                drop_path=sum(dp) / len(dp), heads=heads,
                dropout=dropout,
            ))

        # Bottleneck
        self.bottleneck = Bottleneck(
            dim=dims[-1], era5_dim=era5_dim, mult=mult, heads=heads,
        )
        dp_idx += 2

        # Decoder (reverse order)
        self.decoders = nn.ModuleList()
        for i in range(n_levels - 1, -1, -1):
            dim_in = dims[min(i + 1, n_levels - 1)] if i < n_levels - 1 else dims[i]
            dim_skip = dims[i]
            dim_out = dims[i]
            dp = dp_rates[dp_idx:dp_idx + n_blocks]
            dp_idx += n_blocks
            self.decoders.append(DecoderLevel(
                dim_in=dim_in, dim_skip=dim_skip, dim_out=dim_out,
                n_blocks=n_blocks, mult=mult,
                drop_path=sum(dp) / len(dp) if dp else 0, heads=heads,
                dropout=dropout,
            ))

        # Output head
        if physics_head:
            from models.physics_head import PhysicsConstrainedHead
            self.output_head = PhysicsConstrainedHead(
                dims[0], quantile_mode=quantile_mode,
                n_quantiles=len(quantiles) if quantile_mode else 3)
        elif quantile_mode:
            n_q = len(quantiles)
            self.output_head = nn.Sequential(
                nn.Conv2d(dims[0], dims[0], 3, padding=1),
                LayerNorm2d(dims[0]),
                nn.GELU(),
                nn.Conv2d(dims[0], out_channels * n_q, 1),
            )
        else:
            self.output_head = nn.Conv2d(dims[0], out_channels, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, LayerNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, era5=None, era5_valid=None):
        """
        Args:
            x: (B, in_channels, 64, 64) VIIRS patch
            era5: (B, era5_dim) flattened ERA5 profile, or None
            era5_valid: (B,) binary mask for ERA5 availability, or None
        Returns:
            (B, out_channels, 64, 64) per-pixel predictions
        """
        x = self.init_conv(x)

        # Encoder with skip connections
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck with ERA5 conditioning
        x = self.bottleneck(x, era5, era5_valid)

        # Decoder with skip connections (reverse order)
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = decoder(x, skip)

        out = self.output_head(x)
        if self.quantile_mode and not self.physics_head:
            B, _, H, W = out.shape
            out = out.view(B, self.out_channels, self.n_quantiles, H, W)
        return out


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick test
if __name__ == "__main__":
    model = ConvNextUNet(
        in_channels=10,
        out_channels=30,
        base_dim=96,
        dim_mults=(1, 2, 4),
        era5_dim=104,
    ).cuda()

    print(f"Parameters: {count_params(model) / 1e6:.2f}M")

    # Test forward pass
    x = torch.randn(2, 10, 64, 64).cuda()
    era5 = torch.randn(2, 104).cuda()
    era5_valid = torch.ones(2).cuda()

    with torch.no_grad():
        out = model(x, era5, era5_valid)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 30, 64, 64), f"Expected (2, 30, 64, 64), got {out.shape}"
    print("OK!")
