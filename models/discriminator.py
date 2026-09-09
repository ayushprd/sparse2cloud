"""PatchGAN discriminator for IWC profile prediction.

Discriminates between real (ground truth along track) and fake (model predicted)
IWC profile fields. Conditioned on the input VIIRS patch.

Input:  (B, N_in + N_pcs, 64, 64) — concatenation of VIIRS patch and IWC prediction
Output: (B, 1, H', W') — per-patch real/fake score

Used with WGAN-GP (Wasserstein GAN with gradient penalty) for stable training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNormConv2d(nn.Module):
    """Conv2d with spectral normalization for Lipschitz constraint (WGAN)."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        return self.conv(x)


class DiscriminatorBlock(nn.Module):
    """Downsample block: Conv → LeakyReLU → Conv → LeakyReLU + skip."""
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv1 = SpectralNormConv2d(in_ch, out_ch, 4, stride=stride, padding=1)
        self.conv2 = SpectralNormConv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.skip = SpectralNormConv2d(in_ch, out_ch, 1, stride=stride) if in_ch != out_ch or stride > 1 else nn.Identity()
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        return h + self.skip(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator conditioned on VIIRS input.

    Takes concatenated (VIIRS_patch, IWC_prediction) and outputs a spatial
    grid of real/fake scores.

    Architecture:
        64x64 → 32x32 → 16x16 → 8x8 → 4x4 → 1 score per 4x4 patch

    Args:
        viirs_channels: Number of VIIRS input channels (default: 10)
        iwc_channels: Number of IWC prediction channels (default: 30 PCA)
        base_dim: Base channel width (default: 64)
        n_layers: Number of downsampling layers (default: 4)
        era5_dim: ERA5 dimension for conditioning (default: 104, 0 to disable)
    """
    def __init__(
        self,
        viirs_channels=10,
        iwc_channels=30,
        base_dim=64,
        n_layers=4,
        era5_dim=104,
    ):
        super().__init__()
        in_ch = viirs_channels + iwc_channels

        # ERA5 conditioning via FiLM (feature-wise linear modulation)
        self.use_era5 = era5_dim > 0
        if self.use_era5:
            self.era5_proj = nn.Sequential(
                nn.Linear(era5_dim, base_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(base_dim * 2, base_dim * 2),  # scale + shift
            )

        # Initial conv (no downsampling)
        self.init_conv = nn.Sequential(
            SpectralNormConv2d(in_ch, base_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        # Downsampling blocks
        dims = [base_dim]
        for i in range(n_layers):
            dim_out = min(base_dim * (2 ** (i + 1)), 512)
            dims.append(dim_out)

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            self.blocks.append(DiscriminatorBlock(dims[i], dims[i + 1], stride=2))

        # Final score
        self.final = SpectralNormConv2d(dims[-1], 1, 3, stride=1, padding=1)

    def forward(self, viirs, iwc_pred, era5=None, era5_valid=None):
        """
        Args:
            viirs:      (B, viirs_ch, 64, 64) VIIRS input patch
            iwc_pred:   (B, iwc_ch, 64, 64) IWC prediction (PCA coefficients)
            era5:       (B, era5_dim) or None
            era5_valid: (B,) or None

        Returns:
            (B, 1, H', W') patch-level scores (no sigmoid — use with WGAN)
        """
        x = torch.cat([viirs, iwc_pred], dim=1)  # (B, viirs+iwc, 64, 64)
        x = self.init_conv(x)

        # ERA5 FiLM conditioning after init conv
        if self.use_era5 and era5 is not None:
            film = self.era5_proj(era5)  # (B, base_dim*2)
            if era5_valid is not None:
                film = film * era5_valid.unsqueeze(1)
            scale, shift = film.chunk(2, dim=1)  # each (B, base_dim)
            x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        for block in self.blocks:
            x = block(x)

        return self.final(x)  # (B, 1, H', W')


def gradient_penalty(discriminator, real, fake, viirs, era5=None, era5_valid=None):
    """WGAN-GP gradient penalty.

    Computes gradient penalty on interpolated samples between real and fake.
    """
    B = real.shape[0]
    alpha = torch.rand(B, 1, 1, 1, device=real.device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    d_out = discriminator(viirs, interpolated, era5, era5_valid)
    grads = torch.autograd.grad(
        outputs=d_out,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True,
        retain_graph=True,
    )[0]

    grads = grads.view(B, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    D = PatchDiscriminator(viirs_channels=10, iwc_channels=30, base_dim=64, n_layers=4)
    print(f"Discriminator params: {count_params(D) / 1e6:.2f}M")

    viirs = torch.randn(2, 10, 64, 64)
    iwc = torch.randn(2, 30, 64, 64)
    era5 = torch.randn(2, 104)
    era5_valid = torch.ones(2)

    out = D(viirs, iwc, era5, era5_valid)
    print(f"Input: viirs {viirs.shape}, iwc {iwc.shape}")
    print(f"Output: {out.shape}")

    # Test gradient penalty
    real = torch.randn(2, 30, 64, 64)
    fake = torch.randn(2, 30, 64, 64)
    gp = gradient_penalty(D, real, fake, viirs, era5, era5_valid)
    print(f"Gradient penalty: {gp.item():.4f}")
    print("OK!")
