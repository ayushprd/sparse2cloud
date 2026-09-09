"""Differentiable forward emulator: (IWC profile, ERA5, geometry) → VIIRS radiances.

Approximates the radiative transfer mapping from atmospheric state + cloud
properties to top-of-atmosphere satellite observations. Must be lightweight
(~500K params) since it will be used inside the training loop for the
physics-consistency loss.

Input:  IWC profile (159 levels, log10 mg/m³)
        ERA5 profile (104 dims: 26 levels × 4 vars)
        SZA (1), lat (1), lon (1)
        is_night flag (1)

Output: VIIRS radiances (9 channels):
        - 5 thermal BT: M12, M13, M14, M15, M16 (in Kelvin)
        - 4 reflectance: M07, M08, M10, M11 (dimensionless, 0-1)

Architecture: ResidualMLP with separate thermal/reflective heads.
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """MLP residual block: Linear → LayerNorm → GELU → Linear → skip."""
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ForwardEmulator(nn.Module):
    """Differentiable forward model: atmospheric state → VIIRS radiances.

    Args:
        iwc_dim: IWC profile dimension (default: 159 active levels)
        era5_dim: ERA5 dimension (default: 104 = 26 levels × 4 vars)
        hidden_dim: Hidden layer width (default: 512)
        n_blocks: Number of residual blocks (default: 4)
        n_thermal: Number of thermal BT channels (default: 5)
        n_refl: Number of reflective channels (default: 4)
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(
        self,
        iwc_dim=159,
        era5_dim=104,
        hidden_dim=512,
        n_blocks=4,
        n_thermal=5,
        n_refl=4,
        dropout=0.1,
    ):
        super().__init__()
        self.n_thermal = n_thermal
        self.n_refl = n_refl

        # Input dimension: IWC + ERA5 + SZA + cos(lat) + sin(lon) + cos(lon) + is_night
        input_dim = iwc_dim + era5_dim + 1 + 3 + 1  # 159 + 104 + 1 + 3 + 1 = 268

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(*[
            ResBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)
        ])

        # Thermal head (BT in Kelvin, ~200-340K)
        self.thermal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_thermal),
        )

        # Reflective head (reflectance 0-1, zero at night)
        self.refl_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_refl),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, iwc, era5, sza, lat, lon, is_night):
        """
        Args:
            iwc:      (B, 159) log10 IWC profile
            era5:     (B, 104) ERA5 flattened
            sza:      (B,)     solar zenith angle in degrees
            lat:      (B,)     latitude in degrees
            lon:      (B,)     longitude in degrees
            is_night: (B,)     1.0 if nighttime (SZA > 90), else 0.0

        Returns:
            thermal: (B, 5) brightness temperatures in K
            refl:    (B, 4) reflectances (0-1, zeroed at night)
        """
        # Encode geometry
        sza_norm = sza.unsqueeze(1) / 90.0  # (B, 1)
        lat_rad = lat.unsqueeze(1) * (3.14159 / 180.0)
        lon_rad = lon.unsqueeze(1) * (3.14159 / 180.0)
        geo = torch.cat([
            sza_norm,
            torch.cos(lat_rad),
            torch.sin(lon_rad),
            torch.cos(lon_rad),
            is_night.unsqueeze(1),
        ], dim=1)  # (B, 5)

        # Concatenate all inputs
        x = torch.cat([iwc, era5, geo], dim=1)  # (B, 268)

        # Shared encoding
        h = self.encoder(x)
        h = self.blocks(h)

        # Thermal prediction (always valid)
        thermal = self.thermal_head(h)  # (B, 5)

        # Reflective prediction (zero at night)
        refl = self.refl_head(h)  # (B, 4)
        refl = torch.sigmoid(refl)  # constrain to [0, 1]
        # Zero out nighttime reflectances
        day_mask = (1.0 - is_night).unsqueeze(1)  # (B, 1), 1 for day, 0 for night
        refl = refl * day_mask

        return thermal, refl


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ForwardEmulator()
    print(f"Parameters: {count_params(model) / 1e6:.2f}M")

    # Test forward pass
    B = 4
    iwc = torch.randn(B, 159)
    era5 = torch.randn(B, 104)
    sza = torch.tensor([30.0, 60.0, 90.0, 120.0])
    lat = torch.tensor([45.0, -30.0, 0.0, 70.0])
    lon = torch.tensor([10.0, -90.0, 180.0, 50.0])
    is_night = torch.tensor([0.0, 0.0, 0.0, 1.0])

    thermal, refl = model(iwc, era5, sza, lat, lon, is_night)
    print(f"Thermal: {thermal.shape}, Refl: {refl.shape}")
    print(f"Night sample refl: {refl[3]}")  # Should be all zeros
    assert refl[3].abs().sum() == 0, "Night reflectance should be zero"
    print("OK!")
