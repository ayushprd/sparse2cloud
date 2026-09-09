"""SatVision-TOA Foundation Model fine-tuning for dense IWC profile prediction.

Uses NASA's SatVision-TOA (SwinV2-Giant, 2.6B params) pretrained on 100M MODIS
L1B TOA images as the encoder, with a lightweight FPN decoder for per-pixel
PCA coefficient prediction.

Architecture:
    ChannelAdapter (10 → 14 VIIRS→MODIS mapping)
    → SwinV2-Giant encoder (frozen or fine-tuned with low LR)
    → FPN decoder (multi-scale feature aggregation)
    → Output head (N_pcs per pixel)

Encoder output dims per stage (for 128×128 input):
    Stage 0: 512,  spatial = 32×32
    Stage 1: 1024, spatial = 16×16
    Stage 2: 2048, spatial = 8×8
    Stage 3: 4096, spatial = 4×4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm


# Channel Adapter
class ChannelAdapter(nn.Module):
    """Learnable 1x1 conv mapping VIIRS 10 channels → MODIS 14 channels.

    Initialized with known spectral band correspondences:
        VIIRS Ch → MODIS/SatVision Index
        0 M12 (3.7µm)  → 5 (Band 21, 3.96µm)
        1 M13 (4.05µm) → 5 (Band 21, 3.96µm)
        2 M14 (8.55µm) → 9 (Band 29, 8.55µm)   exact
        3 M15 (10.76µm)→ 11 (Band 31, 11.03µm)
        4 M16 (12.01µm)→ 12 (Band 32, 12.2µm)
        5 M07 (0.865µm)→ 1 (Band 2, 0.865µm)    exact
        6 M08 (1.24µm) → no direct match
        7 M10 (1.61µm) → 3 (Band 6, 1.64µm)
        8 M11 (2.25µm) → 4 (Band 7, 2.13µm)
        9 SZA           → no match (metadata)
    """
    def __init__(self, in_channels=10, out_channels=14):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

        # Initialize with known mapping
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        with torch.no_grad():
            self.conv.weight[5, 0, 0, 0] = 0.5   # M12→Band21
            self.conv.weight[5, 1, 0, 0] = 0.5   # M13→Band21
            self.conv.weight[9, 2, 0, 0] = 1.0   # M14→Band29 exact
            self.conv.weight[11, 3, 0, 0] = 1.0  # M15→Band31
            self.conv.weight[12, 4, 0, 0] = 1.0  # M16→Band32
            self.conv.weight[1, 5, 0, 0] = 1.0   # M07→Band2 exact
            self.conv.weight[3, 7, 0, 0] = 1.0   # M10→Band6
            self.conv.weight[4, 8, 0, 0] = 1.0   # M11→Band7
            # M08 (1.24µm) partial contribution
            self.conv.weight[3, 6, 0, 0] = 0.3   # M08 partial → Band6

    def forward(self, x):
        return self.conv(x)


# FPN Decoder
class FPNDecoder(nn.Module):
    """Feature Pyramid Network decoder for dense prediction from multi-scale features."""
    def __init__(self, encoder_dims=(512, 1024, 2048, 4096),
                 fpn_dim=256, out_channels=30, era5_dim=104, dropout=0.1):
        super().__init__()
        self.n_scales = len(encoder_dims)

        # Lateral connections
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, fpn_dim, 1),
                nn.GroupNorm(32, fpn_dim),
            )
            for dim in encoder_dims
        ])

        # Top-down smoothing
        self.smooths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
                nn.GroupNorm(32, fpn_dim),
                nn.GELU(),
            )
            for _ in encoder_dims
        ])

        # ERA5 conditioning at bottleneck
        if era5_dim > 0:
            self.era5_mlp = nn.Sequential(
                nn.Linear(era5_dim, fpn_dim),
                nn.GELU(),
                nn.Linear(fpn_dim, fpn_dim),
            )
        else:
            self.era5_mlp = None

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Conv2d(fpn_dim * self.n_scales, fpn_dim, 3, padding=1),
            nn.GroupNorm(32, fpn_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
            nn.GroupNorm(32, fpn_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

        self.head = nn.Conv2d(fpn_dim, out_channels, 1)

    def forward(self, features, era5=None, era5_valid=None):
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]

        # ERA5 conditioning on coarsest scale
        if era5 is not None and self.era5_mlp is not None:
            era5_emb = self.era5_mlp(era5)
            if era5_valid is not None:
                era5_emb = era5_emb * era5_valid.unsqueeze(1)
            laterals[-1] = laterals[-1] + era5_emb[:, :, None, None]

        # Top-down pathway
        for i in range(self.n_scales - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:],
                mode='bilinear', align_corners=False
            )

        fpn_outs = [smooth(lat) for smooth, lat in zip(self.smooths, laterals)]

        # Upsample all to highest resolution
        target_size = fpn_outs[0].shape[-2:]
        upsampled = [fpn_outs[0]]
        for feat in fpn_outs[1:]:
            upsampled.append(F.interpolate(feat, size=target_size,
                                            mode='bilinear', align_corners=False))

        fused = torch.cat(upsampled, dim=1)
        fused = self.fusion(fused)
        return self.head(fused)


def load_satvision_weights(model, ckpt_path):
    """Load SatVision-TOA pretrained weights into timm SwinV2 model.

    Handles key remapping between SatVision-TOA checkpoint format and timm
    features_only format:
    - ckpt: encoder.layers.{i}.blocks → timm: layers_{i}.blocks
    - ckpt: encoder.layers.{i}.downsample → timm: layers_{i+1}.downsample
    - ckpt: encoder.patch_embed.xxx → timm: patch_embed.xxx
    - ckpt: encoder.norm.xxx → timm: norm.xxx
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['module']

    timm_state = model.state_dict()
    new_state = {}
    mapped = 0
    shape_mismatch = 0
    not_found = 0

    for ckpt_key, val in state.items():
        if not ckpt_key.startswith('encoder.'):
            continue

        key = ckpt_key[len('encoder.'):]

        # Skip buffers and unsupported keys
        if any(s in key for s in ['mask_token', 'attn_mask', 'norm3',
                                   'relative_coords_table', 'relative_position_index']):
            continue

        # Handle layers.X.downsample → layers_{X+1}.downsample
        if key.startswith('layers.') and '.downsample.' in key:
            parts = key.split('.')
            layer_idx = int(parts[1])
            rest = '.'.join(parts[3:])
            new_key = f'layers_{layer_idx + 1}.downsample.{rest}'
        elif key.startswith('layers.'):
            # layers.X.blocks.Y.xxx → layers_X.blocks.Y.xxx
            parts = key.split('.', 2)
            new_key = f'layers_{parts[1]}.{parts[2]}'
        else:
            # patch_embed.xxx, norm.xxx — keep as-is
            new_key = key

        if new_key in timm_state:
            if val.shape == timm_state[new_key].shape:
                new_state[new_key] = val
                mapped += 1
            else:
                shape_mismatch += 1
        else:
            not_found += 1

    result = model.load_state_dict(new_state, strict=False)

    n_total = len(timm_state)
    print(f"  Loaded {mapped}/{n_total} keys from SatVision-TOA")
    print(f"  Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
    if shape_mismatch > 0:
        print(f"  Shape mismatches: {shape_mismatch}")
    if not_found > 0:
        print(f"  Not found in timm: {not_found}")
    print(f"  Loaded params: {sum(v.numel() for v in new_state.values())/1e9:.2f}B")

    return result


# Full Model
class SatVisionUNet(nn.Module):
    """SatVision-TOA encoder + FPN decoder for dense IWC prediction.

    Args:
        pretrained_path: Path to SatVision-TOA checkpoint
        out_channels: Number of PCA components per pixel
        fpn_dim: FPN feature dimension
        era5_dim: ERA5 input dimension
        freeze_encoder: Whether to freeze encoder weights
        dropout: Dropout rate for decoder
    """
    def __init__(self, pretrained_path=None, out_channels=30, fpn_dim=256,
                 era5_dim=104, freeze_encoder=True, dropout=0.1):
        super().__init__()

        # Channel adapter: 10 VIIRS → 14 MODIS
        self.channel_adapter = ChannelAdapter(in_channels=10, out_channels=14)

        # SwinV2 encoder (timm)
        self.encoder = timm.create_model(
            'swinv2_base_window8_256',
            pretrained=False,
            img_size=128,
            in_chans=14,
            embed_dim=512,
            depths=(2, 2, 42, 2),
            num_heads=(16, 32, 64, 128),
            window_size=8,
            features_only=True,
            strict_img_size=False,
            num_classes=0,
        )

        # Load pretrained weights
        if pretrained_path is not None:
            print(f"  Loading pretrained SatVision-TOA from {pretrained_path}")
            load_satvision_weights(self.encoder, pretrained_path)

        # Freeze encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            n_frozen = sum(p.numel() for p in self.encoder.parameters())
            print(f"  Encoder frozen ({n_frozen/1e9:.2f}B params)")

        # FPN decoder
        encoder_dims = self.encoder.feature_info.channels()
        self.decoder = FPNDecoder(
            encoder_dims=encoder_dims,
            fpn_dim=fpn_dim, out_channels=out_channels,
            era5_dim=era5_dim, dropout=dropout,
        )

        # Print param counts
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable/1e6:.2f}M / Total: {total/1e9:.2f}B")

    def forward(self, x, era5=None, era5_valid=None):
        """
        Args:
            x: (B, 10, 64, 64) VIIRS patches
            era5: (B, era5_dim) or None
            era5_valid: (B,) or None
        Returns:
            (B, out_channels, 64, 64)
        """
        # Upsample 64→128 for SatVision-TOA
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)

        # 10 VIIRS → 14 MODIS channels
        x = self.channel_adapter(x)

        # Encoder: multi-scale features (timm returns channels-last)
        features = self.encoder(x)
        # Convert (B, H, W, C) → (B, C, H, W)
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]

        # Decoder: FPN → predictions at highest encoder resolution
        pred = self.decoder(features, era5, era5_valid)

        # Back to 64×64
        pred = F.interpolate(pred, size=(64, 64), mode='bilinear', align_corners=False)

        return pred

    def unfreeze_encoder(self, stages=None):
        """Unfreeze encoder layers for fine-tuning."""
        if stages is None:
            for p in self.encoder.parameters():
                p.requires_grad = True
        else:
            for idx in stages:
                layer_name = f'layers_{idx}' if hasattr(self.encoder, f'layers_{idx}') else None
                # timm features_only renames layers
                for name, p in self.encoder.named_parameters():
                    if f'layers_{idx}.' in name or f'layers.{idx}.' in name:
                        p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Unfroze encoder. Trainable: {trainable/1e6:.1f}M")

    def get_param_groups(self, base_lr=1e-4, encoder_lr_mult=0.01):
        """Parameter groups with different learning rates."""
        encoder_params = []
        decoder_params = []
        adapter_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('channel_adapter'):
                adapter_params.append(param)
            elif name.startswith('encoder'):
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        groups = [
            {'params': decoder_params, 'lr': base_lr, 'name': 'decoder'},
            {'params': adapter_params, 'lr': base_lr, 'name': 'adapter'},
        ]
        if encoder_params:
            groups.append({
                'params': encoder_params,
                'lr': base_lr * encoder_lr_mult,
                'name': 'encoder',
            })
        return groups


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SatVisionUNet(
        pretrained_path=None, out_channels=30, fpn_dim=256,
        era5_dim=104, freeze_encoder=False, dropout=0.1,
    ).cuda()

    x = torch.randn(2, 10, 64, 64).cuda()
    era5 = torch.randn(2, 104).cuda()
    era5_valid = torch.ones(2).cuda()

    with torch.no_grad():
        out = model(x, era5, era5_valid)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 30, 64, 64)
    print("OK!")
