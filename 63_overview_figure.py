"""Figure 1: Problem overview for ECCV paper.

Creates a 3-panel overview figure:
  (a) EarthCARE narrow track vs VIIRS wide swath on globe
  (b) Architecture schematic (text-based)
  (c) Result teaser: one patch VIIRS -> prediction -> uncertainty

Usage:
    python -u 63_overview_figure.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from pathlib import Path

PAPER_DIR = Path("figures/eccv")
PAPER_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def panel_a_sparse_dense(fig, gs_slot):
    """Panel (a): Illustrate sparse track vs dense swath problem."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ax = fig.add_subplot(gs_slot, projection=ccrs.Orthographic(80, 0))
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='#d4e6f1')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color='#666666')

        # Simulate EarthCARE orbit (narrow track)
        lats = np.linspace(-60, 60, 200)
        lons = 80 + 25 * np.sin(np.radians(lats * 2.5))
        ax.plot(lons, lats, 'r-', linewidth=1.5, transform=ccrs.PlateCarree(),
                label='EarthCARE track', zorder=5)

        # VIIRS swath (wide)
        swath_left = lons - 15
        swath_right = lons + 15
        coords = list(zip(swath_left, lats)) + list(zip(swath_right[::-1], lats[::-1]))
        ax.fill([c[0] for c in coords], [c[1] for c in coords],
                alpha=0.15, color='#2196F3', transform=ccrs.PlateCarree(),
                label='VIIRS swath', zorder=3)

        ax.set_title("(a) Sparse Track vs Dense Swath", fontsize=9, fontweight='bold')
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=1.5, label='EarthCARE (~5 km)'),
            mpatches.Patch(facecolor='#2196F3', alpha=0.3, label='VIIRS (~3000 km)')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=7,
                  frameon=True, fancybox=False, edgecolor='0.8')
        return ax

    except ImportError:
        ax = fig.add_subplot(gs_slot)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')

        theta = np.linspace(0, 2*np.pi, 100)
        ax.fill(np.cos(theta), np.sin(theta), color='#d4e6f1', alpha=0.5)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)

        y = np.linspace(-0.8, 0.8, 50)
        x = 0.1 * np.sin(y * 5)
        ax.plot(x, y, 'r-', linewidth=2, label='EarthCARE\n(~5 km track)')
        ax.fill_betweenx(y, x - 0.3, x + 0.3, alpha=0.2, color='#2196F3',
                          label='VIIRS\n(~3000 km swath)')

        ax.set_title("(a) Sparse Track vs Dense Swath", fontsize=9, fontweight='bold')
        ax.legend(fontsize=7, loc='lower left', frameon=True, fancybox=False, edgecolor='0.8')
        ax.axis('off')
        return ax


def panel_b_architecture(ax):
    """Panel (b): Architecture schematic with explicit box geometry.

    Uses data coordinates with known box positions so arrows connect
    precisely to box edges — no guessing.
    """
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.2, 7.5)
    ax.axis('off')

    fs = 7.5

    # Colors
    C_BLUE = ('#e3f2fd', '#1565c0')
    C_RED = ('#fce4ec', '#c62828')
    C_GREEN = ('#e8f5e9', '#2e7d32')
    C_ORANGE = ('#fff3e0', '#e65100')

    def draw_box(cx, cy, w, h, text, color, bold=False, fs_override=None):
        """Draw a rounded box at (cx, cy) with width w, height h. Returns (cx, cy, w, h)."""
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle='round,pad=0.15', facecolor=color[0],
                              edgecolor=color[1], lw=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy, text, fontsize=fs_override or fs, ha='center', va='center',
                fontweight='bold' if bold else 'normal', zorder=4)
        return (cx, cy, w, h)

    def arrow(x1, y1, x2, y2, color='#333', lw=1.5, style='-', head=True):
        """Draw arrow from (x1,y1) to (x2,y2)."""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->' if head else '-',
                                    color=color, lw=lw, linestyle=style,
                                    mutation_scale=14, shrinkA=0, shrinkB=0),
                    zorder=2)

    # Pipeline row: y=2.0, boxes have explicit widths
    y = 2.0
    bh = 1.6  # box height

    # Define boxes: (x_center, width, label, color, bold)
    boxes_def = [
        (1.5,  2.2, 'VIIRS\n10 ch\n64×64',             C_BLUE,  True),
        (4.8,  2.4, 'ConvNeXt\nEncoder\n96→384',        C_BLUE,  False),
        (8.0,  2.2, 'Linear\nAttn\n384',                C_BLUE,  False),
        (11.2, 2.4, 'ConvNeXt\nDecoder\n384→96',        C_BLUE,  False),
        (14.9, 3.0, '8 targets\n×3 quantiles\n64×64',   C_RED,   True),
        (18.0, 1.8, 'CQR\nCalib.',                      C_GREEN, True),
    ]

    boxes = []
    for (cx, w, label, color, bold) in boxes_def:
        boxes.append(draw_box(cx, y, w, bh, label, color, bold))

    # Horizontal arrows between consecutive boxes (right edge → left edge)
    for i in range(len(boxes) - 1):
        cx1, _, w1, _ = boxes[i]
        cx2, _, w2, _ = boxes[i+1]
        x_from = cx1 + w1/2 + 0.08  # small gap from box edge
        x_to = cx2 - w2/2 - 0.08
        arrow(x_from, y, x_to, y)

    # ERA5 box: above the bottleneck (Linear Attn), centered
    x_bn = boxes[2][0]  # Linear Attn center x
    y_era5 = 5.3
    draw_box(x_bn, y_era5, 2.2, 1.2, 'ERA5\n104-dim', C_ORANGE, bold=True)
    # Vertical arrow: ERA5 bottom → bottleneck top
    arrow(x_bn, y_era5 - 0.6 - 0.08, x_bn, y + bh/2 + 0.08, color=C_ORANGE[1], lw=1.5)

    # Skip connections: curved arc from encoder to decoder, routed ABOVE ERA5
    x_enc = boxes[1][0]  # encoder center
    x_dec = boxes[3][0]  # decoder center
    y_top = y + bh/2 + 0.1

    # Single arc that goes high above everything (above ERA5 box)
    ax.annotate('', xy=(x_dec, y_top), xytext=(x_enc, y_top),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.3,
                                connectionstyle='arc3,rad=-0.7',
                                linestyle='--', mutation_scale=12),
                zorder=1)
    ax.text((x_enc + x_dec) / 2, 6.4, 'skip connections', fontsize=7,
            ha='center', color='#555', style='italic', zorder=1)

    ax.text(10.0, 0.25, '~14M parameters', fontsize=7, ha='center', color='#666')
    ax.set_title("(b) ConvNextUNet + CQR Pipeline", fontsize=9, fontweight='bold',
                 pad=6)


def generate_teaser():
    """Generate teaser sub-images: VIIRS BT, cloud top prediction, uncertainty."""
    from config import OUTPUT_DIR, COLOC_DIR
    from models.convnext_unet import ConvNextUNet
    import torch
    from torch.amp import autocast

    DENSE_DIR = COLOC_DIR / "npy_dense"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tags = ["G-Q", "G-Q2", "G-Q3", "G-Q4", "G-Q5"]
    models = []
    geo_mean = geo_std = era5_mean = era5_std = None

    for tag in tags:
        ckpt_path = OUTPUT_DIR / "models" / f"geom_{tag}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if geo_mean is None:
            geo_mean = np.array(ckpt["geo_mean"], dtype=np.float32)
            geo_std = np.array(ckpt["geo_std"], dtype=np.float32)
            era5_mean = np.array(ckpt["era5_mean"], dtype=np.float32)
            era5_std = np.array(ckpt["era5_std"], dtype=np.float32)

        model = ConvNextUNet(
            in_channels=10, out_channels=8,
            base_dim=ckpt["base_dim"], dim_mults=(1, 2, 4),
            era5_dim=ckpt["era5_dim"],
            quantile_mode=True, physics_head=False,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

    patches = np.load(DENSE_DIR / "test_patches.npy")
    era5_raw = np.load(DENSE_DIR / "test_era5.npy").astype(np.float32)
    nan_mask = np.isnan(era5_raw)
    era5_raw[nan_mask] = 0.0
    era5_valid = (~nan_mask[:, 0, 0]).astype(np.float32)
    era5_flat = era5_raw.reshape(len(era5_raw), -1)
    era5_flat = (era5_flat - era5_mean) / era5_std
    era5_flat[era5_valid < 0.5] = 0.0

    patch_idx = 3128  # Deep convection, Indonesia
    patch = patches[patch_idx]
    era5 = era5_flat[patch_idx]
    era5v = era5_valid[patch_idx]

    patch_t = torch.from_numpy(patch[None]).to(device)
    era5_t = torch.from_numpy(era5[None]).to(device)
    era5v_t = torch.from_numpy(np.array([era5v])).to(device)

    preds = []
    with torch.no_grad():
        for model in models:
            with autocast("cuda", dtype=torch.bfloat16):
                p = model(patch_t, era5_t, era5v_t)
            preds.append(p.float().cpu().numpy()[0])

    pred = np.mean(preds, axis=0)
    for t in range(8):
        pred[t] = pred[t] * geo_std[t] + geo_mean[t]

    def level_to_km(level):
        return 16.2 - (level - 69) * (16.2 - 0.6) / 158

    bt = patch[3]  # M15 BT
    ct_km = level_to_km(pred[1, 1])  # median cloud top
    width_km = np.abs(level_to_km(pred[1, 0]) - level_to_km(pred[1, 2]))

    return bt, ct_km, width_km


def main():
    print("Creating Figure 1: Problem Overview")

    # Try to generate teaser data
    bt, ct_km, width_km = None, None, None
    try:
        print("  Generating teaser predictions...")
        bt, ct_km, width_km = generate_teaser()
        print("  Teaser generated successfully")
    except Exception as e:
        print(f"  Warning: Could not generate teaser: {e}")

    # Figure layout: 2 rows, 3 columns
    #   Row 0: [  globe  ] [    architecture        ] [ teaser images ]
    #   Globe spans both rows; architecture is top-right; teaser is right column
    fig = plt.figure(figsize=(14, 4.0))

    if bt is not None:
        gs = GridSpec(1, 3, figure=fig,
                      width_ratios=[2.0, 5.0, 3.5],
                      wspace=0.08)

        # Panel (a): Globe
        panel_a_sparse_dense(fig, gs[0])

        # Panel (b): Architecture
        ax2 = fig.add_subplot(gs[1])
        panel_b_architecture(ax2)

        # Panel (c): Teaser — 3 sub-panels stacked horizontally
        gs_teaser = gs[2].subgridspec(1, 3, wspace=0.15)

        ax_bt = fig.add_subplot(gs_teaser[0, 0])
        ax_bt.imshow(bt, cmap='gray_r', aspect='equal')
        ax_bt.set_title('VIIRS BT', fontsize=8)
        ax_bt.axis('off')

        ax_ct = fig.add_subplot(gs_teaser[0, 1])
        ax_ct.imshow(ct_km, cmap='RdYlBu_r', vmin=2, vmax=16, aspect='equal')
        ax_ct.set_title('Cloud Top (km)', fontsize=8)
        ax_ct.axis('off')

        ax_uc = fig.add_subplot(gs_teaser[0, 2])
        ax_uc.imshow(width_km, cmap='magma', vmin=0, vmax=8, aspect='equal')
        ax_uc.set_title('Uncertainty (km)', fontsize=8)
        ax_uc.axis('off')

        # (c) label centered above the three teaser panels
        # Use ax_ct (middle panel) as anchor
        ax_ct.text(0.5, 1.55, '(c) Dense Prediction', fontsize=9,
                   ha='center', va='bottom', fontweight='bold',
                   transform=ax_ct.transAxes)
    else:
        gs = GridSpec(1, 2, figure=fig,
                      width_ratios=[3, 5],
                      wspace=0.15)

        panel_a_sparse_dense(fig, gs[0])

        ax2 = fig.add_subplot(gs[1])
        panel_b_architecture(ax2)

    fig.savefig(PAPER_DIR / "overview.pdf", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(PAPER_DIR / "overview.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {PAPER_DIR / 'overview.pdf'}")


if __name__ == "__main__":
    main()
