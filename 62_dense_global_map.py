"""
Generate a DENSE global map using G-QE (5-member ConvNextUNet ensemble).

Downloads VIIRS swaths, tiles into 64x64 patches, runs per-pixel inference,
and accumulates on a 0.25° global grid. Produces cloud top height +
uncertainty global maps.

Adapted from 18_dense_global_map.py for the ConvNextUNet quantile ensemble.

Usage:
    python -u 62_dense_global_map.py [--date 2025-06-01] [--days 3]
"""
import sys, os, time, gc, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from torch.amp import autocast
import netCDF4 as nc
import earthaccess
from scipy.ndimage import gaussian_filter, distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

from config import MODEL_DIR, FIGURE_DIR, OUTPUT_DIR, BT_MEAN, BT_STD, COLOC_DIR, VIIRS_DIR
from models.convnext_unet import ConvNextUNet

CACHE_DIR = VIIRS_DIR / "global_cache"
PATCH_SIZE = 64
STRIDE = 32
BATCH_SIZE = 64  # smaller than old script since 5 models × dense output

THERMAL_BANDS = ["M12", "M13", "M14", "M15", "M16"]
REFL_BANDS = ["M07", "M08", "M10", "M11"]

N_TARGETS = 8
TARGET_NAMES = [
    "centroid", "cloud_top", "cloud_base", "peak_level",
    "thickness", "core_iwc", "mean_iwc", "log_iwp",
]


def load_ensemble(device):
    """Load 5 ConvNextUNet quantile models."""
    tags = ["G-Q", "G-Q2", "G-Q3", "G-Q4", "G-Q5"]
    models = []
    geo_mean = geo_std = era5_dim = None

    for tag in tags:
        ckpt = torch.load(MODEL_DIR / f"geom_{tag}.pt", map_location="cpu",
                          weights_only=False)
        if geo_mean is None:
            geo_mean = np.array(ckpt["geo_mean"], dtype=np.float32)
            geo_std = np.array(ckpt["geo_std"], dtype=np.float32)
            era5_dim = ckpt["era5_dim"]

        model = ConvNextUNet(
            in_channels=10, out_channels=N_TARGETS,
            base_dim=ckpt["base_dim"], dim_mults=(1, 2, 4),
            era5_dim=era5_dim, quantile_mode=True, physics_head=False,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

    return models, geo_mean, geo_std, era5_dim


def extract_patches_with_coords(l1b_path, geo_path):
    """Extract tiled 64x64 patches with per-pixel lat/lon.

    Returns:
        patches: (N, 10, 64, 64)
        patch_lats: (N, 64, 64) per-pixel latitudes
        patch_lons: (N, 64, 64) per-pixel longitudes
    """
    try:
        ds_l1b = nc.Dataset(l1b_path)
        ds_geo = nc.Dataset(geo_path)
    except Exception as e:
        print(f"    Error opening: {e}")
        return None, None, None

    obs = ds_l1b['observation_data']
    geo = ds_geo['geolocation_data']

    lat = geo['latitude'][:]
    lon = geo['longitude'][:]
    nrow, ncol = lat.shape

    if nrow < PATCH_SIZE or ncol < PATCH_SIZE:
        ds_l1b.close(); ds_geo.close()
        return None, None, None

    # Read all bands
    full = np.zeros((10, nrow, ncol), dtype=np.float32)

    for bi, band in enumerate(THERMAL_BANDS):
        try:
            var = obs[band]
            var.set_auto_scale(False)
            raw = var[:].astype(np.uint16)
            bt_lut = obs[f"{band}_brightness_temperature_lut"][:]
            bt = bt_lut[raw]
            bt[raw >= 65528] = 0.0
            full[bi] = (bt - BT_MEAN) / BT_STD
        except Exception:
            pass

    for bi_offset, band in enumerate(REFL_BANDS):
        bi = 5 + bi_offset
        try:
            var = obs[band]
            var.set_auto_scale(False)
            raw = var[:].astype(np.uint16)
            refl = raw.astype(np.float32) * var.scale_factor + var.add_offset
            refl[raw >= 65528] = 0.0
            refl[refl < 0] = 0.0
            full[bi] = refl
        except Exception:
            pass

    try:
        sza = geo['solar_zenith'][:].astype(np.float32)
        sza[sza > 180] = 90.0
        full[9] = sza / 90.0
    except Exception:
        full[9] = 1.0

    ds_l1b.close()
    ds_geo.close()

    # Tile with stride
    row_starts = list(range(0, nrow - PATCH_SIZE + 1, STRIDE))
    col_starts = list(range(0, ncol - PATCH_SIZE + 1, STRIDE))
    n_patches = len(row_starts) * len(col_starts)

    patches = np.zeros((n_patches, 10, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    patch_lats = np.zeros((n_patches, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    patch_lons = np.zeros((n_patches, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

    idx = 0
    for r0 in row_starts:
        for c0 in col_starts:
            patches[idx] = full[:, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
            patch_lats[idx] = lat[r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
            patch_lons[idx] = lon[r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
            idx += 1

    # Filter out mostly-fill patches
    thermal_energy = np.abs(patches[:, :5]).mean(axis=(1, 2, 3))
    valid = thermal_energy > 0.1
    return patches[valid], patch_lats[valid], patch_lons[valid]


def run_ensemble_inference(models, patches, era5_zero, era5v_zero,
                           geo_mean, geo_std, device):
    """Run G-QE ensemble on patches.

    Returns:
        median: (N, 8, 64, 64) denormalized median predictions
        width: (N, 8, 64, 64) denormalized interval width (q90 - q10)
    """
    n = len(patches)
    median_all = np.zeros((n, N_TARGETS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    width_all = np.zeros((n, N_TARGETS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

    for i in range(0, n, BATCH_SIZE):
        bs = min(BATCH_SIZE, n - i)
        batch = torch.from_numpy(patches[i:i+bs]).to(device)
        era5 = era5_zero[:bs]
        era5v = era5v_zero[:bs]

        ens_preds = []
        with torch.no_grad():
            for model in models:
                with autocast("cuda", dtype=torch.bfloat16):
                    pred = model(batch, era5, era5v)
                ens_preds.append(pred.float().cpu().numpy())  # (bs, 8, 3, 64, 64)

        pred = np.mean(ens_preds, axis=0)  # (bs, 8, 3, 64, 64)

        for t in range(N_TARGETS):
            median_all[i:i+bs, t] = pred[:, t, 1] * geo_std[t] + geo_mean[t]
            width_all[i:i+bs, t] = (pred[:, t, 2] - pred[:, t, 0]) * geo_std[t]

        del ens_preds, pred, batch
        torch.cuda.empty_cache()

    return median_all, width_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2025-06-01")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--max-granules", type=int, default=241)
    parser.add_argument("--grid-res", type=float, default=0.25)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip inference, just replot from saved grid")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    grid_path = OUTPUT_DIR / "global_grid_gqe.npz"

    if not args.plot_only:
        print("Loading G-QE ensemble (5 ConvNextUNet models)...")
        models, geo_mean, geo_std, era5_dim = load_ensemble(device)

        # Pre-allocate ERA5 zero tensors (no ERA5 for global inference)
        era5_zero = torch.zeros(BATCH_SIZE, era5_dim, device=device)
        era5v_zero = torch.zeros(BATCH_SIZE, device=device)

        # Output grid
        res = args.grid_res
        lat_bins = np.arange(-90, 90 + res, res)
        lon_bins = np.arange(-180, 180 + res, res)
        nlat, nlon = len(lat_bins) - 1, len(lon_bins) - 1

        grid_sum = np.zeros((N_TARGETS, nlat, nlon), dtype=np.float64)
        grid_width_sum = np.zeros((N_TARGETS, nlat, nlon), dtype=np.float64)
        grid_count = np.zeros((nlat, nlon), dtype=np.float64)

        from datetime import datetime, timedelta
        start_date = datetime.strptime(args.date, "%Y-%m-%d")
        dates = [(start_date + timedelta(days=d)).strftime("%Y-%m-%d")
                 for d in range(args.days)]

        auth = earthaccess.login(strategy='netrc')
        total_patches = 0
        total_pixels = 0
        total_granules = 0
        t_start = time.time()

        for day_idx, date_str in enumerate(dates):
            print(f"\n{'='*60}")
            print(f"  Day {day_idx+1}/{len(dates)}: {date_str}")
            print(f"{'='*60}")

            results_l1b = earthaccess.search_data(
                short_name='VNP02MOD', temporal=(date_str, date_str),
                count=args.max_granules)
            results_geo = earthaccess.search_data(
                short_name='VNP03MOD', temporal=(date_str, date_str),
                count=args.max_granules)
            print(f"  Found {len(results_l1b)} L1B, {len(results_geo)} GEO")

            def get_ts(r):
                links = r.data_links()
                if links:
                    parts = links[0].split('/')[-1].split('.')
                    if len(parts) >= 3:
                        return f"{parts[1]}.{parts[2]}"
                return None

            def get_name(r):
                links = r.data_links()
                return links[0].split('/')[-1] if links else None

            geo_by_time = {get_ts(r): r for r in results_geo if get_ts(r)}
            paired = [(r, geo_by_time[get_ts(r)])
                      for r in results_l1b if get_ts(r) in geo_by_time]
            n_gran = min(len(paired), args.max_granules)
            print(f"  Paired: {len(paired)}, processing {n_gran}")

            def download_granule(item):
                """Download a single granule pair. Returns (l1b_path, geo_path, downloaded, ts)."""
                gi, (l1b_r, geo_r) = item
                ts = get_ts(l1b_r)
                l1b_name, geo_name = get_name(l1b_r), get_name(geo_r)
                if not l1b_name or not geo_name:
                    return (None, None, False, ts)
                # Check local
                for d in [VIIRS_DIR, CACHE_DIR]:
                    if (d / l1b_name).exists() and (d / geo_name).exists():
                        return (str(d / l1b_name), str(d / geo_name), False, ts)
                # Download
                try:
                    earthaccess.download([l1b_r], str(CACHE_DIR))
                    earthaccess.download([geo_r], str(CACHE_DIR))
                    l1b_p = CACHE_DIR / l1b_name
                    geo_p = CACHE_DIR / geo_name
                    if l1b_p.exists() and geo_p.exists():
                        return (str(l1b_p), str(geo_p), True, ts)
                except Exception as e:
                    print(f"  [{gi+1}/{n_gran}] {ts}: download error: {e}")
                return (None, None, False, ts)

            # Parallel download with prefetch queue
            # Download ahead of GPU processing using thread pool
            from queue import Queue
            from threading import Thread

            download_q = Queue(maxsize=8)  # buffer up to 8 ready granules

            def download_worker():
                """Download granules and put results in queue."""
                with ThreadPoolExecutor(max_workers=4) as pool:
                    items = list(enumerate(paired[:n_gran]))
                    futs = {pool.submit(download_granule, item): item for item in items}
                    for fut in as_completed(futs):
                        download_q.put(fut.result())
                download_q.put(None)  # sentinel

            dl_thread = Thread(target=download_worker, daemon=True)
            dl_thread.start()

            processed = 0
            while True:
                item = download_q.get()
                if item is None:
                    break

                l1b_path, geo_path, downloaded, ts = item
                if l1b_path is None:
                    continue

                t0 = time.time()
                patches, plats, plons = extract_patches_with_coords(l1b_path, geo_path)

                if patches is None or len(patches) == 0:
                    if downloaded:
                        Path(l1b_path).unlink(missing_ok=True)
                        Path(geo_path).unlink(missing_ok=True)
                    continue

                median, width = run_ensemble_inference(
                    models, patches, era5_zero, era5v_zero,
                    geo_mean, geo_std, device)

                # Grid per-pixel predictions (vectorized)
                step = 4
                sub_lats = plats[:, ::step, ::step].ravel()
                sub_lons = plons[:, ::step, ::step].ravel()
                li = ((sub_lats + 90) / res).astype(np.int32)
                lj = ((sub_lons + 180) / res).astype(np.int32)
                valid_px = (li >= 0) & (li < nlat) & (lj >= 0) & (lj < nlon)
                li, lj = li[valid_px], lj[valid_px]
                np.add.at(grid_count, (li, lj), 1)
                for t in range(N_TARGETS):
                    sub_med = median[:, t, ::step, ::step].ravel()[valid_px]
                    sub_wid = width[:, t, ::step, ::step].ravel()[valid_px]
                    np.add.at(grid_sum[t], (li, lj), sub_med)
                    np.add.at(grid_width_sum[t], (li, lj), sub_wid)

                n_px = int(valid_px.sum())
                total_patches += len(patches)
                total_pixels += n_px
                elapsed = time.time() - t0
                processed += 1

                if downloaded:
                    Path(l1b_path).unlink(missing_ok=True)
                    Path(geo_path).unlink(missing_ok=True)

                del patches, plats, plons, median, width
                gc.collect()
                torch.cuda.empty_cache()

                if processed % 10 == 0 or processed == 1:
                    covered = (grid_count > 0).sum() / (nlat * nlon) * 100
                    print(f"  [{processed}/{n_gran}] {ts}: {total_pixels:,} px, "
                          f"{covered:.1f}% coverage, {elapsed:.1f}s")

            dl_thread.join(timeout=5)

            total_granules += n_gran
            covered = (grid_count > 0).sum() / (nlat * nlon) * 100
            print(f"  Day {day_idx+1} done: {covered:.1f}% cumulative")

        elapsed_total = time.time() - t_start
        print(f"\nDone: {total_pixels:,} pixels from {total_granules} granules "
              f"in {elapsed_total:.0f}s")

        # Compute grid means
        mask = grid_count > 0
        grid_mean = np.where(mask, grid_sum / np.maximum(grid_count, 1), np.nan)
        grid_width_mean = np.where(mask, grid_width_sum / np.maximum(grid_count, 1), np.nan)

        # Smooth and gap-fill
        grid_mean[:, grid_count < 2] = np.nan
        grid_width_mean[:, grid_count < 2] = np.nan

        sigma = 1.5
        for j in range(N_TARGETS):
            for arr in [grid_mean, grid_width_mean]:
                data = arr[j].copy()
                valid = ~np.isnan(data)
                data[~valid] = 0.0
                sn = gaussian_filter(data, sigma=sigma)
                sd = gaussian_filter(valid.astype(float), sigma=sigma)
                arr[j] = np.where(sd > 0.01, sn / sd, np.nan)

        for j in range(N_TARGETS):
            for arr in [grid_mean, grid_width_mean]:
                data = arr[j]
                nans = np.isnan(data)
                if nans.any():
                    _, idx = distance_transform_edt(nans, return_distances=True,
                                                    return_indices=True)
                    data[nans] = data[tuple(idx[:, nans])]
                arr[j] = gaussian_filter(data, sigma=1.0)

        lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
        lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2

        np.savez(grid_path,
                 grid_mean=grid_mean.astype(np.float32),
                 grid_width_mean=grid_width_mean.astype(np.float32),
                 grid_count=grid_count.astype(np.float32),
                 lat_centers=lat_centers, lon_centers=lon_centers,
                 target_names=TARGET_NAMES, dates=dates)
        print(f"Saved to {grid_path}")

    # PLOTTING
    print("\nGenerating global maps...")

    if grid_path.exists():
        d = np.load(grid_path, allow_pickle=True)
    else:
        print("ERROR: No grid data. Run without --plot-only first.")
        return

    grid_mean = d['grid_mean']
    grid_width_mean = d['grid_width_mean']
    grid_count = d['grid_count']
    lat_centers = d['lat_centers']
    lon_centers = d['lon_centers']
    dates = d['dates']
    nlat, nlon = len(lat_centers), len(lon_centers)

    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    def level_to_km(level):
        return 16.2 - (level - 69) * (16.2 - 0.6) / 158

    ti = {name: i for i, name in enumerate(TARGET_NAMES)}

    centroid_km = level_to_km(grid_mean[ti["centroid"]])
    cloud_top_km = level_to_km(grid_mean[ti["cloud_top"]])
    cloud_base_km = level_to_km(grid_mean[ti["cloud_base"]])
    core_iwc = grid_mean[ti["core_iwc"]]
    top_width_km = grid_width_mean[ti["cloud_top"]] * (16.2 - 0.6) / 158

    coverage = (grid_count > 0).sum() / (nlat * nlon) * 100
    total_px = int(grid_count.sum())
    date_label = f"{dates[0]}" if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"

    PAPER_DIR = Path("figures/eccv")
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    # 4-panel figure (like reference)
    fig = plt.figure(figsize=(20, 24))

    def pct_range(arr, lo=2, hi=98, pad=0.1):
        v = arr[~np.isnan(arr)]
        vmin, vmax = np.percentile(v, lo), np.percentile(v, hi)
        margin = (vmax - vmin) * pad
        return vmin - margin, vmax + margin

    quantities = [
        ("Cloud Centroid Height (km)", centroid_km, 'RdYlBu_r', *pct_range(centroid_km),
         f"R²=0.847"),
        ("Cloud Top Height (km)", cloud_top_km, 'RdYlBu_r', *pct_range(cloud_top_km),
         f"R²=0.826"),
        ("Cloud Base Height (km)", cloud_base_km, 'RdYlBu_r', *pct_range(cloud_base_km),
         f"R²=0.859"),
        ("Core IWC (log₁₀ mg/m³)", core_iwc, 'Blues', *pct_range(core_iwc),
         f"R²=0.812"),
    ]

    for panel_idx, (title, values, cmap, vmin, vmax, r2_str) in enumerate(quantities):
        ax = fig.add_subplot(4, 1, panel_idx + 1, projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color='#333333', zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, color='#999999', zorder=2)
        data = np.ma.masked_invalid(values)
        im = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading='auto',
                           zorder=1, rasterized=True)
        cb = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02, aspect=30)
        cb.ax.tick_params(labelsize=9)
        ax.set_title(f"{title}  ({r2_str})", fontsize=13, fontweight='bold', pad=10)

    fig.suptitle(
        f"G-QE Dense Ice Cloud Geometry — VIIRS Prediction\n"
        f"{date_label}, {total_px:,} pixel predictions, "
        f"{coverage:.0f}% coverage",
        fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = PAPER_DIR / "global_dense_4panel.png"
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(PAPER_DIR / "global_dense_4panel.pdf", dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")

    # Hero: Cloud Top Height + Uncertainty side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5),
                                    subplot_kw={'projection': ccrs.Robinson()})
    for ax in (ax1, ax2):
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color='#333333', zorder=2)

    top_vmin, top_vmax = pct_range(cloud_top_km)
    im1 = ax1.pcolormesh(lon_grid, lat_grid,
                         np.ma.masked_invalid(cloud_top_km),
                         cmap='RdYlBu_r', vmin=top_vmin, vmax=top_vmax,
                         transform=ccrs.PlateCarree(), shading='auto',
                         rasterized=True)
    cb1 = plt.colorbar(im1, ax=ax1, shrink=0.6, pad=0.02, aspect=30)
    cb1.set_label('Cloud Top Height (km)')
    ax1.set_title("(a) G-QE Predicted Cloud Top Height", fontsize=11, fontweight='bold')

    w_vmin, w_vmax = pct_range(top_width_km)
    im2 = ax2.pcolormesh(lon_grid, lat_grid,
                         np.ma.masked_invalid(top_width_km),
                         cmap='magma', vmin=w_vmin, vmax=w_vmax,
                         transform=ccrs.PlateCarree(), shading='auto',
                         rasterized=True)
    cb2 = plt.colorbar(im2, ax=ax2, shrink=0.6, pad=0.02, aspect=30)
    cb2.set_label('Interval Width (km)')
    ax2.set_title("(b) Prediction Uncertainty (90% CI width)", fontsize=11, fontweight='bold')

    fig.suptitle(f"Dense Global Prediction — {date_label}, {coverage:.0f}% coverage",
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out2 = PAPER_DIR / "global_dense_hero.png"
    fig.savefig(out2, dpi=250, bbox_inches='tight', facecolor='white')
    fig.savefig(PAPER_DIR / "global_dense_hero.pdf", dpi=250,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
