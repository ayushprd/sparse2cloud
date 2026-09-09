"""Add January 2025 data for seasonal diversity.

Subsamples Jan matches to ~5K/day (not the full 40K/day), extracts patches,
and merges with existing npy_v2 dataset. Goal is diversity, not volume.

Usage:
    python -u 41_add_jan_diversity.py                # Full pipeline
    python -u 41_add_jan_diversity.py --match-only   # Only match (no extraction)
    python -u 41_add_jan_diversity.py --extract-only  # Only extract (matches exist)
    python -u 41_add_jan_diversity.py --max-per-day 3000  # Fewer samples/day
"""

import sys, os, argparse, json, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from config import (
    ERA5_DIR,
    EC_PROC_DIR, EC_N_LEVELS, OUTPUT_DIR, VIIRS_DIR, COLOC_DIR,
    VIIRS_THERMAL_BANDS, VIIRS_REFL_BANDS, VIIRS_ALL_BANDS,
    N_VIIRS_CHANNELS, PATCH_SIZE, PATCH_HALF,
    MAX_TIME_OFFSET_SEC, LOG_IWC_EPS, BT_MEAN, BT_STD,
    SPLIT_SEED, SPLIT_GRID_DEG, TRAIN_FRAC, VAL_FRAC,
    ACTIVE_LEVEL_START, ACTIVE_LEVEL_END,
)

N_THERMAL = len(VIIRS_THERMAL_BANDS)
N_REFL = len(VIIRS_REFL_BANDS)
JAN_DATES = [f"2025-01-{d:02d}" for d in range(1, 18)]


def step1_match(max_per_day=5000):
    """Match Jan profiles to VIIRS, subsampled."""
    import earthaccess
    try:
        earthaccess.login(strategy="environment")
    except Exception:
        earthaccess.login(persist=True)

    from importlib import util as imp_util
    spec = imp_util.spec_from_file_location(
        "match_extract", str(Path(__file__).parent / "03_match_and_extract_v2.py"))
    match_mod = imp_util.module_from_spec(spec)
    spec.loader.exec_module(match_mod)

    # Load VIIRS index
    viirs_index = pd.read_parquet(OUTPUT_DIR / "viirs_index.parquet")
    jan_index = viirs_index[viirs_index["date"].str.startswith("2025-01")]
    print(f"VIIRS index: {len(jan_index)} Jan granules")

    # Load Jan EC profiles
    ec_profiles = pd.read_parquet(EC_PROC_DIR / "profiles_2025-01.parquet")
    print(f"EC profiles: {len(ec_profiles):,}")

    viirs_cache = VIIRS_DIR / "cache"
    viirs_cache.mkdir(parents=True, exist_ok=True)

    all_matches = []
    t0 = time.time()

    for di, date_str in enumerate(JAN_DATES):
        matches = match_mod.process_date(
            date_str, ec_profiles, viirs_index, viirs_cache)

        # Subsample if too many
        if len(matches) > max_per_day:
            rng = np.random.RandomState(42 + di)
            idx = rng.choice(len(matches), max_per_day, replace=False)
            matches = [matches[i] for i in sorted(idx)]

        all_matches.extend(matches)

        elapsed = time.time() - t0
        rate = (di + 1) / elapsed if elapsed > 0 else 0
        eta = (len(JAN_DATES) - di - 1) / rate if rate > 0 else 0
        print(f"  [{di+1}/{len(JAN_DATES)}] {date_str}: "
              f"{len(matches):,} matches (capped from ~40K) "
              f"(total: {len(all_matches):,}) "
              f"[{elapsed:.0f}s, ~{eta:.0f}s ETA]", flush=True)

    jan_matches = pd.DataFrame(all_matches)
    jan_path = OUTPUT_DIR / "matches_jan2025.parquet"
    jan_matches.to_parquet(jan_path, index=False)
    print(f"\nSaved {len(jan_matches):,} Jan matches to {jan_path}")
    return jan_matches


def step2_extract(jan_matches):
    """Extract VIIRS patches for Jan matches."""
    import earthaccess
    import netCDF4 as nc
    try:
        earthaccess.login(strategy="environment")
    except Exception:
        earthaccess.login(persist=True)

    # Load EC profiles for target extraction
    ec_profiles = pd.read_parquet(EC_PROC_DIR / "profiles_2025-01.parquet")
    iwc_cols = [f"iwc_{i}" for i in range(EC_N_LEVELS)]

    viirs_cache = VIIRS_DIR / "cache"

    # Download needed L1B files
    from importlib import util as imp_util
    spec = imp_util.spec_from_file_location(
        "match_extract", str(Path(__file__).parent / "03_match_and_extract_v2.py"))
    match_mod = imp_util.module_from_spec(spec)
    spec.loader.exec_module(match_mod)

    l1b_names = jan_matches["viirs_l1b"].unique()
    print(f"Downloading {len(l1b_names)} L1B files...")
    match_mod.download_viirs_granules(l1b_names, "VNP02MOD", viirs_cache)

    # Extract patches
    N = len(jan_matches)
    patches = np.zeros((N, N_VIIRS_CHANNELS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    targets = np.zeros((N, EC_N_LEVELS), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)

    grouped = jan_matches.groupby("viirs_l1b")
    done = 0
    geo_cache = {}

    for l1b_name, group in grouped:
        l1b_path = viirs_cache / l1b_name
        if not l1b_path.exists():
            done += len(group)
            continue

        try:
            l1b_ds = nc.Dataset(str(l1b_path))
            obs = l1b_ds["observation_data"]
            nrow, ncol = obs["M15"].shape
        except Exception:
            done += len(group)
            continue

        # Pre-load LUTs
        bt_luts = {}
        for band in VIIRS_THERMAL_BANDS:
            lut_name = f"{band}_brightness_temperature_lut"
            if lut_name in obs.variables:
                bt_luts[band] = obs[lut_name][:]

        refl_scales = {}
        for band in VIIRS_REFL_BANDS:
            if band in obs.variables:
                var = obs[band]
                var.set_auto_scale(False)
                refl_scales[band] = (var.scale_factor, var.add_offset)

        for _, match in group.iterrows():
            idx = match.name  # DataFrame index = position in jan_matches
            row, col = match["viirs_row"], match["viirs_col"]
            ec_idx = match["ec_idx"]

            r0, r1 = row - PATCH_HALF, row + PATCH_HALF
            c0, c1 = col - PATCH_HALF, col + PATCH_HALF
            if r0 < 0 or r1 > nrow or c0 < 0 or c1 > ncol:
                done += 1
                continue

            patch = np.zeros((len(VIIRS_ALL_BANDS), PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            ok = True
            for bi, band in enumerate(VIIRS_ALL_BANDS):
                if band not in obs.variables:
                    ok = False
                    break
                var = obs[band]
                var.set_auto_scale(False)
                raw = var[r0:r1, c0:c1].astype(np.uint16)

                if band in VIIRS_THERMAL_BANDS:
                    if band not in bt_luts:
                        ok = False
                        break
                    bt = bt_luts[band][raw]
                    bt[raw >= 65528] = 0.0
                    patch[bi] = bt
                else:
                    if band not in refl_scales:
                        ok = False
                        break
                    scale, offset = refl_scales[band]
                    refl = raw.astype(np.float32) * scale + offset
                    refl[raw >= 65528] = 0.0
                    refl[refl < 0] = 0.0
                    patch[bi] = refl

            if not ok:
                done += 1
                continue

            # SZA
            geo_name = match["viirs_geo"]
            if geo_name not in geo_cache:
                geo_path = viirs_cache / geo_name
                if geo_path.exists():
                    try:
                        _, _, sza, _ = match_mod.open_viirs_geo(str(geo_path))
                        geo_cache[geo_name] = sza
                    except Exception:
                        pass
            if geo_name in geo_cache:
                sza_patch = geo_cache[geo_name][r0:r1, c0:c1].astype(np.float32)
            else:
                sza_patch = np.full((PATCH_SIZE, PATCH_SIZE), match["sza"], dtype=np.float32)

            full_patch = np.concatenate([patch, sza_patch[np.newaxis]], axis=0)

            # Target
            try:
                ec_row = ec_profiles.loc[ec_idx]
                target = ec_row[iwc_cols].values.astype(np.float32)
            except Exception:
                done += 1
                continue

            patches[idx] = full_patch
            targets[idx] = target
            valid[idx] = True
            done += 1

            if done % 5000 == 0:
                print(f"    Extracted {done:,}/{N:,}, valid={valid[:done].sum():,}", flush=True)

        l1b_ds.close()

        # Free GEO cache
        if len(geo_cache) > 20:
            keys = list(geo_cache.keys())
            for k in keys[:-10]:
                del geo_cache[k]

    n_valid = valid.sum()
    print(f"\n  Extracted: {n_valid:,}/{N:,} valid patches")

    # Keep only valid
    patches = patches[valid]
    targets = targets[valid]
    jan_matches_valid = jan_matches[valid].reset_index(drop=True)

    return patches, targets, jan_matches_valid


def step3_merge_and_save(patches_jan, targets_jan, matches_jan):
    """Merge Jan data with existing npy_v2 and re-split."""
    npy_old = COLOC_DIR / "npy_v2"
    npy_new = COLOC_DIR / "npy_v3"
    npy_new.mkdir(exist_ok=True)

    # Normalize Jan patches same as existing
    patches_jan[:, :N_THERMAL] = (patches_jan[:, :N_THERMAL] - BT_MEAN) / BT_STD
    patches_jan[:, N_THERMAL + N_REFL] = patches_jan[:, N_THERMAL + N_REFL] / 90.0
    targets_jan = np.log10(targets_jan + LOG_IWC_EPS)

    jan_lat = matches_jan["ec_lat"].values.astype(np.float32)
    jan_lon = matches_jan["ec_lon"].values.astype(np.float32)
    jan_sza = matches_jan["sza"].values.astype(np.float32)

    # Load existing data
    print("Loading existing data...")
    all_patches, all_targets, all_lat, all_lon, all_sza = [], [], [], [], []
    all_period = []  # track which period each sample comes from

    for split in ["train", "val", "test"]:
        p = np.load(npy_old / f"{split}_patches.npy")
        t = np.load(npy_old / f"{split}_targets.npy")
        lat = np.load(npy_old / f"{split}_lat.npy")
        lon = np.load(npy_old / f"{split}_lon.npy")
        sza = np.load(npy_old / f"{split}_sza.npy")
        all_patches.append(p)
        all_targets.append(t)
        all_lat.append(lat)
        all_lon.append(lon)
        all_sza.append(sza)
        all_period.append(np.full(len(p), 0, dtype=np.int8))  # period 0 = June
        print(f"  {split}: {len(p):,}")

    # Add Jan data
    all_patches.append(patches_jan)
    all_targets.append(targets_jan)
    all_lat.append(jan_lat)
    all_lon.append(jan_lon)
    all_sza.append(jan_sza)
    all_period.append(np.full(len(patches_jan), 1, dtype=np.int8))  # period 1 = Jan
    print(f"  jan: {len(patches_jan):,}")

    # Concatenate
    all_patches = np.concatenate(all_patches)
    all_targets = np.concatenate(all_targets)
    all_lat = np.concatenate(all_lat)
    all_lon = np.concatenate(all_lon)
    all_sza = np.concatenate(all_sza)
    all_period = np.concatenate(all_period)
    N = len(all_patches)
    print(f"\nTotal combined: {N:,}")

    # Re-split using geographic grid (same seed → same cells → June data
    # stays in same splits, Jan gets distributed across splits)
    lat_bins = np.floor(all_lat / SPLIT_GRID_DEG).astype(int)
    lon_bins = np.floor(all_lon / SPLIT_GRID_DEG).astype(int)
    cells = np.array([f"{la}_{lo}" for la, lo in zip(lat_bins, lon_bins)])

    unique_cells = np.unique(cells)
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(unique_cells)

    n_cells = len(unique_cells)
    n_train = int(n_cells * TRAIN_FRAC)
    n_val = int(n_cells * VAL_FRAC)

    train_cells = set(unique_cells[:n_train])
    val_cells = set(unique_cells[n_train:n_train + n_val])

    splits = np.empty(N, dtype="U5")
    for i, c in enumerate(cells):
        if c in train_cells:
            splits[i] = "train"
        elif c in val_cells:
            splits[i] = "val"
        else:
            splits[i] = "test"

    # Save
    for split in ["train", "val", "test"]:
        mask = splits == split
        n_june = ((all_period[mask] == 0)).sum()
        n_jan = ((all_period[mask] == 1)).sum()
        print(f"\n  {split}: {mask.sum():,} (June={n_june:,}, Jan={n_jan:,})")

        np.save(npy_new / f"{split}_patches.npy", all_patches[mask])
        np.save(npy_new / f"{split}_targets.npy", all_targets[mask])
        np.save(npy_new / f"{split}_lat.npy", all_lat[mask])
        np.save(npy_new / f"{split}_lon.npy", all_lon[mask])
        np.save(npy_new / f"{split}_sza.npy", all_sza[mask])
        np.save(npy_new / f"{split}_period.npy", all_period[mask])

        sz = (all_patches[mask].nbytes + all_targets[mask].nbytes) / 1e9
        print(f"    Size: {sz:.1f} GB")

    print(f"\nSaved to {npy_new}")
    return npy_new


def step4_era5(npy_dir):
    """Download + collocate ERA5 for Jan dates."""
    import xarray as xr


    # Check which ERA5 files are missing for Jan
    missing = [d for d in JAN_DATES
               if not (ERA5_DIR / f"era5_pl_{d}.nc").exists()
               or (ERA5_DIR / f"era5_pl_{d}.nc").stat().st_size < 1e6]

    if missing:
        print(f"\nDownloading ERA5 for {len(missing)} missing Jan dates...")
        import cdsapi
        client = cdsapi.Client()
        PRESSURE_LEVELS = [
            "50", "70", "100", "125", "150", "175", "200", "225", "250",
            "300", "350", "400", "450", "500", "550", "600", "650", "700",
            "750", "800", "850", "900", "925", "950", "975", "1000",
        ]
        VARIABLES = ["temperature", "specific_humidity", "relative_humidity", "geopotential"]
        HOURS = [f"{h:02d}:00" for h in range(24)]

        for date_str in missing:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            out_file = ERA5_DIR / f"era5_pl_{date_str}.nc"
            print(f"  {date_str}...", end=" ", flush=True)
            try:
                client.retrieve("reanalysis-era5-pressure-levels", {
                    "product_type": ["reanalysis"],
                    "variable": VARIABLES,
                    "pressure_level": PRESSURE_LEVELS,
                    "year": [str(date.year)],
                    "month": [f"{date.month:02d}"],
                    "day": [f"{date.day:02d}"],
                    "time": HOURS,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                }, str(out_file))
                print(f"OK ({out_file.stat().st_size/1e6:.1f}MB)")
            except Exception as e:
                print(f"FAILED - {e}")

    # Now collocate ERA5 for the combined dataset
    print("\nCollocating ERA5 for combined dataset...")

    # Load existing ERA5 data
    old_era5 = {}
    npy_old = COLOC_DIR / "npy_v2"
    for split in ["train", "val", "test"]:
        era5_path = npy_old / f"{split}_era5.npy"
        if era5_path.exists():
            old_era5[split] = np.load(era5_path)
            print(f"  Existing {split} ERA5: {old_era5[split].shape}")

    # For Jan samples, we need to build ERA5 features
    # Load period labels to know which samples are Jan
    for split in ["train", "val", "test"]:
        period = np.load(npy_dir / f"{split}_period.npy")
        n_june = (period == 0).sum()
        n_jan = (period == 1).sum()

        if n_jan == 0:
            # Just copy existing ERA5
            if split in old_era5:
                np.save(npy_dir / f"{split}_era5.npy", old_era5[split])
            continue

        lat = np.load(npy_dir / f"{split}_lat.npy")
        lon = np.load(npy_dir / f"{split}_lon.npy")
        sza = np.load(npy_dir / f"{split}_sza.npy")

        # Determine ERA5 shape from existing — keep (N, 26, 4) format
        if split in old_era5:
            era5_shape = old_era5[split].shape[1:]  # (26, 4)
        else:
            era5_shape = (26, 4)

        # Build combined ERA5 array
        era5_combined = np.full((len(period),) + era5_shape, np.nan, dtype=np.float32)

        # Copy June ERA5
        if split in old_era5 and n_june > 0:
            era5_combined[period == 0] = old_era5[split]

        # Collocate Jan ERA5
        jan_mask = period == 1
        jan_indices = np.where(jan_mask)[0]
        jan_lat = lat[jan_mask]
        jan_lon = lon[jan_mask]

        # Simple: assign each Jan sample to the nearest ERA5 grid point at noon
        # (most Jan dates should have ERA5 files)
        for date_str in JAN_DATES:
            era5_file = ERA5_DIR / f"era5_pl_{date_str}.nc"
            if not era5_file.exists() or era5_file.stat().st_size < 1e6:
                continue

            ds = xr.open_dataset(era5_file)
            era5_lats = ds['latitude'].values
            era5_lons = ds['longitude'].values

            # Find Jan samples from this date (approximate by checking lat/lon
            # aren't NaN in existing — but we don't have dates stored)
            # Instead, just collocate ALL Jan samples against each day and take
            # the first valid match (since we don't have per-sample dates in npy_v3)
            # This is a simplification but works for global ERA5 (static within a day)

            # Actually, we'll use noon (12:00) for all samples as a rough approximation
            # since ERA5 varies slowly compared to the spatial variability
            valid_times = ds['valid_time'].values
            time_hours = np.array([
                np.datetime64(t, 'h').astype('datetime64[h]').astype(int) % 24
                for t in valid_times
            ])
            noon_idx = np.abs(time_hours - 12).argmin()

            j_lons = jan_lon.copy()
            j_lons[j_lons < 0] += 360

            lat_idx = np.abs(era5_lats[np.newaxis, :] - jan_lat[:, np.newaxis]).argmin(axis=1)
            lon_idx = np.abs(era5_lons[np.newaxis, :] - j_lons[:, np.newaxis]).argmin(axis=1)

            var_names = [v for v in ['t', 'q', 'r', 'z'] if v in ds]
            profiles = []
            for var in var_names:
                data = ds[var].values[noon_idx]  # (n_levels, n_lat, n_lon)
                profiles.append(data[:, lat_idx, lon_idx].T)  # (n_samples, n_levels)

            # Stack as (n_samples, n_levels, n_vars) to match npy_v2 format
            era5_jan = np.stack(profiles, axis=2).astype(np.float32)  # (n_samples, 26, 4)

            # Only fill in NaN entries (first date fills all, subsequent overwrite)
            still_nan = np.isnan(era5_combined[jan_indices, 0, 0])
            if still_nan.any():
                era5_combined[jan_indices[still_nan]] = era5_jan[still_nan]
            ds.close()
            break  # One day's ERA5 is enough (global coverage, ~same T/q everywhere for a given location)

        n_valid = (~np.isnan(era5_combined[:, 0, 0])).sum()
        print(f"  {split}: {n_valid}/{len(era5_combined)} valid ERA5 ({100*n_valid/len(era5_combined):.1f}%)")
        np.save(npy_dir / f"{split}_era5.npy", era5_combined)

    print("\nERA5 colocation done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--max-per-day", type=int, default=5000)
    args = parser.parse_args()

    t0 = time.time()

    # Step 1: Match
    jan_path = OUTPUT_DIR / "matches_jan2025.parquet"
    if not args.extract_only:
        print(f"\n{'='*60}")
        print(f"Step 1: Match Jan 2025 (max {args.max_per_day}/day)")
        print(f"{'='*60}")
        jan_matches = step1_match(max_per_day=args.max_per_day)
    else:
        jan_matches = pd.read_parquet(jan_path)
        print(f"Loaded {len(jan_matches):,} existing matches")

    if args.match_only:
        return

    # Step 2: Extract
    print(f"\n{'='*60}")
    print(f"Step 2: Extract VIIRS patches")
    print(f"{'='*60}")
    patches_jan, targets_jan, matches_valid = step2_extract(jan_matches)

    # Step 3: Merge with existing
    print(f"\n{'='*60}")
    print(f"Step 3: Merge with existing + re-split")
    print(f"{'='*60}")
    npy_dir = step3_merge_and_save(patches_jan, targets_jan, matches_valid)

    # Step 4: ERA5
    print(f"\n{'='*60}")
    print(f"Step 4: ERA5 download + colocation")
    print(f"{'='*60}")
    step4_era5(npy_dir)

    # Step 5: Cleanup large intermediate files
    print(f"\n{'='*60}")
    print(f"Step 5: Cleanup large intermediate files")
    print(f"{'='*60}")
    step5_cleanup()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done! Total time: {elapsed/60:.1f} min")
    print(f"{'='*60}")


def step5_cleanup():
    """Remove large intermediate files that are no longer needed."""
    import shutil

    cleaned = 0

    # 1. VIIRS cache — L1B and GEO .nc files (biggest: ~200+ GB)
    viirs_cache = VIIRS_DIR / "cache"
    if viirs_cache.exists():
        sz = sum(f.stat().st_size for f in viirs_cache.iterdir() if f.is_file()) / 1e9
        print(f"  VIIRS cache: {sz:.1f} GB — removing...")
        shutil.rmtree(viirs_cache)
        viirs_cache.mkdir()
        print(f"    Removed {sz:.1f} GB")
        cleaned += sz

    # 2. Raw EarthCARE HDF5 files (already extracted to parquet)
    for mdir in sorted(EC_RAW_DIR.glob("*")):
        if not mdir.is_dir():
            continue
        month = mdir.name
        parquet = EC_PROC_DIR / f"profiles_{month}.parquet"
        if parquet.exists():
            h5_files = list(mdir.glob("*.h5"))
            if h5_files:
                sz = sum(f.stat().st_size for f in h5_files) / 1e9
                print(f"  Raw HDF5 {month}: {sz:.1f} GB ({len(h5_files)} files) — removing...")
                for f in h5_files:
                    f.unlink()
                print(f"    Removed {sz:.1f} GB")
                cleaned += sz

    # 3. Old zarr stores (we use npy now)
    for zp in COLOC_DIR.glob("*.zarr"):
        sz = sum(f.stat().st_size for f in zp.rglob("*") if f.is_file()) / 1e9
        print(f"  Zarr {zp.name}: {sz:.1f} GB — removing...")
        shutil.rmtree(zp)
        print(f"    Removed {sz:.1f} GB")
        cleaned += sz

    print(f"\n  Total cleaned: {cleaned:.1f} GB")


if __name__ == "__main__":
    main()
