"""Expand the EarthCARE IWC dataset with new seasonal periods.

Runs the full pipeline for new periods defined in config.PERIODS,
merges with existing data, re-splits, and produces training-ready npy files.

Pipeline steps:
  1. Download EarthCARE ATL_ICE_2A for new periods
  2. Build VIIRS index for new periods
  3. Match + extract VIIRS patches → period-specific zarr
  4. Merge all zarr stores into one combined zarr
  5. Geographic split (train/val/test)
  6. Convert to npy
  7. Download ERA5 for new dates
  8. Collocate ERA5

Usage:
    python -u 40_expand_dataset.py                         # Full pipeline
    python -u 40_expand_dataset.py --step download         # Only step 1
    python -u 40_expand_dataset.py --step viirs-index      # Only step 2
    python -u 40_expand_dataset.py --step match            # Only step 3
    python -u 40_expand_dataset.py --step merge            # Only step 4-6
    python -u 40_expand_dataset.py --step era5             # Only step 7-8
    python -u 40_expand_dataset.py --periods 1,2,3,4       # Specific periods
    python -u 40_expand_dataset.py --skip-existing         # Skip period 0
"""

import sys, os, argparse, json, time, glob
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import zarr

from config import (
    ERA5_DIR,
    PERIODS, EC_RAW_DIR, EC_PROC_DIR, VIIRS_DIR, COLOC_DIR,
    OUTPUT_DIR, EC_PRODUCT, EC_N_LEVELS, EC_FILL_VALUE,
    ESA_USERNAME, ESA_PASSWORD, IWC_NOISE_FLOOR, MIN_ICE_LEVELS,
    VIIRS_L1B_PRODUCT, VIIRS_GEO_PRODUCT,
    VIIRS_THERMAL_BANDS, VIIRS_REFL_BANDS, VIIRS_ALL_BANDS,
    N_VIIRS_CHANNELS, PATCH_SIZE, PATCH_HALF,
    MAX_TIME_OFFSET_SEC, LOG_IWC_EPS, BT_MEAN, BT_STD,
    SPLIT_SEED, SPLIT_GRID_DEG, TRAIN_FRAC, VAL_FRAC,
    ACTIVE_LEVEL_START, ACTIVE_LEVEL_END,
)

N_THERMAL = len(VIIRS_THERMAL_BANDS)
N_REFL = len(VIIRS_REFL_BANDS)


# ══════════════════════════════════════════════════════════════
#  Step 1: Download EarthCARE
# ══════════════════════════════════════════════════════════════

def step_download_earthcare(period_indices):
    """Download EarthCARE ATL_ICE_2A and extract to parquet."""
    from config import EC_EPOCH_STR
    J2000 = datetime(2000, 1, 1, tzinfo=timezone.utc)

    os.environ["ESA_EO_USERNAME"] = ESA_USERNAME
    os.environ["ESA_EO_PASSWORD"] = ESA_PASSWORD
    from earthcare_downloader import search, download

    for pi in period_indices:
        start_str, end_str = PERIODS[pi]
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        n_days = (end - start).days + 1

        print(f"\n{'='*60}")
        print(f"Period {pi}: {start_str} to {end_str}")
        print(f"{'='*60}")

        # Download
        total_files = 0
        for di in range(n_days):
            date = start + timedelta(days=di)
            date_str = date.strftime("%Y-%m-%d")
            month_dir = EC_RAW_DIR / date.strftime("%Y-%m")
            month_dir.mkdir(parents=True, exist_ok=True)

            date_prefix = date.strftime("%Y%m%d")
            existing = list(month_dir.glob(f"*{date_prefix}*.h5"))
            if len(existing) >= 100:
                total_files += len(existing)
                continue

            try:
                files = search(product=EC_PRODUCT, date=date_str)
            except Exception as e:
                print(f"  [{di+1}/{n_days}] {date_str}: search error: {e}")
                continue

            if not files:
                print(f"  [{di+1}/{n_days}] {date_str}: no files")
                continue

            existing_names = {f.stem for f in existing}
            to_download = [f for f in files
                           if f.filename.replace(".ZIP", "") not in existing_names]

            if not to_download:
                total_files += len(files)
                continue

            n_new = 0
            for bi in range(0, len(to_download), 5):
                batch = to_download[bi:bi + 5]
                for attempt in range(5):
                    try:
                        paths = download(batch, output_path=str(month_dir))
                        n_new += sum(1 for p in paths if str(p).endswith(".h5"))
                        break
                    except Exception as e:
                        wait = 10 * (2 ** attempt)  # 10, 20, 40, 80, 160s
                        if attempt < 4:
                            print(f"    batch {bi//5}: attempt {attempt+1} failed (403?), "
                                  f"waiting {wait}s...")
                            time.sleep(wait)
                        else:
                            print(f"    batch failed after 5 attempts: {e}")
                # Small sleep between batches to avoid rate limiting
                time.sleep(2)

            total_files += n_new + len(existing)
            print(f"  [{di+1}/{n_days}] {date_str}: {n_new} new (total: {total_files})")

            # Cool down between days to avoid rate limiting
            if n_new > 0:
                time.sleep(30)

        # Extract to parquet
        print(f"\nExtracting parquets for period {pi}...")
        _extract_all_months()

    return True


def _extract_all_months():
    """Extract all unprocessed month directories to parquet."""
    import h5py
    J2000 = datetime(2000, 1, 1, tzinfo=timezone.utc)
    FILL = EC_FILL_VALUE

    for mdir in sorted(EC_RAW_DIR.glob("*")):
        if not mdir.is_dir():
            continue
        month_name = mdir.name
        output_path = EC_PROC_DIR / f"profiles_{month_name}.parquet"

        if output_path.exists():
            existing = pd.read_parquet(output_path, columns=["lat"])
            print(f"  {month_name}: exists ({len(existing):,} records)")
            continue

        h5_files = sorted(glob.glob(str(mdir / "*.h5")))
        if not h5_files:
            continue

        all_records = []
        t0 = time.time()

        for fi, fpath in enumerate(h5_files):
            try:
                f = h5py.File(fpath, "r")
            except Exception:
                continue

            sd = f["ScienceData"]
            lat = sd["latitude"][:]
            lon = sd["longitude"][:]
            t_sec = sd["time"][:]
            iwc = sd["ice_water_content"][:]
            height = sd["height"][:]
            elev = sd["elevation"][:]

            orbit_id = Path(fpath).stem

            for i in range(len(lat)):
                if lat[i] >= FILL or lon[i] >= FILL or t_sec[i] >= FILL:
                    continue

                iwc_i = iwc[i]
                h_i = height[i]
                valid = (iwc_i < FILL) & (iwc_i > IWC_NOISE_FLOOR) & (h_i < FILL)
                if valid.sum() < MIN_ICE_LEVELS:
                    continue

                h_valid = h_i[valid]
                iwc_valid = iwc_i[valid]
                sort_idx = np.argsort(h_valid)

                iwp = float(np.trapezoid(iwc_valid[sort_idx] * 1e-6, h_valid[sort_idx]))
                cloud_top_h = float(h_valid.max())
                cloud_base_h = float(h_valid.min())
                max_iwc = float(iwc_valid.max())
                max_iwc_alt = float(h_valid[np.argmax(iwc_valid)])

                h_sorted = np.sort(h_valid)
                gaps = np.diff(h_sorted) > 500
                n_layers = int(gaps.sum()) + 1

                iwc_profile = iwc_i.copy()
                iwc_profile[iwc_profile >= FILL] = 0.0
                iwc_profile[iwc_profile < 0] = 0.0
                n_levels_actual = len(iwc_profile)
                if n_levels_actual < EC_N_LEVELS:
                    iwc_profile = np.concatenate([iwc_profile,
                        np.zeros(EC_N_LEVELS - n_levels_actual)])

                time_utc = J2000 + timedelta(seconds=float(t_sec[i]))

                rec = {
                    "orbit_id": orbit_id, "profile_idx": i,
                    "lat": float(lat[i]), "lon": float(lon[i]),
                    "time_utc": time_utc, "iwp": iwp,
                    "cloud_top_h": cloud_top_h, "cloud_base_h": cloud_base_h,
                    "max_iwc": max_iwc, "max_iwc_alt": max_iwc_alt,
                    "n_ice_layers": n_layers, "n_ice_levels": int(valid.sum()),
                    "elevation": float(elev[i]) if elev[i] < FILL else 0.0,
                }
                for li in range(EC_N_LEVELS):
                    rec[f"iwc_{li}"] = float(iwc_profile[li])

                all_records.append(rec)

            f.close()

            if (fi + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"    [{fi+1}/{len(h5_files)}] {len(all_records):,} records ({elapsed:.0f}s)")

        if all_records:
            df = pd.DataFrame(all_records)
            df.to_parquet(output_path, index=False)
            print(f"  Saved {month_name}: {len(df):,} records ({os.path.getsize(output_path)/1e6:.1f} MB)")


# ══════════════════════════════════════════════════════════════
#  Step 2: VIIRS Index
# ══════════════════════════════════════════════════════════════

def step_viirs_index(period_indices):
    """Build VIIRS index for all periods."""
    import earthaccess
    try:
        earthaccess.login(strategy="environment")
    except Exception:
        earthaccess.login(persist=True)

    # Load all EC profiles
    parquet_files = sorted(EC_PROC_DIR.glob("profiles_*.parquet"))
    if not parquet_files:
        print("ERROR: No EarthCARE parquet files found")
        return False

    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=["lat", "lon", "time_utc", "orbit_id"])
        dfs.append(df)
        print(f"  {pf.name}: {len(df):,} profiles")
    profiles = pd.concat(dfs, ignore_index=True)
    profiles["date"] = pd.to_datetime(profiles["time_utc"]).dt.strftime("%Y-%m-%d")

    # Get dates for requested periods
    all_dates = set()
    for pi in period_indices:
        start_str, end_str = PERIODS[pi]
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        for di in range((end - start).days + 1):
            d = (start + timedelta(days=di)).strftime("%Y-%m-%d")
            if d in profiles["date"].values:
                all_dates.add(d)

    dates = sorted(all_dates)
    print(f"\nSearching VIIRS for {len(dates)} dates...")

    # Load existing index if any
    index_path = OUTPUT_DIR / "viirs_index.parquet"
    existing_dates = set()
    existing_records = []
    if index_path.exists():
        existing_df = pd.read_parquet(index_path)
        existing_dates = set(existing_df["date"].unique())
        existing_records = existing_df.to_dict("records")
        print(f"  Existing index: {len(existing_df)} records, {len(existing_dates)} dates")

    new_dates = [d for d in dates if d not in existing_dates]
    if not new_dates:
        print("  All dates already indexed!")
        return True

    print(f"  New dates to index: {len(new_dates)}")

    # Import the search function from 02_viirs_index
    from importlib import util as imp_util
    spec = imp_util.spec_from_file_location(
        "viirs_index", str(Path(__file__).parent / "02_viirs_index.py"))
    viirs_mod = imp_util.module_from_spec(spec)
    spec.loader.exec_module(viirs_mod)

    all_records = list(existing_records)
    for di, date_str in enumerate(new_dates):
        records = viirs_mod.search_viirs_for_date(date_str, profiles)
        all_records.extend(records)
        print(f"  [{di+1}/{len(new_dates)}] {date_str}: {len(records)} granules "
              f"(total: {len(all_records)})")

    index_df = pd.DataFrame(all_records)
    index_df.to_parquet(index_path, index=False)
    print(f"\nSaved index: {len(index_df)} records to {index_path}")
    return True


# ══════════════════════════════════════════════════════════════
#  Step 3: Match + Extract
# ══════════════════════════════════════════════════════════════

def step_match_extract(period_indices):
    """Match EarthCARE to VIIRS and extract patches for new periods."""
    import earthaccess
    try:
        earthaccess.login(strategy="environment")
    except Exception:
        earthaccess.login(persist=True)

    # Import matching functions from 03
    from importlib import util as imp_util
    spec = imp_util.spec_from_file_location(
        "match_extract", str(Path(__file__).parent / "03_match_and_extract_v2.py"))
    match_mod = imp_util.module_from_spec(spec)
    spec.loader.exec_module(match_mod)

    # Load VIIRS index
    index_path = OUTPUT_DIR / "viirs_index.parquet"
    viirs_index = pd.read_parquet(index_path)
    print(f"VIIRS index: {len(viirs_index)} records")

    # Load EC profiles
    parquet_files = sorted(EC_PROC_DIR.glob("profiles_*.parquet"))
    dfs = []
    for pf in parquet_files:
        dfs.append(pd.read_parquet(pf))
    ec_profiles = pd.concat(dfs, ignore_index=True)
    print(f"EC profiles: {len(ec_profiles):,}")

    viirs_cache = VIIRS_DIR / "cache"
    viirs_cache.mkdir(parents=True, exist_ok=True)

    # Get dates for requested periods
    all_dates = set()
    for pi in period_indices:
        start_str, end_str = PERIODS[pi]
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        for di in range((end - start).days + 1):
            all_dates.add((start + timedelta(days=di)).strftime("%Y-%m-%d"))

    # Filter to dates in VIIRS index
    available_dates = sorted(d for d in all_dates if d in viirs_index["date"].values)
    print(f"\nProcessing {len(available_dates)} dates for periods {period_indices}")

    # Check existing matches
    matches_path = OUTPUT_DIR / "matches_v2.parquet"
    existing_dates_matched = set()
    if matches_path.exists():
        existing_matches = pd.read_parquet(matches_path)
        existing_dates_matched = set(existing_matches["date"].unique())
        print(f"  Existing matches: {len(existing_matches):,} across {len(existing_dates_matched)} dates")

    new_dates = [d for d in available_dates if d not in existing_dates_matched]
    if not new_dates:
        print("  All dates already matched!")
        return True

    print(f"  New dates to match: {len(new_dates)}")

    # Match new dates
    all_matches = []
    t0 = time.time()

    for di, date_str in enumerate(new_dates):
        matches = match_mod.process_date(
            date_str, ec_profiles, viirs_index, viirs_cache)
        all_matches.extend(matches)

        elapsed = time.time() - t0
        rate = (di + 1) / elapsed if elapsed > 0 else 0
        eta = (len(new_dates) - di - 1) / rate if rate > 0 else 0
        print(f"  [{di+1}/{len(new_dates)}] {date_str}: "
              f"{len(matches):,} matches (total: {len(all_matches):,}) "
              f"[{elapsed:.0f}s, ~{eta:.0f}s ETA]", flush=True)

    if not all_matches:
        print("No new matches found!")
        return False

    new_matches_df = pd.DataFrame(all_matches)

    # Merge with existing matches
    if matches_path.exists():
        combined = pd.concat([existing_matches, new_matches_df], ignore_index=True)
    else:
        combined = new_matches_df
    combined.to_parquet(matches_path, index=False)
    print(f"\nTotal matches: {len(combined):,} saved to {matches_path}")

    # Extract patches for new matches only → append to zarr
    zarr_path = COLOC_DIR / "patches_v2.zarr"

    # Check if zarr exists and has data
    zarr_exists = zarr_path.exists()
    if zarr_exists:
        existing_store = zarr.open(str(zarr_path), mode="r")
        n_existing = existing_store["patches"].shape[0]
        print(f"  Existing zarr: {n_existing:,} patches")
    else:
        n_existing = 0

    print(f"\nExtracting {len(new_matches_df):,} new patches...")
    # Use a temporary zarr for new patches, then merge
    temp_zarr_path = COLOC_DIR / "patches_new_temp.zarr"
    if temp_zarr_path.exists():
        import shutil
        shutil.rmtree(temp_zarr_path)

    n_written = match_mod.extract_patches_streaming(
        new_matches_df, viirs_cache, ec_profiles, temp_zarr_path)

    if n_written > 0 and zarr_exists:
        print(f"\nMerging {n_written:,} new patches with {n_existing:,} existing...")
        _merge_zarr_stores(zarr_path, temp_zarr_path)
        import shutil
        shutil.rmtree(temp_zarr_path)
    elif n_written > 0 and not zarr_exists:
        # Rename temp to final
        temp_zarr_path.rename(zarr_path)

    return True


def _merge_zarr_stores(existing_path, new_path):
    """Append new zarr patches to existing zarr store."""
    old = zarr.open(str(existing_path), mode="a")
    new = zarr.open(str(new_path), mode="r")

    n_new = new["patches"].shape[0]
    print(f"  Appending {n_new:,} patches...")

    # Append data arrays in chunks
    CHUNK = 5000
    for i in range(0, n_new, CHUNK):
        j = min(i + CHUNK, n_new)
        old["patches"].append(new["patches"][i:j])
        old["targets"].append(new["targets"][i:j])

    # Append metadata
    if "meta" in new:
        for key in new["meta"]:
            if key in old["meta"]:
                old_vals = old["meta"][key][:]
                new_vals = new["meta"][key][:]
                combined = np.concatenate([old_vals, new_vals])
                del old["meta"][key]
                old["meta"].create_array(key, data=combined)
            else:
                old["meta"].create_array(key, data=new["meta"][key][:])

    total = old["patches"].shape[0]
    print(f"  Merged store: {total:,} patches")


# ══════════════════════════════════════════════════════════════
#  Step 4-6: Split + Convert to npy
# ══════════════════════════════════════════════════════════════

def step_merge_and_split():
    """Re-split the combined zarr and convert to npy."""
    zarr_path = COLOC_DIR / "patches_v2.zarr"
    if not zarr_path.exists():
        print("ERROR: No zarr store found!")
        return False

    store = zarr.open(str(zarr_path), mode="a")
    N = store["patches"].shape[0]
    print(f"Total samples in zarr: {N:,}")

    # Get lat/lon for splitting
    lats = store["meta"]["ec_lat"][:]
    lons = store["meta"]["ec_lon"][:]

    # Geographic grid split
    lat_bins = np.floor(lats / SPLIT_GRID_DEG).astype(int)
    lon_bins = np.floor(lons / SPLIT_GRID_DEG).astype(int)
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

    # Update split in zarr
    if "split" in store["meta"]:
        del store["meta"]["split"]
    store["meta"].create_array("split", data=splits)

    for s in ["train", "val", "test"]:
        n_s = (splits == s).sum()
        print(f"  {s}: {n_s:,} ({100*n_s/N:.1f}%)")

    # Convert to npy
    print(f"\nConverting to npy...")
    npy_dir = COLOC_DIR / "npy_v3"  # v3 = expanded dataset
    npy_dir.mkdir(exist_ok=True)

    for split in ["train", "val", "test"]:
        idx = np.where(splits == split)[0]
        print(f"\n  {split}: {len(idx):,} samples")

        t1 = time.time()
        print(f"    Loading patches...", end="", flush=True)
        patches = store["patches"][idx].astype(np.float32)
        print(f" done ({time.time() - t1:.0f}s)")

        # Normalize
        patches[:, :N_THERMAL] = (patches[:, :N_THERMAL] - BT_MEAN) / BT_STD
        patches[:, N_THERMAL + N_REFL] = patches[:, N_THERMAL + N_REFL] / 90.0

        t1 = time.time()
        print(f"    Loading targets...", end="", flush=True)
        targets = store["targets"][idx].astype(np.float32)
        targets = np.log10(targets + LOG_IWC_EPS)
        print(f" done ({time.time() - t1:.0f}s)")

        lat = lats[idx].astype(np.float32)
        lon = lons[idx].astype(np.float32)
        sza = store["meta"]["sza"][idx].astype(np.float32)

        # Get dates for ERA5 matching later
        dates_arr = store["meta"]["date"][idx]
        ec_time_arr = store["meta"]["ec_time"][idx]

        print(f"    Saving...", end="", flush=True)
        np.save(npy_dir / f"{split}_patches.npy", patches)
        np.save(npy_dir / f"{split}_targets.npy", targets)
        np.save(npy_dir / f"{split}_lat.npy", lat)
        np.save(npy_dir / f"{split}_lon.npy", lon)
        np.save(npy_dir / f"{split}_sza.npy", sza)
        np.save(npy_dir / f"{split}_dates.npy", np.array(dates_arr, dtype=str))
        np.save(npy_dir / f"{split}_ec_time.npy", np.array(ec_time_arr, dtype=str))
        print(f" done ({(patches.nbytes + targets.nbytes) / 1e9:.1f} GB)")

    print(f"\nSaved to {npy_dir}")
    return True


# ══════════════════════════════════════════════════════════════
#  Step 7-8: ERA5 download + colocation
# ══════════════════════════════════════════════════════════════

def step_era5():
    """Download ERA5 and collocate with expanded dataset."""
    npy_dir = COLOC_DIR / "npy_v3"
    if not npy_dir.exists():
        print("ERROR: npy_v3 directory not found. Run merge step first.")
        return False

    # Collect all unique dates across splits
    all_dates = set()
    for split in ["train", "val", "test"]:
        dates_path = npy_dir / f"{split}_dates.npy"
        if dates_path.exists():
            dates = np.load(dates_path)
            all_dates.update(dates.tolist())

    all_dates = sorted(all_dates)
    print(f"Unique dates: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")

    # Check which ERA5 files exist
    ERA5_DIR.mkdir(parents=True, exist_ok=True)
    missing = [d for d in all_dates
               if not (ERA5_DIR / f"era5_pl_{d}.nc").exists()
               or (ERA5_DIR / f"era5_pl_{d}.nc").stat().st_size < 1e6]

    if missing:
        print(f"\nDownloading ERA5 for {len(missing)} missing dates...")
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
            print(f"  Requesting {date_str}...", end=" ", flush=True)

            request = {
                "product_type": ["reanalysis"],
                "variable": VARIABLES,
                "pressure_level": PRESSURE_LEVELS,
                "year": [str(date.year)],
                "month": [f"{date.month:02d}"],
                "day": [f"{date.day:02d}"],
                "time": HOURS,
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            try:
                client.retrieve("reanalysis-era5-pressure-levels", request, str(out_file))
                print(f"OK ({out_file.stat().st_size/1e6:.1f}MB)")
            except Exception as e:
                print(f"FAILED - {e}")
    else:
        print("All ERA5 files present!")

    # Collocate ERA5 with npy samples
    print(f"\nCollocating ERA5...")
    import xarray as xr

    for split in ["train", "val", "test"]:
        dates = np.load(npy_dir / f"{split}_dates.npy")
        ec_times = np.load(npy_dir / f"{split}_ec_time.npy")
        lats = np.load(npy_dir / f"{split}_lat.npy")
        lons = np.load(npy_dir / f"{split}_lon.npy")
        N = len(dates)

        print(f"\n  {split}: {N:,} samples")

        # Parse hours from ec_time strings
        hours = np.zeros(N, dtype=np.int32)
        for i in range(N):
            try:
                hours[i] = int(ec_times[i][11:13])
            except Exception:
                hours[i] = 0

        # Determine shape from first available file
        sample_date = dates[0]
        sample_ds = xr.open_dataset(ERA5_DIR / f"era5_pl_{sample_date}.nc")
        var_names = [v for v in ['t', 'q', 'r', 'z'] if v in sample_ds]
        plev_name = 'pressure_level' if 'pressure_level' in sample_ds.dims else 'level'
        n_levels = len(sample_ds[plev_name])
        n_vars = len(var_names)
        sample_ds.close()

        era5_profiles = np.full((N, n_levels, n_vars), np.nan, dtype=np.float32)

        # Process day by day
        unique_dates = sorted(set(dates))
        for date_str in unique_dates:
            era5_file = ERA5_DIR / f"era5_pl_{date_str}.nc"
            if not era5_file.exists() or era5_file.stat().st_size < 1e6:
                continue

            mask = dates == date_str
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            ds = xr.open_dataset(era5_file)
            era5_lats = ds['latitude'].values
            era5_lons = ds['longitude'].values

            day_lats = lats[indices]
            day_lons = lons[indices].copy()
            day_lons[day_lons < 0] += 360  # ERA5 convention
            day_hours = hours[indices]

            lat_idx = np.abs(era5_lats[np.newaxis, :] - day_lats[:, np.newaxis]).argmin(axis=1)
            lon_idx = np.abs(era5_lons[np.newaxis, :] - day_lons[:, np.newaxis]).argmin(axis=1)

            valid_times = ds['valid_time'].values
            time_hours = np.array([
                np.datetime64(t, 'h').astype('datetime64[h]').astype(int) % 24
                for t in valid_times
            ])
            hour_to_tidx = {h: i for i, h in enumerate(time_hours)}

            for v_idx, var_name in enumerate(var_names):
                data = ds[var_name].values

                for j in range(len(indices)):
                    h = day_hours[j]
                    tidx = hour_to_tidx.get(h, None)
                    if tidx is None:
                        tidx = np.abs(time_hours - h).argmin()
                    era5_profiles[indices[j], :, v_idx] = data[tidx, :, lat_idx[j], lon_idx[j]]

            ds.close()

        n_valid = np.sum(~np.isnan(era5_profiles[:, 0, 0]))
        print(f"    Matched: {n_valid:,}/{N:,} ({100*n_valid/N:.1f}%)")

        # Flatten to (N, n_levels*n_vars) for training
        era5_flat = era5_profiles.reshape(N, -1)
        np.save(npy_dir / f"{split}_era5.npy", era5_flat)
        print(f"    Saved: {split}_era5.npy ({era5_flat.shape})")

    # Save metadata
    np.savez(npy_dir / "era5_meta.npz",
             var_names=np.array(var_names), n_levels=n_levels, n_vars=n_vars)
    print("\nERA5 colocation done!")
    return True


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "download", "viirs-index", "match", "merge", "era5"],
                        help="Which pipeline step to run")
    parser.add_argument("--periods", type=str, default=None,
                        help="Comma-separated period indices (e.g., 1,2,3,4)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip period 0 (June, already processed)")
    args = parser.parse_args()

    # Determine which periods to process
    if args.periods:
        period_indices = [int(x) for x in args.periods.split(",")]
    elif args.skip_existing:
        period_indices = list(range(1, len(PERIODS)))
    else:
        period_indices = list(range(len(PERIODS)))

    print(f"Periods to process: {period_indices}")
    for pi in period_indices:
        print(f"  {pi}: {PERIODS[pi][0]} to {PERIODS[pi][1]}")

    t0 = time.time()

    if args.step in ("all", "download"):
        print(f"\n{'#'*60}")
        print("# STEP 1: Download EarthCARE")
        print(f"{'#'*60}")
        step_download_earthcare(period_indices)

    if args.step in ("all", "viirs-index"):
        print(f"\n{'#'*60}")
        print("# STEP 2: Build VIIRS index")
        print(f"{'#'*60}")
        step_viirs_index(period_indices)

    if args.step in ("all", "match"):
        print(f"\n{'#'*60}")
        print("# STEP 3: Match + Extract VIIRS patches")
        print(f"{'#'*60}")
        step_match_extract(period_indices)

    if args.step in ("all", "merge"):
        print(f"\n{'#'*60}")
        print("# STEP 4-6: Split + Convert to npy")
        print(f"{'#'*60}")
        step_merge_and_split()

    if args.step in ("all", "era5"):
        print(f"\n{'#'*60}")
        print("# STEP 7-8: ERA5 download + colocation")
        print(f"{'#'*60}")
        step_era5()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Pipeline complete! Total time: {elapsed/3600:.1f} hours")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
