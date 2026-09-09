"""Download EarthCARE ATL_ICE_2A files and extract to parquet.

Usage:
    python 01_download_earthcare.py                    # Full download + extract
    python 01_download_earthcare.py --test 2           # Test: 2 days only
    python 01_download_earthcare.py --extract-only      # Skip download, just extract
    python 01_download_earthcare.py --period 0          # Only first period
"""

import sys
import os
import argparse
import time
import glob
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import h5py

from config import (
    EC_RAW_DIR, EC_PROC_DIR, EC_PRODUCT, EC_N_LEVELS,
    EC_FILL_VALUE, PERIODS, ESA_USERNAME, ESA_PASSWORD,
    IWC_NOISE_FLOOR, MIN_ICE_LEVELS, LOG_DIR,
)

FILL = EC_FILL_VALUE
J2000 = datetime(2000, 1, 1, tzinfo=timezone.utc)


def download_period(start_date, end_date, test_days=0):
    """Download all ATL_ICE_2A files for a date range."""
    os.environ["ESA_EO_USERNAME"] = ESA_USERNAME
    os.environ["ESA_EO_PASSWORD"] = ESA_PASSWORD
    from earthcare_downloader import search, download

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    n_days = (end - start).days + 1
    if test_days > 0:
        n_days = min(n_days, test_days)

    total_downloaded = 0
    for di in range(n_days):
        date = start + timedelta(days=di)
        date_str = date.strftime("%Y-%m-%d")
        month_dir = EC_RAW_DIR / date.strftime("%Y-%m")
        month_dir.mkdir(parents=True, exist_ok=True)

        # Check existing files for this date
        date_prefix = date.strftime("%Y%m%d")
        existing = list(month_dir.glob(f"*{date_prefix}*.h5"))
        if len(existing) >= 100:
            print(f"  [{di+1}/{n_days}] {date_str}: {len(existing)} files exist, skipping")
            total_downloaded += len(existing)
            continue

        try:
            files = search(product=EC_PRODUCT, date=date_str)
        except Exception as e:
            print(f"  [{di+1}/{n_days}] {date_str}: search error: {e}")
            continue

        if not files:
            print(f"  [{di+1}/{n_days}] {date_str}: no files found")
            continue

        # Filter out already downloaded
        existing_names = {f.stem for f in existing}
        to_download = [f for f in files
                       if f.filename.replace(".ZIP", "") not in existing_names]

        if not to_download:
            print(f"  [{di+1}/{n_days}] {date_str}: all {len(files)} files exist")
            total_downloaded += len(files)
            continue

        # Download in batches of 5 to avoid rate limiting
        BATCH_SIZE = 5
        n_new = 0
        for bi in range(0, len(to_download), BATCH_SIZE):
            batch = to_download[bi:bi + BATCH_SIZE]
            for attempt in range(3):
                try:
                    paths = download(batch, output_path=str(month_dir))
                    n_new += sum(1 for p in paths if str(p).endswith(".h5"))
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(5 * (attempt + 1))
                    else:
                        print(f"    batch {bi//BATCH_SIZE}: failed after 3 attempts: {e}")
        total_downloaded += n_new + len(existing)
        print(f"  [{di+1}/{n_days}] {date_str}: downloaded {n_new} new "
              f"(total: {total_downloaded})")

    return total_downloaded


def extract_file(fpath):
    """Extract ice cloud profiles from a single ATL_ICE_2A HDF5 file.

    Returns a list of dicts (one per ice-bearing profile).
    """
    try:
        f = h5py.File(fpath, "r")
    except Exception:
        return []

    sd = f["ScienceData"]
    lat = sd["latitude"][:]
    lon = sd["longitude"][:]
    t_sec = sd["time"][:]
    iwc = sd["ice_water_content"][:]
    reff = sd["ice_effective_radius"][:]
    height = sd["height"][:]
    qf = sd["quality_status"][:]
    elev = sd["elevation"][:]

    # Orbit ID from filename
    orbit_id = Path(fpath).stem

    records = []
    n_profiles = len(lat)

    for i in range(n_profiles):
        if lat[i] >= FILL or lon[i] >= FILL or t_sec[i] >= FILL:
            continue

        # Ice mask: valid IWC above noise floor
        iwc_i = iwc[i]
        h_i = height[i]
        valid = (iwc_i < FILL) & (iwc_i > IWC_NOISE_FLOOR) & (h_i < FILL)

        if valid.sum() < MIN_ICE_LEVELS:
            continue

        # Compute derived quantities
        h_valid = h_i[valid]
        iwc_valid = iwc_i[valid]
        sort_idx = np.argsort(h_valid)

        iwp = float(np.trapezoid(iwc_valid[sort_idx] * 1e-6, h_valid[sort_idx]))
        cloud_top_h = float(h_valid.max())
        cloud_base_h = float(h_valid.min())
        max_iwc = float(iwc_valid.max())
        max_iwc_alt = float(h_valid[np.argmax(iwc_valid)])

        # Count distinct ice layers (gaps > 500m)
        h_sorted = np.sort(h_valid)
        gaps = np.diff(h_sorted) > 500
        n_layers = int(gaps.sum()) + 1

        # Store full profile arrays as lists
        iwc_profile = iwc_i.copy()
        iwc_profile[iwc_profile >= FILL] = 0.0
        iwc_profile[iwc_profile < 0] = 0.0

        reff_i = reff[i].copy()
        reff_i[reff_i >= FILL] = 0.0
        reff_i[reff_i < 0] = 0.0

        # Pad to EC_N_LEVELS if file has fewer levels
        n_levels_actual = len(iwc_profile)
        if n_levels_actual < EC_N_LEVELS:
            iwc_profile = np.concatenate([iwc_profile,
                np.zeros(EC_N_LEVELS - n_levels_actual)])

        # Time
        time_utc = J2000 + timedelta(seconds=float(t_sec[i]))

        rec = {
            "orbit_id": orbit_id,
            "profile_idx": i,
            "lat": float(lat[i]),
            "lon": float(lon[i]),
            "time_utc": time_utc,
            "iwp": iwp,
            "cloud_top_h": cloud_top_h,
            "cloud_base_h": cloud_base_h,
            "max_iwc": max_iwc,
            "max_iwc_alt": max_iwc_alt,
            "n_ice_layers": n_layers,
            "n_ice_levels": int(valid.sum()),
            "elevation": float(elev[i]) if elev[i] < FILL else 0.0,
        }

        # Store profile as individual columns (more parquet-friendly)
        for li in range(EC_N_LEVELS):
            rec[f"iwc_{li}"] = float(iwc_profile[li])

        records.append(rec)

    f.close()
    return records


def extract_month(month_dir, output_path):
    """Extract all files in a month directory to a single parquet."""
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        print(f"  {output_path.name}: already exists ({len(existing):,} records)")
        return len(existing)

    h5_files = sorted(glob.glob(str(month_dir / "*.h5")))
    if not h5_files:
        print(f"  {month_dir.name}: no HDF5 files")
        return 0

    all_records = []
    t0 = time.time()

    for fi, fpath in enumerate(h5_files):
        records = extract_file(fpath)
        all_records.extend(records)

        if (fi + 1) % 20 == 0 or fi == len(h5_files) - 1:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            eta = (len(h5_files) - fi - 1) / rate if rate > 0 else 0
            print(f"  [{fi+1}/{len(h5_files)}] {len(all_records):,} records "
                  f"({elapsed:.0f}s, ~{eta:.0f}s ETA)")

    if not all_records:
        print(f"  {month_dir.name}: no ice profiles found")
        return 0

    df = pd.DataFrame(all_records)
    df.to_parquet(output_path, index=False)
    print(f"  Saved {output_path.name}: {len(df):,} records "
          f"({os.path.getsize(output_path)/1e6:.1f} MB)")
    return len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0,
                        help="Test mode: process N days only")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--period", type=int, default=-1,
                        help="Only process this period index (0-based)")
    args = parser.parse_args()

    periods = PERIODS
    if args.period >= 0:
        periods = [periods[args.period]]

    # Phase 1: Download
    if not args.extract_only:
        print("=" * 60)
        print("DOWNLOADING EarthCARE ATL_ICE_2A")
        print("=" * 60)
        for pi, (start, end) in enumerate(periods):
            print(f"\nPeriod {pi}: {start} to {end}")
            n = download_period(start, end, test_days=args.test)
            print(f"  Total files: {n}")

    # Phase 2: Extract to parquet
    print("\n" + "=" * 60)
    print("EXTRACTING TO PARQUET")
    print("=" * 60)

    total_records = 0
    month_dirs = sorted(EC_RAW_DIR.glob("*"))
    for mdir in month_dirs:
        if not mdir.is_dir():
            continue
        month_name = mdir.name  # e.g., "2025-06"
        output_path = EC_PROC_DIR / f"profiles_{month_name}.parquet"
        print(f"\n{month_name}:")
        n = extract_month(mdir, output_path)
        total_records += n

    print(f"\n{'=' * 60}")
    print(f"TOTAL EXTRACTED: {total_records:,} ice cloud profiles")

    # Print summary
    print("\nPer-month breakdown:")
    for pq in sorted(EC_PROC_DIR.glob("profiles_*.parquet")):
        df = pd.read_parquet(pq, columns=["lat", "iwp"])
        print(f"  {pq.stem}: {len(df):,} profiles, "
              f"lat=[{df['lat'].min():.1f},{df['lat'].max():.1f}], "
              f"IWP median={df['iwp'].median():.4f} g/m²")


if __name__ == "__main__":
    main()
