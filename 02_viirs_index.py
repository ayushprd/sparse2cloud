"""Build VIIRS granule index for EarthCARE co-location.

For each date with EarthCARE profiles, search for VIIRS VNP02MOD granules
that overlap in time (±30 min) and space. Stores a compact index for fast
lookup during the matching step.

Usage:
    python 02_viirs_index.py              # Full index build
    python 02_viirs_index.py --test 3     # Test: 3 dates only
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import earthaccess

from config import (
    EC_PROC_DIR, OUTPUT_DIR, VIIRS_L1B_PRODUCT, VIIRS_GEO_PRODUCT,
    MAX_TIME_OFFSET_SEC,
)


def earthaccess_login():
    """Login to NASA Earthdata."""
    try:
        earthaccess.login(strategy="environment")
        return True
    except Exception:
        try:
            earthaccess.login(persist=True)
            return True
        except Exception as e:
            print(f"Login failed: {e}")
            return False


def get_orbit_time_ranges(df, group_gap_sec=600):
    """Group profiles into orbit segments with similar times.

    Returns list of (start_time, end_time, lat_min, lat_max, lon_min, lon_max).
    """
    df = df.sort_values("time_utc").copy()
    times = pd.to_datetime(df["time_utc"])

    # Group by time gaps > 10 min (= separate orbits)
    gaps = times.diff().dt.total_seconds() > group_gap_sec
    orbit_ids = gaps.cumsum()

    segments = []
    for _, group in df.groupby(orbit_ids):
        t = pd.to_datetime(group["time_utc"])
        segments.append({
            "start_time": t.min(),
            "end_time": t.max(),
            "lat_min": group["lat"].min(),
            "lat_max": group["lat"].max(),
            "lon_min": group["lon"].min(),
            "lon_max": group["lon"].max(),
            "n_profiles": len(group),
        })

    return segments


def search_viirs_for_segment(segment, time_pad_sec=MAX_TIME_OFFSET_SEC):
    """Search VIIRS granules overlapping an EarthCARE orbit segment."""
    t_min = segment["start_time"] - pd.Timedelta(seconds=time_pad_sec)
    t_max = segment["end_time"] + pd.Timedelta(seconds=time_pad_sec)

    # Spatial bbox with padding
    lat_min = max(segment["lat_min"] - 2, -90)
    lat_max = min(segment["lat_max"] + 2, 90)
    lon_min = max(segment["lon_min"] - 15, -180)
    lon_max = min(segment["lon_max"] + 15, 180)

    try:
        results = earthaccess.search_data(
            short_name=VIIRS_L1B_PRODUCT,
            temporal=(t_min.strftime("%Y-%m-%dT%H:%M:%S"),
                     t_max.strftime("%Y-%m-%dT%H:%M:%S")),
            bounding_box=(lon_min, lat_min, lon_max, lat_max),
            count=100,
        )
    except Exception as e:
        print(f"    VIIRS search error: {e}")
        return []

    records = []
    for r in results:
        # Extract granule metadata
        try:
            umm = r.get("umm", r) if isinstance(r, dict) else r
            # earthaccess returns DataGranule objects
            name = r.data_links()[0].split("/")[-1] if r.data_links() else str(r)
            temporal = r.get("umm", {}).get("TemporalExtent", {})

            # Get time from granule name: VNP02MOD.AYYYYDDD.HHMM...
            # Parse from data links
            link = r.data_links()[0] if r.data_links() else ""
            fname = link.split("/")[-1]

            records.append({
                "granule_name": fname,
                "data_link": r.data_links()[0] if r.data_links() else "",
                "geo_link": "",  # Will be populated separately
            })
        except Exception:
            continue

    return records


def search_viirs_for_date(date_str, profiles_df):
    """Search all VIIRS granules for a given date's EarthCARE profiles."""
    day_profiles = profiles_df[
        pd.to_datetime(profiles_df["time_utc"]).dt.strftime("%Y-%m-%d") == date_str
    ]

    if len(day_profiles) == 0:
        return []

    # Get orbit segments for this date
    segments = get_orbit_time_ranges(day_profiles)

    # Search VIIRS for each segment
    all_granules = {}
    for seg in segments:
        granules = search_viirs_for_segment(seg)
        for g in granules:
            all_granules[g["granule_name"]] = g

    # Also search for matching geolocation granules
    geo_granules = {}
    for seg in segments:
        t_min = seg["start_time"] - pd.Timedelta(seconds=MAX_TIME_OFFSET_SEC)
        t_max = seg["end_time"] + pd.Timedelta(seconds=MAX_TIME_OFFSET_SEC)
        lat_min = max(seg["lat_min"] - 2, -90)
        lat_max = min(seg["lat_max"] + 2, 90)
        lon_min = max(seg["lon_min"] - 15, -180)
        lon_max = min(seg["lon_max"] + 15, 180)

        try:
            geo_results = earthaccess.search_data(
                short_name=VIIRS_GEO_PRODUCT,
                temporal=(t_min.strftime("%Y-%m-%dT%H:%M:%S"),
                         t_max.strftime("%Y-%m-%dT%H:%M:%S")),
                bounding_box=(lon_min, lat_min, lon_max, lat_max),
                count=100,
            )
            for r in geo_results:
                link = r.data_links()[0] if r.data_links() else ""
                fname = link.split("/")[-1]
                geo_granules[fname] = link
        except Exception:
            pass

    # Match L1B to GEO by time stamp (AYYYYDDD.HHMM)
    records = []
    for name, g in all_granules.items():
        # Extract time key from name: VNP02MOD.AYYYYDDD.HHMM
        parts = name.split(".")
        if len(parts) >= 3:
            time_key = f"{parts[1]}.{parts[2]}"
            # Find matching GEO granule
            geo_name = None
            geo_link = ""
            for gn, gl in geo_granules.items():
                if time_key in gn:
                    geo_name = gn
                    geo_link = gl
                    break

            records.append({
                "l1b_granule": name,
                "l1b_link": g["data_link"],
                "geo_granule": geo_name or "",
                "geo_link": geo_link,
                "date": date_str,
                "time_key": time_key,
            })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0,
                        help="Test mode: process N dates only")
    args = parser.parse_args()

    if not earthaccess_login():
        print("ERROR: earthaccess login failed")
        return

    # Load all EarthCARE profiles
    parquet_files = sorted(EC_PROC_DIR.glob("profiles_*.parquet"))
    if not parquet_files:
        print("ERROR: No EarthCARE parquet files found. Run 01_download_earthcare.py first.")
        return

    print("Loading EarthCARE profiles...")
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=["lat", "lon", "time_utc", "orbit_id"])
        dfs.append(df)
        print(f"  {pf.name}: {len(df):,} profiles")
    profiles = pd.concat(dfs, ignore_index=True)

    # Get unique dates
    profiles["date"] = pd.to_datetime(profiles["time_utc"]).dt.strftime("%Y-%m-%d")
    dates = sorted(profiles["date"].unique())
    print(f"\nTotal: {len(profiles):,} profiles across {len(dates)} dates")

    if args.test > 0:
        dates = dates[:args.test]
        print(f"TEST MODE: processing {len(dates)} dates")

    # Search VIIRS for each date
    all_records = []
    t0 = time.time()

    for di, date_str in enumerate(dates):
        records = search_viirs_for_date(date_str, profiles)
        all_records.extend(records)

        elapsed = time.time() - t0
        rate = (di + 1) / elapsed if elapsed > 0 else 0
        eta = (len(dates) - di - 1) / rate if rate > 0 else 0
        print(f"  [{di+1}/{len(dates)}] {date_str}: "
              f"{len(records)} VIIRS granules "
              f"(total: {len(all_records)}) "
              f"[{elapsed:.0f}s, ~{eta:.0f}s ETA]")

    if not all_records:
        print("No VIIRS granules found!")
        return

    # Save index
    index_df = pd.DataFrame(all_records)
    index_path = OUTPUT_DIR / "viirs_index.parquet"
    index_df.to_parquet(index_path, index=False)
    print(f"\nSaved: {index_path} ({len(index_df)} granule records)")

    # Summary
    print(f"\nSummary:")
    print(f"  Dates: {len(dates)}")
    print(f"  VIIRS L1B granules: {index_df['l1b_granule'].nunique()}")
    print(f"  With matching GEO: {(index_df['geo_granule'] != '').sum()}")

    # Save summary
    summary = {
        "n_dates": len(dates),
        "n_granules": len(index_df),
        "n_unique_l1b": int(index_df["l1b_granule"].nunique()),
        "n_with_geo": int((index_df["geo_granule"] != "").sum()),
    }
    with open(OUTPUT_DIR / "viirs_index_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
