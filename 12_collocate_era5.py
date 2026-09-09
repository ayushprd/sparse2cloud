"""
Collocate ERA5 pressure-level data with EarthCARE/VIIRS samples.

For each of the 800K samples, extract the ERA5 temperature and specific humidity
profiles at the nearest grid point and hour. This provides vertical atmospheric
context that VIIRS alone cannot observe.

Output: per-split npy files with ERA5 profiles (N_samples, N_levels, N_vars)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import xarray as xr
import zarr
from pathlib import Path
from datetime import datetime, timezone
from config import COLOC_DIR, OUTPUT_DIR, ERA5_DIR

NPY_DIR = COLOC_DIR / "npy_v2"
ZARR_PATH = COLOC_DIR / "patches_v2.zarr"

def load_era5_day(date_str):
    """Load ERA5 data for one day."""
    fpath = ERA5_DIR / f"era5_pl_{date_str}.nc"
    if not fpath.exists():
        return None
    ds = xr.open_dataset(fpath)
    return ds

def get_nearest_era5(ds, lat, lon, hour):
    """Get ERA5 profile at nearest grid point and hour.

    Returns: dict with 'temperature', 'specific_humidity' arrays (n_levels,)
    """
    # ERA5 grid is 0.25° x 0.25° (1440 x 721 points)
    # Round to nearest 0.25°
    lat_round = round(lat * 4) / 4
    lon_round = round(lon * 4) / 4
    if lon_round < 0:
        lon_round += 360  # ERA5 uses 0-360 longitude

    # Select nearest time, lat, lon
    try:
        profile = ds.sel(
            valid_time=f"2025-06-{int(date_str.split('-')[2]):02d}T{hour:02d}:00",
            latitude=lat_round,
            longitude=lon_round,
            method="nearest"
        )
        result = {
            'temperature': profile['t'].values.astype(np.float32),
            'specific_humidity': profile['q'].values.astype(np.float32),
        }
        if 'r' in profile:
            result['relative_humidity'] = profile['r'].values.astype(np.float32)
        if 'z' in profile:
            result['geopotential'] = profile['z'].values.astype(np.float32)
        return result
    except Exception as e:
        return None


def main():
    print("Loading zarr metadata...")
    z = zarr.open_group(str(ZARR_PATH), mode='r')
    ec_time = np.array(z['meta']['ec_time'][:])
    ec_lat = np.array(z['meta']['ec_lat'][:])
    ec_lon = np.array(z['meta']['ec_lon'][:])
    splits = np.array(z['meta']['split'][:])
    dates = np.array(z['meta']['date'][:])
    N = len(ec_time)
    print(f"  Total samples: {N}")

    # Parse hours from ec_time strings
    hours = np.zeros(N, dtype=np.int32)
    for i in range(N):
        # Format: '2025-06-01 00:05:08.125314+00:00'
        try:
            h = int(ec_time[i][11:13])
            hours[i] = h
        except:
            hours[i] = 0

    unique_dates = sorted(set(dates))
    print(f"  Dates: {unique_dates[0]} to {unique_dates[-1]} ({len(unique_dates)} days)")

    # First, check what ERA5 data is available
    available_dates = []
    for d in unique_dates:
        fpath = ERA5_DIR / f"era5_pl_{d}.nc"
        if fpath.exists() and fpath.stat().st_size > 1e6:
            available_dates.append(d)
    print(f"  ERA5 files available: {len(available_dates)}/{len(unique_dates)}")

    if not available_dates:
        print("ERROR: No ERA5 files available yet. Run 11_download_era5.py first.")
        return

    # Process day by day
    # First pass: determine output shape
    test_ds = load_era5_day(available_dates[0])
    if test_ds is None:
        print("ERROR: Could not load test ERA5 file")
        return

    # Get variable names and pressure levels
    print(f"\n  ERA5 variables: {list(test_ds.data_vars)}")
    if 'pressure_level' in test_ds.dims:
        plev_name = 'pressure_level'
    elif 'level' in test_ds.dims:
        plev_name = 'level'
    else:
        plev_name = list(test_ds.dims.keys())[0]
        print(f"  Warning: guessing pressure level dim = '{plev_name}'")

    n_levels = len(test_ds[plev_name])
    pressure_levels = test_ds[plev_name].values
    print(f"  Pressure levels ({n_levels}): {pressure_levels}")

    # Determine how many variables we'll store
    var_names = []
    for v in ['t', 'q', 'r', 'z']:
        if v in test_ds:
            var_names.append(v)
    n_vars = len(var_names)
    print(f"  Variables to extract: {var_names} ({n_vars})")
    test_ds.close()

    # Allocate output arrays
    era5_profiles = np.full((N, n_levels, n_vars), np.nan, dtype=np.float32)

    # Process each available date
    total_matched = 0
    for date_str in available_dates:
        t0 = time.time()
        mask = dates == date_str
        n_day = mask.sum()
        indices = np.where(mask)[0]

        ds = load_era5_day(date_str)
        if ds is None:
            continue

        # Get the grid coordinates
        era5_lats = ds['latitude'].values
        era5_lons = ds['longitude'].values

        # For each unique (hour, lat_round, lon_round), batch the lookup
        # First, compute the nearest ERA5 indices for all samples on this day
        day_lats = ec_lat[indices]
        day_lons = ec_lon[indices]
        day_hours = hours[indices]

        # Convert lons to ERA5 convention (0-360)
        day_lons_360 = day_lons.copy()
        day_lons_360[day_lons_360 < 0] += 360

        # Find nearest lat/lon indices
        lat_idx = np.abs(era5_lats[np.newaxis, :] - day_lats[:, np.newaxis]).argmin(axis=1)
        lon_idx = np.abs(era5_lons[np.newaxis, :] - day_lons_360[:, np.newaxis]).argmin(axis=1)

        # Get unique times for this day
        valid_times = ds['valid_time'].values
        # Convert hours to time indices
        time_hours = np.array([np.datetime64(t, 'h').astype('datetime64[h]').astype(int) % 24
                              for t in valid_times])
        hour_to_tidx = {h: i for i, h in enumerate(time_hours)}

        matched = 0
        for v_idx, var_name in enumerate(var_names):
            data = ds[var_name].values  # shape: (n_times, n_levels, n_lats, n_lons)

            for j in range(len(indices)):
                h = day_hours[j]
                tidx = hour_to_tidx.get(h, None)
                if tidx is None:
                    # Find nearest hour
                    diffs = np.abs(time_hours - h)
                    tidx = diffs.argmin()

                era5_profiles[indices[j], :, v_idx] = data[tidx, :, lat_idx[j], lon_idx[j]]

            if v_idx == 0:
                matched = np.sum(~np.isnan(era5_profiles[indices, 0, 0]))

        ds.close()
        elapsed = time.time() - t0
        total_matched += matched
        print(f"  {date_str}: {matched}/{n_day} matched ({elapsed:.1f}s)")

    print(f"\nTotal matched: {total_matched}/{N} ({total_matched/N*100:.1f}%)")
    print(f"NaN fraction: {np.isnan(era5_profiles).mean():.4f}")

    # Save per split
    for split in ['train', 'val', 'test']:
        mask = splits == split
        out = era5_profiles[mask]
        out_path = NPY_DIR / f"{split}_era5.npy"
        np.save(out_path, out)
        n_valid = np.sum(~np.isnan(out[:, 0, 0]))
        print(f"  {split}: {out.shape} → {out_path.name} ({n_valid}/{len(out)} valid)")

    # Save metadata
    np.savez(
        NPY_DIR / "era5_meta.npz",
        pressure_levels=pressure_levels,
        var_names=np.array(var_names),
        n_levels=n_levels,
        n_vars=n_vars,
    )
    print(f"\nSaved ERA5 metadata to {NPY_DIR / 'era5_meta.npz'}")
    print("Done!")


if __name__ == "__main__":
    main()
