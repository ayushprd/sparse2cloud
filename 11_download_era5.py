"""
Download ERA5 pressure-level data for EarthCARE IWC colocation.

Downloads T, q, RH, Z on 26 pressure levels (50-1000 hPa) for all dates
in config.PERIODS. Each day is ~3.7 GB.

Usage:
    python 11_download_era5.py                # All periods
    python 11_download_era5.py --period 1     # Only period 1
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import cdsapi
from datetime import datetime, timedelta
from pathlib import Path
from config import PERIODS, ERA5_DIR

OUT_DIR = ERA5_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRESSURE_LEVELS = [
    "50", "70", "100", "125", "150", "175", "200", "225", "250",
    "300", "350", "400", "450", "500", "550", "600", "650", "700",
    "750", "800", "850", "900", "925", "950", "975", "1000",
]

VARIABLES = [
    "temperature",
    "specific_humidity",
    "relative_humidity",
    "geopotential",
]

HOURS = [f"{h:02d}:00" for h in range(24)]


def download_era5_for_period(start_str, end_str, client):
    """Download ERA5 day-by-day for a date range."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    n_days = (end - start).days + 1

    n_downloaded, n_skipped = 0, 0
    for di in range(n_days):
        date = start + timedelta(days=di)
        date_str = date.strftime("%Y-%m-%d")
        out_file = OUT_DIR / f"era5_pl_{date_str}.nc"

        if out_file.exists():
            sz = out_file.stat().st_size
            if sz > 1e6:
                n_skipped += 1
                continue

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
            sz = out_file.stat().st_size
            print(f"OK ({sz/1e6:.1f}MB)")
            n_downloaded += 1
        except Exception as e:
            print(f"FAILED - {e}")

    return n_downloaded, n_skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=int, default=-1,
                        help="Only process this period index (0-based)")
    args = parser.parse_args()

    periods = PERIODS
    if args.period >= 0:
        periods = [(args.period, PERIODS[args.period])]
    else:
        periods = list(enumerate(PERIODS))

    client = cdsapi.Client()

    for pi, (start, end) in periods:
        print(f"\nPeriod {pi}: {start} to {end}")
        n_dl, n_skip = download_era5_for_period(start, end, client)
        print(f"  Downloaded: {n_dl}, Skipped (exist): {n_skip}")

    print(f"\nDone. Files in: {OUT_DIR}")


if __name__ == "__main__":
    main()
