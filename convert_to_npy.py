"""Convert zarr dataset to .npy files for fast loading.

Saves split .npy files that can be mmap'd for zero-copy loading.

Usage:
    python convert_to_npy.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import zarr

from config import COLOC_DIR, LOG_IWC_EPS, BT_MEAN, BT_STD, VIIRS_THERMAL_BANDS, VIIRS_REFL_BANDS

N_THERMAL = len(VIIRS_THERMAL_BANDS)
N_REFL = len(VIIRS_REFL_BANDS)

def main():
    t0 = time.time()
    zarr_path = COLOC_DIR / "patches_v2.zarr"
    npy_dir = COLOC_DIR / "npy_v2"
    npy_dir.mkdir(exist_ok=True)

    store = zarr.open(str(zarr_path), mode="r")

    # Get splits
    splits = store["meta/split"][:]
    if hasattr(splits[0], "decode"):
        splits = np.array([s.decode() for s in splits])

    for split in ["train", "val", "test"]:
        idx = np.where(splits == split)[0]
        print(f"\n{split}: {len(idx):,} samples")

        t1 = time.time()
        print(f"  Loading patches...", end="", flush=True)
        patches = store["patches"][idx].astype(np.float32)
        print(f" done ({time.time() - t1:.0f}s)")

        # Normalize in-place
        patches[:, :N_THERMAL] = (patches[:, :N_THERMAL] - BT_MEAN) / BT_STD
        patches[:, N_THERMAL + N_REFL] = patches[:, N_THERMAL + N_REFL] / 90.0

        t1 = time.time()
        print(f"  Loading targets...", end="", flush=True)
        targets = store["targets"][idx].astype(np.float32)
        targets = np.log10(targets + LOG_IWC_EPS)
        print(f" done ({time.time() - t1:.0f}s)")

        # Metadata
        lat = store["meta/ec_lat"][idx].astype(np.float32)
        sza = store["meta/sza"][idx].astype(np.float32)
        if "meta/ec_lon" in store:
            lon = store["meta/ec_lon"][idx].astype(np.float32)
        else:
            lon = np.zeros(len(idx), dtype=np.float32)

        # Save as npy
        print(f"  Saving...", end="", flush=True)
        np.save(npy_dir / f"{split}_patches.npy", patches)
        np.save(npy_dir / f"{split}_targets.npy", targets)
        np.save(npy_dir / f"{split}_lat.npy", lat)
        np.save(npy_dir / f"{split}_lon.npy", lon)
        np.save(npy_dir / f"{split}_sza.npy", sza)
        print(f" done")

        size_gb = (patches.nbytes + targets.nbytes) / 1e9
        print(f"  Size: {size_gb:.1f} GB")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Saved to {npy_dir}")


if __name__ == "__main__":
    main()
