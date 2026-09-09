"""Build dense dataset from existing npy_v2 + matches_v2.parquet.

Groups nearby npy samples into dense patches where each patch has multiple
EarthCARE IWC profiles as targets (instead of just the center pixel).

No VIIRS re-download needed — reuses existing extracted patches.

For each "patch group":
- Takes the FIRST sample's VIIRS image as the patch (all nearby samples
  have nearly identical images since they overlap by ~90%)
- Records ALL neighbor IWC profiles with their (row, col) positions
- This enables IceCloudNet-style dense profile prediction with masked loss

Usage:
    python -u 30_extract_dense_patches.py
    python -u 30_extract_dense_patches.py --stride 4    # More overlap
    python -u 30_extract_dense_patches.py --stride 16   # Less overlap
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from collections import defaultdict

from config import (
    OUTPUT_DIR, COLOC_DIR,
    ACTIVE_LEVEL_START, ACTIVE_LEVEL_END,
    PATCH_SIZE, PATCH_HALF,
    SPLIT_SEED, SPLIT_GRID_DEG, TRAIN_FRAC, VAL_FRAC,
)

N_ACTIVE = ACTIVE_LEVEL_END - ACTIVE_LEVEL_START  # 159
MAX_PROFILES = 64  # pad to this


def assign_splits(lats, lons, grid_deg=SPLIT_GRID_DEG, train_frac=TRAIN_FRAC,
                  val_frac=VAL_FRAC, seed=SPLIT_SEED):
    """Geographic grid split — same logic as 04_split.py."""
    lat_bins = np.floor(lats / grid_deg).astype(int)
    lon_bins = np.floor(lons / grid_deg).astype(int)
    cells = np.array([f"{la}_{lo}" for la, lo in zip(lat_bins, lon_bins)])

    unique_cells = np.unique(cells)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_cells)

    n = len(unique_cells)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_cells = set(unique_cells[:n_train])
    val_cells = set(unique_cells[n_train:n_train + n_val])

    splits = np.empty(len(cells), dtype="U5")
    for i, c in enumerate(cells):
        if c in train_cells:
            splits[i] = "train"
        elif c in val_cells:
            splits[i] = "val"
        else:
            splits[i] = "test"
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=8,
                        help="Select every K-th profile as group center (default: 8)")
    args = parser.parse_args()

    t_start = time.time()
    npy_dir = COLOC_DIR / "npy_v2"
    dense_dir = COLOC_DIR / "npy_dense"
    dense_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load existing npy data
    print("Loading existing npy data...")
    splits_order = ["train", "val", "test"]
    all_patches, all_targets, all_lats, all_lons, all_era5 = [], [], [], [], []
    split_sizes = {}

    for split in splits_order:
        p = np.load(npy_dir / f"{split}_patches.npy", mmap_mode="r")
        t = np.load(npy_dir / f"{split}_targets.npy", mmap_mode="r")
        lat = np.load(npy_dir / f"{split}_lat.npy")
        lon = np.load(npy_dir / f"{split}_lon.npy")
        era5 = np.load(npy_dir / f"{split}_era5.npy", mmap_mode="r")
        split_sizes[split] = len(p)
        all_patches.append(p)
        all_targets.append(t)
        all_lats.append(lat)
        all_lons.append(lon)
        all_era5.append(era5)
        print(f"  {split}: {len(p):,}")

    # Concatenate metadata (keep patches/targets as mmap references)
    cat_lat = np.concatenate(all_lats)
    cat_lon = np.concatenate(all_lons)
    total = len(cat_lat)
    print(f"  Total: {total:,}")

    # Build cumulative offsets for indexing back into per-split arrays
    offsets = {}
    offset = 0
    for split in splits_order:
        offsets[split] = offset
        offset += split_sizes[split]

    def get_split_and_local_idx(global_idx):
        """Convert global index to (split_name, local_index)."""
        cum = 0
        for s in splits_order:
            if global_idx < cum + split_sizes[s]:
                return s, global_idx - cum
            cum += split_sizes[s]
        return splits_order[-1], global_idx - (total - split_sizes[splits_order[-1]])

    # Step 2: Match npy samples to matches_v2 for viirs_row/col metadata
    print("\nMatching npy samples to matches_v2.parquet...")
    df = pd.read_parquet(OUTPUT_DIR / "matches_v2.parquet")
    print(f"  Matches: {len(df):,}")

    tree = KDTree(np.column_stack([df.ec_lat.values, df.ec_lon.values]))
    dists, match_indices = tree.query(np.column_stack([cat_lat, cat_lon]))
    print(f"  Match quality: mean dist={dists.mean():.7f}°, max={dists.max():.7f}°")

    viirs_rows = df["viirs_row"].values[match_indices]
    viirs_cols = df["viirs_col"].values[match_indices]
    l1b_names = df["viirs_l1b"].values[match_indices]
    orbit_ids = df["orbit_id"].values[match_indices]

    # Step 3: Group by (l1b, orbit), select centers, find neighbors
    print(f"\nGrouping by granule+orbit, stride={args.stride}...")

    # Create a DataFrame for efficient grouping
    meta = pd.DataFrame({
        "global_idx": np.arange(total),
        "viirs_l1b": l1b_names,
        "orbit_id": orbit_ids,
        "viirs_row": viirs_rows,
        "viirs_col": viirs_cols,
        "lat": cat_lat,
        "lon": cat_lon,
    })

    patch_groups = []  # list of (center_global_idx, [neighbor_global_indices])

    for (l1b, orbit), group in meta.groupby(["viirs_l1b", "orbit_id"]):
        group = group.sort_values("viirs_row")
        gidx = group["global_idx"].values
        rows = group["viirs_row"].values
        cols = group["viirs_col"].values

        # Select every stride-th as center
        for ci in range(0, len(gidx), args.stride):
            center_gidx = gidx[ci]
            row_c = rows[ci]
            col_c = cols[ci]

            # Find neighbors within ±32 pixels
            row_ok = np.abs(rows - row_c) < PATCH_HALF
            col_ok = np.abs(cols - col_c) < PATCH_HALF
            neighbor_mask = row_ok & col_ok
            neighbor_gidx = gidx[neighbor_mask]
            neighbor_rows = rows[neighbor_mask]
            neighbor_cols = cols[neighbor_mask]

            # Compute relative positions within 64x64 patch
            rel_rows = neighbor_rows - row_c + PATCH_HALF
            rel_cols = neighbor_cols - col_c + PATCH_HALF

            # Bounds check
            valid = (rel_rows >= 0) & (rel_rows < PATCH_SIZE) & \
                    (rel_cols >= 0) & (rel_cols < PATCH_SIZE)

            if valid.sum() < 1:
                continue

            patch_groups.append({
                "center_gidx": center_gidx,
                "neighbor_gidx": neighbor_gidx[valid],
                "rel_rows": rel_rows[valid].astype(np.int16),
                "rel_cols": rel_cols[valid].astype(np.int16),
            })

    n_patches = len(patch_groups)
    n_profs = [len(g["neighbor_gidx"]) for g in patch_groups]
    print(f"  {n_patches:,} patch groups")
    print(f"  Profiles/patch: mean={np.mean(n_profs):.1f}, "
          f"median={np.median(n_profs):.0f}, "
          f"max={np.max(n_profs)}, min={np.min(n_profs)}")

    # Step 4: Build dense arrays
    print("\nBuilding dense arrays...")

    # Pre-allocate output arrays
    out_patches = np.zeros((n_patches, 10, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    out_targets = np.full((n_patches, MAX_PROFILES, N_ACTIVE), -4.0, dtype=np.float32)
    out_positions = np.zeros((n_patches, MAX_PROFILES, 2), dtype=np.int16)
    out_n_profiles = np.zeros(n_patches, dtype=np.int16)
    out_lats = np.zeros(n_patches, dtype=np.float32)
    out_lons = np.zeros(n_patches, dtype=np.float32)
    out_era5 = np.full((n_patches, 26, 4), np.nan, dtype=np.float32)

    for i, g in enumerate(patch_groups):
        center_gidx = g["center_gidx"]
        neighbor_gidx = g["neighbor_gidx"]

        # Get center patch image
        split_c, local_c = get_split_and_local_idx(center_gidx)
        split_idx_c = splits_order.index(split_c)
        out_patches[i] = all_patches[split_idx_c][local_c]
        out_lats[i] = cat_lat[center_gidx]
        out_lons[i] = cat_lon[center_gidx]
        out_era5[i] = all_era5[split_idx_c][local_c]

        # Get all neighbor profiles
        n = min(len(neighbor_gidx), MAX_PROFILES)
        out_n_profiles[i] = n

        for j in range(n):
            ng = neighbor_gidx[j]
            split_n, local_n = get_split_and_local_idx(ng)
            split_idx_n = splits_order.index(split_n)

            # Get IWC profile (already log10-transformed in npy)
            target_full = all_targets[split_idx_n][local_n]  # (242,)
            out_targets[i, j] = target_full[ACTIVE_LEVEL_START:ACTIVE_LEVEL_END]
            out_positions[i, j, 0] = g["rel_rows"][j]
            out_positions[i, j, 1] = g["rel_cols"][j]

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1:,}/{n_patches:,}] {elapsed:.0f}s")

    print(f"  Done building arrays ({time.time() - t_start:.0f}s)")

    # Step 5: Geographic split
    print("\nApplying geographic split...")
    splits = assign_splits(out_lats, out_lons)

    for s in ["train", "val", "test"]:
        print(f"  {s}: {(splits == s).sum():,}")

    # Step 6: Save
    print("\nSaving npy files...")
    for split in ["train", "val", "test"]:
        mask = splits == split
        n = mask.sum()
        print(f"\n  {split}: {n:,} patches")

        np.save(dense_dir / f"{split}_patches.npy", out_patches[mask])
        np.save(dense_dir / f"{split}_targets.npy", out_targets[mask])
        np.save(dense_dir / f"{split}_positions.npy", out_positions[mask])
        np.save(dense_dir / f"{split}_n_profiles.npy", out_n_profiles[mask])
        np.save(dense_dir / f"{split}_lat.npy", out_lats[mask])
        np.save(dense_dir / f"{split}_lon.npy", out_lons[mask])
        np.save(dense_dir / f"{split}_era5.npy", out_era5[mask])

        size_gb = (out_patches[mask].nbytes + out_targets[mask].nbytes) / 1e9
        print(f"    Size: {size_gb:.1f} GB")

        n_prof = out_n_profiles[mask]
        print(f"    Profiles/patch: mean={n_prof.mean():.1f}, "
              f"median={np.median(n_prof):.0f}, max={n_prof.max()}")

    # Summary
    summary = {
        "n_patches": n_patches,
        "stride": args.stride,
        "max_profiles_per_patch": int(MAX_PROFILES),
        "n_active_levels": int(N_ACTIVE),
        "profiles_per_patch_mean": float(np.mean(n_profs)),
        "profiles_per_patch_median": float(np.median(n_profs)),
        "splits": {s: int((splits == s).sum()) for s in ["train", "val", "test"]},
    }
    with open(dense_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Saved to {dense_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
