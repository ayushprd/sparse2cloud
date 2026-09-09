"""PyTorch dataset for VIIRS patch → IWC profile prediction."""

import numpy as np
import zarr
import torch
from torch.utils.data import Dataset

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    EC_N_LEVELS, N_VIIRS_CHANNELS, PATCH_SIZE,
    BT_MEAN, BT_STD, LOG_IWC_EPS,
    VIIRS_THERMAL_BANDS, VIIRS_REFL_BANDS,
)


N_THERMAL = len(VIIRS_THERMAL_BANDS)
N_REFL = len(VIIRS_REFL_BANDS)


def _load_split_indices(store, split):
    """Get indices for a given split."""
    if split != "all" and "meta/split" in store:
        splits = store["meta/split"][:]
        if hasattr(splits[0], 'decode'):
            splits = np.array([s.decode() for s in splits])
        return np.where(splits == split)[0]
    return np.arange(store["patches"].shape[0])


class IWCDataset(Dataset):
    """Dataset that loads VIIRS patches and IWC profile targets from zarr.

    Preloads all data into RAM for fast training.

    Normalizes inputs:
    - Thermal bands (M12-M16): (BT - BT_MEAN) / BT_STD
    - Reflective bands (M07, M08, M10, M11): as-is (already 0-1 scale)
    - SZA: sza / 90.0 (normalize to ~[0, 2])

    Transforms targets:
    - IWC: log10(IWC + eps) → range [-4, ~2]
    """

    def __init__(self, zarr_path, split="train", augment=False, patch_size=None):
        store = zarr.open(str(zarr_path), mode="r")
        indices = _load_split_indices(store, split)

        # Preload into RAM — ~8 GB for 53K patches, fits easily
        print(f"  Loading {len(indices)} {split} patches into RAM...", flush=True)
        self.patches = store["patches"][indices].astype(np.float32)
        self.targets = store["targets"][indices].astype(np.float32)

        # Normalize in bulk (vectorized)
        self.patches[:, :N_THERMAL] = (self.patches[:, :N_THERMAL] - BT_MEAN) / BT_STD
        self.patches[:, N_THERMAL + N_REFL] = self.patches[:, N_THERMAL + N_REFL] / 90.0

        # Log-transform targets in bulk
        self.targets = np.log10(self.targets + LOG_IWC_EPS)

        self.augment = augment
        self.patch_size = patch_size or PATCH_SIZE

        # Load metadata if available
        self.meta = {}
        for key in ["sza", "ec_lat", "ec_lon"]:
            full_key = f"meta/{key}"
            if full_key in store:
                self.meta[key] = store[full_key][indices]

        print(f"  Loaded: patches {self.patches.shape}, targets {self.targets.shape}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx].copy()
        target = self.targets[idx]

        # Spatial ablation: center crop
        if self.patch_size < PATCH_SIZE:
            offset = (PATCH_SIZE - self.patch_size) // 2
            patch = patch[:, offset:offset + self.patch_size,
                         offset:offset + self.patch_size]

        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                patch = patch[:, :, ::-1].copy()
            if np.random.random() > 0.5:
                patch = patch[:, ::-1, :].copy()

        return torch.from_numpy(patch), torch.from_numpy(target)


class PixelDataset(Dataset):
    """Dataset that extracts only the center pixel for MLP baselines.

    Preloads data into RAM.
    """

    def __init__(self, zarr_path, split="train", context_size=1):
        store = zarr.open(str(zarr_path), mode="r")
        indices = _load_split_indices(store, split)

        print(f"  Loading {len(indices)} {split} pixel features...", flush=True)

        # Load patches and extract features immediately
        patches = store["patches"][indices].astype(np.float32)
        targets = store["targets"][indices].astype(np.float32)

        center = PATCH_SIZE // 2
        h = context_size // 2

        if context_size == 1:
            self.features = patches[:, :, center, center]  # (N, C)
        else:
            region = patches[:, :, center-h:center+h+1, center-h:center+h+1]
            self.features = np.concatenate([
                region.mean(axis=(2, 3)),
                region.std(axis=(2, 3)),
                region.min(axis=(2, 3)),
                region.max(axis=(2, 3)),
            ], axis=1)  # (N, C*4)

        # Normalize thermal bands
        self.features[:, :N_THERMAL] = \
            (self.features[:, :N_THERMAL] - BT_MEAN) / BT_STD
        if context_size > 1:
            for off in [N_VIIRS_CHANNELS, 2*N_VIIRS_CHANNELS, 3*N_VIIRS_CHANNELS]:
                self.features[:, off:off + N_THERMAL] = \
                    (self.features[:, off:off + N_THERMAL] - BT_MEAN) / BT_STD
        # SZA
        self.features[:, N_THERMAL + N_REFL] = \
            self.features[:, N_THERMAL + N_REFL] / 90.0

        # Log-transform targets
        self.targets = np.log10(targets + LOG_IWC_EPS)

        del patches  # free the full patches
        print(f"  Loaded: features {self.features.shape}, targets {self.targets.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.features[idx]),
                torch.from_numpy(self.targets[idx]))
