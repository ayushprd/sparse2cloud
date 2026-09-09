"""Shared paths, dataset constants and hyperparameters."""

import os
from pathlib import Path
import numpy as np

# Set SPARSE2CLOUD_DIR to keep data and outputs outside the repository.
PROJECT_DIR = Path(os.environ.get("SPARSE2CLOUD_DIR", Path(__file__).parent))
DATA_DIR = PROJECT_DIR / "data"
EC_RAW_DIR = DATA_DIR / "earthcare" / "raw"
EC_PROC_DIR = DATA_DIR / "earthcare" / "processed"
VIIRS_DIR = DATA_DIR / "viirs"
COLOC_DIR = DATA_DIR / "colocation"
ERA5_DIR = DATA_DIR / "era5"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = PROJECT_DIR / "figures"
LOG_DIR = PROJECT_DIR / "logs"

for d in [EC_RAW_DIR, EC_PROC_DIR, VIIRS_DIR, COLOC_DIR, ERA5_DIR,
          OUTPUT_DIR, MODEL_DIR, FIGURE_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PYTHON = os.environ.get("PYTHON", "python")

# Register at https://eoiam-idp.eo.esa.int, then export ESA_USERNAME and
# ESA_PASSWORD before running 01_download_earthcare.py or 40_expand_dataset.py.
ESA_USERNAME = os.environ.get("ESA_USERNAME", "")
ESA_PASSWORD = os.environ.get("ESA_PASSWORD", "")

# EarthCARE product
EC_PRODUCT = "ATL_ICE_2A"
EC_N_LEVELS = 242
EC_FILL_VALUE = 9.96921e+36
EC_EPOCH_STR = "2000-01-01T00:00:00"

# Time periods (seasonal diversity)
PERIODS = [
    ("2025-06-01", "2025-06-17"),   # boreal summer (already downloaded)
    ("2025-01-01", "2025-01-17"),   # boreal winter
    ("2024-12-01", "2024-12-17"),   # early winter / different year
    ("2025-03-01", "2025-03-17"),   # boreal spring / shoulder season
    ("2025-09-01", "2025-09-17"),   # boreal autumn / shoulder season
]
# Period 0 (June) already processed. Periods 1-4 are new.

# VIIRS products
VIIRS_L1B_PRODUCT = "VNP02MOD"     # Suomi NPP moderate-resolution radiances
VIIRS_GEO_PRODUCT = "VNP03MOD"     # Suomi NPP geolocation

# VIIRS channels to extract
VIIRS_THERMAL_BANDS = ["M12", "M13", "M14", "M15", "M16"]
VIIRS_REFL_BANDS = ["M07", "M08", "M10", "M11"]
VIIRS_ALL_BANDS = VIIRS_THERMAL_BANDS + VIIRS_REFL_BANDS
N_VIIRS_CHANNELS = len(VIIRS_ALL_BANDS) + 1  # +1 for SZA = 10

# Patch extraction
PATCH_SIZE = 64           # pixels (64 x 64 at 750m = ~48 km)
PATCH_HALF = PATCH_SIZE // 2

# Co-location parameters
MAX_TIME_OFFSET_SEC = 1800   # 30 minutes
MAX_SPATIAL_DIST_DEG = 0.1   # ~11 km

# Quality filtering
IWC_NOISE_FLOOR = 0.005      # mg/m³ — below this, treat as clear
MIN_ICE_LEVELS = 3            # minimum ice-bearing levels to keep profile

# Target samples
TARGET_SAMPLES = 1_000_000   # total co-located samples to collect
MAX_SAMPLES_PER_DAY = 20_000 # subsample to avoid single-day dominance

# Train/val/test split
SPLIT_SEED = 42
SPLIT_GRID_DEG = 10.0         # geographic grid cell size for spatial split
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Normalization constants (from feasibility)
BT_MEAN = 258.0   # K, approximate mean thermal BT
BT_STD = 20.0     # K
LOG_IWC_EPS = 1e-4  # mg/m³, added before log transform
LOG_IWC_FLOOR = np.log10(LOG_IWC_EPS)  # = -4.0

# Active levels (from analyze_targets.py)
ACTIVE_LEVEL_START = 69
ACTIVE_LEVEL_END = 228
N_ACTIVE_LEVELS = ACTIVE_LEVEL_END - ACTIVE_LEVEL_START  # 159

# Training
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 20
WARMUP_STEPS = 1000
