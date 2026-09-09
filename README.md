<h1 align="center">sparse2cloud</h1>

<p align="center">
  <b>Ice Cloud Geometry Retrieval with Calibrated Uncertainty from Passive Satellite Imagery</b><br>
  Ayush Prasad · <b>ECCV 2026</b>
</p>

<p align="center">
  <a href="https://eccv.ecva.net/virtual/2026/poster/5781"><img src="https://img.shields.io/badge/ECCV-2026-1b6675" alt="ECCV 2026"></a>
  <a href="https://ayushprasad.com/projects/sparse2cloud/"><img src="https://img.shields.io/badge/Project-Page-1b6675" alt="Project page"></a>
  <a href="https://github.com/ayushprd/sparse2cloud"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white" alt="Code"></a>
  <img src="https://img.shields.io/badge/license-MIT-informational" alt="License">
</p>

---

Code for the ECCV 2026 spotlight paper. EarthCARE's radar and lidar measure cloud
structure in the vertical, but only under the flight path, which is under one
percent of the planet a day. This trains a ConvNeXt U-Net on co-located VIIRS
patches where labels land on about one percent of pixels, and predicts eight
cloud geometry targets at every pixel with 90 % prediction intervals calibrated
by conformalized quantile regression.


## Overview

Ten VIIRS channels go in (thermal M12-M16, reflective M07/M08/M10/M11, and solar
zenith angle). A three-stage ConvNeXt encoder runs down to 384 channels, ERA5
reanalysis is projected and added at the bottleneck, and a mirrored decoder
returns eight targets at every pixel, each as three quantiles trained with a
pinball loss. The loss is computed only on the labelled pixels; inference runs on
all 64x64. The reported model averages five independently seeded copies.

The eight targets are cloud centroid height, cloud top height, cloud base
height, peak ice concentration level, geometric thickness, core IWC, column-mean
IWC and log IWP.

## Install

```bash
pip install -r requirements.txt
```

`config.py` holds every path and constant. By default data and outputs live
inside the repository; set `SPARSE2CLOUD_DIR` to put them elsewhere. EarthCARE
downloads need ESA credentials in the environment:

```bash
export SPARSE2CLOUD_DIR=/scratch/sparse2cloud
export ESA_USERNAME=... ESA_PASSWORD=...
```

## Data preparation

Runs in numeric order. The numbering follows the working order of the project,
so there are gaps where superseded steps were dropped.

```bash
python 01_download_earthcare.py       # ATL_ICE_2A granules -> parquet
python 02_viirs_index.py              # VIIRS granule index via earthaccess
python 03_match_and_extract_v2.py     # KDTree co-location + 64x64 patch extraction
python 40_expand_dataset.py           # additional seasonal periods
python 41_add_jan_diversity.py        # boreal winter period
python 11_download_era5.py            # ERA5 pressure-level fields
python 12_collocate_era5.py           # ERA5 profiles at each patch centre
python 30_extract_dense_patches.py    # dense per-pixel dataset
python 48_extract_geometry_targets.py # the eight geometry targets
python convert_to_npy.py              # preload arrays; zarr random access is the bottleneck
```

## Training

```bash
python 47_train_geometry_quantile.py --tag G-Q --mode quantile --seed 42
```

Five seeds make the ensemble reported in the paper:

```bash
bash run_quantile_ensemble.sh
```

Uncertainty baselines:

```bash
python 52_train_uncertainty_baselines.py --tag G-HET --mode heteroscedastic
python 52_train_uncertainty_baselines.py --tag G-EVI --mode evidential
python 52_train_uncertainty_baselines.py --tag G-MCD --mode mc-dropout
python 52_train_uncertainty_baselines.py --tag G-DE  --mode ensemble
```

## Calibration and evaluation

Every method goes through the identical conformal step, so the comparison is on
equal terms:

```bash
python 49_calibrate_cqr.py --model-tag G-Q --alphas 0.1
python 53_evaluate_uncertainty.py --model-tag G-QE --mode quantile_ensemble \
    --ensemble-tags G-Q,G-Q2,G-Q3,G-Q4,G-Q5
```

The full evaluation, including the downstream tasks and figures:

```bash
bash run_qe_eval.sh
```

Other analyses:

```bash
python 57_posthoc_physics.py          # ordering projection after training
python 58_crosstrack_uncertainty.py   # quality against distance from the track
python 65_conditional_coverage.py     # coverage by latitude and cloud type
python 64_mlp_geometry_baseline.py    # per-pixel MLP baseline
python 66_bt_cth_baseline.py          # brightness-temperature ridge baseline
```

## Global inference

One day of VIIRS, stride-32 overlapping tiles averaged and gridded at 0.25
degrees:

```bash
python 62_dense_global_map.py --date 2025-06-01
```

## Figures

```bash
python 60_eccv_figures.py
python 61_spatial_figures.py
python 63_overview_figure.py
```

## Citation

```bibtex
@inproceedings{prasad2026icecloud,
  title     = {Ice Cloud Geometry Retrieval with Calibrated Uncertainty
               from Passive Satellite Imagery},
  author    = {Prasad, Ayush},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

MIT. EarthCARE data is from ESA, VIIRS from NASA, and ERA5 from ECMWF, each
under its own terms.
