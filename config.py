"""
config.py
Central configuration for the Flood Inundation Mapping project.
All paths, hyperparameters, and settings live here.
Edit this file before running any notebook or script.
"""

import os

# ─────────────────────────────────────────────
# Google Drive Root (update this to your Drive path)
# ─────────────────────────────────────────────
DRIVE_ROOT = "/content/drive/MyDrive/flood_inundation_project"

# Sub-directories (auto-created by setup)
DATA_DIR        = os.path.join(DRIVE_ROOT, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
PATCHES_DIR     = os.path.join(DATA_DIR, "patches")
MODELS_DIR      = os.path.join(DRIVE_ROOT, "saved_models")
RESULTS_DIR     = os.path.join(DRIVE_ROOT, "results")
LOGS_DIR        = os.path.join(DRIVE_ROOT, "logs")

# ─────────────────────────────────────────────
# Google Earth Engine Study Area
# ─────────────────────────────────────────────
# Bounding box for study area (default: Assam, India - flood-prone)
STUDY_AREA = {
    "lon_min": 89.5,
    "lat_min": 25.0,
    "lon_max": 95.0,
    "lat_max": 28.5,
}

# Date ranges for flood event (modify per event)
FLOOD_START   = "2022-06-01"
FLOOD_END     = "2022-09-30"
PREFLOOD_START = "2022-01-01"
PREFLOOD_END   = "2022-05-31"

# Number of time steps for temporal model
TIME_STEPS = 6

# ─────────────────────────────────────────────
# SAR (Sentinel-1) Settings
# ─────────────────────────────────────────────
SAR_COLLECTION    = "COPERNICUS/S1_GRD"
SAR_POLARIZATIONS = ["VV", "VH"]
SAR_PASS_DIRECTION = "DESCENDING"
SAR_SPECKLE_FILTER = "LEE"       # options: "LEE", "REFINED_LEE", "NONE"
SAR_SPECKLE_WINDOW = 7

# ─────────────────────────────────────────────
# Optical (Sentinel-2) Settings
# ─────────────────────────────────────────────
OPT_COLLECTION  = "COPERNICUS/S2_SR_HARMONIZED"
OPT_CLOUD_THR   = 20              # max cloud cover %
OPT_BANDS       = ["B2", "B3", "B4", "B8", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2
OPT_SCALE       = 10              # metres per pixel

# ─────────────────────────────────────────────
# DEM & HAND Settings
# ─────────────────────────────────────────────
DEM_COLLECTION  = "USGS/SRTMGL1_003"   # SRTM 30m
HAND_COLLECTION = "MERIT/Hydro/v1_0_1" # MERIT Hydro (contains HAND)

# ─────────────────────────────────────────────
# Rainfall Data Settings
# ─────────────────────────────────────────────
RAINFALL_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"  # CHIRPS rainfall

# ─────────────────────────────────────────────
# Patch Generation Settings
# ─────────────────────────────────────────────
PATCH_SIZE    = 256        # pixels × pixels
PATCH_STRIDE  = 128        # overlap stride
MIN_FLOOD_RATIO = 0.02     # discard patches with < 2% flood pixels
MAX_NODATA_RATIO = 0.1     # discard patches with > 10% nodata

# ─────────────────────────────────────────────
# Model Architecture Settings
# ─────────────────────────────────────────────
# Input channels: VV, VH, B2, B3, B4, B8, B11, B12, DEM, HAND, Rainfall = 11
IN_CHANNELS   = 11
OUT_CHANNELS  = 1          # binary: flood / no-flood
BASE_FEATURES = 32         # base feature map count (lightweight)

# ConvLSTM specific
CONVLSTM_HIDDEN  = [32, 64]      # hidden channels per layer
CONVLSTM_KERNEL  = 3

# ─────────────────────────────────────────────
# Training Hyperparameters
# ─────────────────────────────────────────────
BATCH_SIZE     = 8
NUM_EPOCHS     = 50
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
LR_PATIENCE    = 5          # reduce LR after N stagnant epochs
ES_PATIENCE    = 10         # early stopping patience
TRAIN_VAL_TEST = (0.7, 0.15, 0.15)
RANDOM_SEED    = 42

# Loss function: "bce", "dice", "combined"
LOSS_FUNCTION  = "combined"
BCE_WEIGHT     = 0.5
DICE_WEIGHT    = 0.5

# ─────────────────────────────────────────────
# Model to train: "unet", "unet_pp", "convlstm", "spatiotemporal"
# ─────────────────────────────────────────────
ACTIVE_MODEL   = "spatiotemporal"

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
THRESHOLD = 0.5    # sigmoid threshold for binary prediction

# ─────────────────────────────────────────────
# Miscellaneous
# ─────────────────────────────────────────────
NUM_WORKERS = 2    # DataLoader workers (keep low on Colab)
PIN_MEMORY  = True
DEVICE      = "cuda"   # will fall back to cpu automatically in trainer

def print_config():
    """Print all config settings."""
    import json
    settings = {k: v for k, v in globals().items()
                if not k.startswith("_") and not callable(v) and k == k.upper()}
    print(json.dumps(settings, indent=2))
