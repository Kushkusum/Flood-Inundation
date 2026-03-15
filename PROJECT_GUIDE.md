# Flood Inundation Mapping — Project Guide

**Title:** Spatiotemporal Deep Learning Framework for Multi-Source Flood Inundation Mapping Using SAR and Optical Data  
**Funded by:** KSCST  
**Platform:** Google Colab (T4 GPU) + Google Drive + Google Earth Engine

---

## Folder Structure

```
flood_inundation_project/
│
├── config.py                        ← ALL settings live here (edit this first)
│
├── 01_data_collection/
│   └── gee_data_collector.py        ← GEE data download + flood mask generation
│
├── 02_preprocessing/
│   └── preprocessor.py              ← SAR filter, cloud fill, stack all 11 bands
│
├── 03_dataset/
│   ├── patch_generator.py           ← Sliding-window patch extraction
│   └── flood_dataset.py             ← PyTorch Dataset + DataLoader factory
│
├── 04_models/
│   ├── unet.py                      ← Lightweight U-Net (spatial baseline)
│   ├── unet_pp.py                   ← U-Net++ with dense skip connections
│   ├── convlstm.py                  ← ConvLSTM cell, stack, standalone predictor
│   └── spatiotemporal.py            ← FULL MODEL: U-Net encoder + ConvLSTM + decoder
│
├── 05_training/
│   ├── losses.py                    ← BCE, Dice, Combined loss functions
│   ├── metrics.py                   ← IoU, F1, Precision, Recall tracker
│   └── trainer.py                   ← Training loop, early stopping, LR scheduler
│
├── 06_evaluation/
│   └── evaluator.py                 ← Test-set evaluation, flood map visualization
│
├── 07_inference/
│   └── predict.py                   ← Full-scene inference, GeoTIFF export, map
│
├── utils/
│   └── setup_colab.py               ← Drive mount, dir creation, GEE auth helpers
│
└── notebooks/                       ← Open these in Google Colab (in order)
    ├── 01_Setup_and_Data_Collection.ipynb
    ├── 02_Preprocessing.ipynb
    ├── 03_Training.ipynb
    ├── 04_Evaluation.ipynb
    └── 05_Inference.ipynb
```

---

## Step-by-Step Usage

### Step 1: Upload to Google Drive
Copy the entire `flood_inundation_project/` folder to:
```
My Drive/flood_inundation_project/
```

### Step 2: Edit config.py
Open `config.py` and set:
- `STUDY_AREA` — your region of interest (default: Assam, India)
- `FLOOD_START`, `FLOOD_END` — flood event dates
- `ACTIVE_MODEL` — which model to train (`unet`, `unet_pp`, `convlstm`, `spatiotemporal`)

### Step 3: Run Notebooks in Order
| Notebook | What it does | Time |
|---|---|---|
| `01_Setup_and_Data_Collection` | GEE auth, download SAR/optical/DEM/HAND/rainfall | 5 min + 30–60 min GEE export |
| `02_Preprocessing` | Speckle filter, normalization, patch generation | 10–20 min |
| `03_Training` | Train the chosen model (GPU required) | 30–120 min |
| `04_Evaluation` | Test metrics, flood map visualization | 5 min |
| `05_Inference` | Full-scene inference, GeoTIFF export | 5 min |

---

## Input Data (11 Channels)
| # | Source | Band | Description |
|---|---|---|---|
| 0 | Sentinel-1 | VV | SAR vertical-vertical backscatter |
| 1 | Sentinel-1 | VH | SAR vertical-horizontal backscatter |
| 2 | Sentinel-2 | B2 | Blue (492 nm) |
| 3 | Sentinel-2 | B3 | Green (560 nm) |
| 4 | Sentinel-2 | B4 | Red (665 nm) |
| 5 | Sentinel-2 | B8 | NIR (833 nm) |
| 6 | Sentinel-2 | B11 | SWIR-1 (1614 nm) |
| 7 | Sentinel-2 | B12 | SWIR-2 (2202 nm) |
| 8 | SRTM | elevation | Digital Elevation Model (metres) |
| 9 | MERIT Hydro | hnd | Height Above Nearest Drainage (metres) |
| 10 | CHIRPS | rainfall | Cumulative rainfall (mm) |

---

## Models
| Model | Type | Input | Best For |
|---|---|---|---|
| U-Net | Spatial | (B, 11, H, W) | Fast baseline |
| U-Net++ | Spatial | (B, 11, H, W) | Better flood boundaries |
| ConvLSTM | Temporal | (B, T, 11, H, W) | Temporal dynamics |
| SpatiotemporalFloodNet | Both | (B, T, 11, H, W) | **Proposed model — best results** |

---

## Evaluation Metrics
- **IoU** (Intersection over Union): primary metric
- **F1-Score** (Dice): main secondary metric
- **Precision**: false alarm rate
- **Recall**: flood detection rate
- **Pixel Accuracy**: overall

---

## Outputs
After running all notebooks, your Drive will contain:
```
flood_inundation_project/
├── data/
│   ├── raw/               ← Downloaded GeoTIFFs from GEE
│   ├── processed/         ← feature_stack.npy + label_mask.npy
│   └── patches/           ← Training patches (images/ + masks/)
├── saved_models/          ← Best model checkpoints (.pt files)
├── logs/                  ← Training history CSVs
└── results/
    ├── flood_maps.png           ← Visualisation of predictions
    ├── training_curves.png      ← Loss/IoU curves
    ├── flood_probability.tif    ← GeoTIFF for QGIS
    ├── flood_binary_mask.tif    ← Binary flood map
    └── flood_map_interactive.html  ← Web map
```

---

## Hardware Requirements
- Google Colab free tier (T4 GPU) is sufficient
- Patch size 256×256, batch size 8 uses ~3–4 GB VRAM
- Reduce `BATCH_SIZE` to 4 or `PATCH_SIZE` to 128 if you run out of memory

---

## Common Issues & Fixes
| Issue | Fix |
|---|---|
| GEE export fails | Check study area is not too large; reduce resolution |
| CUDA out of memory | Reduce `BATCH_SIZE` in config.py |
| No flood pixels in patches | Lower `MIN_FLOOD_RATIO` in config.py |
| Raw files missing | Copy from `Drive/flood_raw/` to `data/raw/` |
| Drive disconnects | Re-run the mount cell; Colab auto-saves checkpoints |
