"""
01_data_collection/gee_data_collector.py

Downloads multi-source remote sensing data from Google Earth Engine:
  - Sentinel-1 SAR (VV, VH)
  - Sentinel-2 Optical (6 bands)
  - SRTM DEM
  - MERIT Hydro HAND
  - CHIRPS Rainfall

Exports GeoTIFFs to Google Drive for further processing.
"""

import ee
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


def get_study_area():
    """Return the study area as an EE geometry."""
    return ee.Geometry.Rectangle([
        cfg.STUDY_AREA["lon_min"],
        cfg.STUDY_AREA["lat_min"],
        cfg.STUDY_AREA["lon_max"],
        cfg.STUDY_AREA["lat_max"],
    ])


# ─────────────────────────────────────────────────────────────
# SAR: Sentinel-1
# ─────────────────────────────────────────────────────────────

def apply_lee_filter(image, window=7):
    """
    Apply Lee speckle filter to a SAR image.
    Uses a focal mean + focal variance approach (pure EE, no external libs).
    """
    band_names = image.bandNames()
    def filter_band(band_name):
        band = image.select([band_name])
        mean   = band.focal_mean(window, "square", "pixels")
        mean_sq = band.pow(2).focal_mean(window, "square", "pixels")
        variance = mean_sq.subtract(mean.pow(2))
        noise_var = variance.reduce(ee.Reducer.mean())
        weight = variance.subtract(noise_var).divide(variance.add(1e-10))
        filtered = mean.add(weight.multiply(band.subtract(mean)))
        return filtered.rename(band_name)
    filtered_bands = ee.Image.cat(
        [filter_band(b) for b in cfg.SAR_POLARIZATIONS]
    )
    return filtered_bands


def get_sar_image(start_date, end_date, aoi, apply_filter=True):
    """
    Retrieve and mosaic Sentinel-1 SAR image for a date range.
    Returns: EE Image with VV and VH bands (in dB scale, then converted to linear).
    """
    collection = (
        ee.ImageCollection(cfg.SAR_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", cfg.SAR_PASS_DIRECTION))
        .select(cfg.SAR_POLARIZATIONS)
    )

    print(f"  SAR images found: {collection.size().getInfo()}")

    sar = collection.median()  # temporal median to reduce noise

    if apply_filter and cfg.SAR_SPECKLE_FILTER == "LEE":
        sar = apply_lee_filter(sar, cfg.SAR_SPECKLE_WINDOW)

    # Clip to study area
    return sar.clip(aoi)


# ─────────────────────────────────────────────────────────────
# Optical: Sentinel-2
# ─────────────────────────────────────────────────────────────

def mask_clouds_s2(image):
    """Mask clouds and cloud shadows using Sentinel-2 QA60 band."""
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )
    return image.updateMask(mask).divide(10000)  # scale reflectance


def get_optical_image(start_date, end_date, aoi):
    """
    Retrieve cloud-masked Sentinel-2 optical composite.
    Returns: EE Image with 6 bands.
    """
    collection = (
        ee.ImageCollection(cfg.OPT_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.OPT_CLOUD_THR))
        .map(mask_clouds_s2)
        .select(cfg.OPT_BANDS)
    )

    print(f"  Optical images found: {collection.size().getInfo()}")
    return collection.median().clip(aoi)


# ─────────────────────────────────────────────────────────────
# DEM: SRTM
# ─────────────────────────────────────────────────────────────

def get_dem(aoi):
    """Return SRTM DEM clipped to study area."""
    dem = ee.Image(cfg.DEM_COLLECTION).select("elevation").clip(aoi)
    return dem


# ─────────────────────────────────────────────────────────────
# HAND: Height Above Nearest Drainage
# ─────────────────────────────────────────────────────────────

def get_hand(aoi):
    """
    Return HAND from MERIT Hydro dataset.
    Band 'hnd' = Height Above Nearest Drainage (metres).
    """
    hand = (
        ee.Image(cfg.HAND_COLLECTION)
        .select("hnd")
        .clip(aoi)
    )
    return hand


# ─────────────────────────────────────────────────────────────
# Rainfall: CHIRPS
# ─────────────────────────────────────────────────────────────

def get_rainfall(start_date, end_date, aoi):
    """
    Return cumulative rainfall over the period (CHIRPS daily).
    """
    rainfall = (
        ee.ImageCollection(cfg.RAINFALL_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .sum()    # cumulative mm
        .clip(aoi)
    )
    return rainfall


# ─────────────────────────────────────────────────────────────
# Flood Reference Mask (Otsu thresholding on SAR)
# ─────────────────────────────────────────────────────────────

def generate_flood_mask(flood_sar, preflood_sar, aoi):
    """
    Generate a binary flood mask by comparing flood-period SAR
    with pre-flood SAR using a change-detection approach.

    Strategy:
      - Compute difference (flood - preflood) on VV band
      - Pixels below a fixed threshold are classified as flooded
      - Apply terrain mask (slopes > 5° unlikely to be water)

    Returns: Binary EE Image (1=flood, 0=non-flood)
    """
    diff = flood_sar.select("VV").subtract(preflood_sar.select("VV"))

    # Simple threshold: SAR decreases over open water
    flood_mask = diff.lt(-3.0).rename("flood_mask")  # -3 dB change

    # Terrain mask: exclude steep slopes
    dem = get_dem(aoi)
    slope = ee.Terrain.slope(dem)
    terrain_mask = slope.lt(5)

    flood_mask = flood_mask.updateMask(terrain_mask)

    # Morphological cleaning — focal mode to remove speckle
    flood_mask = flood_mask.focal_mode(radius=3, kernelType="square", units="pixels")

    return flood_mask.clip(aoi)


# ─────────────────────────────────────────────────────────────
# Temporal Sequence (for ConvLSTM)
# ─────────────────────────────────────────────────────────────

def get_sar_time_series(aoi, n_steps=6):
    """
    Get a monthly SAR time series for temporal modeling.
    Returns a list of n_steps SAR images covering the flood season.
    """
    import datetime

    start = datetime.datetime.strptime(cfg.FLOOD_START, "%Y-%m-%d")
    images = []

    for i in range(n_steps):
        month_start = (start.replace(day=1) +
                       datetime.timedelta(days=30 * i)).strftime("%Y-%m-%d")
        month_end   = (start.replace(day=1) +
                       datetime.timedelta(days=30 * (i + 1))).strftime("%Y-%m-%d")
        img = get_sar_image(month_start, month_end, aoi, apply_filter=True)
        images.append(img)
        print(f"  Time step {i+1}: {month_start} to {month_end}")

    return images


# ─────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────

def export_to_drive(image, description, folder, region, scale=10):
    """
    Submit an EE export task to Google Drive.
    Call ee_task.start() to begin the export.
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        maxPixels=1e13,
        crs="EPSG:4326",
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"[EXPORT STARTED] {description} → Drive/{folder}/")
    return task


def export_all(aoi=None):
    """
    Export all data layers to Google Drive.
    Run this once; it queues EE tasks (check status in EE Code Editor).
    """
    if aoi is None:
        aoi = get_study_area()

    region = aoi.getInfo()["coordinates"]

    print("\n── SAR (Flood Period) ──")
    flood_sar = get_sar_image(cfg.FLOOD_START, cfg.FLOOD_END, aoi)
    export_to_drive(flood_sar, "flood_sar", "flood_raw", region, scale=10)

    print("\n── SAR (Pre-flood Period) ──")
    preflood_sar = get_sar_image(cfg.PREFLOOD_START, cfg.PREFLOOD_END, aoi)
    export_to_drive(preflood_sar, "preflood_sar", "flood_raw", region, scale=10)

    print("\n── Optical (Flood Period) ──")
    flood_opt = get_optical_image(cfg.FLOOD_START, cfg.FLOOD_END, aoi)
    export_to_drive(flood_opt, "flood_optical", "flood_raw", region, scale=10)

    print("\n── DEM ──")
    dem = get_dem(aoi)
    export_to_drive(dem, "srtm_dem", "flood_raw", region, scale=30)

    print("\n── HAND ──")
    hand = get_hand(aoi)
    export_to_drive(hand, "hand_index", "flood_raw", region, scale=30)

    print("\n── Rainfall ──")
    rain = get_rainfall(cfg.FLOOD_START, cfg.FLOOD_END, aoi)
    export_to_drive(rain, "chirps_rainfall", "flood_raw", region, scale=5000)

    print("\n── Flood Mask (Label) ──")
    flood_mask = generate_flood_mask(flood_sar, preflood_sar, aoi)
    export_to_drive(flood_mask, "flood_label_mask", "flood_raw", region, scale=10)

    print("\n[ALL EXPORTS QUEUED] Monitor at: https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    ee.Initialize()
    export_all()
