from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.draw import polygon as sk_polygon
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# loadImageStack
# ---------------------------------------------------------------------------
def load_image_stack(tif_path: str | Path) -> np.ndarray:
    """Load a TIFF image. Returns (H, W) for single-channel, else (H, W, C).

    Port of loadImageStack.m. Since current data is single-channel, this
    returns a 2D array in the common case.
    """
    tif_path = Path(tif_path)
    if not tif_path.exists():
        raise FileNotFoundError(f"File not found: {tif_path}")

    arr = tifffile.imread(str(tif_path))

    if arr.ndim == 2:                          # single-channel image
        logger.info("Loaded image: %d x %d", arr.shape[0], arr.shape[1])
        return arr
    if arr.ndim == 3:                          # (frames/channels, H, W)
        stack = np.moveaxis(arr, 0, -1)        # -> (H, W, C)
        if stack.shape[2] == 1:
            stack = stack[:, :, 0]             # collapse trivial single channel
            logger.info("Loaded image: %d x %d", stack.shape[0], stack.shape[1])
        else:
            logger.info("Loaded stack: %d x %d x %d",
                        stack.shape[0], stack.shape[1], stack.shape[2])
        return stack
    raise ValueError(f"Unexpected TIFF shape {arr.shape}")


# ---------------------------------------------------------------------------
# pixel size from TIFF metadata
# ---------------------------------------------------------------------------
def read_pixel_size(tif_path: str | Path, default: float = 1.0) -> dict:
    """Read X/Y resolution from TIFF tags; fall back to `default` if missing."""
    x = y = default
    try:
        with tifffile.TiffFile(str(tif_path)) as tf:
            tags = tf.pages[0].tags
            xr = tags.get("XResolution")
            yr = tags.get("YResolution")
            if xr is not None and yr is not None:
                xr_v = _rational_to_float(xr.value)
                yr_v = _rational_to_float(yr.value)
                if xr_v and yr_v and xr_v > 0 and yr_v > 0:
                    x, y = xr_v, yr_v
                    logger.info("Pixel size: %.4f x %.4f (area %.6f)", x, y, x * y)
                else:
                    logger.warning("No valid resolution; using default %s", default)
            else:
                logger.warning("No resolution metadata; using default %s", default)
    except Exception as exc:                   # pragma: no cover
        logger.warning("Could not read resolution (%s); using default %s",
                       exc, default)
    return {"x": x, "y": y, "area": x * y}


def _rational_to_float(val) -> float | None:
    try:
        if isinstance(val, (tuple, list)) and len(val) == 2:
            num, den = val
            return float(num) / float(den) if den else None
        return float(val)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# Fiji ROI loading (.roi / .zip) -> labeled mask + centroids
# ---------------------------------------------------------------------------
def load_fiji_roi(roi_path: str | Path,
                  image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Load Fiji/ImageJ ROI file(s) into a labeled mask and their centroids.

    Each ROI in the file is treated as one manually-counted punctum.

    Returns
    -------
    roi_mask : (H, W) uint16 array, 0 = background, 1..N = ROI labels.
    centroids : (N, 2) array of ROI centroids as [x, y].
    """
    import roifile

    roi_path = Path(roi_path)
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI file not found: {roi_path}")

    h, w = image_size
    roi_mask = np.zeros((h, w), dtype=np.uint16)

    rois = roifile.ImagejRoi.fromfile(str(roi_path))
    if not isinstance(rois, list):
        rois = [rois]

    logger.info("Loading %d ROI(s) from: %s", len(rois), roi_path.name)

    for lbl, roi in enumerate(rois, start=1):
        try:
            coords = roi.coordinates()          # (N, 2) as (x, y)
            _add_roi_to_mask(roi_mask, coords, lbl, (h, w))
        except Exception as exc:
            logger.warning("Failed to rasterize ROI %d: %s", lbl, exc)

    # Centroids from the rasterized mask (consistent with detection centroids)
    centroids = _mask_centroids(roi_mask)
    logger.info("Loaded %d ROI(s); %d centroids computed",
                int(roi_mask.max()), centroids.shape[0])
    return roi_mask, centroids


def load_manual_centroids(roi_path: str | Path,
                          image_size: tuple[int, int]) -> np.ndarray:
    """Convenience: return just the (N, 2) manual centroids [x, y] from ROIs."""
    _, centroids = load_fiji_roi(roi_path, image_size)
    return centroids


def _add_roi_to_mask(roi_mask: np.ndarray, coords: np.ndarray,
                     label_value: int, image_size: tuple[int, int]) -> None:
    """Rasterize one polygon into the mask (port of addROIToMask.m).

    Fiji coords are (x, y); skimage.draw.polygon wants (rows=y, cols=x).
    """
    x = coords[:, 0]
    y = coords[:, 1]
    rr, cc = sk_polygon(y, x, shape=image_size)
    roi_mask[rr, cc] = label_value


def _mask_centroids(roi_mask: np.ndarray) -> np.ndarray:
    """Centroid [x, y] of each labeled region in a mask."""
    props = regionprops(roi_mask)
    if not props:
        return np.empty((0, 2))
    # regionprops centroid is (row, col) -> flip to (x, y)
    return np.array([(p.centroid[1], p.centroid[0]) for p in props],
                    dtype=np.float64)


# ---------------------------------------------------------------------------
# Manual results CSV (Area/Mean/Min/Max) -- count cross-check + stats
# ---------------------------------------------------------------------------
def load_manual_results_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a Fiji 'Analyze Particles' results CSV (Area, Mean, Min, Max).

    The number of rows = number of manually-counted puncta. Coordinates are
    NOT in this file (they live in the ROI zip); this is for count
    cross-checking and per-punctum area/intensity stats.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Loaded manual results CSV: %d puncta", len(df))
    return df


# ---------------------------------------------------------------------------
# load_analysis_data -- path-driven bundle
# ---------------------------------------------------------------------------
def load_analysis_data(
    tif_path: str | Path,
    csv_path: str | Path | None = None,
    roi_path: str | Path | None = None,
    *,
    default_pixel_size: float = 1.0,
) -> dict:
    """Load image + optional manual results CSV + optional ROI zip.

    Non-interactive, path-driven replacement for loadAnalysisData.m.
    Returns a dict with keys:
      image, pixel_size, roi_mask, roi_centroids (manual positions),
      manual_results (DataFrame), filename.
    """
    tif_path = Path(tif_path)
    data: dict = {"filename": str(tif_path)}

    image = load_image_stack(tif_path)
    data["image"] = image
    data["pixel_size"] = read_pixel_size(tif_path, default=default_pixel_size)

    h, w = image.shape[:2]

    if roi_path is not None and Path(roi_path).exists():
        try:
            roi_mask, roi_centroids = load_fiji_roi(roi_path, (h, w))
            data["roi_mask"] = roi_mask
            data["roi_centroids"] = roi_centroids
        except Exception as exc:
            logger.warning("Failed to load ROI file: %s", exc)
            data["roi_mask"] = None
            data["roi_centroids"] = None
    else:
        data["roi_mask"] = None
        data["roi_centroids"] = None

    if csv_path is not None and Path(csv_path).exists():
        try:
            data["manual_results"] = load_manual_results_csv(csv_path)
        except Exception as exc:
            logger.warning("Failed to load manual CSV: %s", exc)
            data["manual_results"] = None
    else:
        data["manual_results"] = None

    # Cross-check: CSV row count vs. number of ROIs
    if data.get("manual_results") is not None and data.get("roi_centroids") is not None:
        n_csv = len(data["manual_results"])
        n_roi = data["roi_centroids"].shape[0]
        if n_csv != n_roi:
            logger.warning("Manual count mismatch: CSV has %d rows but ROI zip "
                           "has %d regions", n_csv, n_roi)
        else:
            logger.info("Manual count cross-check OK: %d puncta", n_csv)

    return data