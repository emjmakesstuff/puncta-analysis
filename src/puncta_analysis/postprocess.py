"""Post-processing: unit conversion, per-punctum tables, and summary output.

Designed for the current single-channel workflow where the headline metrics
are total puncta count and total bright area (in physical units when TIFF
resolution metadata is available).
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .detection import PunctaResults, measure_puncta_blobs
from .config import PunctaConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical-unit conversion
# ---------------------------------------------------------------------------
def convert_area_to_physical(total_bright_pixels: int, pixel_size: dict) -> dict:
    """Convert a bright-pixel area to physical units.

    Parameters
    ----------
    pixel_size : dict with keys x, y, area (area = x*y, in physical units^2).
        If area is 1.0 (default fallback), units are effectively pixels.

    Returns
    -------
    dict with total_area_pixels, total_area_physical, units.
    """
    area_per_pixel = pixel_size.get("area", 1.0) if pixel_size else 1.0
    is_physical = area_per_pixel not in (None, 0, 1.0)
    return {
        "total_area_pixels": int(total_bright_pixels),
        "total_area_physical": float(total_bright_pixels * area_per_pixel),
        "units": "um^2" if is_physical else "pixels",
        "area_per_pixel": float(area_per_pixel),
    }


# ---------------------------------------------------------------------------
# Per-punctum (per-blob) CSV -- mirrors manual Area/Mean/Min/Max
# ---------------------------------------------------------------------------
def write_puncta_table(image: np.ndarray, intensity_threshold: float,
                       config: PunctaConfig, path: str | Path) -> pd.DataFrame:
    """Write per-blob Area/Mean/Min/Max to CSV (mirrors the manual results)."""
    df = measure_puncta_blobs(image, intensity_threshold, config)
    df.to_csv(path, index_label=" ")     # leading blank col like Fiji output
    logger.info("Wrote per-punctum table (%d rows) to %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Per-image result row (for the multi-image summary)
# ---------------------------------------------------------------------------
def build_result_row(
    name: str,
    results: PunctaResults,
    pixel_size: dict,
    config: PunctaConfig,
    similarity: dict | None = None,
    manual_count: int | None = None,
) -> dict:
    """Assemble one summary row for a single processed image."""
    total_bright = results.area_measurements.total_bright
    area_info = convert_area_to_physical(total_bright, pixel_size)

    row = {
        "Image": name,
        "Total_Puncta": results.num_puncta,
        "Total_Area_pixels": area_info["total_area_pixels"],
        "Total_Area_physical": round(area_info["total_area_physical"], 4),
        "Area_Units": area_info["units"],
        "DoG_Sensitivity": round(config.dog_sensitivity, 3),   # the config parameter
        "DoG_Threshold": round(results.detection.threshold, 2),  # computed threshold value
        "Intensity_Threshold": round(results.intensity_threshold, 3),
        "Pixel_Area": area_info["area_per_pixel"],
    }

    # Optional comparison-to-manual columns (only when ground truth available)
    if manual_count is not None:
        row["Manual_Count"] = manual_count
        row["Count_Error_pct"] = round(
            (results.num_puncta - manual_count) / manual_count * 100, 2
        ) if manual_count else np.nan
    if similarity is not None:
        row["Precision"] = round(similarity.get("precision", np.nan), 3)
        row["Recall"] = round(similarity.get("recall", np.nan), 3)
        row["F1_Score"] = round(similarity.get("f1_score", np.nan), 3)
        row["Mean_NN_Distance"] = round(similarity.get("mean_nn", np.nan), 3)

    return row


def write_summary_table(rows: list[dict], path: str | Path) -> pd.DataFrame:
    """Write the multi-image summary table (one row per image) to CSV."""
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Wrote summary table (%d images) to %s", len(df), path)
    return df