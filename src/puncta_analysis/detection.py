from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import logging

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from skimage.morphology import local_maxima, remove_small_objects
from skimage.measure import label, regionprops

from .config import PunctaConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers (replace MATLAB structs)
# ---------------------------------------------------------------------------
@dataclass
class DetectionResult:
    centroids: np.ndarray            # (N, 2) as [x, y]
    areas: np.ndarray                # (N,)
    dog_image: np.ndarray
    binary_mask: np.ndarray
    num_puncta: int
    peak_values: np.ndarray          # (N,)
    threshold: float
    stats: dict = field(default_factory=dict)


@dataclass
class ROIAssignment:
    labels: np.ndarray               # (N,) ROI label per punctum (0 = none)
    counts_per_roi: np.ndarray       # (num_rois,)
    num_rois: int
    roi_mask: np.ndarray | None
    num_assigned: int = 0
    num_unassigned: int = 0


@dataclass
class AreaMeasurements:
    counts_per_roi: np.ndarray       # bright pixels per ROI
    total_bright: int
    binary_mask: np.ndarray
    fraction_per_roi: np.ndarray
    total_area_per_roi: np.ndarray


@dataclass
class PunctaResults:
    detection: DetectionResult
    roi_assignment: ROIAssignment
    area_measurements: AreaMeasurements
    intensity_threshold: float
    dog_threshold: float
    timestamp: datetime
    # convenience top-level fields
    centroids: np.ndarray
    num_puncta: int
    counts_per_roi: np.ndarray
    areas_per_roi: np.ndarray
    # filled in later by postprocessing / pipeline
    pixel_size: dict | None = None

# ---------------------------------------------------------------------------
# removeSmall
# ---------------------------------------------------------------------------
def _remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than `min_area` pixels (8-connectivity).

    Version-proof replacement for bwareaopen / remove_small_objects that avoids
    the scikit-image 0.26 min_size semantics change. Keeps objects with
    area >= min_area.
    """
    if min_area <= 1:
        return mask
    labeled = label(mask, connectivity=2)
    out = np.zeros_like(mask, dtype=bool)
    for region in regionprops(labeled):
        if region.area >= min_area:
            out[labeled == region.label] = True
    return out


# ---------------------------------------------------------------------------
# detectPunctaDoG
# ---------------------------------------------------------------------------
def detect_puncta_dog(image2d: np.ndarray, config: PunctaConfig) -> DetectionResult:
    """Detect puncta using Difference of Gaussians (port of detectPunctaDoG.m)."""
    if not np.all(np.isfinite(image2d)):
        raise ValueError("image2d must be finite and non-sparse")

    logger.info("Detecting Puncta (DoG): sigma1=%.2f sigma2=%.2f",
                config.puncta_sigma1, config.puncta_sigma2)

    img = image2d.astype(np.float64)
    # imgaussfilt default boundary is 'replicate' -> mode='nearest'
    dog1 = gaussian_filter(img, config.puncta_sigma1, mode="nearest")
    dog2 = gaussian_filter(img, config.puncta_sigma2, mode="nearest")
    img_dog = dog1 - dog2

    dog_mean = img_dog.mean()
    dog_std = img_dog.std(ddof=1)          # MATLAB std normalizes by N-1
    threshold = dog_mean + config.dog_sensitivity * dog_std
    logger.debug("DoG mean=%.4f std=%.4f thr=%.4f", dog_mean, dog_std, threshold)

    # imregionalmax (8-connectivity in 2D is skimage default)
    all_maxima = local_maxima(img_dog)
    sig_maxima = all_maxima & (img_dog > threshold)

    binary_mask = sig_maxima
    # bwareaopen(mask, N) removes objects with FEWER than N pixels.
    # min_puncta_area <= 1 means "remove nothing", so only filter when > 1.
    if config.min_puncta_area > 1:
        # scikit-image >=0.26: min_size removes objects <= min_size,
        # whereas MATLAB bwareaopen removes objects < min_size. Subtract 1
        # to preserve the original "keep objects with >= min_puncta_area" behavior.
        binary_mask = _remove_small(binary_mask, config.min_puncta_area)

    labeled = label(binary_mask, connectivity=2)   # 8-connectivity
    props = regionprops(labeled, intensity_image=image2d)
    num_puncta = len(props)

    if num_puncta > 0:
        # regionprops centroid is (row, col) -> flip to (x, y)
        centroids = np.array([(p.centroid[1], p.centroid[0]) for p in props],
                             dtype=np.float64)
        areas = np.array([p.area for p in props], dtype=np.float64)
        peak_values = np.array([p.intensity_max for p in props], dtype=np.float64)
    else:
        centroids = np.empty((0, 2))
        areas = np.empty((0,))
        peak_values = np.empty((0,))

    stats = _summary_stats(areas, peak_values, num_puncta, config)
    logger.info("Final puncta detected: %d", num_puncta)

    return DetectionResult(
        centroids=centroids, areas=areas, dog_image=img_dog,
        binary_mask=binary_mask, num_puncta=num_puncta,
        peak_values=peak_values, threshold=float(threshold), stats=stats,
    )


def _summary_stats(areas, peak_values, num_puncta, config) -> dict:
    if num_puncta > 0:
        return {
            "mean_area": float(np.mean(areas)),
            "median_area": float(np.median(areas)),
            "std_area": float(np.std(areas, ddof=1)),
            "mean_intensity": float(np.mean(peak_values)),
            "median_intensity": float(np.median(peak_values)),
            "std_intensity": float(np.std(peak_values, ddof=1)),
        }
    logger.warning("No puncta detected! Try lowering dog_sensitivity (%.2f) "
                   "or adjusting sigmas (%.2f/%.2f)",
                   config.dog_sensitivity, config.puncta_sigma1, config.puncta_sigma2)
    return {k: 0.0 for k in (
        "mean_area", "median_area", "std_area",
        "mean_intensity", "median_intensity", "std_intensity")}


# ---------------------------------------------------------------------------
# assignPunctaToROIs
# ---------------------------------------------------------------------------
def assign_puncta_to_rois(centroids: np.ndarray,
                          roi_mask: np.ndarray | None) -> ROIAssignment:
    """Assign detected puncta to labeled ROIs (port of assignPunctaToROIs.m)."""
    num_puncta = centroids.shape[0]

    if roi_mask is None or roi_mask.size == 0 or not np.any(roi_mask):
        logger.info("No ROI mask - using entire image as single ROI")
        return ROIAssignment(
            labels=np.ones(num_puncta, dtype=int),
            counts_per_roi=np.array([num_puncta]),
            num_rois=1, roi_mask=None,
            num_assigned=num_puncta, num_unassigned=0,
        )

    h, w = roi_mask.shape
    labels = np.zeros(num_puncta, dtype=int)
    for i in range(num_puncta):
        x = int(round(centroids[i, 0]))
        y = int(round(centroids[i, 1]))
        if 0 <= x < w and 0 <= y < h:
            labels[i] = roi_mask[y, x]

    max_roi = int(roi_mask.max())
    counts = np.array([np.sum(labels == r) for r in range(1, max_roi + 1)])
    num_assigned = int(np.sum(labels > 0))
    num_unassigned = int(np.sum(labels == 0))

    logger.info("%d puncta assigned to %d ROIs (%d outside)",
                num_assigned, max_roi, num_unassigned)
    return ROIAssignment(
        labels=labels, counts_per_roi=counts, num_rois=max_roi,
        roi_mask=roi_mask, num_assigned=num_assigned, num_unassigned=num_unassigned,
    )


# ---------------------------------------------------------------------------
# measurePunctaAreas
# ---------------------------------------------------------------------------
def measure_puncta_areas(image2d: np.ndarray, roi_mask: np.ndarray | None,
                         threshold: float, config: PunctaConfig) -> AreaMeasurements:
    """Measure total bright-pixel area per ROI (port of measurePunctaAreas.m)."""
    if roi_mask is None or roi_mask.size == 0 or not np.any(roi_mask):
        roi_mask = np.ones(image2d.shape, dtype=int)
        num_rois = 1
    else:
        num_rois = int(roi_mask.max())

    logger.info("Thresholding at intensity: %.2f", threshold)
    binary_mask = image2d > threshold
    if config.min_puncta_area > 1:
        binary_mask = _remove_small(binary_mask, config.min_puncta_area)

    counts = np.zeros(num_rois)
    total_area = np.zeros(num_rois)
    fraction = np.zeros(num_rois)
    for idx, r in enumerate(range(1, num_rois + 1)):
        roi_pixels = roi_mask == r
        bright = binary_mask & roi_pixels
        counts[idx] = bright.sum()
        total_area[idx] = roi_pixels.sum()
        if total_area[idx] > 0:
            fraction[idx] = counts[idx] / total_area[idx]

    total_bright = int(counts.sum())
    logger.info("Total bright pixels: %d (mean/ROI %.2f)",
                total_bright, counts.mean() if num_rois else 0.0)

    return AreaMeasurements(
        counts_per_roi=counts, total_bright=total_bright,
        binary_mask=binary_mask, fraction_per_roi=fraction,
        total_area_per_roi=total_area,
    )


# ---------------------------------------------------------------------------
# filterPuncta (optional utility)
# ---------------------------------------------------------------------------
def filter_puncta(centroids: np.ndarray, image2d: np.ndarray,
                  *, edge_buffer: int = 0,
                  min_intensity: float | None = None,
                  max_intensity: float | None = None):
    """Filter puncta by edge proximity / intensity (port of filterPuncta.m)."""
    n = centroids.shape[0]
    keep = np.ones(n, dtype=bool)
    h, w = image2d.shape

    if edge_buffer > 0:
        too_close = (
            (centroids[:, 0] < edge_buffer) | (centroids[:, 0] > w - edge_buffer) |
            (centroids[:, 1] < edge_buffer) | (centroids[:, 1] > h - edge_buffer)
        )
        keep &= ~too_close
        logger.info("Removed %d puncta near edges", int(too_close.sum()))

    if min_intensity is not None or max_intensity is not None:
        xs = np.clip(np.round(centroids[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(centroids[:, 1]).astype(int), 0, h - 1)
        intensities = image2d[ys, xs]
        if min_intensity is not None:
            keep &= intensities >= min_intensity
        if max_intensity is not None:
            keep &= intensities <= max_intensity

    return centroids[keep], keep


# ---------------------------------------------------------------------------
# comparePunctaSets
# ---------------------------------------------------------------------------
def compare_puncta_sets(auto_puncta: np.ndarray, manual_puncta: np.ndarray,
                        max_dist: float = np.inf) -> dict:
    """Compare automated vs manual detections (port of comparePunctaSets.m)."""
    num_auto = 0 if auto_puncta is None else np.asarray(auto_puncta).shape[0]
    num_manual = 0 if manual_puncta is None else np.asarray(manual_puncta).shape[0]

    if num_auto == 0 or num_manual == 0:
        logger.warning("One or both puncta sets are empty")
        return dict(mean_nn=np.nan, median_nn=np.nan, precision=0.0,
                    recall=0.0, f1_score=0.0, num_auto=num_auto,
                    num_manual=num_manual)

    auto = np.asarray(auto_puncta, dtype=float)
    manual = np.asarray(manual_puncta, dtype=float)

    D = cdist(manual, auto)                       # pdist2(manual, auto)
    min_dist_manual = D.min(axis=1)               # manual -> auto
    match_idx_manual = D.argmin(axis=1)
    min_dist_auto = D.min(axis=0)                 # auto -> manual
    match_idx_auto = D.argmin(axis=0)

    tp = int(np.sum(min_dist_auto <= max_dist))
    fp = int(np.sum(min_dist_auto > max_dist))
    fn = int(np.sum(min_dist_manual > max_dist))

    all_d = np.concatenate([min_dist_manual, min_dist_auto])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    result = dict(
        mean_nn=float(all_d.mean()), median_nn=float(np.median(all_d)),
        std_nn=float(all_d.std(ddof=1)), max_nn=float(all_d.max()),
        precision=precision, recall=recall, f1_score=f1,
        num_auto=num_auto, num_manual=num_manual,
        true_positives=tp, false_positives=fp, false_negatives=fn,
        matching_threshold=max_dist,
        manual_to_auto_distances=min_dist_manual,
        auto_to_manual_distances=min_dist_auto,
        manual_match_indices=match_idx_manual,
        auto_match_indices=match_idx_auto,
    )
    logger.info("auto=%d manual=%d meanNN=%.3f P=%.3f R=%.3f F1=%.3f",
                num_auto, num_manual, result["mean_nn"], precision, recall, f1)
    return result


# ---------------------------------------------------------------------------
# analyzePuncta (top-level pipeline)
# ---------------------------------------------------------------------------
def analyze_puncta(channel_image: np.ndarray, roi_mask: np.ndarray | None,
                   intensity_threshold: float,
                   config: PunctaConfig) -> PunctaResults:
    """Complete puncta analysis pipeline (port of analyzePuncta.m)."""
    logger.info("=== PUNCTA ANALYSIS PIPELINE ===")

    detection = detect_puncta_dog(channel_image, config)
    roi_assignment = assign_puncta_to_rois(detection.centroids, roi_mask)
    area_measurements = measure_puncta_areas(
        channel_image, roi_mask, intensity_threshold, config)

    logger.info("ANALYSIS COMPLETE: %d puncta | DoG thr=%.2f | Int thr=%.2f",
                detection.num_puncta, detection.threshold, intensity_threshold)

    return PunctaResults(
        detection=detection,
        roi_assignment=roi_assignment,
        area_measurements=area_measurements,
        intensity_threshold=intensity_threshold,
        dog_threshold=detection.threshold,
        timestamp=datetime.now(),
        centroids=detection.centroids,
        num_puncta=detection.num_puncta,
        counts_per_roi=roi_assignment.counts_per_roi,
        areas_per_roi=area_measurements.counts_per_roi,
    )