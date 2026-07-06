"""Image-only parameter estimation for puncta detection.

These heuristics estimate good starting values for detection parameters using
ONLY the image itself (no ground-truth / hand counts), so they generalize to
future unlabeled images. Hand counts are used elsewhere (validation) only to
verify these estimates after the fact.
"""

from __future__ import annotations
import logging
from dataclasses import replace

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_log
from skimage.filters import threshold_otsu, threshold_triangle

from .config import PunctaConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Puncta scale (blob size) via Laplacian of Gaussian
# ---------------------------------------------------------------------------
def estimate_puncta_scale(
    image: np.ndarray,
    *,
    min_sigma: float = 1.0,
    max_sigma: float = 8.0,
    num_sigma: int = 12,
    threshold_rel: float = 0.2,      # bumped from 0.1 -> 0.2 (noise-robust)
) -> dict:
    """Estimate the characteristic puncta radius (sigma) from the image.

    Uses multiscale Laplacian-of-Gaussian blob detection (skimage.blob_log),
    which returns a best-fit sigma per detected blob. The median of those
    sigmas is a robust estimate of the dominant puncta scale.

    Parameters
    ----------
    min_sigma, max_sigma, num_sigma : blob_log scale-space sampling range.
    threshold_rel : relative intensity threshold for blob_log (0-1); higher
        = only stronger blobs contribute to the scale estimate.

    Returns
    -------
    dict with keys:
        sigma       : median blob sigma (characteristic scale)
        sigma_std   : spread of blob sigmas
        n_blobs     : number of blobs used
        all_sigmas  : the per-blob sigmas (for inspection/plotting)
    """
    img = image.astype(np.float64)
    # Normalize to [0, 1] so threshold_rel behaves consistently across images
    lo, hi = img.min(), img.max()
    img_norm = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)

    blobs = blob_log(
        img_norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=None,
        threshold_rel=threshold_rel,
    )
    # blobs columns: (row, col, sigma)
    if blobs.shape[0] == 0:
        logger.warning("No blobs found for scale estimation; "
                       "falling back to sigma=%.2f", min_sigma)
        return {"sigma": float(min_sigma), "sigma_std": 0.0,
                "n_blobs": 0, "all_sigmas": np.empty(0)}

    sigmas = blobs[:, 2]
    # Use only the stronger half of blobs (by LoG response proxy = larger sigma
    # tends to co-occur with real structure here); robust median resists the
    # noise blobs that pile up at min_sigma.
    med = float(np.median(sigmas))
    # If the median collapsed to the min_sigma floor, the estimate is likely
    # noise-contaminated; fall back to the 75th percentile as a sturdier scale.
    if med <= min_sigma and len(sigmas) > 1:
        med = float(np.percentile(sigmas, 75))
        logger.info("Median hit sigma floor; using 75th-percentile sigma instead")

    logger.info("Puncta-scale estimate: sigma=%.2f (from %d blobs, "
                "range %.2f-%.2f)", med, len(sigmas), sigmas.min(), sigmas.max())
    return {
        "sigma": med,
        "sigma_std": float(np.std(sigmas, ddof=1)) if len(sigmas) > 1 else 0.0,
        "n_blobs": int(len(sigmas)),
        "all_sigmas": sigmas,
    }


# ---------------------------------------------------------------------------
# 2. DoG threshold from image/noise statistics -> equivalent sensitivity
# ---------------------------------------------------------------------------
def estimate_dog_sensitivity(
    image: np.ndarray,
    sigma1: float,
    sigma2: float,
    *,
    method: str = "mad",
    mad_k: float = 3.0,
) -> dict:
    """Estimate an equivalent `dog_sensitivity` from image statistics.

    Your detector uses:  threshold = mean(DoG) + sensitivity * std(DoG).
    This function computes a data-driven threshold on the DoG response, then
    back-solves the sensitivity that reproduces it, so the estimate slots
    directly into PunctaConfig/detect_puncta_dog unchanged.

    Methods (all image-only, no ground truth):
      "mad"      : robust. threshold = median(DoG) + mad_k * (1.4826 * MAD).
                   MAD ignores bright puncta outliers -> estimates *noise*.
                   Best when puncta are a small fraction of pixels (your case).
      "otsu"     : threshold_otsu on the DoG response (natural bg/signal split).
      "triangle" : threshold_triangle on the DoG response (good for skewed
                   histograms where foreground is sparse).

    Returns
    -------
    dict with keys: sensitivity, threshold, dog_mean, dog_std, method
    """
    img = image.astype(np.float64)
    dog = gaussian_filter(img, sigma1, mode="nearest") - \
        gaussian_filter(img, sigma2, mode="nearest")

    dog_mean = float(dog.mean())
    dog_std = float(dog.std(ddof=1))

    method = method.lower()
    if method == "mad":
        med = float(np.median(dog))
        mad = float(np.median(np.abs(dog - med)))
        robust_std = 1.4826 * mad          # MAD -> std for normal noise
        threshold = med + mad_k * robust_std
    elif method == "otsu":
        threshold = float(threshold_otsu(dog))
    elif method == "triangle":
        threshold = float(threshold_triangle(dog))
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    # Back-solve sensitivity: threshold = mean + sensitivity * std
    if dog_std > 0:
        sensitivity = (threshold - dog_mean) / dog_std
    else:
        sensitivity = 0.0

    logger.info("DoG threshold estimate (%s): thr=%.4f -> sensitivity=%.3f "
                "(dog mean=%.4f std=%.4f)", method, threshold, sensitivity,
                dog_mean, dog_std)
    return {
        "sensitivity": float(sensitivity),
        "threshold": threshold,
        "dog_mean": dog_mean,
        "dog_std": dog_std,
        "method": method,
    }


# ---------------------------------------------------------------------------
# 3. Minimum puncta area from the estimated scale
# ---------------------------------------------------------------------------
def estimate_min_area(sigma: float, *, fraction: float = 0.5) -> int:
    """Estimate a minimum-area floor (pixels) from the puncta scale.

    A blob of radius ~sigma covers ~pi*sigma^2 pixels. We keep objects at
    least `fraction` of that, filtering noise-sized detections. Never below 1.
    """
    expected_area = np.pi * (sigma ** 2)
    min_area = max(1, int(round(fraction * expected_area)))
    logger.info("Min-area estimate: %d px (sigma=%.2f, expected area=%.1f)",
                min_area, sigma, expected_area)
    return min_area


# ---------------------------------------------------------------------------
# 4. Full parameter estimation -> ready-to-use PunctaConfig
# ---------------------------------------------------------------------------
def estimate_parameters(
    image: np.ndarray,
    base_config: PunctaConfig | None = None,
    *,
    threshold_method: str = "otsu",
    sigma_ratio: float = 2.0,
) -> tuple[PunctaConfig, dict]:
    """Estimate detection parameters from the image alone (no ground truth).

    Pipeline:
      1. Estimate characteristic puncta sigma via LoG blobs (noise-robust).
      2. Set sigma1 = scale, sigma2 = sigma_ratio * sigma1.
      3. Estimate dog_sensitivity via Otsu-on-DoG (adapts per image).

    Note: min_puncta_area is intentionally left at the base default. In
    maxima-based detection the peaks are single pixels, so area-filtering them
    is meaningless and removes valid puncta.

    Returns
    -------
    config : PunctaConfig with estimated sigma1, sigma2, dog_sensitivity.
    info   : dict of intermediate estimates (for plotting/inspection).
    """
    if base_config is None:
        base_config = PunctaConfig()

    logger.info("=== Estimating parameters from image (no ground truth) ===")

    scale = estimate_puncta_scale(image)
    sigma1 = scale["sigma"]
    sigma2 = sigma_ratio * sigma1

    thr = estimate_dog_sensitivity(image, sigma1, sigma2, method=threshold_method)

    config = replace(
        base_config,
        puncta_sigma1=round(sigma1, 3),
        puncta_sigma2=round(sigma2, 3),
        dog_sensitivity=round(thr["sensitivity"], 3),
        # min_puncta_area left at base default (see note above)
    )

    info = {
        "scale": scale,
        "threshold": thr,
        "estimated": {
            "puncta_sigma1": config.puncta_sigma1,
            "puncta_sigma2": config.puncta_sigma2,
            "dog_sensitivity": config.dog_sensitivity,
            "min_puncta_area": config.min_puncta_area,
        },
    }

    logger.info("Estimated config: sigma1=%.2f sigma2=%.2f sensitivity=%.2f",
                config.puncta_sigma1, config.puncta_sigma2, config.dog_sensitivity)
    return config, info