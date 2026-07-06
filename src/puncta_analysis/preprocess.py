from __future__ import annotations
import logging

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from skimage.exposure import (
    equalize_hist,
    equalize_adapthist,
    rescale_intensity,
)
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture

from .config import PunctaConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# fitGaussianMixture
# ---------------------------------------------------------------------------
def fit_gaussian_mixture(image2d: np.ndarray, config: PunctaConfig,
                         show_plot: bool = False):
    """Fit a GMM to pixel intensities and return an intensity threshold.

    Port of fitGaussianMixture.m. Threshold = max of the fitted component means
    (the brightest component). Falls back to Otsu if the GMM fit fails.

    Returns
    -------
    threshold : float
    gm_model  : sklearn.mixture.GaussianMixture | None
    """
    logger.info("Fitting Gaussian Mixture Model...")

    img = image2d.astype(np.float64)

    # Downsample for speed (bilinear + anti-aliasing to mimic imresize)
    scale = config.gmm_downsample_factor
    downsampled = rescale(img, scale, order=1, anti_aliasing=True,
                          preserve_range=True)
    logger.info("  Downsampled to %d x %d pixels",
                downsampled.shape[0], downsampled.shape[1])

    # Flatten to a column of finite values
    pixel_vector = downsampled.ravel()
    pixel_vector = pixel_vector[np.isfinite(pixel_vector)]

    logger.info("  Fitting %d components...", config.num_gmm_components)
    try:
        gm = GaussianMixture(
            n_components=config.num_gmm_components,
            reg_covar=config.gmm_regularization,     # RegularizationValue
            n_init=config.gmm_replicates,            # Replicates
            max_iter=config.gmm_max_iter,            # MaxIter
            covariance_type="full",
        )
        gm.fit(pixel_vector.reshape(-1, 1))

        means = np.sort(gm.means_.ravel())           # ascending
        threshold = float(means.max())               # brightest component

        logger.info("  GMM means: %s", np.round(means, 3).tolist())
        logger.info("  Threshold set to: %.2f", threshold)
        return threshold, gm

    except Exception as exc:                         # pragma: no cover
        logger.warning("GMM fitting failed: %s", exc)
        logger.warning("Using Otsu threshold instead")
        # graythresh returns a level in [0,1] scaled by image max in MATLAB;
        # threshold_otsu works directly in the image's intensity units.
        threshold = float(threshold_otsu(image2d))
        return threshold, None


# ---------------------------------------------------------------------------
# selectChannel (config-driven + interactive fallback)
# ---------------------------------------------------------------------------
def select_channel(image_stack: np.ndarray, channel: int):
    """Select a single channel by index (0-based) from an (H, W, C) stack.

    This is the non-interactive, config-driven replacement for the MATLAB
    listdlg GUI. `channel` is 0-based to match the YAML config convention.

    Returns
    -------
    selected_idx    : list[int]         (0-based, single element)
    selected_images : list[np.ndarray]  (2D arrays)
    """
    num_channels = _num_channels(image_stack)

    if num_channels == 1:
        logger.info("Single channel detected - auto-selected")
        return [0], [_get_channel(image_stack, 0)]

    if not (0 <= channel < num_channels):
        raise ValueError(
            f"Requested channel {channel} is out of range "
            f"(image has {num_channels} channels, valid 0..{num_channels - 1})"
        )

    img = _get_channel(image_stack, channel)
    logger.info("Selected Channel %d (0-based) of %d", channel, num_channels)
    return [channel], [img]


def select_channel_interactive(image_stack: np.ndarray,
                               channel_names: list[str] | None = None,
                               allow_multiple: bool = False):
    """Terminal-based channel selection (optional interactive fallback).

    Returns (selected_idx, selected_images) with 0-based indices.
    """
    num_channels = _num_channels(image_stack)
    if channel_names is None or len(channel_names) != num_channels:
        channel_names = [f"Channel {k}" for k in range(num_channels)]

    if num_channels == 1:
        logger.info("Single channel detected - auto-selected")
        return [0], [_get_channel(image_stack, 0)]

    print("\nAvailable channels:")
    for k, name in enumerate(channel_names):
        print(f"  [{k}] {name}")

    if allow_multiple:
        raw = input("Select channel(s), comma-separated (e.g. 0,2): ").strip()
        idx = [int(x) for x in raw.split(",") if x.strip() != ""]
    else:
        idx = [int(input("Select ONE channel index: ").strip())]

    for i in idx:
        if not (0 <= i < num_channels):
            raise ValueError(f"Channel {i} out of range 0..{num_channels - 1}")

    images = [_get_channel(image_stack, i) for i in idx]
    return idx, images


def _num_channels(stack: np.ndarray) -> int:
    """Number of channels for an (H, W) or (H, W, C) array."""
    return 1 if stack.ndim == 2 else stack.shape[2]


def _get_channel(stack: np.ndarray, idx: int) -> np.ndarray:
    return stack if stack.ndim == 2 else stack[:, :, idx]


# ---------------------------------------------------------------------------
# enhanceContrast  (optional utility)
# ---------------------------------------------------------------------------
def enhance_contrast(image2d: np.ndarray, method: str = "adapthisteq") -> np.ndarray:
    """Enhance image contrast (port of enhanceContrast.m)."""
    img = _to_unit_float(image2d)
    method = method.lower()

    if method == "histeq":
        out = equalize_hist(img)
    elif method == "adapthisteq":
        # NOTE: MATLAB used Distribution='rayleigh'; skimage uses uniform.
        # ClipLimit 0.02 maps directly to clip_limit.
        out = equalize_adapthist(img, clip_limit=0.02)
    elif method == "imadjust":
        p1, p99 = np.percentile(img, [1, 99])
        out = rescale_intensity(img, in_range=(p1, p99), out_range=(0.0, 1.0))
    else:
        raise ValueError(f"Unknown enhancement method: {method}")

    logger.info("Contrast enhanced using %s method", method)
    return out


def _to_unit_float(image2d: np.ndarray) -> np.ndarray:
    """Mimic the MATLAB dtype handling: scale to [0,1]."""
    if image2d.dtype == np.uint16:
        m = image2d.max()
        return image2d.astype(np.float64) / (float(m) if m > 0 else 1.0)
    if image2d.dtype == np.uint8:
        return image2d.astype(np.float64) / 255.0
    # mat2gray-style min-max
    img = image2d.astype(np.float64)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


# ---------------------------------------------------------------------------
# normalizeImage  (optional utility)
# ---------------------------------------------------------------------------
def normalize_image(image2d: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize image intensity (port of normalizeImage.m)."""
    img = image2d.astype(np.float64)
    method = method.lower()

    if method == "minmax":
        lo, hi = img.min(), img.max()
        out = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    elif method == "zscore":
        mu, sd = img.mean(), img.std(ddof=1)
        out = (img - mu) / sd if sd > 0 else img - mu
    elif method == "percentile":
        p1, p99 = np.percentile(img, [1, 99])
        if p99 > p1:
            out = np.clip((img - p1) / (p99 - p1), 0.0, 1.0)
        else:
            out = np.zeros_like(img)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    logger.info("Image normalized using %s method", method)
    return out


# ---------------------------------------------------------------------------
# preprocessImage  (optional pipeline utility)
# ---------------------------------------------------------------------------
def preprocess_image(image2d: np.ndarray, config: PunctaConfig, *,
                     normalize: bool = False,
                     enhance_contrast_flag: bool = False,
                     remove_background: bool = False):
    """Complete preprocessing pipeline (port of preprocessImage.m).

    Returns (processed_img, info_dict).
    """
    processed = image2d.astype(np.float64)
    steps: list[str] = []
    logger.info("--- Preprocessing Image ---")

    if normalize:
        processed = normalize_image(processed, "percentile")
        steps.append("normalized")

    if enhance_contrast_flag:
        processed = enhance_contrast(processed, "adapthisteq")
        steps.append("contrast_enhanced")

    if remove_background:
        logger.info("Removing background...")
        background = gaussian_filter(processed, sigma=50, mode="nearest")
        processed = np.maximum(processed - background, 0.0)
        steps.append("background_removed")

    info = {
        "steps": steps,
        "original_range": (float(image2d.min()), float(image2d.max())),
        "processed_range": (float(processed.min()), float(processed.max())),
    }
    logger.info("Preprocessing complete: %s", ", ".join(steps) or "(none)")
    return processed, info