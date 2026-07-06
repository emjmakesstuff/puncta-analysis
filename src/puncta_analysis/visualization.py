"""Visualization: per-image overlay PNGs and optional detailed multi-panel figures."""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .detection import PunctaResults

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Overlay PNG (always saved in batch mode)
# ---------------------------------------------------------------------------
def save_overlay_png(
    image: np.ndarray,
    results: PunctaResults,
    path: str | Path,
    manual_centroids: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    """Save a lightweight overlay: image + detections (+ bright area + manual).

    - Red circles: auto-detected puncta
    - Faint red shading: thresholded bright area (what's counted as area)
    - Green x: manual annotations (when available)
    """
    if image.ndim != 2:
        raise ValueError(f"save_overlay_png expects a 2D image, got shape {image.shape}")
    
    vmax = np.percentile(image, 99)
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(image, cmap="gray", vmax=vmax)

    # Bright-area overlay (semi-transparent red)
    bright = results.area_measurements.binary_mask
    overlay = np.zeros((*bright.shape, 4))
    overlay[bright] = [1, 0, 0, 0.25]        # red, alpha 0.25
    ax.imshow(overlay)

    # Manual markers (green x)
    if manual_centroids is not None and len(manual_centroids) > 0:
        ax.scatter(manual_centroids[:, 0], manual_centroids[:, 1],
                   marker="x", s=30, c="lime", linewidths=0.8,
                   label=f"manual ({len(manual_centroids)})")

    # Auto detections (red circles)
    if results.num_puncta > 0:
        ax.scatter(results.centroids[:, 0], results.centroids[:, 1],
                   facecolors="none", edgecolors="red", s=50, linewidths=0.8,
                   label=f"auto ({results.num_puncta})")

    if title:
        ax.set_title(title)
    if (manual_centroids is not None and len(manual_centroids) > 0) or results.num_puncta > 0:
        ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved overlay PNG: %s", path)


# ---------------------------------------------------------------------------
# Detailed multi-panel figure (optional / on-demand)
# ---------------------------------------------------------------------------
def save_detail_figure(
    image: np.ndarray,
    results: PunctaResults,
    path: str | Path,
    manual_centroids: np.ndarray | None = None,
    similarity: dict | None = None,
    title: str | None = None,
) -> None:
    """Save a detailed multi-panel figure for close inspection of one image.

    Panels:
      1. Image + auto detections (+ manual)
      2. DoG response
      3. Bright-area overlay
      4. Auto-vs-manual comparison + NN-distance histogram (if manual present)
    """
    has_manual = manual_centroids is not None and len(manual_centroids) > 0
    vmax = np.percentile(image, 99)

    fig, axes = plt.subplots(2, 2, figsize=(16, 18))

    # Panel 1: detections
    ax = axes[0, 0]
    ax.imshow(image, cmap="gray", vmax=vmax)
    if has_manual:
        ax.scatter(manual_centroids[:, 0], manual_centroids[:, 1],
                   marker="x", s=30, c="lime", linewidths=0.8,
                   label=f"manual ({len(manual_centroids)})")
    if results.num_puncta > 0:
        ax.scatter(results.centroids[:, 0], results.centroids[:, 1],
                   facecolors="none", edgecolors="red", s=50, linewidths=0.8,
                   label=f"auto ({results.num_puncta})")
    ax.set_title(f"Detected puncta (n={results.num_puncta})")
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")

    # Panel 2: DoG response
    ax = axes[0, 1]
    im = ax.imshow(results.detection.dog_image, cmap="magma")
    ax.set_title("Difference-of-Gaussians response")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3: bright-area overlay
    ax = axes[1, 0]
    ax.imshow(image, cmap="gray", vmax=vmax)
    bright = results.area_measurements.binary_mask
    overlay = np.zeros((*bright.shape, 4))
    overlay[bright] = [1, 0, 0, 0.4]
    ax.imshow(overlay)
    ax.set_title(f"Bright area ({results.area_measurements.total_bright} px)")
    ax.axis("off")

    # Panel 4: comparison / distance histogram
    ax = axes[1, 1]
    if has_manual and similarity is not None:
        d_manual = similarity.get("manual_to_auto_distances", np.array([]))
        d_auto = similarity.get("auto_to_manual_distances", np.array([]))
        all_d = np.concatenate([np.asarray(d_manual), np.asarray(d_auto)])
        if all_d.size > 0:
            ax.hist(all_d, bins=30, color="steelblue", alpha=0.75)
            ax.axvline(similarity["mean_nn"], color="red", ls="--",
                       label=f"mean={similarity['mean_nn']:.2f}")
            ax.axvline(similarity["median_nn"], color="green", ls="--",
                       label=f"median={similarity['median_nn']:.2f}")
        ax.set_xlabel("Nearest-neighbor distance (px)")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"P={similarity['precision']:.2f} R={similarity['recall']:.2f} "
            f"F1={similarity['f1_score']:.2f}"
        )
        ax.legend()
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No manual data", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved detail figure: %s", path)