"""Batch pipeline: discover images, process each, collect results."""

from __future__ import annotations
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed

from .config import PunctaConfig
from .io_utils import load_image_stack, read_pixel_size
from .preprocess import fit_gaussian_mixture
from .detection import analyze_puncta
from .postprocess import (
    build_result_row, write_summary_table, write_puncta_table,
)
from .visualization import save_overlay_png, save_detail_figure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------
def discover_images(input_dir: str | Path, glob: str = "*.tif",
                    recursive: bool = True) -> list[Path]:
    """Find all image files under input_dir (recursively by default)."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(input_dir.rglob(glob) if recursive else input_dir.glob(glob))
    logger.info("Discovered %d image(s) under %s", len(files), input_dir)
    return files


# ---------------------------------------------------------------------------
# Process a single image
# ---------------------------------------------------------------------------
def process_one_image(
    tif_path: Path,
    config: PunctaConfig,
    output_dir: Path,
    *,
    config_channel: int | None = None,
    pixel_size_um: float | None = None,
    tune_each: bool = False,
    save_overlay: bool = True,
    save_detail: bool = False,
    save_table: bool = True,
) -> dict | None:
    """Run the full detection pipeline on one image; write its outputs.

    Returns a summary row dict, or None if processing failed.
    """
    name = tif_path.stem
    try:
        logger.info("Processing %s", tif_path.name)
        image = load_image_stack(tif_path)

        # Multi-channel images: extract the configured channel.
        # Single-channel (2D) images pass through unchanged.
        if image.ndim == 3:
            ch = config_channel if config_channel is not None else 0
            if ch >= image.shape[2]:
                raise ValueError(
                    f"Requested channel {ch} but image only has "
                    f"{image.shape[2]} channels"
                )
            logger.info("Multi-channel image (%d ch); using channel %d",
                        image.shape[2], ch)
            image = image[:, :, ch]

        # Interactive per-image tuning (workflow b). Auto-estimates first,
        # then lets the user adjust before this image is processed.
        if tune_each:
            from .tuning import interactive_tune
            from .estimation import estimate_parameters
            logger.info("Launching interactive tuner for %s", tif_path.name)
            est_config, _ = estimate_parameters(image)
            # Start the tuner from the estimated params; don't overwrite the
            # shared config.yaml (pass a per-image save path or None).
            config = interactive_tune(
                image,
                manual_centroids=None,
                config=est_config,
                save_path=output_dir / f"{tif_path.stem}_config.yaml",
            )
            logger.info("Tuned: sensitivity=%.2f sigma1=%.2f",
                        config.dog_sensitivity, config.puncta_sigma1)

        # Pixel size: config override takes precedence, else TIFF metadata
        if pixel_size_um is not None:
            pixel_size = {"x": pixel_size_um, "y": pixel_size_um,
                          "area": pixel_size_um ** 2}
        else:
            pixel_size = read_pixel_size(tif_path)

        # GMM intensity threshold (for area measurement)
        intensity_threshold, _ = fit_gaussian_mixture(image, config)

        # Full analysis (detection + area). No ROI mask in production.
        results = analyze_puncta(image, None, intensity_threshold, config)
        results.pixel_size = pixel_size

        # --- Outputs ---
        if save_overlay:
            save_overlay_png(image, results, output_dir / f"{name}_overlay.png",
                             title=name)
        if save_detail:
            save_detail_figure(image, results, output_dir / f"{name}_detail.png",
                               title=name)
        if save_table:
            write_puncta_table(image, intensity_threshold, config,
                               output_dir / f"{name}_puncta.csv")

        row = build_result_row(name, results, pixel_size, config)
        logger.info("  %s: %d puncta", name, results.num_puncta)
        return row

    except Exception as exc:
        logger.error("FAILED on %s: %s", tif_path.name, exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
def run_batch(config_dict: dict) -> Path:
    """Run the full batch analysis described by a config dict.

    Returns the timestamped output directory.
    """
    input_dir = config_dict["input_dir"]
    glob = config_dict.get("glob", "*.tif")
    recursive = config_dict.get("recursive", True)
    workers = int(config_dict.get("workers", 1))
    pixel_size_um = config_dict.get("pixel_size_um", None)

    save_overlay = config_dict.get("save_overlay_png", True)
    save_detail = config_dict.get("save_detail_figure", False)
    save_table = config_dict.get("save_puncta_table", True)

    config = PunctaConfig.from_dict({
        **config_dict.get("detection", {}),
        **config_dict.get("gmm", {}),
    })

    channel = config_dict.get("channel", None)    # <- is this indented 4 spaces?

    tune_each = config_dict.get("tune_each", False)
    if tune_each and workers > 1:
        logger.info("Interactive tuning requires serial processing; "
                    "setting workers=1")
        workers = 1

    # Timestamped output folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config_dict.get("output_dir", "results")) / f"run_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Copy the config used (reproducibility)
    import yaml
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(config_dict,
                                                                sort_keys=False))

    # Discover images
    images = discover_images(input_dir, glob=glob, recursive=recursive)
    if not images:
        logger.warning("No images found — nothing to do.")
        return output_dir

    # Process (parallel or serial)
    def _work(p):
        return process_one_image(
            p, config, output_dir,
            config_channel=channel,
            pixel_size_um=pixel_size_um,
            tune_each=tune_each,
            save_overlay=save_overlay, save_detail=save_detail,
            save_table=save_table,
        )

    if workers > 1:
        logger.info("Processing %d images with %d workers", len(images), workers)
        rows = Parallel(n_jobs=workers)(delayed(_work)(p) for p in images)
    else:
        rows = [_work(p) for p in images]

    rows = [r for r in rows if r is not None]

    # Summary table
    if rows:
        write_summary_table(rows, output_dir / "summary.csv")
        logger.info("✓ Processed %d/%d images successfully",
                    len(rows), len(images))
    else:
        logger.warning("No images processed successfully.")

    return output_dir

# ---------------------------------------------------------------------------
# Validation against manual ROI counts
# ---------------------------------------------------------------------------
def load_counters(image_dir: Path, image_size: tuple[int, int]) -> list[dict]:
    """Load every ROI .zip in a folder as a separate 'counter'.

    Folder-based rule: each .zip = one person's manual counts for the image.
    Returns a list of dicts: {name, count, area, centroids}.
    """
    from .io_utils import load_fiji_roi

    counters = []
    for zpath in sorted(image_dir.glob("*.zip")):
        try:
            roi_mask, centroids = load_fiji_roi(zpath, image_size)
            counters.append({
                "name": zpath.stem,
                "count": int(centroids.shape[0]),
                "area": int((roi_mask > 0).sum()),
                "centroids": centroids,
            })
            logger.info("  Counter '%s': %d ROIs, %d px area",
                        zpath.stem, centroids.shape[0], int((roi_mask > 0).sum()))
        except Exception as exc:
            logger.warning("  Failed to load %s: %s", zpath.name, exc)
    return counters


def validate_one_image(
    tif_path: Path,
    config: PunctaConfig,
    output_dir: Path,
    *,
    config_channel: int | None = None,
    do_spatial: bool = False,
    matching_distance: float = 5.0,
) -> list[dict]:
    """Validate auto + tuned detection against all manual counters for one image.

    Returns a list of summary rows (one per algorithm mode: 'auto', 'tuned').
    """
    from .estimation import estimate_parameters
    from .tuning import interactive_tune
    from .preprocess import fit_gaussian_mixture
    from .detection import analyze_puncta, compare_puncta_sets
    from .visualization import save_overlay_png
    import numpy as np

    name = tif_path.stem
    logger.info("=== Validating %s ===", name)

    image = load_image_stack(tif_path)
    if image.ndim == 3:
        ch = config_channel if config_channel is not None else 0
        image = image[:, :, ch]
        logger.info("Multi-channel; using channel %d", ch)

    # --- Load all manual counters from the folder ---
    counters = load_counters(tif_path.parent, image.shape[:2])
    if not counters:
        logger.warning("No manual ROI zips found for %s; skipping", name)
        return []

    human_counts = [c["count"] for c in counters]
    human_areas = [c["area"] for c in counters]
    human_summary = {
        "n_counters": len(counters),
        "count_mean": float(np.mean(human_counts)),
        "count_min": int(np.min(human_counts)),
        "count_max": int(np.max(human_counts)),
        "count_std": float(np.std(human_counts, ddof=1)) if len(human_counts) > 1 else 0.0,
        "area_mean": float(np.mean(human_areas)),
        "area_min": int(np.min(human_areas)),
        "area_max": int(np.max(human_areas)),
    }
    logger.info("Human consensus: count %.0f (range %d-%d), %d counters",
                human_summary["count_mean"], human_summary["count_min"],
                human_summary["count_max"], human_summary["n_counters"])

    # --- Get auto + tuned configs ---
    auto_config, _ = estimate_parameters(image)
    logger.info("Launching tuner for %s (tune, then close)", name)
    tuned_config = interactive_tune(
        image, manual_centroids=None, config=auto_config,
        save_path=output_dir / f"{name}_tuned_config.yaml",
    )

    rows = []
    for mode, cfg in [("auto", auto_config), ("tuned", tuned_config)]:
        threshold, _ = fit_gaussian_mixture(image, cfg)
        results = analyze_puncta(image, None, threshold, cfg)

        algo_count = results.num_puncta
        algo_area = results.area_measurements.total_bright

        row = {
            "Image": name,
            "Mode": mode,
            "Algo_Count": algo_count,
            "Algo_Area": algo_area,
            "Human_Count_Mean": round(human_summary["count_mean"], 1),
            "Human_Count_Min": human_summary["count_min"],
            "Human_Count_Max": human_summary["count_max"],
            "Human_Count_Std": round(human_summary["count_std"], 1),
            "Human_Area_Mean": round(human_summary["area_mean"], 1),
            "N_Counters": human_summary["n_counters"],
            # Is the algorithm within the human range?
            "Count_In_Human_Range": (human_summary["count_min"] <= algo_count
                                     <= human_summary["count_max"]),
            "Count_vs_HumanMean_pct": round(
                (algo_count - human_summary["count_mean"])
                / human_summary["count_mean"] * 100, 1),
            "Area_vs_HumanMean_pct": round(
                (algo_area - human_summary["area_mean"])
                / human_summary["area_mean"] * 100, 1)
                if human_summary["area_mean"] else float("nan"),
            "Sensitivity": round(cfg.dog_sensitivity, 3),
            "Sigma1": round(cfg.puncta_sigma1, 3),
        }

        # --- Spatial metrics against each counter (optional) ---
        if do_spatial:
            precisions, recalls, f1s = [], [], []
            for c in counters:
                sim = compare_puncta_sets(results.centroids, c["centroids"],
                                          max_dist=matching_distance)
                precisions.append(sim["precision"])
                recalls.append(sim["recall"])
                f1s.append(sim["f1_score"])
            row["Spatial_Precision_Mean"] = round(float(np.mean(precisions)), 3)
            row["Spatial_Recall_Mean"] = round(float(np.mean(recalls)), 3)
            row["Spatial_F1_Mean"] = round(float(np.mean(f1s)), 3)

        rows.append(row)

        # Save overlay for this mode
        save_overlay_png(image, results,
                         output_dir / f"{name}_{mode}_overlay.png", title=f"{name} ({mode})")

    return rows


def run_validation(config_dict: dict, do_spatial: bool = False) -> Path:
    """Run validation across all images: auto + tuned vs. manual counters."""
    import numpy as np

    input_dir = config_dict["input_dir"]
    glob = config_dict.get("glob", "*.tif")
    recursive = config_dict.get("recursive", True)
    channel = config_dict.get("channel", None)
    matching_distance = config_dict.get("matching_distance", 5.0)

    config = PunctaConfig.from_dict({
        **config_dict.get("detection", {}),
        **config_dict.get("gmm", {}),
    })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config_dict.get("output_dir", "results")) / f"validation_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Validation output: %s", output_dir)

    images = discover_images(input_dir, glob=glob, recursive=recursive)

    all_rows = []
    for tif in images:
        try:
            rows = validate_one_image(
                tif, config, output_dir,
                config_channel=channel,
                do_spatial=do_spatial,
                matching_distance=matching_distance,
            )
            all_rows.extend(rows)
        except Exception as exc:
            logger.error("Validation failed on %s: %s", tif.name, exc, exc_info=True)

    if all_rows:
        write_summary_table(all_rows, output_dir / "validation_summary.csv")
        logger.info("✓ Validation complete: %d rows", len(all_rows))
    return output_dir