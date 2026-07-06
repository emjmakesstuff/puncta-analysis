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