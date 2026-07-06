"""Command-line interface for puncta analysis.

Commands:
  puncta analyze --config config.yaml       Batch-process a folder of images.
  puncta tune --image X.tif [--rois Y.zip]  Interactive parameter tuning.
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import yaml


def _setup_logging(output_dir: Path | None = None, level=logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    if output_dir is not None:
        handlers.append(logging.FileHandler(output_dir / "run.log"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        sys.exit(f"Config file not found: {path}\n"
                 f"Tip: copy config.example.yaml to config.yaml and edit it.")
    with open(p) as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
def cmd_analyze(args):
    from .pipeline import run_batch

    config_dict = _load_config(args.config)

    # CLI overrides
    if args.input_dir:
        config_dict["input_dir"] = args.input_dir
    if args.output_dir:
        config_dict["output_dir"] = args.output_dir
    if args.workers is not None:
        config_dict["workers"] = args.workers
    if args.tune_each:
        config_dict["tune_each"] = True

    # Only force headless backend when NOT tuning (tuning needs windows)
    if not config_dict.get("tune_each", False):
        import matplotlib
        matplotlib.use("Agg")

    _setup_logging()
    output_dir = run_batch(config_dict)
    print(f"\n✓ Done. Results in: {output_dir}")


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------
def cmd_tune(args):
    from .io_utils import load_image_stack, load_fiji_roi
    from .tuning import interactive_tune
    from .config import PunctaConfig

    _setup_logging()

    image = load_image_stack(args.image)

    manual_centroids = None
    if args.rois:
        _, manual_centroids = load_fiji_roi(args.rois, image.shape[:2])

    # Start from existing config if given, else auto-estimate inside the tuner
    start_config = None
    if args.config and Path(args.config).exists():
        cfg_dict = _load_config(args.config)
        start_config = PunctaConfig.from_dict({
            **cfg_dict.get("detection", {}),
            **cfg_dict.get("gmm", {}),
        })

    save_path = args.config or "config.yaml"
    interactive_tune(image, manual_centroids=manual_centroids,
                     config=start_config, save_path=save_path)


# ---------------------------------------------------------------------------
# main / arg parsing
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="puncta",
        description="Automated cell/puncta counting from microscopy images.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_an = sub.add_parser("analyze", help="Batch-process a folder of images.")
    p_an.add_argument("--config", default="config.yaml",
                      help="Path to config YAML (default: config.yaml)")
    p_an.add_argument("--input-dir", help="Override input_dir from config.")
    p_an.add_argument("--output-dir", help="Override output_dir from config.")
    p_an.add_argument("--workers", type=int, help="Override number of workers.")
    p_an.set_defaults(func=cmd_analyze)
    p_an.add_argument("--tune-each", action="store_true",
                      help="Interactively tune parameters for each image "
                           "before processing it.")
    p_an.set_defaults(func=cmd_analyze)

    # tune
    p_tn = sub.add_parser("tune", help="Interactive parameter tuning.")
    p_tn.add_argument("--image", required=True, help="Path to a .tif image.")
    p_tn.add_argument("--rois", help="Optional Fiji ROI .zip to overlay (manual).")
    p_tn.add_argument("--config", default="config.yaml",
                      help="Config to start from / save to (default: config.yaml)")
    p_tn.set_defaults(func=cmd_tune)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()