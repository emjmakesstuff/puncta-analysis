"""Puncta Analysis - automated cell and puncta counting from microscopy images."""

from .config import PunctaConfig
from .detection import (
    analyze_puncta,
    detect_puncta_dog,
    assign_puncta_to_rois,
    measure_puncta_areas,
    filter_puncta,
    compare_puncta_sets,
)
from .preprocess import (
    fit_gaussian_mixture,
    select_channel,
    select_channel_interactive,
    enhance_contrast,
    normalize_image,
    preprocess_image,
)
from .io_utils import (
    load_image_stack,
    read_pixel_size,
    load_fiji_roi,
    load_manual_centroids,
    load_manual_results_csv,
    load_analysis_data,
)
from .estimation import (
    estimate_puncta_scale,
    estimate_dog_sensitivity,
    estimate_min_area,
    estimate_parameters,
)
from .tuning import interactive_tune
from .postprocess import (
    convert_area_to_physical,
    write_puncta_table,
    build_result_row,
    write_summary_table,
)
from .visualization import (
    save_overlay_png,
    save_detail_figure,
)
from .pipeline import discover_images, process_one_image, run_batch

__version__ = "0.1.0"

__all__ = [
    # config
    "PunctaConfig",
    # detection
    "analyze_puncta",
    "detect_puncta_dog",
    "assign_puncta_to_rois",
    "measure_puncta_areas",
    "filter_puncta",
    "compare_puncta_sets",
    # preprocess
    "fit_gaussian_mixture",
    "select_channel",
    "select_channel_interactive",
    "enhance_contrast",
    "normalize_image",
    "preprocess_image",
    # io
    "load_image_stack",
    "read_pixel_size",
    "load_fiji_roi",
    "load_manual_centroids",
    "load_manual_results_csv",
    "load_analysis_data",
    # estimation
    "estimate_puncta_scale",
    "estimate_dog_sensitivity",
    "estimate_min_area",
    "estimate_parameters",
    # tuning
    "interactive_tune",
    # postprocess
    "convert_area_to_physical",
    "write_puncta_table",
    "build_result_row",
    "write_summary_table",
    # visualization
    "save_overlay_png",
    "save_detail_figure",
    # pipeline
    "discover_images",
    "process_one_image",
    "run_batch",
]