"""Interactive visual tuning of detection parameters via matplotlib sliders.

Starts from auto-estimated parameters (image-only), then lets the user adjust
dog_sensitivity and puncta_sigma1 with live overlay + count feedback, and save
the chosen parameters to a config file.
"""

from __future__ import annotations
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from .config import PunctaConfig
from .detection import detect_puncta_dog
from .estimation import estimate_parameters

logger = logging.getLogger(__name__)


def interactive_tune(
    image: np.ndarray,
    manual_centroids: np.ndarray | None = None,
    config: PunctaConfig | None = None,
    *,
    save_path: str | Path = "config.yaml",
    sensitivity_range: tuple[float, float] = (0.0, 5.0),
    sigma_range: tuple[float, float] = (0.5, 6.0),
) -> PunctaConfig:
    """Launch an interactive slider UI to tune detection parameters.

    Parameters
    ----------
    image : 2D image to tune on.
    manual_centroids : optional (N, 2) ground-truth points [x, y]. If provided,
        shown as green markers so you can gauge agreement (validation images).
    config : starting config. If None, parameters are auto-estimated.
    save_path : where the "Save" button writes the tuned config (YAML).
    sensitivity_range, sigma_range : slider min/max.

    Returns
    -------
    The final PunctaConfig (also returned after the window closes).
    """
    # --- Starting point: auto-estimate if no config given ---
    if config is None:
        config, _ = estimate_parameters(image)
        logger.info("Starting from auto-estimated parameters")

    # Mutable holder so nested callbacks can update the "current" config
    state = {"config": config}

    vmax = np.percentile(image, 99)   # display contrast clip

    # --- Figure layout: big image on top, sliders + button below ---
    fig = plt.figure(figsize=(10, 11))
    ax_img = fig.add_axes([0.08, 0.28, 0.84, 0.66])   # [left, bottom, w, h]
    ax_img.imshow(image, cmap="gray", vmax=vmax)
    ax_img.axis("off")

    # Optional manual ground-truth markers
    if manual_centroids is not None and len(manual_centroids) > 0:
        ax_img.scatter(manual_centroids[:, 0], manual_centroids[:, 1],
                       marker="x", s=30, c="lime", linewidths=0.8,
                       label=f"manual ({len(manual_centroids)})")

    # Detected-puncta scatter (updated live). Start empty; filled by update().
    auto_scatter = ax_img.scatter([], [], facecolors="none", edgecolors="red",
                                  s=50, linewidths=0.8)

    title = ax_img.set_title("")   # updated with live count

    # --- Sliders ---
    ax_sens = fig.add_axes([0.15, 0.16, 0.70, 0.03])
    ax_sig = fig.add_axes([0.15, 0.10, 0.70, 0.03])

    s_sens = Slider(ax_sens, "dog_sensitivity", *sensitivity_range,
                    valinit=config.dog_sensitivity, valstep=0.05)
    s_sig = Slider(ax_sig, "puncta_sigma1", *sigma_range,
                   valinit=config.puncta_sigma1, valstep=0.05)

    # --- Save button ---
    ax_btn = fig.add_axes([0.40, 0.03, 0.20, 0.04])
    b_save = Button(ax_btn, "Save to config")

    # --- Core update function: recompute detection + redraw ---
    def update(_event=None):
        cfg = replace(
            state["config"],
            dog_sensitivity=float(s_sens.val),
            puncta_sigma1=float(s_sig.val),
            puncta_sigma2=float(s_sig.val) * 2.0,   # sigma2 follows sigma1
        )
        state["config"] = cfg

        det = detect_puncta_dog(image, cfg)
        if det.num_puncta > 0:
            auto_scatter.set_offsets(det.centroids)
        else:
            auto_scatter.set_offsets(np.empty((0, 2)))

        # Title with live count (+ comparison if manual available)
        if manual_centroids is not None and len(manual_centroids) > 0:
            n_man = len(manual_centroids)
            err = (det.num_puncta - n_man) / n_man * 100
            title.set_text(
                f"Detected: {det.num_puncta}   Manual: {n_man}   "
                f"({err:+.1f}%)   |   sens={cfg.dog_sensitivity:.2f}, "
                f"sigma1={cfg.puncta_sigma1:.2f}"
            )
        else:
            title.set_text(
                f"Detected: {det.num_puncta}   |   "
                f"sens={cfg.dog_sensitivity:.2f}, "
                f"sigma1={cfg.puncta_sigma1:.2f}"
            )
        fig.canvas.draw_idle()

    # --- Save callback ---
    def save(_event):
        cfg = state["config"]
        _save_config_yaml(cfg, save_path)
        print(f"\n✓ Saved tuned parameters to {save_path}")
        print(f"  dog_sensitivity = {cfg.dog_sensitivity:.3f}")
        print(f"  puncta_sigma1   = {cfg.puncta_sigma1:.3f}")
        print(f"  puncta_sigma2   = {cfg.puncta_sigma2:.3f}")
        # brief visual confirmation on the button
        b_save.label.set_text("Saved ✓")
        fig.canvas.draw_idle()

    s_sens.on_changed(update)
    s_sig.on_changed(update)
    b_save.on_clicked(save)

    if manual_centroids is not None and len(manual_centroids) > 0:
        ax_img.legend(loc="upper right", fontsize=8)

    update()          # initial render
    plt.show()        # blocks until window closed

    return state["config"]


def _save_config_yaml(config: PunctaConfig, path: str | Path) -> None:
    """Write a PunctaConfig to YAML (detection section + full flat dump)."""
    import yaml
    from dataclasses import asdict

    data = asdict(config)
    # Organize the most-tuned params under a 'detection' section for clarity,
    # while still writing everything so the file round-trips.
    out = {
        "detection": {
            "puncta_sigma1": config.puncta_sigma1,
            "puncta_sigma2": config.puncta_sigma2,
            "dog_sensitivity": config.dog_sensitivity,
            "min_puncta_area": config.min_puncta_area,
        },
        "_full_config": data,
    }
    Path(path).write_text(yaml.safe_dump(out, sort_keys=False))