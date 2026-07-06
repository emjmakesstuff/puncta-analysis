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
    """Launch an interactive slider UI to tune detection parameters."""
    from matplotlib.patches import FancyBboxPatch

    if config is None:
        config, _ = estimate_parameters(image)
        logger.info("Starting from auto-estimated parameters")

    init_sens = config.dog_sensitivity
    init_sigma = config.puncta_sigma1

    state = {"config": config}
    vmax = np.percentile(image, 99)

    # --- Palette ---
    NAVY = "#1a3a6b"
    BLUE = "#2f6fed"
    BG = "#f4f7fc"
    TEXT = "#2b3a55"
    GREEN = "#22c55e"
    RED = "#ef4444"
    BTN = BLUE
    BTN_HOVER = NAVY

    # Narrower figure -> less whitespace beside the image
    fig = plt.figure(figsize=(6.2, 8.2))
    fig.patch.set_facecolor(BG)

    # Image panel spans nearly full width now
    ax_img = fig.add_axes([0.04, 0.36, 0.92, 0.58])
    ax_img.imshow(image, cmap="gray", vmax=vmax)
    ax_img.axis("off")
    # store original limits for zoom reset
    x0, x1 = ax_img.get_xlim()
    y0, y1 = ax_img.get_ylim()
    home_limits = {"x": (x0, x1), "y": (y0, y1)}

    if manual_centroids is not None and len(manual_centroids) > 0:
        ax_img.scatter(manual_centroids[:, 0], manual_centroids[:, 1],
                       marker="x", s=28, c=GREEN, linewidths=0.9,
                       label=f"manual ({len(manual_centroids)})")

    # (4) give the detected scatter a label so it shows in the legend
    auto_scatter = ax_img.scatter([], [], facecolors="none", edgecolors=RED,
                                  s=45, linewidths=0.9, label="detected")
    title = ax_img.set_title("", fontsize=11, color=NAVY, fontweight="bold", pad=8)

    # --- Slider 1: sensitivity ---
    fig.text(0.04, 0.315, "Detection sensitivity", fontsize=10,
             color=NAVY, fontweight="bold")
    fig.text(0.04, 0.294,
             "How bright a spot must be to count. Higher → fewer, stronger puncta.",
             fontsize=7.5, color=TEXT, style="italic")
    ax_sens = fig.add_axes([0.04, 0.262, 0.84, 0.022])
    s_sens = Slider(ax_sens, "", *sensitivity_range, valinit=init_sens,
                    valstep=0.05, color=BLUE)
    ax_sens.set_facecolor("#d9e2ec")

    # --- Slider 2: sigma ---
    fig.text(0.04, 0.220, "Puncta size (σ)", fontsize=10,
             color=NAVY, fontweight="bold")
    fig.text(0.04, 0.199,
             "Expected puncta radius in pixels. Match to your real puncta size.",
             fontsize=7.5, color=TEXT, style="italic")
    ax_sig = fig.add_axes([0.04, 0.167, 0.84, 0.022])
    s_sig = Slider(ax_sig, "", *sigma_range, valinit=init_sigma,
                   valstep=0.05, color=BLUE)
    ax_sig.set_facecolor("#d9e2ec")

    # --- Buttons: subtle small radius (not pill) ---
    def make_button(rect, label):
        ax = fig.add_axes(rect)
        ax.set_facecolor(BG)
        ax.axis("off")
        patch = FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0,
            boxstyle="round,pad=0,rounding_size=0.06",   # small radius
            transform=ax.transAxes, facecolor=BTN, edgecolor="none",
            mutation_aspect=2.5,        # keeps radius subtle on wide boxes
            clip_on=False,
        )
        ax.add_patch(patch)
        ax.text(0.5, 0.5, label, ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                color="white")
        return ax, patch

    ax_reset, p_reset = make_button([0.04, 0.06, 0.26, 0.05], "Reset")
    ax_save, p_save = make_button([0.36, 0.06, 0.26, 0.05], "Save")
    ax_done, p_done = make_button([0.68, 0.06, 0.26, 0.05], "Done")

    status = fig.text(0.5, 0.02,
                      "Scroll to zoom, drag to pan.  Adjust sliders, then Done.",
                      ha="center", fontsize=8, color="#64748b")

    # --- Core update ---
    def update(_event=None):
        cfg = replace(
            state["config"],
            dog_sensitivity=float(s_sens.val),
            puncta_sigma1=float(s_sig.val),
            puncta_sigma2=float(s_sig.val) * 2.0,
        )
        state["config"] = cfg
        det = detect_puncta_dog(image, cfg)
        auto_scatter.set_offsets(det.centroids if det.num_puncta > 0
                                 else np.empty((0, 2)))
        # update legend label with live count
        auto_scatter.set_label(f"detected ({det.num_puncta})")
        _refresh_legend()
        if manual_centroids is not None and len(manual_centroids) > 0:
            n_man = len(manual_centroids)
            err = (det.num_puncta - n_man) / n_man * 100
            title.set_text(f"Detected: {det.num_puncta}    Manual: {n_man}    "
                           f"({err:+.1f}%)")
        else:
            title.set_text(f"Detected: {det.num_puncta} puncta")
        fig.canvas.draw_idle()

    def _refresh_legend():
        leg = ax_img.legend(loc="upper right", fontsize=8, framealpha=0.9)
        leg.get_frame().set_facecolor("white")

    # --- Button clicks ---
    def on_click(event):
        if event.inaxes is ax_reset:
            s_sens.reset(); s_sig.reset()
            status.set_text("Reset to auto-estimated values.")
            fig.canvas.draw_idle()
        elif event.inaxes is ax_save:
            _save_config_yaml(state["config"], save_path)
            status.set_text(f"Saved to {save_path}")
            print(f"✓ Saved to {save_path}")
            fig.canvas.draw_idle()
        elif event.inaxes is ax_done:
            _save_config_yaml(state["config"], save_path)
            print(f"✓ Saved to {save_path} and closing.")
            plt.close(fig)

    # --- Hover on buttons ---
    def on_move(event):
        for ax, patch in [(ax_reset, p_reset), (ax_save, p_save), (ax_done, p_done)]:
            patch.set_facecolor(BTN_HOVER if event.inaxes is ax else BTN)
        fig.canvas.draw_idle()

    # --- (3) Scroll-to-zoom on the image ---
    def on_scroll(event):
        if event.inaxes is not ax_img:
            return
        scale = 0.8 if event.button == "up" else 1.25   # zoom in / out
        cur_x = ax_img.get_xlim()
        cur_y = ax_img.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_w = (cur_x[1] - cur_x[0]) * scale
        new_h = (cur_y[1] - cur_y[0]) * scale
        # keep cursor position stable while zooming
        relx = (cur_x[1] - xdata) / (cur_x[1] - cur_x[0])
        rely = (cur_y[1] - ydata) / (cur_y[1] - cur_y[0])
        ax_img.set_xlim([xdata - new_w * (1 - relx), xdata + new_w * relx])
        ax_img.set_ylim([ydata - new_h * (1 - rely), ydata + new_h * rely])
        fig.canvas.draw_idle()

    # --- Double-click to reset zoom ---
    def on_double(event):
        if event.inaxes is ax_img and event.dblclick:
            ax_img.set_xlim(home_limits["x"])
            ax_img.set_ylim(home_limits["y"])
            fig.canvas.draw_idle()

    s_sens.on_changed(update)
    s_sig.on_changed(update)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("button_press_event", on_double)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    update()
    plt.show()
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