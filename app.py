"""Streamlit app for puncta analysis — four-phase guided workflow.

Tabs (persistent):
  Home            : what the program is + how it works
  Image Selection : file-structure guide + folder picker
  Results         : auto-counts for all images (table + overlays + export)
  Manual Tuning   : per-image slider tuning (optional refinement)

Run with:  streamlit run app.py
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")          # must come before pyplot import
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#cdd5e8",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#eef2f9",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

from puncta_analysis import (
    PunctaConfig,
    load_image_stack,
    load_fiji_roi,
    detect_puncta_dog,
    estimate_parameters,
    fit_gaussian_mixture,
    analyze_puncta,
)
from puncta_analysis.postprocess import build_result_row, write_summary_table
from puncta_analysis.visualization import save_overlay_png

# ---------------------------------------------------------------------------
# Page config + theming
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Puncta Analysis", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    /* ---- Import a nicer font (like your inspo sites) ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }
            
    /* Add these to your <style> block */
    [data-testid="stHeader"] {
        background: transparent;
        height: 0;
    }
    [data-testid="stToolbar"] {
        display: none;
    }

    /* ---- Background: soft gradient instead of flat white ---- */
    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef2f9 100%);
    }

    .block-container {
        padding-top: 2.5rem !important;
        max-width: 1200px;
    }

    /* ---- Headings ---- */
    h1 { color: #132157; font-weight: 700; letter-spacing: -0.02em; }
    h2, h3 { color: #1c2b63; font-weight: 600; letter-spacing: -0.01em; }

    /* ---- Buttons: primary vs secondary ---- */
    .stButton>button {
        border-radius: 10px;
        border: 1px solid transparent;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
        transition: all 0.15s ease-in-out;
        box-shadow: 0 1px 2px rgba(19,33,87,0.06);
    }
    /* primary */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #6b8cef 0%, #2f6fed 100%);
        color: white;
    }
    .stButton>button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(47,111,237,0.35);
    }
    /* secondary (nav tabs when inactive) */
    .stButton>button[kind="secondary"] {
        background: #ffffff;
        color: #5a6482;
        border: 1px solid #e2e8f5;
    }
    .stButton>button[kind="secondary"]:hover {
        border-color: #6b8cef;
        color: #2f6fed;
    }

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6ebf5;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(19,33,87,0.05);
    }
    [data-testid="stMetricValue"] { color: #132157; font-weight: 700; }

    /* ---- Dataframe: rounded corners ---- */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(19,33,87,0.06);
    }

    /* ---- Images / thumbnails: card look ---- */
    [data-testid="stImage"] img {
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(19,33,87,0.08);
    }

    /* ---- Expanders ---- */
    [data-testid="stExpander"] {
        border: 1px solid #e6ebf5;
        border-radius: 12px;
        background: #ffffff;
    }

    /* ---- Info / success / warning callouts: softer ---- */
    [data-testid="stAlert"] { border-radius: 12px; }

    /* ---- Sliders: match brand color ---- */
    [data-testid="stSlider"] [role="slider"] { background-color: #2f6fed; }

    /* ---- Divider: subtle ---- */
    hr { border-color: #e2e8f5; }

    /* ---- Hide the default Streamlit menu/footer for a cleaner look ---- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

COUNTER_COLORS = ["#1BB6AF", "#FFAD0A", "#132157", "#EE6100", "#9093A2"]
AUTO_COLOR = "#D72000"

PAGES = ["Home", "Image Selection", "Results", "Manual Tuning"]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("page", "Home")
    ss.setdefault("master_folder", None)
    ss.setdefault("images", [])
    ss.setdefault("idx", 0)
    ss.setdefault("settings", {})      # name -> {sensitivity, sigma1}
    ss.setdefault("channel", 1)

_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def step_cards():
    steps = [
        ("Image Selection", "Pick a folder — we find every image and hand-count file automatically."),
        ("Results", "Puncta are counted in every image automatically, with overlays and exports."),
        ("Manual Tuning", "Optionally fine-tune any image with live interactive sliders."),
    ]
    card_style = (
        "background:#fff; border:1px solid #e6ebf5; border-radius:16px; "
        "padding:1.4rem; height:100%; box-shadow:0 2px 12px rgba(19,33,87,0.05);"
    )
    cols = st.columns(3)
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div style="{card_style}">'
                f'<div style="font-size:2rem;" </div>'
                f'<h3 style="margin:0.4rem 0 0.4rem 0;">{title}</h3>'
                f'<p style="color:#5a6482; font-size:0.92rem; margin:0;">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

def hero(title, subtitle):
    st.markdown(f"""
    <div style="
        background: #132157;
        border-radius: 20px;
        padding: 1.5rem 1.5rem;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 30px rgba(19,33,87,0.25);
    ">
        <div style="font-size: 3rem; line-height:1;" </div>
        <h1 style="color:white; margin: 0.3rem 0.3rem 0.3rem 0; font-size:2.2rem;">{title}</h1>
        <p style="color:#c7d4f5; font-size:1.15rem; margin:0; font-weight:400;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def dog_histogram_png(tif_str, channel, sigma1, current_sens, auto_sens) -> bytes:
    """DoG-response histogram with current + auto (reference) threshold lines.

    The count threshold = dog_mean + sensitivity * dog_std, so we mark where
    the current and auto sensitivity values place that cutoff.
    """
    import io

    image = load_image_stack(tif_str)
    if image.ndim == 3:
        image = image[:, :, min(channel, image.shape[2] - 1)]

    # Compute the DoG response (same as detection uses)
    cfg = PunctaConfig(dog_sensitivity=current_sens, puncta_sigma1=sigma1,
                       puncta_sigma2=sigma1 * 2.0)
    det = detect_puncta_dog(image, cfg)
    dog = det.dog_image.ravel()

    dmean, dstd = float(dog.mean()), float(dog.std())
    cur_thr = dmean + current_sens * dstd
    auto_thr = dmean + auto_sens * dstd

    # Focus the x-range on the informative upper tail (where puncta live)
    hi = np.percentile(dog, 99.5)
    lo = np.percentile(dog, 50)

    fig, ax = plt.subplots(figsize=(4, 1.8))
    ax.hist(dog, bins=80, range=(lo, hi), color="#9093A2", alpha=0.7)
    ax.axvline(auto_thr, color="#9093A2", ls="--", lw=1.5,
               label=f"auto ({auto_sens:.2f})")
    ax.axvline(cur_thr, color="#D72000", lw=2,
               label=f"current ({current_sens:.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("DoG response (brighter →)", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

@st.cache_data(show_spinner=False)
def get_gmm_threshold(tif_str, channel, sigma1, sensitivity) -> float:
    """Return the auto GMM intensity threshold for an image (cached)."""
    image = load_image_stack(tif_str)
    if image.ndim == 3:
        image = image[:, :, min(channel, image.shape[2] - 1)]
    cfg = PunctaConfig(dog_sensitivity=sensitivity, puncta_sigma1=sigma1,
                       puncta_sigma2=sigma1 * 2.0)
    thr, _ = fit_gaussian_mixture(image, cfg)
    return float(thr)

@st.cache_data(show_spinner=False)
def gmm_histogram_png(tif_str, channel, current_thr, auto_thr) -> bytes:
    """Intensity histogram with current + auto (reference) threshold lines."""
    import io

    image = load_image_stack(tif_str)
    if image.ndim == 3:
        image = image[:, :, min(channel, image.shape[2] - 1)]
    vals = image.ravel()
    vmax = np.percentile(vals, 99.5)

    fig, ax = plt.subplots(figsize=(4, 1.8))
    ax.hist(vals, bins=80, range=(0, vmax), color="#9093A2", alpha=0.7)
    ax.axvline(auto_thr, color="#9093A2", ls="--", lw=1.5,
               label=f"auto ({auto_thr:.0f})")
    ax.axvline(current_thr, color="#D72000", lw=2,
               label=f"current ({current_thr:.0f})")
    ax.set_yscale("log")
    ax.set_xlabel("Pixel intensity", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

@st.cache_data(show_spinner=False)
def make_overlay_image(tif_str, channel, sensitivity, sigma1) -> bytes:
    """Render an overlay (image + auto detections) as PNG bytes, cached."""
    import io

    image = load_image_stack(tif_str)
    if image.ndim == 3:
        image = image[:, :, min(channel, image.shape[2] - 1)]
    cfg = PunctaConfig(dog_sensitivity=sensitivity, puncta_sigma1=sigma1,
                       puncta_sigma2=sigma1 * 2.0)
    det = detect_puncta_dog(image, cfg)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray", vmax=np.percentile(image, 99))
    if det.num_puncta > 0:
        ax.scatter(det.centroids[:, 0], det.centroids[:, 1],
                   facecolors="none", edgecolors=AUTO_COLOR, s=40,
                   linewidths=0.8)
    ax.set_title(f"{det.num_puncta} puncta", color="#132157")
    ax.axis("off")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@st.dialog("Detection preview", width="large")
def show_enlarged(name, png_bytes):
    """Modal popup showing the full-size overlay."""
    st.markdown(f"### {name}")
    st.image(png_bytes, use_container_width=True)

def pick_folder() -> str | None:
    """Open a native OS folder-picker in a separate process (won't crash Streamlit)."""
    import subprocess, sys, platform

    helper = (
        "import tkinter as tk;"
        "from tkinter import filedialog;"
        "r = tk.Tk();"
        "r.withdraw();"
        "r.wm_attributes('-topmost', 1);"
        "r.update();"
        "p = filedialog.askdirectory(master=r);"
        "r.destroy();"
        "print(p)"
    )
    try:
        proc = subprocess.Popen([sys.executable, "-c", helper],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        if platform.system() == "Darwin":
            try:
                subprocess.run([
                    "osascript", "-e",
                    'tell application "System Events" to set frontmost of '
                    'the first process whose unix id is {} to true'.format(proc.pid)
                ], timeout=3, capture_output=True)
            except Exception:
                pass
        out, _ = proc.communicate(timeout=120)
        return out.strip() or None
    except Exception as exc:
        st.error(f"Folder picker failed: {exc}")
        return None


def _reset_image(name, image):
    """Reset callback: restore all sliders to auto-estimated values."""
    est, _ = estimate_parameters(image)
    st.session_state[f"sens_{name}"] = round(est.dog_sensitivity, 2)
    st.session_state[f"sig_{name}"] = round(est.puncta_sigma1, 2)
    # reset area threshold to the GMM auto value
    auto_area = get_gmm_threshold(
        st.session_state[f"_tif_{name}"], int(st.session_state.channel),
        round(est.puncta_sigma1, 2), round(est.dog_sensitivity, 2))
    st.session_state[f"area_{name}"] = float(auto_area)
    st.session_state.settings[name] = {
        "sensitivity": round(est.dog_sensitivity, 2),
        "sigma1": round(est.puncta_sigma1, 2),
        "area_threshold": float(auto_area),
    }


def discover_dataset(master: Path) -> list[dict]:
    """Each subfolder = one image (.tif) + its handcount zips. Flat folder ok too."""
    images = []
    subdirs = [d for d in sorted(master.iterdir()) if d.is_dir()]
    if subdirs:
        for d in subdirs:
            tifs = sorted(d.glob("*.tif")) + sorted(d.glob("*.tiff"))
            if not tifs:
                continue
            images.append({"name": d.name, "tif": tifs[0],
                           "zips": sorted(d.glob("*.zip"))})
    else:
        for tif in sorted(master.glob("*.tif")) + sorted(master.glob("*.tiff")):
            images.append({"name": tif.stem, "tif": tif, "zips": []})
    return images


@st.cache_data(show_spinner=False)
def load_image_cached(tif_str: str, channel: int) -> np.ndarray:
    img = load_image_stack(tif_str)
    if img.ndim == 3:
        ch = channel if channel is not None else 0
        img = img[:, :, min(ch, img.shape[2] - 1)]
    return img


@st.cache_data(show_spinner=False)
def load_counters_cached(zip_strs: tuple[str, ...],
                         shape: tuple[int, int]) -> list[np.ndarray]:
    out = []
    for zs in zip_strs:
        try:
            _, cent = load_fiji_roi(zs, shape)
            out.append(cent)
        except Exception:
            out.append(np.empty((0, 2)))
    return out


def get_config_for(name: str, image: np.ndarray) -> PunctaConfig:
    """Return the config for an image: saved (tuned) if present, else auto-estimated."""
    ss = st.session_state
    if name not in ss.settings:
        est, _ = estimate_parameters(image)
        ss.settings[name] = {"sensitivity": round(est.dog_sensitivity, 2),
                             "sigma1": round(est.puncta_sigma1, 2)}
    s = ss.settings[name]
    return PunctaConfig(dog_sensitivity=s["sensitivity"],
                        puncta_sigma1=s["sigma1"],
                        puncta_sigma2=s["sigma1"] * 2.0)


def make_preview(image, centroids, counter_overlays, show_auto=True, zoom=None,
                 area_threshold=None):
    """Plotly preview sized to the image aspect ratio, optional red area mask."""
    import plotly.graph_objects as go

    h, w = image.shape
    vmax = float(np.percentile(image, 99))
    fig = go.Figure()

    fig.add_trace(go.Heatmap(z=image, colorscale="gray", zmax=vmax,
                             zmin=float(image.min()), showscale=False,
                             hoverinfo="skip"))

    # Red highlight for pixels above the area threshold (display-approximate:
    # downsample large images for speed).
    if area_threshold is not None:
        disp = image
        step = max(1, int(max(h, w) / 700))     # downsample for display
        if step > 1:
            disp = image[::step, ::step]
        mask = (disp > area_threshold).astype(float)
        # overlay red where mask==1, transparent elsewhere
        fig.add_trace(go.Heatmap(
            z=mask,
            x=np.arange(0, w, step), y=np.arange(0, h, step),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(215,32,0,0.35)"]],
            showscale=False, hoverinfo="skip", zmin=0, zmax=1,
        ))

    for label, cent, color in counter_overlays:
        if len(cent) > 0:
            fig.add_trace(go.Scatter(
                x=cent[:, 0], y=cent[:, 1], mode="markers",
                marker=dict(symbol="x", color=color, size=7),
                name=f"{label} ({len(cent)})"))

    if show_auto and len(centroids) > 0:
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1], mode="markers",
            marker=dict(symbol="circle-open", color=AUTO_COLOR, size=8,
                        line=dict(width=1.5)),
            name=f"auto ({len(centroids)})"))

    if zoom:
        xr, yr = [zoom["x0"], zoom["x1"]], [zoom["y0"], zoom["y1"]]
    else:
        xr, yr = [0, w], [h, 0]

    fig.update_xaxes(range=xr, constrain="domain", minallowed=0, maxallowed=w,
                     showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=yr, constrain="domain", scaleanchor="x",
                     scaleratio=1, minallowed=0, maxallowed=h,
                     showgrid=False, zeroline=False, visible=False)
    

    aspect = w / h
    BASE_WIDTH = 700
    fig_height = max(300, min(int(BASE_WIDTH / aspect), 820))

    fig.update_layout(
        height=fig_height, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        dragmode="pan", plot_bgcolor="white", paper_bgcolor="white",
        shapes=[dict(type="rect", xref="x", yref="y", x0=0, y0=0, x1=w, y1=h,
                     line=dict(color="#132157", width=2))],
    )
    return fig


def _capture_zoom(event, zoom_key, shape):
    if not event or not isinstance(event, dict):
        return
    relayout = (event.get("relayoutData") or event.get("relayout")
                or event.get("selection"))
    if not isinstance(relayout, dict):
        return
    if relayout.get("xaxis.autorange") or relayout.get("yaxis.autorange"):
        st.session_state.pop(zoom_key, None)
        return
    try:
        x0 = relayout.get("xaxis.range[0]"); x1 = relayout.get("xaxis.range[1]")
        y0 = relayout.get("yaxis.range[0]"); y1 = relayout.get("yaxis.range[1]")
        if None not in (x0, x1, y0, y1):
            st.session_state[zoom_key] = {"x0": float(x0), "x1": float(x1),
                                          "y0": float(y0), "y1": float(y1)}
    except (TypeError, ValueError):
        pass


def render_color_key(show_auto, num_auto, overlays, show_area=False,
                     area_px=0):
    parts = []
    if show_auto and num_auto > 0:
        parts.append(f"<span style='color:{AUTO_COLOR};font-weight:600'>"
                     f"○ auto ({num_auto})</span>")
    if show_area:
        parts.append(f"<span style='color:{AUTO_COLOR};font-weight:600'>"
                     f"▧ area ({area_px:,} px)</span>")
    for label, cent, color in overlays:
        parts.append(f"<span style='color:{color};font-weight:600'>"
                     f"✕ {label} ({len(cent)})</span>")
    if parts:
        st.markdown(
            "<div style='text-align:center;font-size:15px;"
            "padding-top:0px;padding-bottom:4px'>" +
            "&nbsp;&nbsp;&nbsp;".join(parts) + "</div>",
            unsafe_allow_html=True)


def go_to(page: str):
    st.session_state.page = page
    st.rerun()


# ---------------------------------------------------------------------------
# Persistent top navigation ("tabs")
# ---------------------------------------------------------------------------
def render_nav():
    ss = st.session_state
    has_images = bool(ss.images)

    cols = st.columns(len(PAGES))
    for col, page in zip(cols, PAGES):
        with col:
            is_active = (ss.page == page)
            # Guard: Results / Manual Tuning need a loaded dataset
            disabled = (page in ("Results", "Manual Tuning") and not has_images)
            if st.button(
                ("● " if is_active else "") + page,
                key=f"nav_{page}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                disabled=disabled,
            ):
                go_to(page)
    st.divider()


# ===========================================================================
# PAGE: HOME
# ===========================================================================
def page_home():
    hero("Puncta Analysis",
         "Automated cell & puncta counting from microscopy images")

    st.markdown("Welcome! This tool automatically counts fluorescent puncta "
                "(or cells) in your microscopy images, so you don't have to "
                "count them by hand.")

    st.markdown("### How it works")
    step_cards()          # <-- the cards render here

    st.markdown("""
    ### What makes it reliable

    - **Adapts to each image:** detection parameters are estimated from the
      image itself, so it generalizes to new data.
    - **Compare to hand counts:** overlay your manual annotations to verify
      the automatic counts match what a human would find.
    - **Export everything:** counts, per-punctum measurements, and annotated
      overlay images are saved for your records.
    """)

    if st.button("Get started — Select images", type="primary"):
        go_to("Image Selection")


# ===========================================================================
# PAGE: IMAGE SELECTION
# ===========================================================================
def page_selection():
    ss = st.session_state
    st.title("Image Selection")
    st.write("Choose the master folder containing your images. Each image "
             "should be in its own subfolder with any hand-count ROI files "
             "(.zip) for that image.")

    st.markdown("""
    **Expected folder structure:**
    ```
    Master folder/
    ├── Image 1/
    │   ├── image1.tif
    │   ├── handcounts_A.zip
    │   └── handcounts_B.zip
    ├── Image 2/
    │   ├── image2.tif
    │   └── ...
    ```
    *Each subfolder is treated as one image plus its hand counts. Images
    without hand counts work fine too — those files are just optional.*
    """)

    ss.channel = st.number_input("Channel to analyze (0-based)", min_value=0,
                                 value=int(ss.channel), step=1,
                                 help="For multi-channel images. Ignored for "
                                      "single-channel images.")

    if st.button("Choose folder…", type="primary"):
        folder = pick_folder()
        if folder:
            images = discover_dataset(Path(folder))
            if not images:
                st.error("No .tif images found in that folder or its subfolders.")
            else:
                ss.master_folder = folder
                ss.images = images
                ss.idx = 0
                ss.settings = {}      # fresh settings for a new dataset
                st.success(f"Loaded {len(images)} image(s).")

    if ss.master_folder:
        st.success(f"Selected: {ss.master_folder}")
        st.write(f"Found **{len(ss.images)}** image(s):")
        for im in ss.images:
            st.write(f"- **{im['name']}** ({len(im['zips'])} hand-count set(s))")
        st.info("Next: view the automatic counts on the **Results** page. "
                "You can fine-tune any image afterward in **Manual Tuning**.")
        if st.button("View Results", type="primary"):
            go_to("Results")


# ===========================================================================
# PAGE: RESULTS
# ===========================================================================
@st.cache_data(show_spinner=False)
def _analyze_image(tif_str, channel, sensitivity, sigma1):
    """Run detection + area for one image at given params (cached)."""
    image = load_image_stack(tif_str)
    if image.ndim == 3:
        image = image[:, :, min(channel, image.shape[2] - 1)]
    cfg = PunctaConfig(dog_sensitivity=sensitivity, puncta_sigma1=sigma1,
                       puncta_sigma2=sigma1 * 2.0)
    thr, _ = fit_gaussian_mixture(image, cfg)
    results = analyze_puncta(image, None, thr, cfg)
    return results.num_puncta, results.area_measurements.total_bright


def page_results():
    ss = st.session_state
    st.title("Results")

    if not ss.images:
        st.warning("No images loaded. Go to **Image Selection** first.")
        return

    st.write("Automatic counts for every image. Parameters are estimated from "
             "each image unless you've tuned it manually.")

    # Build the results table (auto or tuned settings per image)
    rows = []
    progress = st.progress(0.0, text="Analyzing images…")
    for i, im in enumerate(ss.images):
        name = im["name"]
        image = load_image_cached(str(im["tif"]), int(ss.channel))
        cfg = get_config_for(name, image)
        count, area = _analyze_image(str(im["tif"]), int(ss.channel),
                                     cfg.dog_sensitivity, cfg.puncta_sigma1)
        rows.append({
            "Image": name,
            "Puncta Count": count,
            "Total Area (px)": area,
            "Sensitivity": round(cfg.dog_sensitivity, 2),
            "Sigma": round(cfg.puncta_sigma1, 2),
            "Tuned": "✓" if name in ss.settings and _is_tuned(name) else "auto",
        })
        progress.progress((i + 1) / len(ss.images), text=f"Analyzed {name}")
    progress.empty()

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    # ---- Visual thumbnails: inspect each result, click to enlarge ----
    st.subheader("Visual inspection")
    st.caption("Review each image's detections. Click **Enlarge** to inspect "
               "closely, then use **Manual Tuning** if any need adjustment.")

    n_cols = 3
    grid = st.columns(n_cols)
    for i, im in enumerate(ss.images):
        name = im["name"]
        image = load_image_cached(str(im["tif"]), int(ss.channel))
        cfg = get_config_for(name, image)
        png = make_overlay_image(str(im["tif"]), int(ss.channel),
                                 cfg.dog_sensitivity, cfg.puncta_sigma1)
        with grid[i % n_cols]:
            st.image(png, caption=name, use_container_width=True)
            if st.button("🔍 Enlarge", key=f"enlarge_{name}",
                         use_container_width=True):
                show_enlarged(name, png)

    st.info("Not happy with a result? Fine-tune any image in the "
            "**Manual Tuning** tab, then come back here.")

    # Export
    st.divider()
    output_dir = st.text_input("Output folder", value="results")
    if st.button("Process All & Export", type="primary"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(output_dir) / f"run_{ts}"
        out.mkdir(parents=True, exist_ok=True)
        prog = st.progress(0.0)
        result_rows = []
        for i, im in enumerate(ss.images):
            name = im["name"]
            image = load_image_cached(str(im["tif"]), int(ss.channel))
            cfg = get_config_for(name, image)
            thr, _ = fit_gaussian_mixture(image, cfg)
            results = analyze_puncta(image, None, thr, cfg)
            save_overlay_png(image, results, out / f"{name}_overlay.png",
                             title=name)
            result_rows.append(build_result_row(
                name, results, {"x": 1, "y": 1, "area": 1}, cfg))
            prog.progress((i + 1) / len(ss.images))
        write_summary_table(result_rows, out / "summary.csv")
        prog.empty()
        st.success(f"✓ Exported to: {out}")

        # Show overlay thumbnails
        st.subheader("Overlay images")
        for ov in sorted(out.glob("*_overlay.png")):
            st.image(str(ov), caption=ov.stem, width=500)


def _is_tuned(name):
    """Heuristic: an image is 'tuned' if the user explicitly saved it.

    We track this via a separate set so auto-estimated defaults aren't marked
    as tuned. (Set is populated when Save is clicked in Manual Tuning.)
    """
    return name in st.session_state.get("tuned_names", set())


# ===========================================================================
# PAGE: MANUAL TUNING
# ===========================================================================
def page_tune():
    ss = st.session_state
    if not ss.images:
        st.warning("No images loaded. Go to **Image Selection** first.")
        return

    n = len(ss.images)
    im = ss.images[ss.idx]
    name = im["name"]

    st.title("Manual Tuning")
    st.caption(f"Image {ss.idx + 1} of {n}: **{name}**")

    image = load_image_cached(str(im["tif"]), int(ss.channel))
    counters = load_counters_cached(tuple(str(z) for z in im["zips"]),
                                    image.shape[:2])

    # stash tif path so the reset callback can recompute the GMM area value
    ss[f"_tif_{name}"] = str(im["tif"])

    # settings init (includes area_threshold); backfill old entries missing it
    if name not in ss.settings:
        est, _ = estimate_parameters(image)
        ss.settings[name] = {"sensitivity": round(est.dog_sensitivity, 2),
                             "sigma1": round(est.puncta_sigma1, 2)}

    saved = ss.settings[name]

    # Backfill area_threshold if this entry predates that field
    if "area_threshold" not in saved:
        auto_area = get_gmm_threshold(
            str(im["tif"]), int(ss.channel),
            round(saved["sigma1"], 2), round(saved["sensitivity"], 2))
        saved["area_threshold"] = float(auto_area)
        ss.settings[name] = saved

    sens_key = f"sens_{name}"
    sig_key = f"sig_{name}"
    area_key = f"area_{name}"
    ss.setdefault(sens_key, float(saved["sensitivity"]))
    ss.setdefault(sig_key, float(saved["sigma1"]))
    ss.setdefault(area_key, float(saved["area_threshold"]))

    auto_area_ref = float(saved["area_threshold"])   # reference (auto) value
    img_max = float(image.max())

    col_ctrl, col_view = st.columns([1, 1.8])

    # ---------------- LEFT: controls ----------------
    with col_ctrl:
        st.subheader("Parameters")

        # --- Count sensitivity + DoG histogram ---
        sens = st.slider("Brightness threshold (sensitivity)", 0.0, 5.0,
                         step=0.05, key=sens_key,
                         help="Higher → fewer, brighter puncta (affects count).")
        with st.expander("Show count-threshold plot (DoG)"):
            dog_png = dog_histogram_png(
                str(im["tif"]), int(ss.channel),
                round(float(ss[sig_key]), 2), round(sens, 2),
                round(float(saved["sensitivity"]), 2))
            st.image(dog_png, use_container_width=True)

        # --- Area threshold + GMM histogram ---
        area = st.slider("Area threshold (intensity)", 0.0, img_max,
                         step=max(1.0, img_max / 500.0), key=area_key,
                         help="Pixels brighter than this count as puncta area "
                              "(shown as the red region on the image).")
        with st.expander("Show area-threshold plot (intensity)"):
            gmm_png = gmm_histogram_png(str(im["tif"]), int(ss.channel),
                                        round(area, 1), round(auto_area_ref, 1))
            st.image(gmm_png, use_container_width=True)

        # --- Puncta size ---
        sig = st.slider("Puncta size (σ)", 0.5, 6.0, step=0.05, key=sig_key,
                        help="Expected puncta radius in pixels.")

        with st.expander("How are the starting values chosen?"):
            st.markdown("""
            Starting values are estimated automatically from each image:

            - **Puncta size (σ):** typical puncta size.
            - **Puncta Brightness threshold:** an [Otsu cutoff](https://en.wikipedia.org/wiki/Otsu%27s_method) on the
              filtered image separates bright puncta from dark background.
            - **Area threshold:** a background/signal split on raw intensities controls the red area region.

            On the histogram, the dashed line is the automatic value; the solid
            red line is your current setting. Adjust the sliders to fine-tune.
            """)

        st.subheader("Show overlays")
        toggle_specs = [("Auto counts", "auto", AUTO_COLOR),
                        ("Area region", "area", AUTO_COLOR)]
        for i in range(len(counters)):
            toggle_specs.append((f"Manual count {i + 1}", i,
                                 COUNTER_COLORS[i % len(COUNTER_COLORS)]))

        show_auto = True
        show_area = True
        overlays = []
        tcols = st.columns(2)
        for j, (label, ref, color) in enumerate(toggle_specs):
            with tcols[j % 2]:
                default = (ref in ("auto", "area"))    # both on by default
                on = st.toggle(label, value=default, key=f"toggle_{name}_{ref}")
                if ref == "auto":
                    show_auto = on
                elif ref == "area":
                    show_area = on
                elif on:
                    overlays.append((label, counters[ref], color))
        if not counters:
            st.caption("No hand counts in this folder.")

        # live detection (count) — uses sensitivity + sigma, NOT area
        cfg = PunctaConfig(dog_sensitivity=sens, puncta_sigma1=sig,
                           puncta_sigma2=sig * 2.0)
        det = detect_puncta_dog(image, cfg)

        # live area (bright pixels above the area threshold)
        area_pixels = int((image > area).sum())

        m1, m2 = st.columns(2)
        m1.metric("Auto-detected puncta", det.num_puncta)
        m2.metric("Area (bright px)", f"{area_pixels:,}")

        # Save + Reset
        c_save, c_reset = st.columns(2)
        with c_save:
            if st.button("Save", type="primary", key=f"save_{name}"):
                ss.settings[name] = {"sensitivity": round(sens, 3),
                                     "sigma1": round(sig, 3),
                                     "area_threshold": float(area)}
                ss.setdefault("tuned_names", set()).add(name)
                _analyze_image.clear()
                make_overlay_image.clear()
                st.success("Saved! Results will use these settings.")
        with c_reset:
            st.button("↺ Reset", key=f"reset_{name}",
                      on_click=_reset_image, args=(name, image))

    # ---------------- RIGHT: preview + key + nav ----------------
    with col_view:
        zoom_key = f"zoom_{name}"
        fig = make_preview(image, det.centroids, overlays,
                           show_auto=show_auto, zoom=ss.get(zoom_key),
                           area_threshold=area if show_area else None)
        event = st.plotly_chart(fig, use_container_width=True,
                                key=f"plot_{name}", on_select="rerun",
                                config={"scrollZoom": True,
                                        "displayModeBar": True,
                                        "doubleClick": "reset",
                                        "modeBarButtonsToRemove": ["lasso2d",
                                                                   "select2d"]})
        _capture_zoom(event, zoom_key, image.shape)
        render_color_key(show_auto, det.num_puncta, overlays,
                         show_area=show_area, area_px=area_pixels)

        _, nav_prev, nav_over, nav_next, _ = st.columns([0.5, 1, 1, 1, 0.5])
        with nav_prev:
            if st.button("⬅ Previous", disabled=(ss.idx == 0),
                         use_container_width=True):
                ss.idx -= 1; st.rerun()
        with nav_over:
            if st.button("Results", use_container_width=True):
                go_to("Results")
        with nav_next:
            if st.button("Next ⮕", disabled=(ss.idx == n - 1),
                         use_container_width=True):
                ss.idx += 1; st.rerun()


# ===========================================================================
# ROUTER
# ===========================================================================
render_nav()

page = st.session_state.page
if page == "Home":
    page_home()
elif page == "Image Selection":
    page_selection()
elif page == "Results":
    page_results()
elif page == "Manual Tuning":
    page_tune()