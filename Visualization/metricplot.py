"""
metricplot.py
-------------
Walks Inputs.root_dir recursively, finds every ring_metrics_summary.csv
produced by cwt_statistics.py, and generates one composite figure per
subfolder with one subplot per metric plotted against scan point integer.

Output figures are saved alongside each CSV as ring_metrics_composite.png.

Run independently after cwt_statistics.py:
    python metricplot.py
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

import Config.Inputs as Inputs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUMMARY_FILENAME = "ring_metrics_summary.csv"
OUTPUT_FILENAME  = "ring_metrics_composite.png"

# Ring indices and display labels are derived automatically from Inputs.tth_ranges
# so this list stays in sync without manual editing. Override here if you want
# custom labels or a subset of rings.
RINGS = [
    (i, f"Ring {i} ({lo}–{hi}°)")
    for i, (lo, hi) in enumerate(Inputs.tth_ranges)
]

# Base metric keys and display labels. Do not include ring prefixes here —
# the ring{i}_ prefix is applied automatically when reading the CSV.
METRIC_CONFIG = [
    # --- Original metrics ---
    ("mean",             "Mean Intensity",           "linear"),
    ("std",              "Std Intensity",             "linear"),
    ("cv",               "CV (std/mean)",             "linear"),
    ("peak_valley",      "Peak/Valley Ratio",         "linear"),
    ("skewness",         "Skewness",                  "linear"),
    ("kurtosis",         "Kurtosis",                  "linear"),
    ("entropy",          "Shannon Entropy (nats)",    "linear"),
    ("acf_length_deg",   "ACF Length (deg)",          "linear"),
    ("n_texture_peaks",  "N Texture Peaks",           "linear"),
    ("completeness",     "Completeness",              "linear"),
    ("integrated",       "Integrated Intensity",      "linear"),
    # --- Texture / ODF metrics ---
    ("texture_index",    "Texture Index F2",          "linear"),
    ("fourier_c2",       "Fourier C2 (two-fold)",     "linear"),
    ("fourier_c4",       "Fourier C4 (four-fold)",    "linear"),
    ("fourier_c6",       "Fourier C6 (six-fold)",     "linear"),
    # --- Peak shape metrics ---
    ("peak_fwhm_mean_deg",  "Peak FWHM Mean (deg)",  "linear"),
    ("peak_fwhm_std_deg",   "Peak FWHM Std (deg)",   "linear"),
    ("peak_asymmetry_mean", "Peak Asymmetry Mean",   "linear"),
    # --- Symmetry / balance metrics ---
    ("fiber_symmetry_index", "Fiber Symmetry Index", "linear"),
    ("arc_imbalance",        "Arc Imbalance",         "linear"),
    # --- Grain statistics ---
    ("warren_grain_proxy",   "Warren Grain Proxy",    "linear"),
    # --- CWT metrics ---
    ("cwt_n_peaks",           "CWT N Peaks",              "linear"),
    ("cwt_dominant_scale_deg","CWT Dominant Scale (deg)", "linear"),
    ("cwt_total_power",       "CWT Total Power",          "linear"),
    ("cwt_scale_entropy",     "CWT Scale Entropy",        "linear"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_summary(csv_path: str) -> tuple[list[int], dict]:
    """
    Load a ring_metrics_summary.csv into a nested dict keyed by ring index
    then base metric name.

    Parameters
    ----------
    csv_path : str
        Path to the summary CSV.

    Returns
    -------
    scan_points : list of int
        Scan point integers in sorted order.
    data : dict
        {ring_idx: {base_key: list of float}} aligned to scan_points.
        Only ring indices present in RINGS are extracted.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rows.sort(key=lambda r: int(r["scan_point"]))
    scan_points = [int(r["scan_point"]) for r in rows]

    data = {}
    for ring_idx, _ in RINGS:
        ring_data = {}
        for key, _, _ in METRIC_CONFIG:
            csv_col = f"ring{ring_idx}_{key}"
            if csv_col in rows[0]:
                ring_data[key] = [float(r[csv_col]) for r in rows]
        data[ring_idx] = ring_data

    return scan_points, data


def plot_composite(
    scan_points: list[int],
    data: dict,
    output_png: str,
    title: str,
) -> None:
    """
    Plot all metrics as a grid: n_metrics rows x n_rings columns.

    Each row is one metric; each column is one ring. This layout lets you
    compare how a given metric varies with scan point across different
    reflections simultaneously.

    Parameters
    ----------
    scan_points : list of int
        X-axis values.
    data : dict
        {ring_idx: {base_key: list of float}} from load_summary().
    output_png : str
        Path to save the figure.
    title : str
        Suptitle for the figure, typically the subfolder name.
    """
    n_metrics = len(METRIC_CONFIG)
    n_rings   = len(RINGS)
    colors    = plt.cm.tab10(np.linspace(0, 0.9, n_rings))
    x         = np.array(scan_points)

    fig, axes = plt.subplots(
        n_metrics, n_rings,
        figsize=(7 * n_rings, 3.5 * n_metrics),
        sharex=True,
        squeeze=False,  # always 2-D axes array even for n_rings == 1
    )

    for row, (key, label, scale) in enumerate(METRIC_CONFIG):
        for col, (ring_idx, ring_label) in enumerate(RINGS):
            ax        = axes[row, col]
            ring_data = data.get(ring_idx, {})

            if key not in ring_data:
                ax.set_visible(False)
                continue

            y = np.array(ring_data[key])
            ax.plot(x, y, marker="o", lw=1.8, ms=5, color=colors[col])
            ax.set_yscale(scale)
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.grid(axis="y", lw=0.5, alpha=0.4)

            # y-axis label only on leftmost column
            if col == 0:
                ax.set_ylabel(label, fontsize=11)

            # Column header only on top row
            if row == 0:
                ax.set_title(ring_label, fontsize=12, fontweight="bold")

    # x-axis label on bottom row only
    for col in range(n_rings):
        axes[n_metrics - 1, col].set_xlabel("Scan Point", fontsize=11)

    fig.suptitle(f"Ring Metrics vs. Scan Point — {title}", fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved composite figure: {output_png}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    found = False
    for dirpath, _, filenames in os.walk(Inputs.root_dir):
        if SUMMARY_FILENAME not in filenames:
            continue
        found = True
        csv_path    = os.path.join(dirpath, SUMMARY_FILENAME)
        output_png  = os.path.join(dirpath, OUTPUT_FILENAME)
        subfolder_label = os.path.relpath(dirpath, Inputs.root_dir) or "root"

        try:
            scan_points, data = load_summary(csv_path)
            print(
                f"[{subfolder_label}] Loaded {len(scan_points)} scan points "
                f"from {csv_path}"
            )
            plot_composite(scan_points, data, output_png, title=subfolder_label)
        except Exception as exc:
            print(f"Failed to process {csv_path}: {exc}")

    if not found:
        raise FileNotFoundError(
            f"No {SUMMARY_FILENAME} files found under {Inputs.root_dir}.\n"
            "Run azimuthal_ring_statistics.py first to generate them."
        )
