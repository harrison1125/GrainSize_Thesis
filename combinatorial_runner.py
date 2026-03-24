"""
combinatorial_runner.py
-----------------------
Reads ring_metrics_summary.csv from each listed subdirectory, filters to
scan points SCAN_MIN–SCAN_MAX, and produces a single composite figure with
one subplot per metric.

All six directories are concatenated into a SINGLE continuous line using
position-stacking and flip logic that matches the reference combinatorial
figure (ax_xrf panel). Directories indexed in FLIP_INDICES are reversed
before stacking; the y-axis is position in mm.

Paths are read from Inputs.py. To run on a new machine, update Inputs.py
only — nothing in this file needs to change.

Run independently after cwt_statistics.py:
    python combinatorial_runner.py
"""

import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import Inputs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCAN_MIN             = 4
SCAN_MAX             = 30
DIRECTORY_GAP        = 13    # mm gap between directories, matching reference plot
CURRENT_BOTTOM_START = 10

SUMMARY_FILENAME = "ring_metrics_summary.csv"
OUTPUT_PNG       = str(Inputs.combinatorial_output)

# Scan row directories are discovered automatically from Inputs.combinatorial_dirs
# (all SampleData* folders under Config/, sorted lexicographically).
# No folder names are hardcoded here.
DIRECTORIES = [str(p) for p in Inputs.combinatorial_dirs]

# Indices (into DIRECTORIES) whose scan-point axis is flipped before stacking,
# matching the orientation of the reference combinatorial figure.
# NOTE: docstring previously listed {0,1,2,5}; currently only index 0 is
# flipped. Update this set if the reference orientation changes.
FLIP_INDICES = {0}

# Ring indices and display labels derived automatically from Inputs.tth_ranges
# so this list stays in sync without manual editing. Override here if you want
# custom labels or a subset of rings.
RINGS = [
    (i, f"Ring {i}  ({lo}–{hi}°)")
    for i, (lo, hi) in enumerate(Inputs.tth_ranges)
]

METRIC_CONFIG = [
    # --- Original metrics ---
    ("mean",                   "Mean Intensity",            "linear"),
    ("std",                    "Std Intensity",             "linear"),
    ("cv",                     "CV (std/mean)",             "linear"),
    ("peak_valley",            "Peak/Valley Ratio",         "linear"),
    ("skewness",               "Skewness",                  "linear"),
    ("kurtosis",               "Kurtosis",                  "linear"),
    ("entropy",                "Shannon Entropy (nats)",    "linear"),
    ("acf_length_deg",         "ACF Length (deg)",          "linear"),
    ("n_texture_peaks",        "N Texture Peaks",           "linear"),
    ("completeness",           "Completeness",              "linear"),
    ("integrated",             "Integrated Intensity",      "linear"),
    # --- Texture / ODF metrics ---
    ("texture_index",          "Texture Index F2",          "linear"),
    ("fourier_c2",             "Fourier C2 (two-fold)",     "linear"),
    ("fourier_c4",             "Fourier C4 (four-fold)",    "linear"),
    ("fourier_c6",             "Fourier C6 (six-fold)",     "linear"),
    # --- Peak shape metrics ---
    ("peak_fwhm_mean_deg",     "Peak FWHM Mean (deg)",      "linear"),
    ("peak_fwhm_std_deg",      "Peak FWHM Std (deg)",       "linear"),
    ("peak_asymmetry_mean",    "Peak Asymmetry Mean",       "linear"),
    # --- Symmetry / balance metrics ---
    ("fiber_symmetry_index",   "Fiber Symmetry Index",      "linear"),
    ("arc_imbalance",          "Arc Imbalance",             "linear"),
    # --- Grain statistics ---
    ("warren_grain_proxy",     "Warren Grain Proxy",        "linear"),
    # --- CWT metrics ---
    ("cwt_n_peaks",            "CWT N Peaks",               "linear"),
    ("cwt_dominant_scale_deg", "CWT Dominant Scale (deg)",  "linear"),
    ("cwt_total_power",        "CWT Total Power",           "linear"),
    ("cwt_scale_entropy",      "CWT Scale Entropy",         "linear"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_label(directory: str) -> str:
    folder = os.path.basename(directory)
    match = re.search(r"(S\d+R\d+C\d+)", folder)
    return match.group(1) if match else folder[:20]


def load_summary(csv_path: str, scan_min: int, scan_max: int) -> tuple[list[int], dict]:
    """
    Load and filter a ring_metrics_summary.csv to a scan point range.

    Returns scan_points sorted ascending and a dict mapping every CSV column
    (excluding sample and scan_point) to a float array. Column names in
    multi-ring CSVs are prefixed ring0_*, ring1_*, etc.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = int(row["scan_point"])
            if scan_min <= sp <= scan_max:
                rows.append(row)

    rows.sort(key=lambda r: int(r["scan_point"]))
    if not rows:
        return [], {}

    scan_points = [int(r["scan_point"]) for r in rows]
    data = {
        key: np.array([float(r[key]) for r in rows])
        for key in rows[0].keys()
        if key not in ("sample", "scan_point")
    }
    return scan_points, data


# ---------------------------------------------------------------------------
# Build single concatenated position + metric arrays
# ---------------------------------------------------------------------------

def build_combined_series(directories: list[str]) -> tuple[np.ndarray, dict]:
    """
    Load all directories, apply flip and stacking logic matching the
    reference combinatorial figure, and return a single position array
    and a nested dict of ring_index -> metric_key -> concatenated array.

    Parameters
    ----------
    directories : list of str
        Directories in R-number order (R1 first).

    Returns
    -------
    all_positions : np.ndarray
        Stacked y-positions in mm for every scan point across all directories.
    all_metrics : dict
        {ring_idx: {base_key: np.ndarray}} aligned to all_positions.
        ring_idx corresponds to entries in RINGS.
    """
    all_positions = []
    all_metrics = {
        ring_idx: {key: [] for key, _, _ in METRIC_CONFIG}
        for ring_idx, _ in RINGS
    }
    current_bottom = CURRENT_BOTTOM_START

    for i, directory in enumerate(directories):
        csv_path = os.path.join(directory, SUMMARY_FILENAME)
        if not os.path.exists(csv_path):
            print(f"WARNING: No CSV found in {directory}, skipping.")
            continue

        scan_points, data = load_summary(csv_path, SCAN_MIN, SCAN_MAX)
        if not scan_points:
            print(f"WARNING: No scan points in range for {_parse_label(directory)}, skipping.")
            continue

        sp = np.array(scan_points, dtype=float)

        if i in FLIP_INDICES:
            sp = sp.max() - (sp - sp.min())

        sp_shifted = sp - sp.min() + current_bottom
        current_bottom = sp_shifted.max() + DIRECTORY_GAP
        all_positions.append(sp_shifted)

        for ring_idx, _ in RINGS:
            for key, _, _ in METRIC_CONFIG:
                csv_col = f"ring{ring_idx}_{key}"
                if csv_col in data:
                    all_metrics[ring_idx][key].append(data[csv_col])
                else:
                    all_metrics[ring_idx][key].append(np.full(len(sp), np.nan))

        print(f"Loaded {len(scan_points)} points from {_parse_label(directory)}")

    if not all_positions:
        raise RuntimeError("No data loaded from any directory.")

    combined_positions = np.concatenate(all_positions)
    combined_metrics = {
        ring_idx: {
            key: np.concatenate(vals)
            for key, vals in ring_data.items()
        }
        for ring_idx, ring_data in all_metrics.items()
    }
    return combined_positions, combined_metrics


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_composite(
    positions: np.ndarray,
    metrics: dict,
    output_png: str,
) -> None:
    """
    Plot all metrics as a grid of subplots: n_metrics rows x n_rings columns.

    Each row is one metric; each column is one ring. This layout lets you
    compare how a given metric varies with position across different
    reflections simultaneously.

    Parameters
    ----------
    positions : np.ndarray
        Stacked position axis in mm.
    metrics : dict
        {ring_idx: {base_key: np.ndarray}} from build_combined_series().
    output_png : str
        Output file path.
    """
    n_metrics = len(METRIC_CONFIG)
    n_rings   = len(RINGS)
    colors    = plt.cm.tab10(np.linspace(0, 0.9, n_rings))

    fig, axes = plt.subplots(
        n_metrics, n_rings,
        figsize=(6 * n_rings, 3.5 * n_metrics),
        sharex=True,
        squeeze=False,
    )

    x_min, x_max = positions.min(), positions.max()

    for row, (key, label, scale) in enumerate(METRIC_CONFIG):
        for col, (ring_idx, ring_label) in enumerate(RINGS):
            ax = axes[row, col]
            ring_data = metrics.get(ring_idx, {})
            y = ring_data.get(key, None)

            if y is None:
                ax.set_visible(False)
                continue

            ax.plot(
                positions, y,
                marker="o", lw=1.5, ms=3,
                color=colors[col],
            )
            ax.set_yscale(scale)
            ax.set_xlim(x_max, x_min)   # reversed: low position on right
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.grid(axis="y", lw=0.5, alpha=0.4)

            if col == 0:
                ax.set_ylabel(label, fontsize=11)

            if row == 0:
                ax.set_title(ring_label, fontsize=12, fontweight="bold")

    for col in range(n_rings):
        axes[n_metrics - 1, col].set_xlabel("Position (mm)", fontsize=11)

    fig.suptitle(
        f"Ring Metrics vs. Position  |  scan points {SCAN_MIN}–{SCAN_MAX}",
        fontsize=13, y=1.005,
    )
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_png}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    positions, metrics = build_combined_series(DIRECTORIES)
    plot_composite(positions, metrics, OUTPUT_PNG)
