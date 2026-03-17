"""
plot_combined_directories.py  —  TEMPORARY SCRIPT
--------------------------------------------------
Reads ring_metrics_summary.csv from each listed subdirectory, filters to
scan points 4–30, and produces a single composite figure with one subplot
per metric.

All six directories are concatenated into a SINGLE continuous line using
the same position-stacking and flip logic as the existing combinatorial
figure (ax_xrf panel). Directories indexed 0,1,2,5 (R1,R2,R3,R6) are
flipped before stacking; the y-axis is position in mm, matching the
reference plot exactly.

Run independently after azimuthal_ring_statistics.py:
    python plot_combined_directories.py
"""

import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCAN_MIN = 4
SCAN_MAX = 30
DIRECTORY_GAP = 13       # mm gap between directories, matching reference plot
CURRENT_BOTTOM_START = 10

SUMMARY_FILENAME = "ring_metrics_summary.csv"
OUTPUT_PNG = (
    "/Users/hpark108/Desktop/Data in MAXIMA Paper/Combinatorial Figure/"
    "combined_ring_metrics.png"
)

# Listed in R-number order (R1 first). Index in this list determines
# flip logic: indices 0,1,2,5 are flipped to match the reference plot.
DIRECTORIES = [
    "/Users/hpark108/Desktop/Data in MAXIMA Paper/Combinatorial Figure/"
    "JHAMAC00001-S3R1C1_JHAMAC00001-S3R1C1_1_1_2025-09-29_19-42-34",
    "/Users/hpark108/Desktop/Data in MAXIMA Paper/Combinatorial Figure/"
    "JHAMAC00001-S3R2C1_JHAMAC00001-S3R2C1_1_1_2025-09-29_19-18-56",
    "/Users/hpark108/Desktop/Data in MAXIMA Paper/Combinatorial Figure/"
    "JHAMAC00001-S3R3C1_JHAMAC00001-S3R3C1_1_1_2025-09-29_18-51-46",
    "/Users/hpark108/Desktop/Data in MAXIMA Paper/Combinatorial Figure/"
    "JHAMAC00001-S3R4C1_JHAMAC00001-S3R4C1_1_1_2025-09-29_18-27-12",
    "/Users/hpark108/Desktop/Data in MAXIMA Paper/Combinatorial Figure/"
    "JHAMAC00001-S3R5C1_JHAMAC00001-S3R5C1_1_1_2025-09-29_17-59-03",
    "/Users/hpark108/Desktop/Data in MAXIMA Paper/Combinatorial Figure/"
    "JHAMAC00001-S3R6C1_JHAMAC00001-S3R6C1_1_1_2025-09-29_17-28-24",
]

# Indices to flip (matching reference plot logic: if i in [0, 1, 2, 5])
FLIP_INDICES = {0}

METRIC_CONFIG = [
    ("mean",                   "Mean Intensity",           "linear"),
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

    Returns scan_points sorted ascending and a dict of metric -> float list.
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
    and a dict of metric -> concatenated value array.

    Parameters
    ----------
    directories : list of str
        Directories in R-number order (R1 first).

    Returns
    -------
    all_positions : np.ndarray
        Stacked y-positions in mm for every scan point across all directories.
    all_metrics : dict
        Metric name -> np.ndarray aligned to all_positions.
    """
    all_positions = []
    all_metrics = {key: [] for key, _, _ in METRIC_CONFIG}
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

        # Apply flip matching reference plot
        if i in FLIP_INDICES:
            sp = sp.max() - (sp - sp.min())

        # Stack vertically
        sp_shifted = sp - sp.min() + current_bottom
        current_bottom = sp_shifted.max() + DIRECTORY_GAP

        all_positions.append(sp_shifted)
        for key, _, _ in METRIC_CONFIG:
            if key in data:
                all_metrics[key].append(data[key])
            else:
                all_metrics[key].append(np.full(len(sp), np.nan))

        print(f"Loaded {len(scan_points)} points from {_parse_label(directory)}")

    if not all_positions:
        raise RuntimeError("No data loaded from any directory.")

    combined_positions = np.concatenate(all_positions)
    combined_metrics = {
        key: np.concatenate(vals) for key, vals in all_metrics.items() if vals
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
    n_metrics = len(METRIC_CONFIG)
    ncols = 2
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 3.5 * nrows),
        sharex=True,
    )
    axes_flat = axes.flatten()

    x_min, x_max = positions.min(), positions.max()

    for idx, (key, label, scale) in enumerate(METRIC_CONFIG):
        ax = axes_flat[idx]
        if key not in metrics:
            ax.set_visible(False)
            continue

        ax.plot(positions, metrics[key], marker="o", lw=1.5, ms=3, color="steelblue")
        ax.set_ylabel(label, fontsize=11)
        ax.set_yscale(scale)
        ax.set_xlim(x_max, x_min)   # reversed so low position is on the right
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.grid(axis="y", lw=0.5, alpha=0.4)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # x-axis label on bottom row only
    for ax in axes_flat[(nrows - 1) * ncols:]:
        ax.set_xlabel("Position (mm)", fontsize=11)

    fig.suptitle(
        f"Ring Metrics vs. Position  |  scan points {SCAN_MIN}–{SCAN_MAX}",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    positions, metrics = build_combined_series(DIRECTORIES)
    plot_composite(positions, metrics, OUTPUT_PNG)
