"""
plot_metrics_composite.py
-------------------------
Walks input_directory recursively, finds every ring_metrics_summary.csv
produced by azimuthal_ring_statistics.py, and generates one composite figure
per subfolder with one subplot per metric plotted against scan point integer.

Output figures are saved alongside each CSV as ring_metrics_composite.png.

Run independently after azimuthal_ring_statistics.py:
    python plot_metrics_composite.py
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import Inputs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUMMARY_FILENAME = "ring_metrics_summary.csv"
OUTPUT_FILENAME  = "ring_metrics_composite.png"

# Display label and y-axis scale for each metric column
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

def load_summary(csv_path: str) -> tuple[list[int], dict[str, list]]:
    """
    Load a ring_metrics_summary.csv into parallel lists keyed by metric name.

    Parameters
    ----------
    csv_path : str
        Path to the summary CSV.

    Returns
    -------
    scan_points : list of int
        Scan point integers in sorted order.
    data : dict
        Keys are metric column names, values are lists of float aligned to
        scan_points.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rows.sort(key=lambda r: int(r["scan_point"]))

    scan_points = [int(r["scan_point"]) for r in rows]
    data = {
        key: [float(r[key]) for r in rows]
        for key in rows[0].keys()
        if key not in ("sample", "scan_point")
    }
    return scan_points, data


def plot_composite(
    scan_points: list[int],
    data: dict[str, list],
    output_png: str,
    title: str,
) -> None:
    """
    Plot each metric as its own subplot against scan point integer.

    Parameters
    ----------
    scan_points : list of int
        X-axis values.
    data : dict
        Metric name -> list of float, aligned to scan_points.
    output_png : str
        Path to save the figure.
    title : str
        Suptitle for the figure, typically the subfolder name.
    """
    n_metrics = len(METRIC_CONFIG)
    ncols = 2
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 3.5 * nrows),
        sharex=True,
    )
    axes_flat = axes.flatten()

    x = np.array(scan_points)

    for idx, (key, label, scale) in enumerate(METRIC_CONFIG):
        ax = axes_flat[idx]
        if key not in data:
            ax.set_visible(False)
            continue
        y = np.array(data[key])
        ax.plot(x, y, marker="o", lw=1.8, ms=5, color="steelblue")
        ax.set_ylabel(label, fontsize=11)
        ax.set_yscale(scale)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.grid(axis="y", lw=0.5, alpha=0.4)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # x-axis label on bottom row only
    for ax in axes_flat[(nrows - 1) * ncols:]:
        ax.set_xlabel("Scan Point", fontsize=11)

    fig.suptitle(f"Ring Metrics vs. Scan Point — {title}", fontsize=13, y=1.01)
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
        csv_path = os.path.join(dirpath, SUMMARY_FILENAME)
        output_png = os.path.join(dirpath, OUTPUT_FILENAME)
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
