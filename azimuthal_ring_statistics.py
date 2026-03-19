import os
import re
import csv
import fabio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from ring_metrics import compute_ring_statistics, plot_ring_metrics
import Inputs


def _extract_scan_point(base_name: str) -> int:
    """
    Extract the trailing integer scan point index from a filename stem.

    For example, 'sample_042' returns 42. If no trailing number is found,
    returns -1 so the file sorts to the front and is easy to spot.

    Parameters
    ----------
    base_name : str
        Filename without extension, e.g. 'sample_042'.

    Returns
    -------
    int
        Trailing integer, or -1 if none found.
    """
    match = re.search(r"(\d+)$", base_name)
    return int(match.group(1)) if match else -1


def _write_subfolder_csv(
    csv_path: str,
    rows: list[dict],
    n_rings: int,
    scalar_keys: list[str],
) -> None:
    """
    Write a ring metrics summary CSV for a single subfolder.

    Column names are prefixed with ring{i}_ for each ring index i so that
    metricplot.py and temp_runner.py can unambiguously address any ring.

    Parameters
    ----------
    csv_path : str
        Full output path for the CSV file.
    rows : list of dict
        One dict per scan point, already sorted by scan_point.
        Each dict must contain 'sample', 'scan_point', and for every ring
        index i a nested dict at key i with all scalar_keys.
    n_rings : int
        Number of rings (determines column prefix count).
    scalar_keys : list of str
        Metric column names to include (excluding sample and scan_point).
    """
    ring_cols = [
        f"ring{i}_{key}" for i in range(n_rings) for key in scalar_keys
    ]
    fieldnames = ["sample", "scan_point"] + ring_cols

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {"sample": row["sample"], "scan_point": row["scan_point"]}
            for i in range(n_rings):
                ring_metrics = row.get(i, {})
                for key in scalar_keys:
                    flat[f"ring{i}_{key}"] = ring_metrics.get(key, "")
            writer.writerow(flat)
    print(f"Saved summary: {csv_path}")


def azimuthal_ring_statistics(
    input_directory: str,
    poni_file: str,
    tth_ranges: list[tuple] = ((15.5, 17),),
    npt_rad: int = 200,
    npt_azim: int = 360,
) -> dict:
    """
    Analyze intensity along azimuthal rings for all TIFF images in a directory tree.

    Walks input_directory recursively. For each subfolder containing TIFFs:
      - Runs integrate2d once per image (expensive cake operation).
      - For each entry in tth_ranges, slices the cached 2D result to that ROI
        and computes per-bin statistics and ring metrics.
      - Saves one .dat file per image per ring and one multi-column .png per
        image (n_rings columns, 5 rows: caked image, ROI heatmap, intensity
        profile, CWT scalogram, Fourier spectrum).
      - Writes one ring_metrics_summary.csv per subfolder with ring{i}_-prefixed
        columns, sorted by scan point (trailing integer in filename).

    Parameters
    ----------
    input_directory : str
        Root directory containing subfolders of TIFF images.
    poni_file : str
        Path to the pyFAI calibration (.poni) file.
    tth_ranges : list of tuple of float, optional
        Each entry is a (tth_lo, tth_hi) pair in degrees defining one ring ROI.
        Defaults to [(15.5, 17)] (single ring, backward-compatible).
    npt_rad : int, optional
        Number of radial bins for integrate2d.
    npt_azim : int, optional
        Number of azimuthal bins. 360 gives ~1 degree resolution.

    Returns
    -------
    dict
        Two-level nested dict: results[subfolder][base_name] contains
        'scan_point' (int), shared 'tth' and 'azimuth' axes, 'intensity_2d',
        and per-ring data under key 'rings' (list of dicts, one per tth_range).
    """
    scalar_keys = [
        # Original metrics
        "mean", "std", "cv", "peak_valley", "skewness", "kurtosis",
        "entropy", "acf_length_deg", "n_texture_peaks", "completeness", "integrated",
        # Texture / ODF metrics
        "texture_index",
        "peak_fwhm_mean_deg", "peak_fwhm_std_deg", "peak_asymmetry_mean",
        "fiber_symmetry_index",
        "fourier_c2", "fourier_c4", "fourier_c6",
        "arc_imbalance",
        "warren_grain_proxy",
        # CWT metrics
        "cwt_n_peaks", "cwt_dominant_scale_deg", "cwt_total_power", "cwt_scale_entropy",
    ]

    n_rings = len(tth_ranges)

    ai = AzimuthalIntegrator()
    ai.load(poni_file)

    # Group TIFF files by subfolder, sorted by scan point within each
    subfolders: dict[str, list] = {}
    for dirpath, _, filenames in os.walk(input_directory):
        for filename in filenames:
            base_name, ext = os.path.splitext(filename)
            if ext.lower() not in (".tif", ".tiff"):
                continue
            scan_point = _extract_scan_point(base_name)
            subfolders.setdefault(dirpath, []).append(
                (scan_point, filename, base_name)
            )

    for dirpath in subfolders:
        subfolders[dirpath].sort(key=lambda x: x[0])

    all_results = {}

    for dirpath, file_list in subfolders.items():
        subfolder_label = os.path.relpath(dirpath, input_directory) or "root"
        subfolder_results = {}
        csv_rows = []

        for scan_point, filename, base_name in file_list:
            input_path = os.path.join(dirpath, filename)
            output_png = os.path.join(dirpath, f"{base_name}_ring_stats.png")

            try:
                image = fabio.open(input_path).data

                # --- integrate2d: run once, reuse for all rings ---
                # Returns I[azimuth, radial], tth, azimuth
                intensity_2d, tth, azimuth = ai.integrate2d(
                    image,
                    npt_rad=npt_rad,
                    npt_azim=npt_azim,
                    azimuth_range=(-180, 180),
                    unit="2th_deg",
                )

                # --- Per-ring processing ---
                rings_data = []      # accumulates plot_ring_metrics input dicts
                csv_ring_metrics = {}  # {ring_idx: {scalar_key: value}}

                for ring_idx, tth_range in enumerate(tth_ranges):
                    # Slice radial axis to this ring's ROI
                    tth_mask    = (tth >= tth_range[0]) & (tth <= tth_range[1])
                    tth_roi     = tth[tth_mask]
                    intensity_roi = intensity_2d[:, tth_mask]  # (npt_azim, n_roi_bins)

                    # Per-azimuthal-bin statistics collapsed over the radial ROI
                    mean_i = np.nanmean(intensity_roi, axis=1)  # (npt_azim,)
                    std_i  = np.nanstd(intensity_roi, axis=1)

                    metrics = compute_ring_statistics(mean_i, azimuth)

                    # Save per-bin profile for this ring
                    output_dat = os.path.join(
                        dirpath, f"{base_name}_ring{ring_idx}_stats.dat"
                    )
                    np.savetxt(
                        output_dat,
                        np.column_stack((azimuth, mean_i, std_i)),
                        header="gamma  mean  std",
                        comments="",
                    )
                    print(f"Saved: {output_dat}")

                    rings_data.append({
                        "mean_i":        mean_i,
                        "std_i":         std_i,
                        "azimuth":       azimuth,
                        "metrics":       metrics,
                        "tth_range":     tth_range,
                        "intensity_2d":  intensity_2d,   # full caked image (shared)
                        "tth":           tth,             # full radial axis (shared)
                        "intensity_roi": intensity_roi,
                        "tth_roi":       tth_roi,
                    })

                    csv_ring_metrics[ring_idx] = {k: metrics[k] for k in scalar_keys}

                # --- Multi-column diagnostic plot (one PNG per image) ---
                plot_ring_metrics(rings_data, base_name, output_png)

                # --- Accumulate CSV row ---
                csv_row = {
                    "sample":     base_name,
                    "scan_point": scan_point,
                }
                for ring_idx in range(n_rings):
                    csv_row[ring_idx] = csv_ring_metrics[ring_idx]
                csv_rows.append(csv_row)

                # --- Store in result dict ---
                subfolder_results[base_name] = {
                    "scan_point":   scan_point,
                    "tth":          tth,
                    "azimuth":      azimuth,
                    "intensity_2d": intensity_2d,
                    "rings":        rings_data,
                }

            except Exception as exc:
                print(f"Failed to process {input_path}: {exc}")

        # Write one CSV per subfolder
        if csv_rows:
            csv_path = os.path.join(dirpath, "ring_metrics_summary.csv")
            _write_subfolder_csv(csv_path, csv_rows, n_rings, scalar_keys)

        all_results[subfolder_label] = subfolder_results

    return all_results


if __name__ == "__main__":
    azimuthal_ring_statistics(
        input_directory=Inputs.root_dir,
        poni_file=Inputs.poni_file,
        tth_ranges=Inputs.tth_ranges,
    )
