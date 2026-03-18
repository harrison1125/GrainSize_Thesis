import os
import re
import csv
import fabio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from ring_metrics import compute_ring_statistics
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


def _write_subfolder_csv(csv_path: str, rows: list[dict], scalar_keys: list[str]) -> None:
    """
    Write a ring metrics summary CSV for a single subfolder.

    Parameters
    ----------
    csv_path : str
        Full output path for the CSV file.
    rows : list of dict
        One dict per scan point, already sorted by scan_point.
    scalar_keys : list of str
        Metric column names to include (excluding sample and scan_point).
    """
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["sample", "scan_point"] + scalar_keys
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved summary: {csv_path}")


def azimuthal_ring_statistics(
    input_directory: str,
    poni_file: str,
    tth_range: tuple = (15.5, 17),
    npt_rad: int = 200,
    npt_azim: int = 360,
) -> dict:
    """
    Analyze intensity along azimuthal rings for all TIFF images in a directory tree.

    Walks input_directory recursively. For each subfolder containing TIFFs, cakes
    each image, computes per-azimuthal-bin statistics and ring metrics, saves
    per-image .dat and .png files, and writes one ring_metrics_summary.csv per
    subfolder sorted by scan point (trailing integer in filename).

    Parameters
    ----------
    input_directory : str
        Root directory containing subfolders of TIFF images.
    poni_file : str
        Path to the pyFAI calibration (.poni) file.
    tth_range : tuple of float, optional
        2theta range (degrees) defining the ring ROI.
    npt_rad : int, optional
        Number of radial bins for integrate2d.
    npt_azim : int, optional
        Number of azimuthal bins. 360 gives ~1 degree resolution.

    Returns
    -------
    dict
        Two-level nested dict: results[subfolder][base_name] contains
        per-bin arrays, scalar metrics, and 'scan_point' (int).
    """
    scalar_keys = [
        # Original metrics
        "mean", "std", "cv", "peak_valley", "skewness", "kurtosis",
        "entropy", "acf_length_deg", "n_texture_peaks", "completeness", "integrated",
        # New metrics
        "texture_index",
        "peak_fwhm_mean_deg", "peak_fwhm_std_deg", "peak_asymmetry_mean",
        "fiber_symmetry_index",
        "fourier_c2", "fourier_c4", "fourier_c6",
        "arc_imbalance",
        "warren_grain_proxy",
        # CWT metrics
        "cwt_n_peaks", "cwt_dominant_scale_deg", "cwt_total_power", "cwt_scale_entropy",
    ]

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
            output_dat = os.path.join(dirpath, f"{base_name}_ring_stats.dat")
            output_png = os.path.join(dirpath, f"{base_name}_ring_stats.png")

            try:
                image = fabio.open(input_path).data

                # integrate2d returns I[azimuth, radial], tth, azimuth
                intensity_2d, tth, azimuth = ai.integrate2d(
                    image,
                    npt_rad=npt_rad,
                    npt_azim=npt_azim,
                    azimuth_range=(-180, 180),
                    unit="2th_deg",
                )

                # Restrict radial axis to 2theta ROI
                tth_mask = (tth >= tth_range[0]) & (tth <= tth_range[1])
                tth_roi  = tth[tth_mask]
                intensity_roi = intensity_2d[:, tth_mask]  # (npt_azim, n_roi_bins)

                # Per-azimuthal-bin statistics collapsed over the radial ROI
                mean_i = np.nanmean(intensity_roi, axis=1)  # shape: (npt_azim,)
                std_i  = np.nanstd(intensity_roi, axis=1)

                metrics = compute_ring_statistics(mean_i, azimuth)

                # Save per-bin profile
                np.savetxt(
                    output_dat,
                    np.column_stack((azimuth, mean_i, std_i)),
                    header="gamma  mean  std",
                    comments="",
                )

                # --- Plot ---
                fig, axes = plt.subplots(
                    4, 1, figsize=(9, 17),
                    gridspec_kw={"height_ratios": [3, 1, 3, 2]},
                )

                # Panel 1: Caked image (log scale)
                pos_2d = intensity_2d.copy()
                pos_2d[pos_2d <= 0] = np.nan
                axes[0].imshow(
                    pos_2d,
                    aspect="auto",
                    origin="lower",
                    extent=[tth.min(), tth.max(), azimuth.min(), azimuth.max()],
                    cmap="viridis",
                    norm=LogNorm(
                        vmin=np.nanpercentile(pos_2d, 1),
                        vmax=np.nanpercentile(pos_2d, 99),
                    ),
                )
                axes[0].axvline(tth_range[0], color="red", lw=1.2, ls="--", label="ROI")
                axes[0].axvline(tth_range[1], color="red", lw=1.2, ls="--")
                axes[0].set_xlabel(r"2$\theta$ (deg)", fontsize=13)
                axes[0].set_ylabel(r"$\gamma$ (deg)", fontsize=13)
                axes[0].set_title(
                    f"{base_name} (scan {scan_point}) — Caked Image", fontsize=13
                )
                axes[0].legend(fontsize=10, loc="upper right")

                # Panel 2: ROI interior heatmap (1/3 height, log scale)
                roi_pos = intensity_roi.T.copy()
                roi_pos[roi_pos <= 0] = np.nan
                axes[1].imshow(
                    roi_pos,
                    aspect="auto",
                    origin="lower",
                    extent=[azimuth.min(), azimuth.max(), tth_roi.min(), tth_roi.max()],
                    cmap="viridis",
                    norm=LogNorm(
                        vmin=np.nanpercentile(roi_pos, 1),
                        vmax=np.nanpercentile(roi_pos, 99),
                    ),
                )
                axes[1].set_xlabel(r"$\gamma$ (deg)", fontsize=13)
                axes[1].set_ylabel(r"2$\theta$ (deg)", fontsize=13)
                axes[1].set_title("ROI Interior — radial vs. azimuthal", fontsize=13)

                # Panel 3: Intensity vs gamma (log scale)
                axes[2].plot(azimuth, mean_i, lw=2, color="steelblue", label="Mean over ROI")
                axes[2].fill_between(
                    azimuth,
                    mean_i - std_i,
                    mean_i + std_i,
                    alpha=0.3,
                    color="steelblue",
                    label="±1 std",
                )

                # Mark detected texture peaks
                if metrics["n_texture_peaks"] > 0:
                    peak_intensities = mean_i[
                        np.round(
                            np.interp(
                                metrics["peak_positions"],
                                azimuth,
                                np.arange(len(azimuth)),
                            )
                        ).astype(int)
                    ]
                    axes[2].scatter(
                        metrics["peak_positions"],
                        peak_intensities,
                        color="red", zorder=5, s=50,
                        label=f"Texture peaks (n={metrics['n_texture_peaks']})",
                    )
                    for pos, intensity in zip(metrics["peak_positions"], peak_intensities):
                        axes[2].annotate(
                            f"{pos:.1f}°",
                            xy=(pos, intensity),
                            xytext=(0, 8),
                            textcoords="offset points",
                            ha="center", fontsize=8, color="red",
                        )
                axes[2].set_xlabel(r"$\gamma$ (deg)", fontsize=13)
                axes[2].set_ylabel("Intensity", fontsize=13)
                axes[2].set_title(
                    f"Intensity vs. gamma  ROI: {tth_range[0]}-{tth_range[1]} deg",
                    fontsize=13,
                )
                axes[2].set_yscale("log")
                axes[2].legend(fontsize=10)

                # Panel 4: CWT scalogram
                coeffs = metrics["cwt_coefficients"]  # (n_scales, npt_azim)
                n_scales = coeffs.shape[0]
                d_gamma = float(np.median(np.diff(azimuth)))
                scale_axis_min = 1 * d_gamma
                scale_axis_max = n_scales * d_gamma
                axes[3].imshow(
                    np.abs(coeffs),
                    aspect="auto",
                    origin="lower",
                    extent=[azimuth.min(), azimuth.max(), scale_axis_min, scale_axis_max],
                    cmap="hot",
                )
                axes[3].set_xlabel(r"$\gamma$ (deg)", fontsize=13)
                axes[3].set_ylabel("Scale (deg)", fontsize=13)
                axes[3].set_title(
                    f"CWT Scalogram (Mexican hat)  —  dominant scale: "
                    f"{metrics['cwt_dominant_scale_deg']:.1f}°",
                    fontsize=11,
                )

                for ax in axes:
                    ax.tick_params(axis="both", which="major", labelsize=12)

                plt.tight_layout()
                plt.savefig(output_png, dpi=150)
                plt.close()

                print(f"Saved: {output_dat}, {output_png}")

                subfolder_results[base_name] = {
                    "scan_point": scan_point,
                    "tth": tth,
                    "azimuth": azimuth,
                    "intensity_2d": intensity_2d,
                    "intensity_roi": intensity_roi,
                    "mean": mean_i,
                    "std": std_i,
                    **metrics,
                }

                csv_rows.append({
                    "sample": base_name,
                    "scan_point": scan_point,
                    **{k: metrics[k] for k in scalar_keys},
                })

            except Exception as exc:
                print(f"Failed to process {input_path}: {exc}")

        # Write one CSV per subfolder
        if csv_rows:
            csv_path = os.path.join(dirpath, "ring_metrics_summary.csv")
            _write_subfolder_csv(csv_path, csv_rows, scalar_keys)

        all_results[subfolder_label] = subfolder_results

    return all_results


if __name__ == "__main__":
    azimuthal_ring_statistics(
        input_directory=Inputs.root_dir,
        poni_file=Inputs.poni_file,
    )
