"""
cwt_statistics.py
-----------------
Walks the raw/ subdirectory of each SampleData folder under Inputs.root_dir,
finds every *_master.h5 file (Dectris Eiger NeXus format), reads the detector
image from entry/data/data_000001, integrates azimuthally via pyFAI, computes
ring metrics via ring_metrics_cwt.compute_ring_statistics(), and writes:

  - Per-image diagnostic PNGs with eight rows:
      0 — Full caked image (log) with ROI boundaries
      1 — ROI interior heatmap
      2 — Intensity vs gamma with peak markers and metrics box
      3 — CWT scalogram
      4 — Fourier spectrum
      5 — find_peaks intensity histogram
      6 — CWT intensity histogram (same quantity as row 5, different detector)
      7 — CWT scale-max power histogram

  - Per-subfolder ring_metrics_summary.csv

Comparing rows 5 and 6 reveals agreement or disagreement between the
find_peaks and CWT detectors. Row 7 confirms which CWT detections have
strong, scale-consistent wavelet responses (likely real texture spots).

Run independently:
    python cwt_statistics.py
"""

import os
import re
import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from ring_metrics_cwt import compute_ring_statistics
import Inputs


# ---------------------------------------------------------------------------
# Private helpers  (unchanged from azimuthal_ring_statistics.py)
# ---------------------------------------------------------------------------

def _extract_scan_point(base_name: str) -> int:
    """
    Extract the trailing integer scan point index from a filename stem.

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

    Parameters
    ----------
    csv_path : str
        Full output path for the CSV file.
    rows : list of dict
        One dict per scan point, already sorted by scan_point.
    n_rings : int
        Number of rings.
    scalar_keys : list of str
        Metric column names to include (excluding sample and scan_point).
    """
    ring_cols  = [f"ring{i}_{key}" for i in range(n_rings) for key in scalar_keys]
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


# ---------------------------------------------------------------------------
# Extended plot function — rows 0-4 identical to plot_ring_metrics,
# row 5 adds per-peak intensity histogram
# ---------------------------------------------------------------------------

def _plot_ring_metrics_with_cwt_histograms(
    rings: list[dict],
    base_name: str,
    output_png: str,
) -> None:
    """
    Plot ring diagnostics with three appended histogram rows.

    Rows 0-4 reproduce the layout of plot_ring_metrics() exactly:
      0 — Full caked image (log) with ROI boundaries
      1 — ROI interior heatmap
      2 — Intensity vs gamma with peak markers and metrics box
      3 — CWT scalogram
      4 — Fourier spectrum

    Row 5 (from GSD_statistics.py):
      Histogram of mean_i at each find_peaks detection (n_texture_peaks).

    Row 6 (new):
      Histogram of mean_i at each CWT peak position. Same x-axis quantity
      as Row 5, different detector — directly comparable.

    Row 7 (new):
      Histogram of scale-averaged CWT power at each CWT peak position.
      Computed as the mean across all scales of coeffs[:, peak_idx]**2,
      giving a per-peak measure of wavelet response strength independent
      of absolute intensity.

    Parameters
    ----------
    rings : list of dict
        Same structure as accepted by plot_ring_metrics(). Each dict must
        contain mean_i, std_i, azimuth, metrics, tth_range, intensity_2d,
        tth, intensity_roi, tth_roi.
    base_name : str
        Sample identifier used in the suptitle.
    output_png : str
        Full path for the saved PNG.
    """
    def _fmt(v, digits=3):
        return f"{v:.{digits}f}" if np.isfinite(v) else "nan"

    n_rings   = len(rings)
    fig_width = max(11, 7 * n_rings)

    fig, axes = plt.subplots(
        8, n_rings,
        figsize=(fig_width, 34),
        gridspec_kw={"height_ratios": [3, 1, 3, 2, 2, 2, 2, 2]},
        squeeze=False,
    )

    for ci, ring in enumerate(rings):
        mean_i        = ring["mean_i"]
        std_i         = ring["std_i"]
        azimuth       = ring["azimuth"]
        metrics       = ring["metrics"]
        tth_range     = ring.get("tth_range")
        intensity_2d  = ring.get("intensity_2d")
        tth           = ring.get("tth")
        intensity_roi = ring.get("intensity_roi")
        tth_roi       = ring.get("tth_roi")

        d_gamma    = float(np.median(np.diff(azimuth)))
        n_scales   = metrics["cwt_coefficients"].shape[0]
        scale_axis = np.arange(1, n_scales + 1) * d_gamma

        col_title = (
            f"ROI: {tth_range[0]}–{tth_range[1]}°"
            if tth_range is not None else f"Ring {ci}"
        )

        # ------------------------------------------------------------------
        # Row 0: Full caked image (log) with ROI boundaries
        # ------------------------------------------------------------------
        ax0 = axes[0, ci]
        if intensity_2d is not None and tth is not None:
            pos_2d = intensity_2d.copy()
            pos_2d[pos_2d <= 0] = np.nan
            ax0.imshow(
                pos_2d,
                aspect="auto", origin="lower",
                extent=[tth.min(), tth.max(), azimuth.min(), azimuth.max()],
                cmap="viridis",
                norm=LogNorm(
                    vmin=np.nanpercentile(pos_2d, 1),
                    vmax=np.nanpercentile(pos_2d, 99),
                ),
            )
            if tth_range is not None:
                ax0.axvline(tth_range[0], color="red", lw=1.2, ls="--", label="ROI")
                ax0.axvline(tth_range[1], color="red", lw=1.2, ls="--")
                ax0.legend(fontsize=9, loc="upper right")
            ax0.set_xlabel(r"2$\theta$ (deg)", fontsize=11)
            if ci == 0:
                ax0.set_ylabel(r"$\gamma$ (deg)", fontsize=11)
        else:
            ax0.set_visible(False)
        ax0.set_title(col_title, fontsize=12, fontweight="bold")
        ax0.tick_params(axis="both", which="major", labelsize=10)

        # ------------------------------------------------------------------
        # Row 1: ROI heatmap
        # ------------------------------------------------------------------
        ax1 = axes[1, ci]
        if intensity_roi is not None and tth_roi is not None:
            roi_pos = intensity_roi.T.copy()
            roi_pos[roi_pos <= 0] = np.nan
            ax1.imshow(
                roi_pos,
                aspect="auto", origin="lower",
                extent=[azimuth.min(), azimuth.max(), tth_roi.min(), tth_roi.max()],
                cmap="viridis",
                norm=LogNorm(
                    vmin=np.nanpercentile(roi_pos, 1),
                    vmax=np.nanpercentile(roi_pos, 99),
                ),
            )
            ax1.set_xlabel(r"$\gamma$ (deg)", fontsize=11)
            if ci == 0:
                ax1.set_ylabel(r"2$\theta$ (deg)", fontsize=11)
            ax1.set_title("ROI heatmap", fontsize=10)
        else:
            ax1.set_visible(False)
        ax1.tick_params(axis="both", which="major", labelsize=10)

        # ------------------------------------------------------------------
        # Row 2: Intensity vs gamma with peak markers
        # ------------------------------------------------------------------
        ax2 = axes[2, ci]
        ax2.plot(azimuth, mean_i, lw=2, color="steelblue", label="Mean over ROI")
        ax2.fill_between(
            azimuth, mean_i - std_i, mean_i + std_i,
            alpha=0.3, color="steelblue", label="±1 std",
        )

        if metrics["n_texture_peaks"] > 0:
            peak_intensities = mean_i[
                np.round(np.interp(
                    metrics["peak_positions"], azimuth, np.arange(len(azimuth))
                )).astype(int)
            ]
            ax2.scatter(
                metrics["peak_positions"], peak_intensities,
                color="red", zorder=5, s=50,
                label=f"find_peaks (n={metrics['n_texture_peaks']})",
            )
            for pos, intensity in zip(metrics["peak_positions"], peak_intensities):
                ax2.annotate(
                    f"{pos:.1f}°", xy=(pos, intensity),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="red",
                )

        if metrics["cwt_n_peaks"] > 0:
            cwt_intensities = mean_i[
                np.round(np.interp(
                    metrics["cwt_peak_positions"], azimuth, np.arange(len(azimuth))
                )).astype(int)
            ]
            ax2.scatter(
                metrics["cwt_peak_positions"], cwt_intensities,
                color="orange", zorder=5, s=50, marker="^",
                label=f"CWT peaks (n={metrics['cwt_n_peaks']})",
            )

        metrics_str = (
            f"CV={_fmt(metrics['cv'])}  P/V={_fmt(metrics['peak_valley'], 2)}  "
            f"TI={_fmt(metrics['texture_index'], 3)}\n"
            f"Kurt={_fmt(metrics['kurtosis'], 2)}  Skew={_fmt(metrics['skewness'], 2)}  "
            f"Warren={_fmt(metrics['warren_grain_proxy'], 4)}\n"
            f"Entropy={_fmt(metrics['entropy'])}  ACF={_fmt(metrics['acf_length_deg'], 1)}°  "
            f"FSI={_fmt(metrics['fiber_symmetry_index'], 3)}\n"
            f"C2={_fmt(metrics['fourier_c2'], 3)}  C4={_fmt(metrics['fourier_c4'], 3)}  "
            f"C6={_fmt(metrics['fourier_c6'], 3)}  ArcImbal={_fmt(metrics['arc_imbalance'], 3)}\n"
            f"FWHM={_fmt(metrics['peak_fwhm_mean_deg'], 1)}°"
            f"±{_fmt(metrics['peak_fwhm_std_deg'], 1)}°  "
            f"PkAsym={_fmt(metrics['peak_asymmetry_mean'], 3)}\n"
            f"CWT power={_fmt(metrics['cwt_total_power'], 2)}  "
            f"dom.scale={_fmt(metrics['cwt_dominant_scale_deg'], 1)}°  "
            f"scale_ent={_fmt(metrics['cwt_scale_entropy'], 2)}"
        )
        ax2.text(
            0.02, 0.97, metrics_str, transform=ax2.transAxes,
            fontsize=7.5, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.78),
        )
        ax2.set_xlabel(r"$\gamma$ (deg)", fontsize=11)
        ax2.set_yscale("log")
        ax2.legend(fontsize=8)
        ax2.tick_params(axis="both", which="major", labelsize=10)
        if ci == 0:
            ax2.set_ylabel("Intensity", fontsize=12)

        # ------------------------------------------------------------------
        # Row 3: CWT scalogram
        # ------------------------------------------------------------------
        ax3 = axes[3, ci]
        coeffs = metrics["cwt_coefficients"]
        ax3.imshow(
            np.abs(coeffs),
            aspect="auto", origin="lower",
            extent=[azimuth.min(), azimuth.max(), scale_axis.min(), scale_axis.max()],
            cmap="hot",
        )
        ax3.set_xlabel(r"$\gamma$ (deg)", fontsize=11)
        ax3.set_title(
            f"CWT Scalogram — dom. scale: {metrics['cwt_dominant_scale_deg']:.1f}°",
            fontsize=10,
        )
        ax3.tick_params(axis="both", which="major", labelsize=10)
        if ci == 0:
            ax3.set_ylabel("Scale (deg)", fontsize=11)

        # ------------------------------------------------------------------
        # Row 4: Fourier spectrum
        # ------------------------------------------------------------------
        ax4 = axes[4, ci]
        s        = np.nan_to_num(mean_i, nan=float(np.nanmean(mean_i)))
        fft_amps = np.abs(np.fft.rfft(s)) / len(s)
        c0_amp   = fft_amps[0]
        orders   = np.arange(len(fft_amps))
        norm_amps = (
            np.where(orders == 0, fft_amps / c0_amp, 2 * fft_amps / c0_amp)
            if c0_amp > 0 else fft_amps
        )
        max_order = min(13, len(orders))
        ax4.bar(
            orders[:max_order], norm_amps[:max_order],
            color="steelblue", alpha=0.75, width=0.6,
        )
        for order, key in [(2, "fourier_c2"), (4, "fourier_c4"), (6, "fourier_c6")]:
            if order < max_order and np.isfinite(metrics[key]):
                ax4.annotate(
                    f"C{order}={metrics[key]:.3f}",
                    xy=(order, norm_amps[order]),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=8, color="darkred",
                )
        ax4.set_xlabel("Fourier Order", fontsize=11)
        ax4.set_title(
            "Fourier Spectrum  (C2=two-fold, C4=four-fold, C6=six-fold)", fontsize=9,
        )
        ax4.set_xticks(orders[:max_order])
        ax4.tick_params(axis="both", which="major", labelsize=10)
        ax4.grid(axis="y", lw=0.5, alpha=0.4)
        if ci == 0:
            ax4.set_ylabel("Normalized Amplitude", fontsize=11)

        # ------------------------------------------------------------------
        # Row 5 (NEW): Histogram of intensities at find_peaks detections
        # ------------------------------------------------------------------
        ax5 = axes[5, ci]
        n_peaks = metrics["n_texture_peaks"]

        if n_peaks == 0:
            ax5.text(
                0.5, 0.5, "No texture peaks detected",
                transform=ax5.transAxes, ha="center", va="center",
                fontsize=11, color="gray",
            )
            ax5.set_axis_off()
        else:
            # Map peak_positions (azimuth degrees) back to array indices,
            # matching the interp+round approach used in Row 2 above.
            peak_idx = np.round(
                np.interp(
                    metrics["peak_positions"],
                    azimuth,
                    np.arange(len(azimuth)),
                )
            ).astype(int)
            peak_intensities = mean_i[peak_idx]   # one value per detected peak

            # 'auto' binning: with small n_peaks this naturally gives
            # one bin per peak, which is the most informative view.
            ax5.hist(
                peak_intensities,
                bins="auto",
                color="tomato",
                edgecolor="white",
                linewidth=0.6,
                alpha=0.85,
            )
            ax5.set_xlabel("Intensity at peak position", fontsize=11)
            ax5.set_title(
                f"Peak intensity histogram  (n_texture_peaks = {n_peaks})",
                fontsize=10,
            )
            ax5.tick_params(axis="both", which="major", labelsize=10)
            ax5.grid(axis="x", lw=0.5, alpha=0.35)

        if ci == 0:
            ax5.set_ylabel("Count", fontsize=11)

        # ------------------------------------------------------------------
        # Row 6 (NEW): Histogram of mean_i at CWT peak positions
        # ------------------------------------------------------------------
        ax6 = axes[6, ci]
        cwt_n = metrics["cwt_n_peaks"]

        if cwt_n == 0:
            ax6.text(
                0.5, 0.5, "No CWT peaks detected",
                transform=ax6.transAxes, ha="center", va="center",
                fontsize=11, color="gray",
            )
            ax6.set_axis_off()
        else:
            cwt_idx = np.round(
                np.interp(
                    metrics["cwt_peak_positions"],
                    azimuth,
                    np.arange(len(azimuth)),
                )
            ).astype(int)
            cwt_mean_i = mean_i[cwt_idx]

            ax6.hist(
                cwt_mean_i,
                bins="auto",
                color="darkorange",
                edgecolor="white",
                linewidth=0.6,
                alpha=0.85,
            )
            ax6.set_xlabel("mean_i at CWT peak position", fontsize=11)
            ax6.set_title(
                f"CWT peak intensity histogram  (cwt_n_peaks = {cwt_n})",
                fontsize=10,
            )
            ax6.tick_params(axis="both", which="major", labelsize=10)
            ax6.grid(axis="x", lw=0.5, alpha=0.35)

        if ci == 0:
            ax6.set_ylabel("Count", fontsize=11)

        # ------------------------------------------------------------------
        # Row 7 (NEW): Histogram of scale-max CWT power at CWT peaks
        # ------------------------------------------------------------------
        ax7 = axes[7, ci]

        if cwt_n == 0:
            ax7.text(
                0.5, 0.5, "No CWT peaks detected",
                transform=ax7.transAxes, ha="center", va="center",
                fontsize=11, color="gray",
            )
            ax7.set_axis_off()
        else:
            # coeffs shape: (n_scales, npt_azim)
            # Reproduce the scale_max_power signal that find_peaks ran on,
            # then sample it at the detected peak indices.
            coeffs = metrics["cwt_coefficients"]
            scale_max_power = np.max(coeffs ** 2, axis=0)
            cwt_peak_power = scale_max_power[cwt_idx]

            ax7.hist(
                cwt_peak_power,
                bins="auto",
                color="mediumpurple",
                edgecolor="white",
                linewidth=0.6,
                alpha=0.85,
            )
            ax7.set_xlabel("Scale-max CWT power at peak", fontsize=11)
            ax7.set_title(
                f"CWT scale-max power histogram  (cwt_n_peaks = {cwt_n})",
                fontsize=10,
            )
            ax7.tick_params(axis="both", which="major", labelsize=10)
            ax7.grid(axis="x", lw=0.5, alpha=0.35)

        if ci == 0:
            ax7.set_ylabel("Count", fontsize=11)

    fig.suptitle(f"{base_name} — Ring Metrics", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def cwt_statistics(
    input_directory: str,
    poni_file: str,
    tth_ranges: list[tuple] = ((15.5, 17),),
    npt_rad: int = 200,
    npt_azim: int = 360,
) -> dict:
    """
    Analyze intensity along azimuthal rings for all TIFF images in a directory
    tree, producing the same outputs as GSD_statistics() plus two additional
    CWT histogram rows (6 and 7) in each per-image diagnostic PNG.

    Parameters
    ----------
    input_directory : str
        Root directory containing subfolders of TIFF images.
    poni_file : str
        Path to the pyFAI calibration (.poni) file.
    tth_ranges : list of tuple of float, optional
        Each entry is a (tth_lo, tth_hi) pair in degrees.
    npt_rad : int, optional
        Number of radial bins for integrate2d.
    npt_azim : int, optional
        Number of azimuthal bins.

    Returns
    -------
    dict
        results[subfolder][base_name] with 'scan_point', 'tth', 'azimuth',
        'intensity_2d', and 'rings'.
    """
    scalar_keys = [
        "mean", "std", "cv", "peak_valley", "skewness", "kurtosis",
        "entropy", "acf_length_deg", "n_texture_peaks", "completeness", "integrated",
        "texture_index",
        "peak_fwhm_mean_deg", "peak_fwhm_std_deg", "peak_asymmetry_mean",
        "fiber_symmetry_index",
        "fourier_c2", "fourier_c4", "fourier_c6",
        "arc_imbalance",
        "warren_grain_proxy",
        "cwt_n_peaks", "cwt_dominant_scale_deg", "cwt_total_power", "cwt_scale_entropy",
    ]

    n_rings = len(tth_ranges)

    ai = AzimuthalIntegrator()
    ai.load(str(poni_file))

    subfolders: dict[str, list] = {}
    for dirpath, _, filenames in os.walk(input_directory):
        # Only look inside raw/ subdirectories for master HDF5 files.
        if os.path.basename(dirpath) != "raw":
            continue
        for filename in filenames:
            if not filename.endswith("_master.h5"):
                continue
            base_name = filename[: -len("_master.h5")]   # e.g. "scan_point_0"
            scan_point = _extract_scan_point(base_name)
            subfolders.setdefault(dirpath, []).append(
                (scan_point, filename, base_name)
            )

    for dirpath in subfolders:
        subfolders[dirpath].sort(key=lambda x: x[0])

    all_results = {}

    for dirpath, file_list in subfolders.items():
        # Output goes to the SampleData folder (parent of raw/).
        output_dir = os.path.dirname(dirpath)
        subfolder_label = os.path.relpath(output_dir, input_directory) or "root"
        subfolder_results = {}
        csv_rows = []

        for scan_point, filename, base_name in file_list:
            input_path = os.path.join(dirpath, filename)
            output_png = os.path.join(output_dir, f"{base_name}_ring_stats.png")

            try:
                with h5py.File(input_path, "r") as f:
                    image = f["entry/data/data_000001"][0].astype(np.float32)

                intensity_2d, tth, azimuth = ai.integrate2d(
                    image,
                    npt_rad=npt_rad,
                    npt_azim=npt_azim,
                    azimuth_range=(-180, 180),
                    unit="2th_deg",
                )

                rings_data       = []
                csv_ring_metrics = {}

                for ring_idx, tth_range in enumerate(tth_ranges):
                    tth_mask      = (tth >= tth_range[0]) & (tth <= tth_range[1])
                    tth_roi       = tth[tth_mask]
                    intensity_roi = intensity_2d[:, tth_mask]

                    mean_i = np.nanmean(intensity_roi, axis=1)
                    std_i  = np.nanstd(intensity_roi, axis=1)

                    metrics = compute_ring_statistics(mean_i, azimuth)

                    output_dat = os.path.join(
                        output_dir, f"{base_name}_ring{ring_idx}_stats.dat"
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
                        "intensity_2d":  intensity_2d,
                        "tth":           tth,
                        "intensity_roi": intensity_roi,
                        "tth_roi":       tth_roi,
                    })

                    csv_ring_metrics[ring_idx] = {k: metrics[k] for k in scalar_keys}

                # Use local plot function (rows 0-4 unchanged, rows 5-7 new)
                _plot_ring_metrics_with_cwt_histograms(rings_data, base_name, output_png)

                csv_row = {"sample": base_name, "scan_point": scan_point}
                for ring_idx in range(n_rings):
                    csv_row[ring_idx] = csv_ring_metrics[ring_idx]
                csv_rows.append(csv_row)

                subfolder_results[base_name] = {
                    "scan_point":   scan_point,
                    "tth":          tth,
                    "azimuth":      azimuth,
                    "intensity_2d": intensity_2d,
                    "rings":        rings_data,
                }

            except Exception as exc:
                print(f"Failed to process {input_path}: {exc}")

        if csv_rows:
            csv_path = os.path.join(output_dir, "ring_metrics_summary.csv")
            _write_subfolder_csv(csv_path, csv_rows, n_rings, scalar_keys)

        all_results[subfolder_label] = subfolder_results

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cwt_statistics(
        input_directory=Inputs.root_dir,
        poni_file=Inputs.poni_file,
        tth_ranges=Inputs.tth_ranges,
    )
