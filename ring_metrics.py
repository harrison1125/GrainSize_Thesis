"""
ring_metrics.py
---------------
Statistical metrics for characterizing diffraction ring profiles.

Intended to be called with the output of azimuthal_ring_statistics.py.
Pass the per-azimuthal-bin mean intensity array and azimuth axis to
compute_ring_statistics() to get a dict of scalar metrics, then call
plot_ring_metrics() to produce the annotated diagnostic plot.

Typical usage
-------------
    from ring_analysis.azimuthal_ring_statistics import azimuthal_ring_statistics
    from ring_analysis.ring_metrics import compute_ring_statistics, plot_ring_metrics

    results = azimuthal_ring_statistics(input_dir, poni_file)
    for name, r in results.items():
        metrics = compute_ring_statistics(r["mean"], r["azimuth"])
        plot_ring_metrics(r["mean"], r["std"], r["azimuth"], metrics, name, output_png)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, cwt


def ricker(points: int, a: float) -> np.ndarray:
    """
    Mexican hat (Ricker) wavelet, equivalent to the removed scipy.signal.ricker.

    Parameters
    ----------
    points : int
        Number of points in the wavelet.
    a : float
        Width parameter (scale).

    Returns
    -------
    np.ndarray
        Wavelet values.
    """
    A = 2 / (np.sqrt(3 * a) * np.pi ** 0.25)
    wsq = a ** 2
    vec = np.arange(0, points) - (points - 1.0) / 2
    tsq = vec ** 2
    mod = 1 - tsq / wsq
    gauss = np.exp(-tsq / (2 * wsq))
    return A * mod * gauss
from scipy.stats import kurtosis, skew


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _autocorrelation_length(signal: np.ndarray) -> float:
    """
    Estimate the autocorrelation length of a 1D periodic signal (ring).

    Computes the normalized circular autocorrelation via FFT and returns the
    first lag (in bins) at which it drops below 1/e. A short ACF length
    indicates many fine grains contributing per azimuthal bin; a long ACF
    indicates few coarse grains dominating the ring.

    Parameters
    ----------
    signal : np.ndarray
        1D intensity array sampled uniformly around the ring.

    Returns
    -------
    float
        Lag index at which autocorrelation falls below 1/e, or len(signal)/2
        if it never does (signal is constant or has very long-range order).
    """
    s = signal - np.nanmean(signal)
    n = len(s)
    f = np.fft.fft(s, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    acf /= acf[0] if acf[0] != 0 else 1.0
    below = np.where(acf < 1 / np.e)[0]
    return float(below[0]) if len(below) > 0 else float(n // 2)


def _ring_entropy(signal: np.ndarray) -> float:
    """
    Compute the Shannon entropy of the normalized azimuthal intensity distribution.

    A uniform powder ring has maximum entropy. A strongly textured sample
    concentrates intensity at specific azimuths and therefore has low entropy.

    Parameters
    ----------
    signal : np.ndarray
        1D intensity array sampled around the ring. Must be non-negative.

    Returns
    -------
    float
        Shannon entropy in nats. Returns nan if the signal sums to zero.
    """
    s = np.nan_to_num(signal, nan=0.0)
    s = np.clip(s, 0, None)
    total = s.sum()
    if total == 0:
        return np.nan
    p = s / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _wavelet_peak_analysis(
    signal: np.ndarray,
    d_gamma: float,
    scales: np.ndarray = None,
) -> dict:
    """
    Characterize the azimuthal intensity peak distribution using a
    continuous wavelet transform (CWT) with the Mexican hat (Ricker) wavelet.

    The Mexican hat wavelet is the negative normalized second derivative of a
    Gaussian. It responds maximally to localized intensity bumps whose width
    matches the wavelet scale, making it well-suited for detecting texture
    spots of varying angular width on a smooth background.

    The CWT produces a 2D coefficient matrix (scale x position). Peaks in
    the scale-averaged power indicate where texture spots are located
    regardless of their angular width. The dominant scale at each peak
    reflects the angular width of the corresponding grain cluster.

    Parameters
    ----------
    signal : np.ndarray
        1D mean intensity array sampled uniformly around the ring,
        shape (npt_azim,).
    d_gamma : float
        Angular spacing between bins in degrees.
    scales : np.ndarray, optional
        Wavelet scales to evaluate (in bins). Defaults to 1–30 bins
        (1°–30° at 1°/bin resolution), covering single-grain spots up
        to broad texture components.

    Returns
    -------
    dict with keys:
        cwt_n_peaks         : int
            Number of peaks detected in the scale-averaged CWT power profile.
            More robust than simple find_peaks because it integrates evidence
            across multiple scales before detecting.
        cwt_peak_positions  : list of float
            Gamma positions (degrees) of CWT-detected peaks.
        cwt_dominant_scale_deg : float
            The wavelet scale (degrees) at which total CWT power is maximized,
            i.e. the most prevalent angular width of intensity features.
            Small values (~1–5°) indicate sharp spots (coarse grains);
            large values (~10–30°) indicate broad texture lobes (fiber texture).
        cwt_total_power     : float
            Sum of squared CWT coefficients across all scales and positions,
            normalized by signal length. Measures the total non-uniformity
            energy in the ring — high values indicate strong localized features,
            near zero indicates a flat uniform ring.
        cwt_scale_entropy   : float
            Shannon entropy of the scale-averaged power spectrum. Low entropy
            means power is concentrated at one scale (peaks have a characteristic
            width); high entropy means features exist at many scales simultaneously
            (complex multi-scale texture or noise).
        cwt_coefficients    : np.ndarray
            Full 2D CWT coefficient array (n_scales x npt_azim), retained for
            optional downstream visualization.
    """
    if scales is None:
        scales = np.arange(1, 31)  # 1–30 bins

    s = np.nan_to_num(signal, nan=float(np.nanmean(signal)))

    # Compute CWT with Ricker (Mexican hat) wavelet
    coeffs = cwt(s, ricker, scales)  # shape: (n_scales, npt_azim)

    # Scale-averaged power: mean squared coefficient per azimuthal position
    scale_avg_power = np.mean(coeffs ** 2, axis=0)  # shape: (npt_azim,)

    # Detect peaks in scale-averaged power
    if scale_avg_power.max() > 0:
        prominence = scale_avg_power.std() * 0.5
        peaks_idx, _ = find_peaks(scale_avg_power, prominence=prominence, distance=5)
    else:
        peaks_idx = np.array([], dtype=int)

    cwt_peak_positions = (peaks_idx * d_gamma - 180.0).tolist()

    # Dominant scale: which scale has the most total power
    scale_power = np.sum(coeffs ** 2, axis=1)  # shape: (n_scales,)
    dominant_scale_idx = int(np.argmax(scale_power))
    dominant_scale_deg = float(scales[dominant_scale_idx] * d_gamma)

    # Total normalized power
    cwt_total_power = float(np.sum(coeffs ** 2) / len(s))

    # Scale entropy: how spread is the power across scales
    scale_power_norm = scale_power / scale_power.sum() if scale_power.sum() > 0 else scale_power
    scale_power_norm = scale_power_norm[scale_power_norm > 0]
    cwt_scale_entropy = float(-np.sum(scale_power_norm * np.log(scale_power_norm)))

    return {
        "cwt_n_peaks":            int(len(peaks_idx)),
        "cwt_peak_positions":     cwt_peak_positions,
        "cwt_dominant_scale_deg": dominant_scale_deg,
        "cwt_total_power":        cwt_total_power,
        "cwt_scale_entropy":      cwt_scale_entropy,
        "cwt_coefficients":       coeffs,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ring_statistics(
    mean_i: np.ndarray,
    azimuth: np.ndarray,
    noise_threshold: float = None,
) -> dict:
    """
    Compute a suite of statistics from the azimuthal intensity profile of a ring.

    Parameters
    ----------
    mean_i : np.ndarray
        Mean intensity per azimuthal bin, shape (npt_azim,).
    azimuth : np.ndarray
        Azimuthal angle axis in degrees, shape (npt_azim,).
    noise_threshold : float, optional
        Intensity below which bins are considered noise/background.
        Defaults to the 5th percentile of mean_i.

    Returns
    -------
    dict
        Scalar metrics:

        mean                  : float
            Mean intensity around the ring.
        std                   : float
            Standard deviation of intensity around the ring.
        cv                    : float
            Coefficient of variation (std/mean). Primary texture proxy;
            CV ~ 0 for a uniform powder, higher for preferred orientation.
        peak_valley           : float
            max/min intensity ratio. Intuitive texture metric; 1.0 = flat ring.
        skewness              : float
            Asymmetry of the gamma intensity distribution.
        kurtosis              : float
            Excess kurtosis. High positive = spotty ring from coarse grains.
        entropy               : float
            Shannon entropy of the normalized intensity distribution (nats).
        acf_length_deg        : float
            Autocorrelation decay length in degrees.
        n_texture_peaks       : int
            Number of distinct intensity maxima from simple find_peaks.
        peak_positions        : list of float
            Gamma positions (degrees) of find_peaks detections.
        completeness          : float
            Fraction of azimuthal bins above noise threshold (0-1).
        integrated            : float
            Sum of mean_i * d_gamma. Phase volume fraction proxy.
        cwt_n_peaks           : int
            Number of peaks in scale-averaged CWT power. More robust than
            n_texture_peaks because it integrates across scales.
        cwt_peak_positions    : list of float
            Gamma positions (degrees) of CWT-detected peaks.
        cwt_dominant_scale_deg : float
            Most prevalent angular feature width in degrees. Small = sharp
            spots (coarse grains); large = broad lobes (fiber texture).
        cwt_total_power       : float
            Total normalized CWT power. Near zero = flat ring; high = strong
            localized features.
        cwt_scale_entropy     : float
            Entropy of scale power distribution. Low = single characteristic
            feature width; high = multi-scale or noisy texture.
        cwt_coefficients      : np.ndarray
            Full 2D CWT array (n_scales x npt_azim) for visualization.
    """
    if noise_threshold is None:
        noise_threshold = np.nanpercentile(mean_i, 5)

    d_gamma = float(np.median(np.diff(azimuth)))

    mean_val    = float(np.nanmean(mean_i))
    std_val     = float(np.nanstd(mean_i))
    cv          = std_val / mean_val if mean_val > 0 else np.nan
    peak_valley = float(np.nanmax(mean_i) / np.nanmax([np.nanmin(mean_i), 1e-9]))
    skewness    = float(skew(mean_i[~np.isnan(mean_i)]))
    kurt        = float(kurtosis(mean_i[~np.isnan(mean_i)]))
    entropy     = _ring_entropy(mean_i)
    acf_bins    = _autocorrelation_length(mean_i)
    acf_deg     = acf_bins * d_gamma

    prominence = std_val * 0.5
    peaks_idx, _ = find_peaks(mean_i, prominence=prominence, distance=5)
    peak_positions = azimuth[peaks_idx].tolist()

    completeness = float(np.sum(mean_i > noise_threshold) / len(mean_i))
    integrated   = float(np.nansum(mean_i) * d_gamma)

    wavelet = _wavelet_peak_analysis(mean_i, d_gamma)

    return {
        "mean":                   mean_val,
        "std":                    std_val,
        "cv":                     cv,
        "peak_valley":            peak_valley,
        "skewness":               skewness,
        "kurtosis":               kurt,
        "entropy":                entropy,
        "acf_length_deg":         acf_deg,
        "n_texture_peaks":        len(peaks_idx),
        "peak_positions":         peak_positions,
        "completeness":           completeness,
        "integrated":             integrated,
        "cwt_n_peaks":            wavelet["cwt_n_peaks"],
        "cwt_peak_positions":     wavelet["cwt_peak_positions"],
        "cwt_dominant_scale_deg": wavelet["cwt_dominant_scale_deg"],
        "cwt_total_power":        wavelet["cwt_total_power"],
        "cwt_scale_entropy":      wavelet["cwt_scale_entropy"],
        "cwt_coefficients":       wavelet["cwt_coefficients"],
    }


def plot_ring_metrics(
    mean_i: np.ndarray,
    std_i: np.ndarray,
    azimuth: np.ndarray,
    metrics: dict,
    base_name: str,
    output_png: str,
    tth_range: tuple = None,
) -> None:
    """
    Plot the azimuthal intensity profile with annotated ring metrics,
    including a CWT scalogram panel.

    Parameters
    ----------
    mean_i : np.ndarray
        Mean intensity per azimuthal bin.
    std_i : np.ndarray
        Standard deviation per azimuthal bin.
    azimuth : np.ndarray
        Azimuthal angle axis in degrees.
    metrics : dict
        Output of compute_ring_statistics().
    base_name : str
        Sample name used in the plot title.
    output_png : str
        Full path for the saved figure.
    tth_range : tuple of float, optional
        If provided, shown in the plot title for context.
    """
    d_gamma = float(np.median(np.diff(azimuth)))
    n_scales = metrics["cwt_coefficients"].shape[0]
    scale_axis = np.arange(1, n_scales + 1) * d_gamma  # degrees

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # Panel 1: intensity profile with peak markers
    axes[0].plot(azimuth, mean_i, lw=2, color="steelblue", label="Mean over ROI")
    axes[0].fill_between(
        azimuth,
        mean_i - std_i,
        mean_i + std_i,
        alpha=0.3, color="steelblue", label="±1 std",
    )

    # find_peaks markers (red)
    if metrics["n_texture_peaks"] > 0:
        peak_intensities = mean_i[
            np.round(np.interp(
                metrics["peak_positions"], azimuth, np.arange(len(azimuth))
            )).astype(int)
        ]
        axes[0].scatter(
            metrics["peak_positions"], peak_intensities,
            color="red", zorder=5, s=50,
            label=f"find_peaks (n={metrics['n_texture_peaks']})",
        )
        for pos, intensity in zip(metrics["peak_positions"], peak_intensities):
            axes[0].annotate(
                f"{pos:.1f}°", xy=(pos, intensity),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, color="red",
            )

    # CWT peak markers (orange)
    if metrics["cwt_n_peaks"] > 0:
        cwt_intensities = mean_i[
            np.round(np.interp(
                metrics["cwt_peak_positions"], azimuth, np.arange(len(azimuth))
            )).astype(int)
        ]
        axes[0].scatter(
            metrics["cwt_peak_positions"], cwt_intensities,
            color="orange", zorder=5, s=50, marker="^",
            label=f"CWT peaks (n={metrics['cwt_n_peaks']})",
        )

    metrics_str = (
        f"CV={metrics['cv']:.3f}  P/V={metrics['peak_valley']:.2f}\n"
        f"Kurt={metrics['kurtosis']:.2f}  Skew={metrics['skewness']:.2f}\n"
        f"Entropy={metrics['entropy']:.3f}  ACF={metrics['acf_length_deg']:.1f}°\n"
        f"CWT power={metrics['cwt_total_power']:.2f}  "
        f"dom.scale={metrics['cwt_dominant_scale_deg']:.1f}°  "
        f"scale_ent={metrics['cwt_scale_entropy']:.2f}"
    )
    axes[0].text(
        0.02, 0.97, metrics_str, transform=axes[0].transAxes,
        fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7),
    )
    axes[0].set_ylabel("Intensity", fontsize=13)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[0].tick_params(axis="both", which="major", labelsize=11)

    title = f"{base_name} — Ring Metrics"
    if tth_range is not None:
        title += f"  |  ROI: {tth_range[0]}–{tth_range[1]} deg"
    axes[0].set_title(title, fontsize=13)

    # Panel 2: CWT scalogram
    coeffs = metrics["cwt_coefficients"]
    axes[1].imshow(
        np.abs(coeffs),
        aspect="auto",
        origin="lower",
        extent=[azimuth.min(), azimuth.max(), scale_axis.min(), scale_axis.max()],
        cmap="hot",
    )
    axes[1].set_xlabel(r"$\gamma$ (deg)", fontsize=13)
    axes[1].set_ylabel("Scale (deg)", fontsize=11)
    axes[1].set_title("CWT Scalogram (Mexican hat)", fontsize=11)
    axes[1].tick_params(axis="both", which="major", labelsize=11)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Saved: {output_png}")
