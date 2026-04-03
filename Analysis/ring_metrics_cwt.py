"""
ring_metrics_cwt.py
--------------------
Core metric computation module for azimuthal ring analysis.

Computes a suite of statistics from the azimuthal intensity profile of each
diffraction ring, including CWT-based peak detection, texture index, Fourier
ODF coefficients, fiber symmetry index, and Warren grain proxy.

CWT peak detection note
-----------------------
Peak detection runs on the per-position MAXIMUM power across all scales
(scale_max_power) rather than the mean. Taking the max preserves the full
CWT response at whichever scale best matches each feature's angular width,
rather than diluting it by averaging across scales where the feature has
little response. Background variation at large scales no longer inflates the
effective detection threshold.

The returned dict key cwt_scale_max_power_profile records which signal was
used; all other key names match the original azimuthal_ring_statistics output
so this module is a drop-in replacement.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import kurtosis, skew


# ---------------------------------------------------------------------------
# Wavelet primitives
# ---------------------------------------------------------------------------

def _cwt(signal: np.ndarray, wavelet_fn, scales: np.ndarray) -> np.ndarray:
    """
    Continuous wavelet transform via direct convolution.

    Reimplements the removed scipy.signal.cwt using numpy convolve,
    keeping the same interface: wavelet_fn(length, scale) -> array.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.
    wavelet_fn : callable
        Function of (points, scale) returning the wavelet at that scale.
    scales : np.ndarray
        Array of scales to evaluate.

    Returns
    -------
    np.ndarray
        CWT coefficients, shape (len(scales), len(signal)).
    """
    out = np.empty((len(scales), len(signal)), dtype=float)
    for i, scale in enumerate(scales):
        wav = wavelet_fn(min(10 * int(scale), len(signal)), scale)
        out[i] = np.convolve(signal, wav[::-1], mode="same")
    return out


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

    The CWT produces a 2D coefficient matrix (scale x position). This
    implementation runs find_peaks on the per-position MAXIMUM power across
    all scales (scale_max_power) rather than the mean. Taking the max
    preserves the full CWT response at whichever scale best matches each
    feature's angular width, rather than diluting it by averaging across
    scales where the feature has little response. This improves sensitivity
    for narrow texture spots and eliminates the suppression caused by
    broad-scale background variation inflating the effective threshold.

    Parameters
    ----------
    signal : np.ndarray
        1D mean intensity array sampled uniformly around the ring,
        shape (npt_azim,).
    d_gamma : float
        Angular spacing between bins in degrees.
    scales : np.ndarray, optional
        Wavelet scales to evaluate (in bins). Defaults to 1–30 bins
        (1 deg–30 deg at 1 deg/bin resolution), covering single-grain spots up
        to broad texture components.

    Returns
    -------
    dict with keys:
        cwt_n_peaks            : int
        cwt_peak_positions     : list of float
        cwt_dominant_scale_deg : float
        cwt_total_power        : float
        cwt_scale_entropy      : float
        cwt_coefficients       : np.ndarray
    """
    if scales is None:
        scales = np.arange(1, 31)  # 1–30 bins

    s = np.nan_to_num(signal, nan=float(np.nanmean(signal)))

    coeffs = _cwt(s, ricker, scales)  # shape: (n_scales, npt_azim)

    # KEY CHANGE: max across scales instead of mean.
    # Each position retains the power from its single most-responsive scale,
    # preventing averaging dilution across irrelevant scales.
    scale_max_power = np.max(coeffs ** 2, axis=0)  # shape: (npt_azim,)

    if scale_max_power.max() > 0:
        prominence = scale_max_power.std() * 0.1
        peaks_idx, _ = find_peaks(scale_max_power, prominence=prominence, distance=1)
    else:
        peaks_idx = np.array([], dtype=int)

    cwt_peak_positions = (peaks_idx * d_gamma - 180.0).tolist()

    scale_power = np.sum(coeffs ** 2, axis=1)  # shape: (n_scales,)
    dominant_scale_idx = int(np.argmax(scale_power))
    dominant_scale_deg = float(scales[dominant_scale_idx] * d_gamma)

    cwt_total_power = float(np.sum(coeffs ** 2) / len(s))

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


def _texture_index(signal: np.ndarray) -> float:
    """
    Compute the texture index F2 = mean((I / I_mean)^2).

    F2 = 1.0 for a perfectly uniform powder ring. Values above 1 indicate
    preferred orientation; the excess scales with texture sharpness. This is
    the azimuthal analog of the standard pole figure texture index used in
    quantitative ODF analysis (Bunge, 1982).

    Parameters
    ----------
    signal : np.ndarray
        1D intensity array sampled uniformly around the ring.

    Returns
    -------
    float
        Texture index. Returns nan if the mean is zero.
    """
    mean_val = np.nanmean(signal)
    if mean_val <= 0:
        return np.nan
    return float(np.nanmean((signal / mean_val) ** 2))


def _fit_peak_gaussians(
    signal: np.ndarray,
    azimuth: np.ndarray,
    peaks_idx: np.ndarray,
    window_bins: int = 20,
) -> tuple[list[float], list[float]]:
    """
    Fit a Gaussian to each detected peak and return FWHM and asymmetry values.

    A narrow FWHM (few degrees) indicates a sharp, well-defined texture
    component from few coarse grains or strong alignment. A wide FWHM (tens
    of degrees) indicates a broad fiber lobe or fine-grained material with
    partial texture.

    Asymmetry is computed from the data within ±FWHM of the fitted center as:
        asymmetry = (I_right - I_left) / (I_right + I_left)
    A nonzero asymmetry indicates an orientation gradient within the
    diffracting volume, a signature of dislocation density gradients or
    residual stress.

    Parameters
    ----------
    signal : np.ndarray
        1D mean intensity array sampled uniformly around the ring.
    azimuth : np.ndarray
        Azimuthal angle axis in degrees, shape (npt_azim,).
    peaks_idx : np.ndarray
        Indices of detected peaks in signal/azimuth.
    window_bins : int, optional
        Half-width (in bins) of the fitting window around each peak.

    Returns
    -------
    fwhm_list : list of float
        FWHM in degrees for each peak. nan if fit fails.
    asymmetry_list : list of float
        Asymmetry index for each peak. nan if fit fails.
    """
    def _gaussian(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    fwhm_list = []
    asymmetry_list = []

    for idx in peaks_idx:
        lo = max(0, idx - window_bins)
        hi = min(len(signal), idx + window_bins + 1)
        x = azimuth[lo:hi]
        y = signal[lo:hi]

        try:
            p0 = [signal[idx], azimuth[idx], 3.0]
            popt, _ = curve_fit(_gaussian, x, y, p0=p0, maxfev=2000)
            amp, mu, sigma = popt
            if sigma <= 0 or abs(sigma) > 90:
                raise ValueError("Unphysical sigma")

            fwhm = abs(sigma) * 2 * np.sqrt(2 * np.log(2))
            fwhm_list.append(fwhm)

            # Asymmetry: integrated intensity left vs right of center within ±FWHM
            half = fwhm / 2
            left_mask  = (x >= mu - half) & (x < mu)
            right_mask = (x >= mu) & (x <= mu + half)
            i_left  = np.trapz(y[left_mask],  x[left_mask])  if left_mask.any()  else 0.0
            i_right = np.trapz(y[right_mask], x[right_mask]) if right_mask.any() else 0.0
            denom = i_left + i_right
            asymmetry_list.append(
                float((i_right - i_left) / denom) if denom > 0 else np.nan
            )
        except Exception:
            fwhm_list.append(np.nan)
            asymmetry_list.append(np.nan)

    return fwhm_list, asymmetry_list


def _fiber_symmetry_index(signal: np.ndarray, azimuth: np.ndarray) -> float:
    """
    Measure the deviation from fiber (inversion) symmetry: I(gamma) = I(-gamma).

    For a sample with axial symmetry (fiber texture, rolled sheet, most
    combinatorial libraries), the ring should be symmetric about gamma = 0.
    The residual after enforcing symmetry is:

        FSI = sum(|I(gamma) - I(-gamma)|) / sum(I(gamma))

    FSI ~ 0 indicates clean fiber symmetry. High FSI flags sample tilt,
    detector misalignment, a gradient in illuminated volume across the ring,
    or genuinely broken symmetry such as a shear texture component.

    Parameters
    ----------
    signal : np.ndarray
        1D intensity array sampled uniformly around the ring.
    azimuth : np.ndarray
        Azimuthal angle axis in degrees. Assumed to span ~ (-180, 180).

    Returns
    -------
    float
        Fiber symmetry index in [0, 2]. Returns nan if interpolation fails.
    """
    try:
        i_neg = np.interp(-azimuth, azimuth, signal)
        total = np.nansum(signal)
        if total == 0:
            return np.nan
        return float(np.nansum(np.abs(signal - i_neg)) / total)
    except Exception:
        return np.nan


def _fourier_coefficients(signal: np.ndarray) -> tuple[float, float, float]:
    """
    Compute the normalized even-order Fourier coefficients C2, C4, C6 of the
    azimuthal intensity profile.

    The even harmonics of the azimuthal distribution correspond directly to
    the C_l coefficients used in spherical harmonic ODF analysis (Bunge, 1982).
    C2 captures the dominant two-fold symmetry present in most deformation
    textures. The ratio C4/C2 distinguishes texture types — for example,
    copper-type vs. brass-type textures in FCC metals — with direct implications
    for predicted r-value anisotropy and yield locus shape. C6 is sensitive
    to higher-order texture components and is often near zero for weak textures.

    Coefficients are normalized by the DC component (C0) so they are
    dimensionless and independent of absolute intensity.

    Parameters
    ----------
    signal : np.ndarray
        1D intensity array sampled uniformly around the ring, shape (npt_azim,).

    Returns
    -------
    c2 : float
    c4 : float
    c6 : float
        Normalized Fourier amplitudes at orders 2, 4, 6.
    """
    s = np.nan_to_num(signal, nan=float(np.nanmean(signal)))
    fft = np.fft.rfft(s)
    amps = np.abs(fft) / len(s)
    c0 = amps[0]
    if c0 == 0:
        return np.nan, np.nan, np.nan
    # Factor of 2 for one-sided spectrum (all orders except DC)
    c2 = float(2 * amps[2] / c0) if len(amps) > 2 else np.nan
    c4 = float(2 * amps[4] / c0) if len(amps) > 4 else np.nan
    c6 = float(2 * amps[6] / c0) if len(amps) > 6 else np.nan
    return c2, c4, c6


def _arc_imbalance(signal: np.ndarray, azimuth: np.ndarray) -> float:
    """
    Compute the fractional intensity imbalance between opposing azimuthal hemispheres.

    For a reflection where both the upper (gamma > 0) and lower (gamma < 0)
    hemispheres should contribute equally, any persistent imbalance indicates:

        arc_imbalance = (I_upper - I_lower) / (I_upper + I_lower)

    Near zero for symmetric rings. Positive if the upper hemisphere is brighter,
    negative if the lower is. Flags sample tilt, detector shadowing, or an
    asymmetric illuminated volume.

    Parameters
    ----------
    signal : np.ndarray
        1D intensity array sampled uniformly around the ring.
    azimuth : np.ndarray
        Azimuthal angle axis in degrees. Assumed to span ~ (-180, 180).

    Returns
    -------
    float
        Arc imbalance in [-1, 1]. Returns nan if both hemispheres are zero.
    """
    upper = signal[azimuth >= 0]
    lower = signal[azimuth < 0]
    i_upper = np.nansum(upper)
    i_lower = np.nansum(lower)
    denom = i_upper + i_lower
    return float((i_upper - i_lower) / denom) if denom > 0 else np.nan


def _warren_grain_estimate(signal: np.ndarray) -> float:
    """
    Estimate a relative grain count proxy using the Warren variance method.

    When N independent grains contribute to an azimuthal bin, Poisson
    counting statistics give Var(I) / Mean(I)^2 ~ 1/N. This assumes each
    grain's diffracted intensity is an independent random variable with the
    same mean, which holds when grain orientations are random (powder limit).
    The estimate is therefore most physically meaningful for near-random rings;
    for strongly textured samples it provides a lower bound on the number of
    contributing crystallites.

    The returned value is dimensionless and inversely proportional to the
    estimated grain count. Smaller values indicate more grains (fine-grained);
    larger values indicate fewer, coarser grains dominating the ring.

    Parameters
    ----------
    signal : np.ndarray
        1D intensity array sampled uniformly around the ring.

    Returns
    -------
    float
        Variance / Mean^2. Returns nan if mean is zero.
    """
    s = np.nan_to_num(signal, nan=float(np.nanmean(signal)))
    mean_val = float(np.nanmean(s))
    if mean_val <= 0:
        return np.nan
    return float(np.nanvar(s) / mean_val ** 2)


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
        Scalar metrics (see module docstring for full descriptions):
        mean, std, cv, peak_valley, skewness, kurtosis, entropy,
        acf_length_deg, n_texture_peaks, peak_positions, completeness,
        integrated, texture_index, peak_fwhm_mean_deg, peak_fwhm_std_deg,
        peak_asymmetry_mean, fiber_symmetry_index, fourier_c2, fourier_c4,
        fourier_c6, arc_imbalance, warren_grain_proxy,
        cwt_n_peaks, cwt_peak_positions, cwt_dominant_scale_deg,
        cwt_total_power, cwt_scale_entropy, cwt_coefficients.
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

    prominence = std_val * 0.1
    peaks_idx, _ = find_peaks(mean_i, prominence=prominence, distance=1)
    peak_positions = azimuth[peaks_idx].tolist()

    completeness = float(np.sum(mean_i > noise_threshold) / len(mean_i))
    integrated   = float(np.nansum(mean_i) * d_gamma)

    # --- New metrics ---
    tex_index = _texture_index(mean_i)

    fwhm_list, asymmetry_list = _fit_peak_gaussians(mean_i, azimuth, peaks_idx)
    valid_fwhm = [v for v in fwhm_list if np.isfinite(v)]
    peak_fwhm_mean_deg  = float(np.mean(valid_fwhm))  if valid_fwhm             else np.nan
    peak_fwhm_std_deg   = float(np.std(valid_fwhm))   if len(valid_fwhm) > 1    else np.nan
    valid_asym          = [v for v in asymmetry_list if np.isfinite(v)]
    peak_asymmetry_mean = float(np.mean(valid_asym))  if valid_asym             else np.nan

    fsi          = _fiber_symmetry_index(mean_i, azimuth)
    c2, c4, c6   = _fourier_coefficients(mean_i)
    arc_imbal    = _arc_imbalance(mean_i, azimuth)
    warren_proxy = _warren_grain_estimate(mean_i)

    wavelet = _wavelet_peak_analysis(mean_i, d_gamma)

    return {
        # Original metrics
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
        # New metrics
        "texture_index":          tex_index,
        "peak_fwhm_mean_deg":     peak_fwhm_mean_deg,
        "peak_fwhm_std_deg":      peak_fwhm_std_deg,
        "peak_asymmetry_mean":    peak_asymmetry_mean,
        "fiber_symmetry_index":   fsi,
        "fourier_c2":             c2,
        "fourier_c4":             c4,
        "fourier_c6":             c6,
        "arc_imbalance":          arc_imbal,
        "warren_grain_proxy":     warren_proxy,
        # CWT metrics
        "cwt_n_peaks":            wavelet["cwt_n_peaks"],
        "cwt_peak_positions":     wavelet["cwt_peak_positions"],
        "cwt_dominant_scale_deg": wavelet["cwt_dominant_scale_deg"],
        "cwt_total_power":        wavelet["cwt_total_power"],
        "cwt_scale_entropy":      wavelet["cwt_scale_entropy"],
        "cwt_coefficients":       wavelet["cwt_coefficients"],
    }


def plot_ring_metrics(
    rings: list[dict],
    base_name: str,
    output_png: str,
) -> None:
    """
    Plot ring metrics for one or more azimuthal ROIs as side-by-side columns.

    Layout: 5 rows x n_rings columns.

      Row 0 — Full caked image (log scale) with ROI boundaries marked.
               Repeated per column so each ring panel is self-contained.
      Row 1 — ROI interior heatmap (radial vs. azimuthal, log scale).
               Each column shows only its own tth_range slice.
      Row 2 — Intensity vs gamma with find_peaks (red) and CWT (orange) peak
               markers, plus a full metrics annotation box.
      Row 3 — CWT scalogram (scale vs gamma).
      Row 4 — Normalized Fourier amplitude spectrum (orders 0-12) with C2,
               C4, C6 annotated.

    Parameters
    ----------
    rings : list of dict
        Each dict must contain:
            mean_i        : np.ndarray  — mean intensity per azimuthal bin.
            std_i         : np.ndarray  — std intensity per azimuthal bin.
            azimuth       : np.ndarray  — azimuthal angle axis in degrees.
            metrics       : dict        — output of compute_ring_statistics().
            tth_range     : tuple       — (tth_lo, tth_hi) for ROI labels and
                                          vertical lines on the caked image.
            intensity_2d  : np.ndarray  — full caked image, shape
                                          (npt_azim, npt_rad).
            tth           : np.ndarray  — full radial axis in degrees.
            intensity_roi : np.ndarray  — ROI slice, shape
                                          (npt_azim, n_roi_bins).
            tth_roi       : np.ndarray  — radial axis clipped to tth_range.
    base_name : str
        Sample name used in the figure suptitle.
    output_png : str
        Full path for the saved figure.
    """
    def _fmt(v, digits=3):
        return f"{v:.{digits}f}" if np.isfinite(v) else "nan"

    n_rings   = len(rings)
    fig_width = max(11, 7 * n_rings)

    # Row heights: caked image | ROI heatmap | intensity profile | CWT | Fourier
    fig, axes = plt.subplots(
        5, n_rings,
        figsize=(fig_width, 22),
        gridspec_kw={"height_ratios": [3, 1, 3, 2, 2]},
        squeeze=False,   # always 2-D axes array even for n_rings == 1
    )

    for ci, ring in enumerate(rings):
        mean_i        = ring["mean_i"]
        std_i         = ring["std_i"]
        azimuth       = ring["azimuth"]
        metrics       = ring["metrics"]
        tth_range     = ring.get("tth_range", None)
        intensity_2d  = ring.get("intensity_2d", None)
        tth           = ring.get("tth", None)
        intensity_roi = ring.get("intensity_roi", None)
        tth_roi       = ring.get("tth_roi", None)

        d_gamma    = float(np.median(np.diff(azimuth)))
        n_scales   = metrics["cwt_coefficients"].shape[0]
        scale_axis = np.arange(1, n_scales + 1) * d_gamma

        col_title = (
            f"ROI: {tth_range[0]}–{tth_range[1]}°"
            if tth_range is not None else f"Ring {ci}"
        )

        # ------------------------------------------------------------------
        # Row 0: Full caked image (log scale) with ROI boundaries
        # ------------------------------------------------------------------
        ax0 = axes[0, ci]
        if intensity_2d is not None and tth is not None:
            pos_2d = intensity_2d.copy()
            pos_2d[pos_2d <= 0] = np.nan
            ax0.imshow(
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
        # Row 1: ROI interior heatmap (radial vs. azimuthal, log scale)
        # ------------------------------------------------------------------
        ax1 = axes[1, ci]
        if intensity_roi is not None and tth_roi is not None:
            roi_pos = intensity_roi.T.copy()   # (n_roi_bins, npt_azim)
            roi_pos[roi_pos <= 0] = np.nan
            ax1.imshow(
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
            ax1.set_xlabel(r"$\gamma$ (deg)", fontsize=11)
            if ci == 0:
                ax1.set_ylabel(r"2$\theta$ (deg)", fontsize=11)
            ax1.set_title("ROI heatmap", fontsize=10)
        else:
            ax1.set_visible(False)
        ax1.tick_params(axis="both", which="major", labelsize=10)

        # ------------------------------------------------------------------
        # Row 2: Intensity vs gamma with peak markers and metrics annotation
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
            aspect="auto",
            origin="lower",
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
        # Row 4: Fourier spectrum (binned bar chart)
        # ------------------------------------------------------------------
        ax4 = axes[4, ci]
        s = np.nan_to_num(mean_i, nan=float(np.nanmean(mean_i)))
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
            "Fourier Spectrum  (C2=two-fold, C4=four-fold, C6=six-fold)",
            fontsize=9,
        )
        ax4.set_xticks(orders[:max_order])
        ax4.tick_params(axis="both", which="major", labelsize=10)
        ax4.grid(axis="y", lw=0.5, alpha=0.4)
        if ci == 0:
            ax4.set_ylabel("Normalized Amplitude", fontsize=11)

    fig.suptitle(f"{base_name} — Ring Metrics", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")
