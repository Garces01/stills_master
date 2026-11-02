"""
rfi_pipeline.py
===============
End-to-end RFI characterisation for SKAMPI time-frequency data.

For each detected peak the pipeline computes the full set of metrics:

    fc   — power-weighted centroid frequency
    B99  — 99 % occupied bandwidth  (ITU-R SM.328-12)
    B10  — 10 dB bandwidth
    B20  — 20 dB bandwidth
    B30  — 30 dB bandwidth
    FWHM — full width at half maximum (scipy peak_widths, rel_height=0.5)
    RBW  — ratio of 99 % occupied bandwidth to FWHM, B99 / FWHM (dimensionless)
    A    — integrated power (sum of clipped detrended power inside the feature)
    S    — moment skewness  (γ₁, power-weighted 3rd standardised central moment)
    SNR  — local signal-to-noise ratio (noise from flanking channels, robust MAD)

Usage
-----
    from rfi_pipeline import analyze_time_bins

    results = analyze_time_bins(
        data,          # (n_time, n_freq) numpy array — RAW power
        freq,          # (n_freq,) frequency axis in MHz
        time_bins=[2, 5, 10],
    )
    # returns a pandas DataFrame, one row per (time_bin × peak)
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from scipy.stats import kurtosis, median_abs_deviation, skew
from scipy.ndimage import uniform_filter1d


# ─────────────────────────────────────────────────────────────────────────────
# 1. Detrending  (rolling median baseline subtraction)
# ─────────────────────────────────────────────────────────────────────────────

def detrend_spectrum(power_raw, freq=None, method="asls",
                     lam=1e7, p=0.02, window_factor=10):
    """
    Estimate and subtract a smooth baseline from a 1-D raw power spectrum.
    The result is in LINEAR power units with background ≈ 0.

    Parameters
    ----------
    power_raw     : 1-D array   Raw linear power.
    freq          : 1-D array   Frequency axis (used by ASLS).
    method        : str         "asls"   — Asymmetric Least Squares
                                           (pybaselines, lam / p).
                                "median" — running median, window =
                                           len // window_factor.
    lam, p        : float       ASLS smoothness and asymmetry.
                                Default 1e7, 0.02.
    window_factor : int         Running-median window relative to array
                                length. Default 10 → 10 % of the band.

    Returns
    -------
    power_detrended : 1-D ndarray   background ≈ 0 in linear power.
    baseline        : 1-D ndarray   Estimated smooth baseline.
    """
    power_raw = np.asarray(power_raw, dtype=float)

    if method == "asls":
        from pybaselines import Baseline
        baseline, _ = Baseline(x_data=freq).asls(power_raw, lam=lam, p=p)
    elif method == "median":
        n = len(power_raw)
        win = max(51, (n // window_factor) | 1)   # must be odd
        baseline = _rolling_median(power_raw, win)
    else:
        raise ValueError(f"Unknown detrend method '{method}'.")

    power_detrended = power_raw - baseline
    return power_detrended.astype(float), np.asarray(baseline, dtype=float)


def _rolling_median(x, win):
    """Efficient rolling median via sorting on a sliding window."""
    from scipy.ndimage import median_filter
    return median_filter(x.astype(float), size=win, mode="nearest")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Peak detection
# ─────────────────────────────────────────────────────────────────────────────

SIGMA_BY_LEVEL = {"high": 3.0, "low": 5.0}


def resolve_sigma(sigma_value=None, rf_level=None):
    """
    Return the detection σ. Uses ``sigma_value`` if given, otherwise the σ for
    ``rf_level``; if both are None, asks the user for ``high`` or ``low``.
    """
    if sigma_value is not None:
        return float(sigma_value)
    while rf_level is None:
        answer = input("RF level to evaluate — high (3σ) or low (5σ)? ").strip().lower()
        if answer in SIGMA_BY_LEVEL:
            rf_level = answer
        else:
            print("Please type 'low' or 'high'.")
    try:
        return SIGMA_BY_LEVEL[rf_level.lower()]
    except (AttributeError, KeyError):
        raise ValueError("rf_level must be 'high' or 'low'.") from None


def detect_peaks(power_detrended, freq=None, sigma_value=None,
                 prominence=0.5, distance=80, edge_threshold=0.05,
                 rel_height=0.5, rf_level=None, exclude_mask=None):
    """
    Detect RFI peaks in a detrended 1-D linear power spectrum.

    Detection threshold = MAD(p) + sigma_value × std(p). If ``sigma_value``
    is omitted, ``rf_level`` selects 3σ for high level or 5σ for low level.
    ``exclude_mask`` marks channels to omit from threshold estimation and peak
    detection; detections and feature edges are confined to unmasked segments.

    Parameters
    ----------
    power_detrended : 1-D array   Detrended power (background ≈ 0).
    freq            : 1-D array   Frequency axis (MHz). If given, the FWHM
                                  is also returned in MHz.
    sigma_value     : float or None  Number of σ above the MAD. Overrides
                                     ``rf_level`` when provided.
    prominence      : float       Minimum prominence (find_peaks).
    distance        : int         Minimum separation between peaks, channels.
    edge_threshold  : float       Feature edges: walk outward from the peak
                                  until power drops to this level.
    rel_height      : float       Relative height for peak_widths
                                  (0.5 → FWHM).
    rf_level        : str or None ``"high"`` (3σ) or ``"low"`` (5σ), used
                                  when ``sigma_value`` is None. If both are
                                  None, the user is asked interactively.
    exclude_mask    : 1-D bool array or None  Channels excluded from detection.

    Returns
    -------
    list of dict — each dict has ``peak_index``, ``start_index``,
    ``end_index``, ``peak_height``, ``detection_threshold``,
    ``fwhm_channels``, ``fwhm_height`` and, if ``freq`` is given,
    ``FWHM``, ``FWHM_start``, ``FWHM_end`` (MHz).
    """
    p = np.asarray(power_detrended, dtype=float)
    if exclude_mask is None:
        exclude_mask = np.zeros(len(p), dtype=bool)
    else:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
        if exclude_mask.shape != p.shape:
            raise ValueError("exclude_mask must have the same shape as power_detrended.")

    sigma_value = resolve_sigma(sigma_value, rf_level)
    valid_power = p[~exclude_mask]
    if len(valid_power) == 0:
        return []
    threshold = median_abs_deviation(valid_power) + sigma_value * np.std(valid_power)

    if freq is not None:
        freq = np.asarray(freq, dtype=float)
        if freq.shape != p.shape:
            raise ValueError("freq must have the same shape as power_detrended.")
        channel_spacing = freq[1] - freq[0]
        channels = np.arange(len(freq))

    # Process each unmasked run separately so excluded high-level features
    # cannot create artificial low-level peaks at their boundaries.
    valid_indices = np.flatnonzero(~exclude_mask)
    if len(valid_indices) == 0:
        return []
    split_points = np.flatnonzero(np.diff(valid_indices) > 1) + 1
    segments = np.split(valid_indices, split_points)

    peaks = []
    for segment in segments:
        seg_start, seg_end = int(segment[0]), int(segment[-1]) + 1
        segment_power = p[seg_start:seg_end]
        local_indices, _ = find_peaks(
            segment_power,
            height=threshold,
            prominence=prominence,
            distance=distance,
        )
        if len(local_indices) == 0:
            continue

        widths_idx, width_heights, left_ips, right_ips = peak_widths(
            segment_power, local_indices, rel_height=rel_height)

        for i, local_peak in enumerate(local_indices):
            pk = int(local_peak + seg_start)
            left = pk
            while left > seg_start and p[left] > edge_threshold:
                left -= 1
            right = pk
            while right < seg_end - 1 and p[right] > edge_threshold:
                right += 1

            row = {
                "peak_index":          pk,
                "start_index":         int(left),
                "end_index":           int(right),
                "peak_height":         float(p[pk]),
                "detection_threshold": float(threshold),
                "fwhm_channels":       float(widths_idx[i]),
                "fwhm_height":         float(width_heights[i]),
            }
            if freq is not None:
                fwhm_lo = np.interp(left_ips[i] + seg_start, channels, freq)
                fwhm_hi = np.interp(right_ips[i] + seg_start, channels, freq)
                row.update({
                    "FWHM":       float(widths_idx[i] * channel_spacing),
                    "FWHM_start": float(fwhm_lo),
                    "FWHM_end":   float(fwhm_hi),
                })
            peaks.append(row)

    return peaks


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metric computation per peak
# ─────────────────────────────────────────────────────────────────────────────

def _compute_b99(freq, power_detrended, start_index, end_index):
    """
    99 % occupied bandwidth (ITU-R SM.328-12 §3.2).

    The occupied bandwidth is the band within which 99 % of the total
    mean power of the emission is contained; 0.5 % is excluded on each
    side by integrating the cumulative power distribution.
    """
    f   = freq [start_index : end_index + 1]
    p   = np.maximum(power_detrended[start_index : end_index + 1], 0.0)
    total = p.sum()
    if total <= 0 or len(p) < 3:
        return np.nan, np.nan, np.nan

    cum   = np.cumsum(p)
    lo_thr = 0.005 * total
    hi_thr = 0.995 * total

    i_lo = np.searchsorted(cum, lo_thr, side="left")
    i_hi = np.searchsorted(cum, hi_thr, side="right") - 1

    i_lo = max(0, min(i_lo, len(f) - 1))
    i_hi = max(0, min(i_hi, len(f) - 1))

    f_lo = float(f[i_lo])
    f_hi = float(f[i_hi])
    return f_lo, f_hi, float(f_hi - f_lo)


def _compute_xdb(freq, power_detrended, peak_index, start_index, end_index,
                 levels=(10, 20, 30)):
    """
    x-dB bandwidths on the LINEAR power spectrum.

    Threshold at level N:  target = peak × 10^(−N/10)
    (Same physical crossing as the dB scale without any conversion.)
    """
    p         = np.asarray(power_detrended, dtype=float)
    peak_val  = float(p[peak_index])
    f         = np.asarray(freq, dtype=float)
    results   = {}

    if peak_val <= 0:
        for N in levels:
            results[f"B{N}_start"] = np.nan
            results[f"B{N}_end"]   = np.nan
            results[f"B{N}"]       = np.nan
        return results

    for N in levels:
        target = peak_val * 10.0 ** (-N / 10.0)

        # Walk left
        left = peak_index
        while left > 0 and p[left] > target:
            left -= 1

        # Walk right
        right = peak_index
        while right < len(p) - 1 and p[right] > target:
            right += 1

        left_reached  = p[left]  <= target
        right_reached = p[right] <= target

        if not left_reached or not right_reached:
            f_lo = f_hi = bw = np.nan
        else:
            p0, p1 = p[left],     p[left + 1]
            f0, f1 = f[left],     f[left + 1]
            f_lo   = f0 + (target - p0) * (f1 - f0) / (p1 - p0) if p1 != p0 else (f0 + f1) / 2

            p0, p1 = p[right - 1], p[right]
            f0, f1 = f[right - 1], f[right]
            f_hi   = f0 + (target - p0) * (f1 - f0) / (p1 - p0) if p1 != p0 else (f0 + f1) / 2

            bw = float(f_hi - f_lo)

        results[f"B{N}_start"] = f_lo
        results[f"B{N}_end"]   = f_hi
        results[f"B{N}"]       = bw

    return results


def _compute_moment_metrics(freq, power_detrended, peak_index,
                            start_index, end_index):
    """
    Power-weighted moment metrics: centroid (fc), skewness (S).
    """
    f  = np.asarray(freq, dtype=float)[start_index : end_index + 1]
    p  = np.maximum(power_detrended[start_index : end_index + 1], 0.0)
    total = p.sum()

    if total <= 0:
        return dict(fc=np.nan, S=np.nan,
                    moment_variance=np.nan, moment_std=np.nan)

    w          = p / total
    fc         = float(np.sum(w * f))
    delta      = f - fc
    mu2        = float(np.sum(w * delta ** 2))
    mu3        = float(np.sum(w * delta ** 3))
    gamma1     = float(mu3 / mu2 ** 1.5) if mu2 > 0 else np.nan
    sigma_spec = float(np.sqrt(mu2))     if mu2 > 0 else np.nan

    return dict(fc=fc, S=gamma1,
                moment_variance=mu2, moment_std=sigma_spec)


def _compute_local_snr(power_detrended, peak_index, start_index, end_index,
                       noise_window=50, guard=5, exclude_mask=None):
    """
    Local signal-to-noise ratio of one peak.

    The noise is estimated from two flanking windows of ``noise_window``
    channels on each side of the feature, separated from the feature edges
    by ``guard`` channels.  Channels flagged in ``exclude_mask`` (e.g. other
    detected peaks) are dropped.  Noise level and local offset are robust:

        local_median = median(noise channels)
        local_sigma  = 1.4826 × MAD(noise channels)

    Returns
    -------
    dict with
        snr_local       — (peak − local_median) / local_sigma
        snr_integrated  — Σ(p − local_median) over the feature
                          / (local_sigma × √N_feature)
        local_noise_sigma, local_noise_median, n_noise_channels
    """
    p = np.asarray(power_detrended, dtype=float)
    n = len(p)

    lo_a = max(0, start_index - guard - noise_window)
    lo_b = max(0, start_index - guard)
    hi_a = min(n, end_index + guard + 1)
    hi_b = min(n, end_index + guard + 1 + noise_window)
    idx  = np.r_[lo_a:lo_b, hi_a:hi_b]

    if exclude_mask is not None and len(idx):
        idx = idx[~exclude_mask[idx]]

    nan_out = dict(snr_local=np.nan, snr_integrated=np.nan,
                   local_noise_sigma=np.nan, local_noise_median=np.nan,
                   n_noise_channels=int(len(idx)))
    if len(idx) < 10:
        return nan_out

    noise  = p[idx]
    median = float(np.median(noise))
    sigma  = float(1.4826 * np.median(np.abs(noise - median)))
    if sigma <= 0:
        return nan_out

    feature = p[start_index : end_index + 1] - median
    snr_pk  = float((p[peak_index] - median) / sigma)
    snr_int = float(feature.sum() / (sigma * np.sqrt(len(feature))))

    return dict(snr_local=snr_pk, snr_integrated=snr_int,
                local_noise_sigma=sigma, local_noise_median=median,
                n_noise_channels=int(len(idx)))


def characterise_peak(freq, power_detrended, peak_index,
                      start_index, end_index,
                      levels=(10, 20, 30),
                      noise_window=50, guard=5, exclude_mask=None,
                      fwhm=np.nan):
    """
    Compute the full metric set for one RFI peak.

    Returns a flat dict with keys:
        fc, B99, B99_start, B99_end,
        B10, B20, B30  (and *_start/*_end variants),
        RBW, channel_spacing, A, S,
        snr_local, snr_integrated, local_noise_sigma, local_noise_median,
        n_noise_channels
    """
    freq = np.asarray(freq, dtype=float)
    p    = np.asarray(power_detrended, dtype=float)

    channel_spacing = float(freq[1] - freq[0]) if len(freq) > 1 else np.nan

    # Integrated power (clipped to ≥ 0)
    A = float(np.maximum(p[start_index : end_index + 1], 0.0).sum())

    # B99
    b99_lo, b99_hi, b99 = _compute_b99(freq, p, start_index, end_index)

    # x-dB bandwidths
    xdb = _compute_xdb(freq, p, peak_index, start_index, end_index, levels)

    # Moment metrics (fc, S)
    mom = _compute_moment_metrics(freq, p, peak_index, start_index, end_index)

    # Local SNR
    snr = _compute_local_snr(p, peak_index, start_index, end_index,
                             noise_window=noise_window, guard=guard,
                             exclude_mask=exclude_mask)

    row = {
        "f_peak":     float(freq[peak_index]),
        "fc":         mom["fc"],
        "B99_start":  b99_lo,
        "B99_end":    b99_hi,
        "B99":        b99,
        "RBW":        b99 / fwhm if np.isfinite(fwhm) and fwhm > 0 else np.nan,
        "channel_spacing": channel_spacing,
        "A":          A,
        "S":          mom["S"],
        "moment_std": mom["moment_std"],
    }
    row.update(xdb)   # adds B10_*, B20_*, B30_*
    row.update(snr)   # adds snr_local, snr_integrated, local_noise_*

    return row


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyze_time_bins(
    data,
    freq,
    time_bins,
    detrend_method="asls",
    asls_lam=1e7,
    asls_p=0.02,
    detrend_window_factor=10,
    sigma_value=None,
    sigma_values=None,
    prominence=0.5,
    distance=80,
    edge_threshold=0.05,
    levels=(10, 20, 30),
    snr_noise_window=50,
    snr_guard=5,
    rf_level="both",
):
    """
    Run the full RFI characterisation pipeline for selected time bins.

    Parameters
    ----------
    data                 : (n_time, n_freq) array   Raw power spectrum.
    freq                 : (n_freq,) array           Frequency axis (MHz).
    time_bins            : list of int               Time-bin indices to process.
    detrend_method       : str    "asls" (default) or "median".
    asls_lam, asls_p     : float  ASLS smoothness / asymmetry (1e7, 0.02).
    detrend_window_factor: int    Rolling-median window = n_freq // factor.
    sigma_value          : float or None  Explicit threshold multiplier. When
                                         provided, runs one detection pass.
    sigma_values         : dict or None  Separate multipliers for ``high``
                                         and ``low`` detections. Defaults to
                                         3σ and 5σ when omitted per level.
    rf_level             : str  ``"both"`` (default) detects high peaks at 3σ,
                                masks their feature ranges, then detects low
                                peaks at 5σ on the remaining channels. Use
                                ``"high"`` or ``"low"`` for one pass only.
    prominence           : float  Minimum peak prominence.
    distance             : int    Minimum peak separation in channels.
    edge_threshold       : float  Power level that defines feature edges.
    levels               : tuple  dB levels for x-dB bandwidths.
    snr_noise_window     : int    Channels per side used for the local noise.
    snr_guard            : int    Channels skipped between feature edge and
                                  the noise windows.

    Returns
    -------
    pd.DataFrame
        One row per (time_bin × peak).  Columns:
            time_bin, peak_index, start_index, end_index, peak_height,
            detection_threshold, f_peak, fc,
            FWHM, FWHM_start, FWHM_end, fwhm_channels, fwhm_height, B99, B99_start, B99_end,
            B10, …, B20, …, B30, …,
            RBW, channel_spacing, A, S, moment_std,
            snr_local, snr_integrated, local_noise_sigma,
            local_noise_median, n_noise_channels
    """
    data = np.asarray(data, dtype=float)
    freq = np.asarray(freq, dtype=float)
    rows = []
    rf_level = rf_level.lower() if isinstance(rf_level, str) else rf_level
    if sigma_value is not None and sigma_values is not None:
        raise ValueError("Use either sigma_value or sigma_values, not both.")
    if sigma_value is not None:
        detection_levels = [("custom", float(sigma_value))]
    elif sigma_values is not None:
        if not isinstance(sigma_values, dict) or set(sigma_values) - set(SIGMA_BY_LEVEL):
            raise ValueError("sigma_values must be a dict with 'high' and/or 'low' keys.")
        selected_levels = (tuple(SIGMA_BY_LEVEL) if rf_level == "both"
                           else (rf_level,) if rf_level in SIGMA_BY_LEVEL
                           else ())
        if not selected_levels:
            raise ValueError("rf_level must be 'both', 'high', or 'low'.")
        detection_levels = [
            (level, float(sigma_values.get(level, SIGMA_BY_LEVEL[level])))
            for level in selected_levels
        ]
    elif rf_level == "both":
        detection_levels = [(level, resolve_sigma(rf_level=level))
                            for level in ("high", "low")]
    elif rf_level in SIGMA_BY_LEVEL:
        detection_levels = [(rf_level, resolve_sigma(rf_level=rf_level))]
    else:
        raise ValueError("rf_level must be 'both', 'high', or 'low'.")

    for t in time_bins:
        raw = data[t]
        power_det, baseline = detrend_spectrum(
            raw, freq=freq, method=detrend_method,
            lam=asls_lam, p=asls_p, window_factor=detrend_window_factor)

        high_peaks = []
        peaks = []
        if detection_levels[0][0] == "high" and len(detection_levels) == 2:
            high_peaks = detect_peaks(
                power_det,
                freq=freq,
                sigma_value=detection_levels[0][1],
                prominence=prominence,
                distance=distance,
                edge_threshold=edge_threshold,
            )
            for pk in high_peaks:
                pk["detection_level"] = "high"
                peaks.append(pk)

            high_mask = np.zeros(len(power_det), dtype=bool)
            for pk in high_peaks:
                high_mask[pk["start_index"] : pk["end_index"] + 1] = True

            low_peaks = detect_peaks(
                power_det,
                freq=freq,
                sigma_value=detection_levels[1][1],
                prominence=prominence,
                distance=distance,
                edge_threshold=edge_threshold,
                exclude_mask=high_mask,
            )
            for pk in low_peaks:
                pk["detection_level"] = "low"
                peaks.append(pk)
        else:
            level, level_sigma = detection_levels[0]
            peaks = detect_peaks(
                power_det,
                freq=freq,
                sigma_value=level_sigma,
                prominence=prominence,
                distance=distance,
                edge_threshold=edge_threshold,
            )
            for pk in peaks:
                pk["detection_level"] = level

        # Mask every detected feature so it is excluded from local SNR noise windows.
        peak_mask = np.zeros(len(power_det), dtype=bool)
        for pk in peaks:
            peak_mask[pk["start_index"] : pk["end_index"] + 1] = True

        for pk in peaks:
            metrics = characterise_peak(
                freq, power_det,
                peak_index=pk["peak_index"],
                start_index=pk["start_index"],
                end_index=pk["end_index"],
                fwhm=pk.get("FWHM", np.nan),
                levels=levels,
                noise_window=snr_noise_window,
                guard=snr_guard,
                exclude_mask=peak_mask,
            )
            rows.append({
                "time_bin":   t,
                **pk,
                **metrics,
            })

    if not rows:
        return pd.DataFrame()

    # Reorder columns for readability
    col_order = [
        "time_bin", "detection_level", "peak_index", "start_index", "end_index", "peak_height",
        "detection_threshold", "f_peak", "fc",
        "FWHM", "FWHM_start", "FWHM_end", "fwhm_channels", "fwhm_height",
        "B99", "B99_start", "B99_end",
        "B10", "B10_start", "B10_end",
        "B20", "B20_start", "B20_end",
        "B30", "B30_start", "B30_end",
        "RBW", "channel_spacing", "A", "S", "moment_std",
        "snr_local", "snr_integrated",
        "local_noise_sigma", "local_noise_median", "n_noise_channels",
    ]
    df = pd.DataFrame(rows)
    existing = [c for c in col_order if c in df.columns]
    extra    = [c for c in df.columns if c not in col_order]
    return df[existing + extra]

# ─────────────────────────────────────────────────────────────────────────────
# 5. Time binning  (average consecutive spectra, then characterise each bin)
# ─────────────────────────────────────────────────────────────────────────────

def bin_spectrum(data, time, bin_size=None, bin_seconds=None):
    """
    Average consecutive, non-overlapping blocks of spectra.

    Give either ``bin_size`` (number of spectra per bin) or ``bin_seconds``
    (bin duration; converted to spectra using the median time cadence).
    Trailing spectra that do not fill a whole bin are dropped.

    Parameters
    ----------
    data        : (n_time, n_freq) array   LINEAR power (not dB).
    time        : (n_time,) array          Timestamps in seconds.
    bin_size    : int                      Spectra per bin.
    bin_seconds : float                    Bin duration in seconds.

    Returns
    -------
    binned      : (n_bins, n_freq) array   Mean spectrum of each bin.
    bin_info    : pd.DataFrame             One row per bin: timing and row
                  metadata, plus mean, population std, skewness, and Fisher
                  excess kurtosis over the bin's linear-power samples.
    """
    data = np.asarray(data, dtype=float)
    time = np.asarray(time, dtype=float)

    if (bin_size is None) == (bin_seconds is None):
        raise ValueError("Give exactly one of bin_size or bin_seconds.")
    if bin_seconds is not None:
        dt = np.median(np.diff(time))
        bin_size = max(1, int(round(bin_seconds / dt)))

    n_bins = data.shape[0] // bin_size
    if n_bins == 0:
        raise ValueError(f"bin_size={bin_size} exceeds n_time={data.shape[0]}.")
    n_used = n_bins * bin_size

    blocks = data[:n_used].reshape(n_bins, bin_size, -1)
    binned = blocks.mean(axis=1)
    block_values = blocks.reshape(n_bins, -1)
    t_blocks = time[:n_used].reshape(n_bins, bin_size)

    bin_info = pd.DataFrame({
        "time_bin":  np.arange(n_bins),
        "t_start":   t_blocks[:, 0],
        "t_end":     t_blocks[:, -1],
        "t_mid":     t_blocks.mean(axis=1),
        "row_start": np.arange(n_bins) * bin_size,
        "row_end":   np.arange(n_bins) * bin_size + bin_size - 1,
        "n_spectra": bin_size,
        "mean":      np.nanmean(block_values, axis=1),
        "std":       np.nanstd(block_values, axis=1),
        "skewness":  skew(block_values, axis=1, nan_policy="omit"),
        "kurtosis":  kurtosis(block_values, axis=1, fisher=True, nan_policy="omit"),
    })
    return binned, bin_info


def analyze_binned_spectrum(data, freq, time, bin_size=None, bin_seconds=None,
                            **pipeline_kwargs):
    """
    Bin the spectrum in time, then run :func:`analyze_time_bins` on every bin.

    Returns one row per (bin × peak) with the bin timing columns
    (t_start, t_end, t_mid, t_rel, row_start, row_end, n_spectra) attached;
    ``t_rel`` is t_mid relative to the first spectrum, in seconds.
    Extra keyword arguments are passed to :func:`analyze_time_bins`.
    """
    binned, bin_info = bin_spectrum(data, time, bin_size=bin_size,
                                    bin_seconds=bin_seconds)
    df = analyze_time_bins(binned, freq, time_bins=range(len(binned)),
                           **pipeline_kwargs)
    if df.empty:
        return df

    bin_info["t_rel"] = bin_info["t_mid"] - float(np.asarray(time)[0])
    return df.merge(bin_info, on="time_bin", how="left")


def assign_peak_groups(df, frequency_column="f_peak", tolerance_mhz=2.0):
    """
    Match peaks across time bins by frequency proximity.

    Frequencies are sorted and assigned sequentially; a frequency joins the
    current group when it lies within ``tolerance_mhz`` of the group's running
    median, otherwise it starts a new group.

    Adds ``peak_group`` (int id) and ``peak_reference_MHz`` (group median).
    """
    result = df.copy()
    freqs = np.sort(result[frequency_column].dropna().unique())

    groups, current = [], []
    for f in freqs:
        if current and abs(f - np.median(current)) > tolerance_mhz:
            groups.append(current)
            current = []
        current.append(f)
    if current:
        groups.append(current)

    group_id = {f: i for i, g in enumerate(groups) for f in g}
    reference = {f: float(np.median(g)) for g in groups for f in g}
    result["peak_group"] = result[frequency_column].map(group_id).astype("Int64")
    result["peak_reference_MHz"] = result[frequency_column].map(reference)
    return result
