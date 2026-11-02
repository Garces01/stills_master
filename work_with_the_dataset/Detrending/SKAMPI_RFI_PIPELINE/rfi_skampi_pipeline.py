"""
RFI Analysis Script
-------------------
This script performs spectral RFI detection on MeerKAT / SKAMPI HDF5 data.

Workflow:
1. Load spectral data from a selected group
2. Convert power to dB
3. Select representative time bins
4. Remove spectral baseline
5. Detect RFI peaks (mean + 3σ)
6. Print detected frequencies
7. Plot dynamic spectrum

To analyze a different dataset:
    change the filename

To analyze another group:
    change the group variable
"""

# ==============================
# Import Libraries
# ==============================

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from pybaselines import Baseline


# ==============================
# USER INPUT
# ==============================

filename = "/home/chavango/Master/DATA/SKAMPI_data/EDD_2023-03-11T14_58_01.567317UTC_ErQFG.hdf5"

group = "scan/001/P1_ND0"

time_step_seconds = 15


# ==============================
# LOAD DATA
# ==============================

def load_group(filename, group):
    """
    Load spectrum, frequency, and timestamps
    from the selected HDF5 group.
    """

    with h5py.File(filename, "r") as f:

        spectrum = f[f"{group}/spectrum"][30:, 1600:-60]

        frequency = f[f"{group}/frequency"][1600:-60] / 1e9

        timestamp = f[f"{group}/timestamp"][30:, 0]

    # Convert power to dB
    spectrum_db = 10 * np.log10(spectrum**2)

    # Convert timestamps to UTC datetime
    time_utc = pd.to_datetime(timestamp, unit='s', origin='unix', utc=True)

    return spectrum_db, frequency, time_utc


# ==============================
# BASELINE REMOVAL
# ==============================

def remove_baseline(freq, spectrum):

    baseline_fitter = Baseline(x_data=freq)

    baseline, _ = baseline_fitter.asls(
        spectrum,
        lam=1e7,
        p=0.02
    )

    detrended = spectrum - baseline

    return detrended


# ==============================
# PEAK DETECTION
# ==============================

def detect_rfi_peaks(freq, spectrum, sigma=3):
    """
    Detect peaks using statistical threshold.
    """

    mean = np.mean(spectrum)

    std = np.std(spectrum)

    threshold = mean + sigma * std

    peaks, properties = find_peaks(
        spectrum,
        height=threshold
    )

    peak_freqs = freq[peaks]

    return peaks, peak_freqs


# ==============================
# SELECT TIME BINS
# ==============================

def select_time_indices(time_utc, step):

    target_times = pd.date_range(
        start=time_utc[0],
        end=time_utc[-1],
        freq=f"{step}s"
    )

    indices = time_utc.get_indexer(
        target_times,
        method="nearest"
    )

    return indices


# ==============================
# MAIN ANALYSIS
# ==============================

def analyze_dataset(filename, group):

    print("\nLoading data...")

    spectrum, freq, time_utc = load_group(filename, group)

    print("Data loaded")
    print("Number of time samples:", spectrum.shape[0])
    print("Number of frequency channels:", spectrum.shape[1])

    indices = select_time_indices(time_utc, time_step_seconds)

    print("\nRunning RFI detection...\n")

    results = []

    for i in indices:

        power = spectrum[i, :]

        detrended = remove_baseline(freq, power)

        peaks, peak_freqs = detect_rfi_peaks(freq, detrended)

        formatted = [f"{f:.6f}" for f in peak_freqs]

        print(
            f"Peaks detected at [{time_utc[i].strftime('%H:%M:%S')}]: {formatted}"
        )

        results.append({
            "time": time_utc[i],
            "peaks": peak_freqs
        })

    return spectrum, freq, time_utc, results


# ==============================
# PLOT DYNAMIC SPECTRUM
# ==============================

def plot_dynamic_spectrum(freq, time_utc, spectrum):

    plt.figure(figsize=(12,6))

    plt.imshow(
        spectrum,
        aspect='auto',
        extent=[freq[0], freq[-1], time_utc[0], time_utc[-1]],
        origin='lower',
        cmap='viridis'
    )

    plt.xlabel("Frequency [GHz]", fontsize=14)

    plt.ylabel(
        f"Time [UTC] on {time_utc[0].strftime('%Y-%m-%d')}",
        fontsize=14
    )

    cbar = plt.colorbar()
    cbar.set_label("Power [dB]")

    plt.title("Dynamic Spectrum")

    plt.tight_layout()

    plt.show()


# ==============================
# RUN ANALYSIS
# ==============================

if __name__ == "__main__":

    spectrum, freq, time_utc, results = analyze_dataset(
        filename,
        group
    )

    plot_dynamic_spectrum(
        freq,
        time_utc,
        spectrum
    )