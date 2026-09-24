import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyabf
from scipy.signal import butter, sosfiltfilt

# ---- settings ----------------------------------------------------
DATE = "14092026" 
FOLDER = Path("acute") / DATE
FILE_PATTERN = "*sr*.abf"
# -----------------------------------------------------------------------



def butter_lowpass(signal, sampling_rate, cutoff=500, order=4):
    sos = butter(
        N=order,
        Wn=cutoff,
        btype="lowpass",
        fs=sampling_rate,
        output="sos"
    )

    return sosfiltfilt(sos, signal)


def stim_time_s(abf):
    """Stimulus pulse time (s) = onset of the shortest step epoch in the command waveform."""
    ep = abf.sweepEpochs
    durations = [p2 - p1 for p1, p2 in zip(ep.p1s, ep.p2s)]
    idx = durations.index(min(durations))
    return ep.p1s[idx] / abf.dataRate


def compute_slope(time, signal, stim_time, onset_offset=0.001, search_window=0.025, low_frac=0.2, high_frac=0.8):
    """Searches a 25 ms window starting 1 ms after the stimulus,
      to avoid the artifact. It identifies the dominant deflection in that window
        (the negative fEPSP trough, or a positive population spike if that is larger), 
        and fits a straight line by least squares to the rising phase between the
        first crossings of 20 % and 80 % of that peak amplitude. 
        The slope is returned in signal units per second, with its original sign."""
    
    search_start = stim_time + onset_offset

    post_mask = (time >= search_start) & (time <= search_start + search_window)
    segment_time = time[post_mask]
    segment_signal = signal[post_mask]

    if len(segment_signal) < 2:
        return np.nan

    # dominant deflection may be negative (fEPSP) or positive (population spike
    # overtaking the fEPSP at higher stimulus strengths) — pick whichever is larger
    idx_min = np.argmin(segment_signal)
    idx_max = np.argmax(segment_signal)
    if abs(segment_signal[idx_min]) >= abs(segment_signal[idx_max]):
        sign, peak_idx = 1.0, idx_min
    else:
        sign, peak_idx = -1.0, idx_max

    working_signal = segment_signal * sign  # dominant deflection now always a negative peak
    peak_value = working_signal[peak_idx]

    if peak_idx < 1:
        return np.nan

    # scan FORWARD from response onset to the peak for the first 20% and 80%
    # threshold crossings. This only needs a first crossing, not an unbroken
    # descent, so a small non-monotonic wobble (e.g. an early shoulder) before
    # the real peak doesn't throw off where the fit window starts — unlike a
    # backward walk from the peak, which is sensitive to exactly where that
    # wobble sits and can silently give a very short, unstable fit window.
    rising = working_signal[:peak_idx + 1]
    low_val = low_frac * peak_value
    high_val = high_frac * peak_value

    low_candidates = np.where(rising <= low_val)[0]
    high_candidates = np.where(rising <= high_val)[0]
    if len(low_candidates) == 0 or len(high_candidates) == 0:
        return np.nan

    idx_low = low_candidates[0]
    idx_high = high_candidates[0]
    if idx_high <= idx_low:
        return np.nan

    fit_time = segment_time[idx_low:idx_high + 1]
    fit_signal = working_signal[idx_low:idx_high + 1]

    if len(fit_signal) < 2:
        return np.nan

    slope, intercept = np.polyfit(fit_time, fit_signal, 1)
    return slope * sign

# recording
rec_to_intensity_2602026 = {
    "00": 25,
    "01": 50,
    "02": 75,
    "03": 100,
    "04": 125,
    "05": 150,
    "06": 175,
    "07": 200,
    "08": 225,
    "09": 250,
    "10": 275,
    "11": 300,
    "12": 350,
    "13": 1000,
    "14": 500,
    "15": 400,
    "16": 750,
}

rec_to_intensity_14092026 = {
    "00": 150,
    "01": 175,
    "02": 200,
    "03": 250,
    "04": 300,
    "05": 350,
    "06": 400,
    "07": 450,
    "08": 500,
    "09": 550,
    "10": 600,
    "11": 650,
    "12": 700,
    "13": 750,
    "14": 800,
    "15": 850,
    "16": 900,
    "17": 950,
    "18": 1000,
}

rec_to_intensity = rec_to_intensity_14092026 if DATE == "14092026" else rec_to_intensity_2602026

SR_numbers = list(rec_to_intensity.keys())
amplitudes = []
slope_by_rec = {}
slope_sd_by_rec = {}

files = sorted(
    p for p in FOLDER.rglob("*.abf")
    if p.is_file() and "sr" in p.name.lower()
)

if not files:
    raise SystemExit(f"Fant ingen SR .abf filer i {FOLDER}. Sjekk at mappen finnes og at filnavnene inneholder 'sr'.")

plots_dir = FOLDER / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

print(f"Fant {len(files)} SR-filer i {FOLDER}")
for p in files:
    print(f" - {p.name}")

for file_path in files:
    fig, ax = plt.subplots(2, 1, figsize=(5, 6))
    stem = file_path.stem
    match = re.search(r"sr_(\d+)$", stem) or re.search(r"(\d+)$", stem)
    rec = f"{int(match.group(1)) :02d}" if match else "00"
    slopes = []
    abf = pyabf.ABF(str(file_path))
    for sweep_number in abf.sweepList:

        if sweep_number == 0:
            continue
        abf.setSweep(sweepNumber=sweep_number, channel=0)


        time = abf.sweepX       # timepoints
        signal = abf.sweepY     # signal 


        peak_time = stim_time_s(abf)

        signal_filtered = butter_lowpass(
            signal,
            sampling_rate=abf.dataRate,
            cutoff=500,
            order=4,
        )

        signal_length = 50 * 1e-3

        baseline_mask = (time >= peak_time + signal_length / 2)
        baseline = np.mean(signal_filtered[baseline_mask])
        signal_filtered = signal_filtered - baseline

        mask = (time >= peak_time - signal_length / 4) & (time <= peak_time + signal_length / 4 * 3)

        time_window = time[mask]
        signal_window = signal_filtered[mask]

        slope = compute_slope(time, signal_filtered, peak_time)
        slopes.append(slope)

        ax[0].plot(time, signal, alpha=0.3, label=f"Sweep {sweep_number}")
        ax[0].set_xlabel("Time (ms)")
        ax[0].set_ylabel("Membrane potential (mV)")
        ax[0].set_title(f"FP recording - SR - {DATE} - raw signal")
        ax[0].legend(loc="upper right", fontsize=6)
        ax[0].set_ylim(-1.5,0)
        ax[0].set_xlim(0.2, 0.25)

        ax[1].plot(time_window, signal_window, alpha=0.7, label=f"Sweep {sweep_number}")
        ax[1].set_xlim(time_window[0], time_window[-1])
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Membrane potential (mV)")
        ax[1].legend(loc="upper right", fontsize=6)
        ax[1].set_ylim(-0.4, 0.1)

    if np.all(np.isnan(slopes)):
        print(f"Warning: file {file_path.name} has no valid slope in any sweep — excluded from I/O curve")
        mean_slope = np.nan
    else:
        mean_slope = np.nanmean(slopes)
        sd_slope = np.nanstd(slopes)
        if rec in SR_numbers:
            slope_by_rec[rec] = mean_slope
            slope_sd_by_rec[rec] = sd_slope

    current_strength = rec_to_intensity.get(rec, "unknown")
    ax[1].set_title(
        f"FP recording in SR - {DATE} - current strength: {current_strength} uA "
        f"(mean slope: {mean_slope:.4f} mV/s)",
        fontsize=8,
    )
    fig.suptitle(f"SR / {DATE}", fontsize=10, y=0.98)
    plt.tight_layout()
    plt.savefig(plots_dir / f"all_sweeps_{file_path.stem}.png", dpi=300)
    plt.close()

#-------------------------I/O-CURVE-----------------------------------
mean_slope_global = np.nanmean(list(slope_by_rec.values()))
plt.figure()
sorted_recs = sorted(slope_by_rec, key=lambda r: rec_to_intensity[r])
x = [rec_to_intensity[r] for r in sorted_recs]
y = [np.abs(slope_by_rec[r]) for r in sorted_recs]
yerr = [slope_sd_by_rec[r] for r in sorted_recs]
plt.errorbar(x, y, yerr=yerr, marker='o', capsize=3)
plt.xlabel("Stimulus Intensity (µA)")
plt.ylabel("|Slope| (mV/s)")
plt.title(f"I/O Curve - SR ({DATE}) - Mean Slope: {mean_slope_global:.4f} mV/s")
plt.grid()
plt.savefig(plots_dir / "io_curve.png", dpi=300)


