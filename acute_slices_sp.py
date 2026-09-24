import numpy as np
import matplotlib.pyplot as plt
import pyabf
import scipy
from pathlib import Path
from scipy.signal import butter, sosfiltfilt

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


def compute_slope(time, signal, peak_time, onset_offset=0.001, search_window=0.025, low_frac=0.2, high_frac=0.8):
    # start litt etter artefakten, ikke ved peak_time selv
    search_start = peak_time + onset_offset

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
date = "21092026"
base_dir = ""  # set to "" for dates that aren't under acute/ (e.g. 21092026)
rec_numbers = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17"]

# rec 04 (30 uA) ga ingen respons ("ingenting") ifolge notatene
rec_to_intensity = {
    "00": 20, "01": 10, "02": 5, "03": 5, "04": 30, "05": 30, "06": 40, "07": 50,
    "08": 60, "09": 70, "10": 80, "11": 90, "12": 100, "13": 120, "14": 140,
    "15": 160, "16": 180, "17": 200,
}

filename = f"{date}_cell1_sp_00"
series_name = "sp"  # subfolder under plots/, keeps this series' output separate from others
exclude_recs = ("04",)  # ga ingen respons ("ingenting") ifolge notatene - ren stoy, ikke ekte slope

valid_recs = list(rec_to_intensity.keys())
slope_by_rec = {}
slope_sd_by_rec = {}

plots_root = f"{base_dir}/{date}" if base_dir else date
Path(f"{plots_root}/plots/{series_name}").mkdir(parents=True, exist_ok=True)

# fixed y-limits for the raw-signal panel, shared across all recs, excluding the
# stimulus artifact itself (stim to stim+3ms) so it doesn't blow out the scale
raw_min, raw_max = np.inf, -np.inf
for rec in rec_numbers:
    abf = pyabf.ABF(str(f"{plots_root}/{filename}{rec}.abf"))
    stim = stim_time_s(abf)
    for sweep_number in abf.sweepList:
        if sweep_number == 0:
            continue
        abf.setSweep(sweepNumber=sweep_number, channel=0)
        time, signal = abf.sweepX, abf.sweepY
        no_artifact_mask = (time < stim) | (time > stim + 0.003)
        raw_min = min(raw_min, signal[no_artifact_mask].min())
        raw_max = max(raw_max, signal[no_artifact_mask].max())
raw_pad = 0.1 * (raw_max - raw_min)
raw_ylim = (raw_min - raw_pad, raw_max + raw_pad)

for rec in rec_numbers:
    fig, ax = plt.subplots(2, 1, figsize=(5, 6))
    file_path = f"{plots_root}/{filename}{rec}.abf"
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
                order=4,)

        signal_length = 50 * 1e-3

        baseline_mask = (time >= peak_time + signal_length / 2)
        baseline = np.mean(signal_filtered[baseline_mask])
        signal_filtered = signal_filtered - baseline

        mask = (time >= peak_time - signal_length / 4) & (time <= peak_time + signal_length / 4 * 3)

        time_window = time[mask]
        signal_window = signal_filtered[mask]

        # slope
        slope = compute_slope(time, signal_filtered, peak_time)
        slopes.append(slope)

        # raw signal
        ax[0].plot(time, signal, alpha=0.3, label=f"Sweep {sweep_number}")
        ax[0].set_ylim(raw_ylim)
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Membrane potential (mV)")
        ax[0].set_title("FP recording - sp - raw signal")
        ax[0].legend(loc="upper right", fontsize=6)

        ax[1].plot(time_window, signal_window, alpha=0.7, label=f"Sweep {sweep_number}")
        ax[1].set_xlim(time_window[0], time_window[-1])
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Membrane potential (mV)")
        ax[1].legend(loc="upper right", fontsize=6)

    if np.all(np.isnan(slopes)):
        print(f"Warning: rec {rec} has no valid slope in any sweep (probably below response threshold) — excluded from I/O curve")
        mean_slope = np.nan
    else:
        mean_slope = np.nanmean(slopes)
        sd_slope = np.nanstd(slopes)
        if rec in valid_recs and rec not in exclude_recs:
            slope_by_rec[rec] = mean_slope
            slope_sd_by_rec[rec] = sd_slope

    ax[1].set_title(f"FP recording in SP - current strength: {rec_to_intensity[rec]} uA (mean slope: {mean_slope:.4f} mV/s)", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{plots_root}/plots/{series_name}/all_sweeps{rec}.png", dpi=300)
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
plt.title(f"I/O Curve - SP (acute slices) - Mean Slope: {mean_slope_global:.4f} mV/s")
plt.grid()
plt.savefig(f"{plots_root}/plots/{series_name}/io_curve.png", dpi=300)
