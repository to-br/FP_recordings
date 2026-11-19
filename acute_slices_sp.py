import numpy as np
import matplotlib.pyplot as plt
import pyabf
from pathlib import Path

from fp_analysis import stim_time_s, preprocess, mean_trace, find_slope_window, compute_slope, pop_spike_amplitude

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

# fixed slope window: defined once on the mean response of ref_rec, then applied to
# every sweep and intensity. In SP the fEPSP is positive-going (fepsp_polarity=+1).
ref_rec = "10"
fepsp_search_ms = (8.5, 12.0)
fepsp_polarity = +1
# population spike (tangent method) is measured separately, never mixed into the slope
pop_spike_search_ms = (6.0, 12.0)

valid_recs = list(rec_to_intensity.keys())
slope_by_rec = {}
slope_sd_by_rec = {}
pop_by_rec = {}
pop_sd_by_rec = {}

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

# ---- fixed slope window from the reference recording ----
ref_abf = pyabf.ABF(str(f"{plots_root}/{filename}{ref_rec}.abf"))
ref_time, ref_signal, ref_stim = mean_trace(ref_abf, ref_abf.sweepList[1:])
slope_window = find_slope_window(ref_time, ref_signal, ref_stim,
                                 search_s=(fepsp_search_ms[0] / 1000, fepsp_search_ms[1] / 1000),
                                 polarity=fepsp_polarity)
if slope_window is None:
    raise SystemExit(f"Fant ikke fEPSP-flanke i referansen rec {ref_rec} - sjekk ref_rec / fepsp_search_ms")
print(f"Fast slope-vindu fra rec {ref_rec} ({rec_to_intensity[ref_rec]} uA): "
      f"{slope_window[0] * 1000:.2f}-{slope_window[1] * 1000:.2f} ms etter stimulus")
pop_search_s = (pop_spike_search_ms[0] / 1000, pop_spike_search_ms[1] / 1000)

for rec in rec_numbers:
    fig, ax = plt.subplots(2, 1, figsize=(5, 6))
    file_path = f"{plots_root}/{filename}{rec}.abf"
    slopes = []
    pop_amps = []
    abf = pyabf.ABF(str(file_path))
    for sweep_number in abf.sweepList:

        if sweep_number == 0:
            continue
        abf.setSweep(sweepNumber=sweep_number, channel=0)

        time = abf.sweepX       # timepoints
        signal = abf.sweepY     # signal

        peak_time = stim_time_s(abf)

        signal_filtered = preprocess(time, signal, peak_time, abf.dataRate)

        signal_length = 50 * 1e-3

        mask = (time >= peak_time - signal_length / 4) & (time <= peak_time + signal_length / 4 * 3)

        time_window = time[mask]
        signal_window = signal_filtered[mask]

        # slope
        slope = compute_slope(time, signal_filtered, peak_time, slope_window)
        slopes.append(slope)

        # population spike amplitude (separate measure)
        pop_amp, _ = pop_spike_amplitude(time, signal_filtered, peak_time, pop_search_s)
        pop_amps.append(pop_amp)

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

    if np.all(np.isnan(pop_amps)):
        mean_pop = np.nan
    else:
        mean_pop = np.nanmean(pop_amps)
        if rec in valid_recs and rec not in exclude_recs:
            pop_by_rec[rec] = mean_pop
            pop_sd_by_rec[rec] = np.nanstd(pop_amps)

    ax[1].axvspan(peak_time + slope_window[0], peak_time + slope_window[1], color="grey", alpha=0.2, lw=0)
    ax[1].set_title(f"FP recording in SP - current strength: {rec_to_intensity[rec]} uA (mean slope: {mean_slope:.4f} mV/s, pop spike: {mean_pop:.3f} mV)", fontsize=8)
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
plt.title(f"I/O Curve - SP - Mean Slope: {mean_slope_global:.4f} mV/s")
plt.grid()
plt.savefig(f"{plots_root}/plots/{series_name}/io_curve.png", dpi=300)

#-------------------------I/O-CURVE POP SPIKE---------------------------
plt.figure()
sorted_recs = sorted(pop_by_rec, key=lambda r: rec_to_intensity[r])
x = [rec_to_intensity[r] for r in sorted_recs]
y = [pop_by_rec[r] for r in sorted_recs]
yerr = [pop_sd_by_rec[r] for r in sorted_recs]
plt.errorbar(x, y, yerr=yerr, marker='o', capsize=3)
plt.xlabel("Stimulus Intensity (µA)")
plt.ylabel("Pop spike amplitude (mV)")
plt.title("I/O Curve - SP population spike (tangent method)", fontsize=9)
plt.grid()
plt.savefig(f"{plots_root}/plots/{series_name}/io_curve_pop_spike.png", dpi=300)
