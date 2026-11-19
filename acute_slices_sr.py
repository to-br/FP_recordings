import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyabf

from fp_analysis import stim_time_s, preprocess, mean_trace, find_slope_window, compute_slope

# ---- settings ----------------------------------------------------
DATE = "12082026"
FOLDER = Path("acute") / {"12082026": "120826"}.get(DATE, DATE)   # 12.08 folder is named 120826
FILE_PATTERN = "*sr*.abf"

# fixed slope window: defined once on the mean response of REF_REC, then applied to
# every sweep and intensity. FEPSP_SEARCH_MS must start AFTER the fibre volley.
FEPSP_SEARCH_MS = {"14092026": (4.0, 15.0), "26082026": (3.0, 15.0), "12082026": (3.0, 15.0)}[DATE]
REF_REC = {"14092026": "09", "26082026": "08", "12082026": "10"}[DATE]
# -----------------------------------------------------------------------



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

# 12.08: recs 04-10 (from the original analysis; rec 12 was listed as 300 uA there but is left out here,
# as in the later 12.08 I/O curve). Other recs (00-03, 11, 13-16) are not part of the I/O series.
rec_to_intensity_12082026 = {
    "04": 100,
    "05": 75,
    "06": 50,
    "07": 25,
    "08": 150,
    "09": 200,
    "10": 250,
}

rec_to_intensity = {"14092026": rec_to_intensity_14092026,
                    "12082026": rec_to_intensity_12082026}.get(DATE, rec_to_intensity_2602026)

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

def rec_of(file_path):
    match = re.search(r"sr_(\d+)$", file_path.stem) or re.search(r"(\d+)$", file_path.stem)
    return f"{int(match.group(1)) :02d}" if match else "00"

# ---- fixed slope window from the reference recording ----
ref_file = [p for p in files if rec_of(p) == REF_REC][-1]
ref_abf = pyabf.ABF(str(ref_file))
ref_time, ref_signal, ref_stim = mean_trace(ref_abf, ref_abf.sweepList[1:])
slope_window = find_slope_window(ref_time, ref_signal, ref_stim,
                                 search_s=(FEPSP_SEARCH_MS[0] / 1000, FEPSP_SEARCH_MS[1] / 1000))
if slope_window is None:
    raise SystemExit(f"Fant ikke fEPSP-flanke i referansen {ref_file.name} - sjekk REF_REC / FEPSP_SEARCH_MS")
print(f"Fast slope-vindu fra {ref_file.name} ({rec_to_intensity[REF_REC]} uA): "
      f"{slope_window[0] * 1000:.2f}-{slope_window[1] * 1000:.2f} ms etter stimulus")

for file_path in files:
    fig, ax = plt.subplots(2, 1, figsize=(5, 6))
    rec = rec_of(file_path)
    slopes = []
    raw_no_artifact = []   # raw samples in the plotted window, artifact excluded (for y-limits)
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

        slope = compute_slope(time, signal_filtered, peak_time, slope_window)
        slopes.append(slope)

        # exclude the artifact itself (stim to stim+3ms) when figuring out sensible y-limits
        keep = (time >= 0.2) & (time <= 0.25) & ((time < peak_time) | (time > peak_time + 0.003))
        raw_no_artifact.append(signal[keep])

        ax[0].plot(time, signal, alpha=0.3, label=f"Sweep {sweep_number}")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Membrane potential (mV)")
        ax[0].set_title(f"FP recording - SR - {DATE} - raw signal")
        ax[0].legend(loc="upper right", fontsize=6)
        ax[0].set_xlim(0.2, 0.25)

        ax[1].plot(time_window, signal_window, alpha=0.7, label=f"Sweep {sweep_number}")
        ax[1].set_xlim(time_window[0], time_window[-1])
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Membrane potential (mV)")
        ax[1].legend(loc="upper right", fontsize=6)

    if np.all(np.isnan(slopes)):
        print(f"Warning: file {file_path.name} has no valid slope in any sweep — excluded from I/O curve")
        mean_slope = np.nan
    else:
        mean_slope = np.nanmean(slopes)
        sd_slope = np.nanstd(slopes)
        if rec in SR_numbers:
            slope_by_rec[rec] = mean_slope
            slope_sd_by_rec[rec] = sd_slope

    raw_no_artifact = np.concatenate(raw_no_artifact)
    pad = 0.1 * (raw_no_artifact.max() - raw_no_artifact.min())
    ax[0].set_ylim(raw_no_artifact.min() - pad, raw_no_artifact.max() + pad)

    ax[1].axvspan(peak_time + slope_window[0], peak_time + slope_window[1], color="grey", alpha=0.2, lw=0)

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
plt.title(f"I/O Curve - SR - Mean Slope: {mean_slope_global:.4f} mV/s")
plt.grid()
plt.savefig(plots_dir / "io_curve.png", dpi=300)


