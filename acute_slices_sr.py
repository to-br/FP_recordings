import numpy as np
import matplotlib.pyplot as plt
import pyabf
import scipy 
from pathlib import Path


# reading all abf files 

#file_paths = sorted(Path("rec15072026").glob("20260714_cell*.abf"))
#
#print(f"Found {len(file_paths)} files")
#
#for file_path in file_paths:
#    print(file_path)
#    abf = pyabf.ABF(str(file_path))
#
#    print(abf)
#    print("Antall sweeps:", abf.sweepCount)
#    print("Antall kanaler:", abf.channelCount)
#    print("Samplingsfrekvens:", abf.dataRate, "Hz")


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


def compute_slope(time, signal, peak_time, onset_offset=0.001, search_window=0.015, low_frac=0.2, high_frac=0.8):
    # start litt etter artefakten, ikke ved peak_time selv
    search_start = peak_time + onset_offset

    post_mask = (time >= search_start) & (time <= search_start + search_window)
    segment_time = time[post_mask]
    segment_signal = signal[post_mask]

    if len(segment_signal) < 2:
        return np.nan

    trough_idx = np.argmin(segment_signal)
    trough_value = segment_signal[trough_idx]

    # walk back from the trough to where its final, uninterrupted descent starts —
    # skips over any earlier shoulder (e.g. fiber volley) that would otherwise get
    # mixed into the 20-80% fit once it's large enough to fall inside that band
    descent_start_idx = trough_idx
    while descent_start_idx > 0 and segment_signal[descent_start_idx - 1] > segment_signal[descent_start_idx]:
        descent_start_idx -= 1

    falling_time = segment_time[descent_start_idx:trough_idx + 1]
    falling_signal = segment_signal[descent_start_idx:trough_idx + 1]

    low_val = low_frac * trough_value
    high_val = high_frac * trough_value
    fit_mask = (falling_signal <= low_val) & (falling_signal >= high_val)

    if np.sum(fit_mask) < 2:
        return np.nan

    slope, intercept = np.polyfit(falling_time[fit_mask], falling_signal[fit_mask], 1)
    return slope

# recording
date = "26082026"
rec_numbers = ["00","01","02","03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
#rec_to_intensity = {
#    "04": 100,
#    "05": 75,
#    "06": 50,
#    "07": 25,
#    "08": 150,
#    "09": 200,
#    "10": 250,
#}

rec_to_intensity = {
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

filename = f"{date}_FP_cell2_acute_sr__00"

SR_numbers = list(rec_to_intensity.keys())
amplitudes = []
slope_by_rec = {}
slope_sd_by_rec = {}

for rec in rec_numbers:
    fig, ax = plt.subplots(2,1, figsize=(5, 6))
    file_path = f"acute/{date}/{filename}{rec}.abf"
    slopes = []
    abf = pyabf.ABF(str(file_path))
    for sweep_number in abf.sweepList:

        if sweep_number == 0:
            continue
        abf.setSweep(sweepNumber=sweep_number, channel=0)


        time = abf.sweepX       # timepoints
        signal = abf.sweepY     # signal 


        peak_index = np.argmax(np.abs(signal))
        peak_time = time[peak_index]

        #print(sweep_number, peak_time, time.max())

       # signal_length = 50*1e-3
#
       # mask = (time >= peak_time - signal_length/4) & (time <= peak_time + signal_length/4 * 3)
#
       # time_window = time[mask]
       # signal_window = signal[mask]




        signal_filtered = butter_lowpass(
                signal,
                sampling_rate=abf.dataRate,
                cutoff=500,
                order=4,)


        #print(sweep_number, peak_time, time.max())

        signal_length = 50*1e-3

        baseline_mask =  (time >= peak_time + signal_length/2) 
        baseline = np.mean(signal_filtered[baseline_mask])
        signal_filtered = signal_filtered - baseline

        mask = (time >= peak_time - signal_length/4) & (time <= peak_time + signal_length/4 * 3)

        time_window = time[mask]
        signal_window = signal_filtered[mask]

        # slope
        slope = compute_slope(time, signal_filtered, peak_time)
        slopes.append(slope)

        # for the I/O curve 
        
        #if sweep_number == 1 and rec in SR_numbers:
        #    mask_amplitude = (time >= 0.215)
#
        #    time_window_amplitude = time[mask_amplitude]
        #    signal_window_amplitude = signal[mask_amplitude]
        #    amplitude = np.abs(np.min(signal_window_amplitude) - baseline)
        #    amplitudes.append(amplitude)

        # raw signal
        ax[0].plot(time, signal, alpha=0.3, label=f"Sweep {sweep_number}")
        ax[0].set_xlabel("Time (ms)")
        ax[0].set_ylabel(abf.sweepLabelY, fontsize=5)
        ax[0].set_title("FP recording - organotypic slices - raw signal")
        ax[0].legend(loc="upper right", fontsize=6)

        #plt.plot(time * 1000, signal) # the whole signal
        ax[1].plot(time_window, signal_window, alpha=0.7, label=f"Sweep {sweep_number}")
        ax[1].set_xlim(time_window[0],time_window[-1])
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Membrane potential (mV)")
        #ax[1].set_title(f"FP recording - organotypic slices (mean slope: {mean_slope:.4f} mV/s)", fontsize=8)
        ax[1].legend(loc="upper right", fontsize=6)

    if np.all(np.isnan(slopes)):
        print(f"Warning: rec {rec} has no valid slope in any sweep (probably below response threshold) — excluded from I/O curve")
        mean_slope = np.nan
    else:
        mean_slope = np.nanmean(slopes)
        sd_slope = np.nanstd(slopes)
        if rec in SR_numbers:
            slope_by_rec[rec] = mean_slope
            slope_sd_by_rec[rec] = sd_slope


    ax[1].set_title(f"FP recording in SR - current strength: {rec_to_intensity[rec]} uA (mean slope: {mean_slope:.4f} mV/s)", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"acute/{date}/plots/all_sweeps{rec}.png", dpi=300)
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
plt.title(f"I/O Curve - SR (acute slices) - Mean Slope: {mean_slope_global:.4f} mV/s")
plt.grid()
plt.savefig(f"acute/{date}/plots/io_curve.png", dpi=300)


