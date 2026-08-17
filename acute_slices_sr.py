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
    trough_time = segment_time[trough_idx]

    falling_mask = (time >= search_start) & (time <= trough_time)
    falling_time = time[falling_mask]
    falling_signal = signal[falling_mask]

    low_val = low_frac * trough_value
    high_val = high_frac * trough_value
    fit_mask = (falling_signal <= low_val) & (falling_signal >= high_val)

    if np.sum(fit_mask) < 2:
        return np.nan

    slope, intercept = np.polyfit(falling_time[fit_mask], falling_signal[fit_mask], 1)
    return slope

# recording 
rec_numbers = ["00","01","02","03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
SR_numbers = ["04", "05", "06", "07", "08", "09", "10",  "12"]
stim_intensities = [100, 75, 50, 25, 150, 200, 250, 300]
amplitudes = []
slope_global = []

for rec in rec_numbers:
    plt.figure()
    fig, ax = plt.subplots(2,1, figsize=(5, 6))
    file_path = f"acute_slices_120826/12082026_FP_SR_cell2_activation_00{rec}.abf"
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

        signal_length = 50*1e-3

        mask = (time >= peak_time - signal_length/4) & (time <= peak_time + signal_length/4 * 3)

        time_window = time[mask]
        signal_window = signal[mask]




        signal_filtered = butter_lowpass(
                signal,
                sampling_rate=abf.dataRate,
                cutoff=500,
                order=4,)


        #print(sweep_number, peak_time, time.max())

        signal_length = 50*1e-3

        baseline_mask =  (time >= peak_time + signal_length/2) # til 1 ms før stim
        baseline = np.mean(signal_filtered[baseline_mask])
        signal_filtered = signal_filtered - baseline

        mask = (time >= peak_time - signal_length/4) & (time <= peak_time + signal_length/4 * 3)

        time_window = time[mask]
        signal_window = signal_filtered[mask]

        # slope
        slope = compute_slope(time, signal_filtered, peak_time)
        slopes.append(slope)

        # for the I/O curve 
        
        if sweep_number == 1 and rec in SR_numbers:
            mask_amplitude = (time >= 0.215)

            time_window_amplitude = time[mask_amplitude]
            signal_window_amplitude = signal[mask_amplitude]
            amplitude = np.abs(np.min(signal_window_amplitude) - baseline)
            amplitudes.append(amplitude)


        mean_slope = np.nanmean(slopes)
        slope_global.append(mean_slope)

        # raw signal
        ax[0].plot(time, signal, alpha=0.3, label=f"Sweep {sweep_number}")
        ax[0].set_xlabel("Time (ms)")
        ax[0].set_ylabel(abf.sweepLabelY, fontsize=5)
        ax[0].set_title("FP recording - organotypic slices - raw signal")
        ax[0].legend(loc="upper right", fontsize=6)

        #plt.plot(time * 1000, signal) # the whole signal
        ax[1].plot(time_window, signal_window, alpha=0.7, label=f"Sweep {sweep_number}")
        ax[1].set_xlim(time_window[0],time_window[-1])
        ax[1].set_xlabel("Time (ms)")
        ax[1].set_ylabel(abf.sweepLabelY)
        ax[1].set_title(f"FP recording - organotypic slices (mean slope: {mean_slope:.4f} mV/s)", fontsize=8)
        ax[1].legend(loc="upper right", fontsize=6)


    plt.tight_layout()
    plt.savefig(f"acute_slices_plots/all_sweeps{rec}.png", dpi=300)
    plt.close()



#-------------------------I/O-CURVE-----------------------------------
mean_slope_global = np.nanmean(slope_global)    
plt.figure()
sorted_pairs = sorted(zip(stim_intensities, amplitudes))
x, y = zip(*sorted_pairs)
plt.plot(x, y, marker='o')
plt.xlabel("Stimulus Intensity (µA)")
plt.ylabel("Amplitude (mV)")
plt.title(f"I/O Curve - SR (acute slices) - Mean Slope: {mean_slope_global:.4f} mV/s")
plt.grid()
plt.savefig(f"acute_slices_plots/io_curve.png", dpi=300)


