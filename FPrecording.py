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

# recording 
rec_number = "06"
date = "21082026"
#file_path = f"C:\Users\tonje\Experiments\210826\rec210826\21082026_FP_cell1_organotypic_sp__00{rec_number}.abf"
#file_path = f"Experiments\{date}\rec{date}\{date}_FP_cell1_organotypic_sp_00{rec_number}.abf"

file_path = Path(date) / f"rec{date}" / f"{date}_FP_cell1_organotypic_sp__00{rec_number}.abf"
abf = pyabf.ABF(str(file_path))


plt.figure()
for sweep_number in abf.sweepList:
    #if sweep_number == 0:
     #   continue
    abf.setSweep(sweepNumber=sweep_number, channel=0)


    time = abf.sweepX       # timepoints
    signal = abf.sweepY     # signal 


    signal = butter_lowpass(
            signal,
            sampling_rate=abf.dataRate,
            cutoff=500,
            order=4,)

    peak_index = np.argmax(np.abs(signal))
    peak_time = time[peak_index]

    signal_length = 50*1e-3

    baseline_mask =  (time >= peak_time + signal_length) # til 1 ms før stim
    baseline = np.mean(signal[baseline_mask])
    signal = signal - baseline

    mask = (time >= peak_time - signal_length/4) & (time <= peak_time + signal_length/4 * 3)

    time_window = time[mask]
    signal_window = signal[mask]

    
    #plt.plot(time * 1000, signal) # the whole signal
    plt.plot(
    time_window,
    signal_window,
    alpha=0.7,
    label=f"Sweep {sweep_number}")

plt.xlabel("Time (ms)")
plt.ylabel(abf.sweepLabelY)
plt.title("FP recording - organotypic slices")
plt.legend(loc="upper right")
#plt.xlim(0.03,0.08)
plt.tight_layout()
plt.xlim(0.0325,0.07)
plt.savefig(f"{date}/plots/all_sweeps_{rec_number}.png", dpi=300)
plt.close()



all_sweeps = []

for sweep_number in abf.sweepList:
    abf.setSweep(sweepNumber=sweep_number, channel=0)

    time = abf.sweepX.copy()
    signal = abf.sweepY.copy()

   # signal = butter_lowpass(
    #    signal,
    #    sampling_rate=abf.dataRate,
    #    cutoff=5000,
    #    order=4)

    time = time[mask]
    signal = signal[mask]

    all_sweeps.append(signal)

# Form: (antall sweeps, antall tidspunkter)
all_sweeps = np.array(all_sweeps)

# Gjennomsnitt over sweep-aksen
mean_signal = np.mean(all_sweeps, axis=0)

plt.figure()
plt.plot(time * 1000, mean_signal, color="black")
plt.xlabel("Tid (ms)")
plt.ylabel(abf.sweepLabelY)
plt.title(f"Mean of {abf.sweepCount} sweeps")
plt.tight_layout()
plt.savefig(f"{date}/plots/mean_{rec_number}.png", dpi=300)
plt.close()