
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pyabf
from scipy.signal import butter, sosfiltfilt

# ---- settings ----------------------------------------------------
folder = "22092026"
T0, T1 = 200, 250          # time window (ms) for plotting
BASELINE = 200             # baseline = average of signal before this time (ms)
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


files = sorted(glob.glob(os.path.join(folder, "**", "*TA_stim*.abf"), recursive=True))
if not files:
    raise SystemExit(f"Fant ingen TA_stim .abf i {os.path.abspath(folder)}")
print(f"Fant {len(files)} filer i {folder}")

plots_dir = os.path.join(folder, "plots")
os.makedirs(plots_dir, exist_ok=True)

for f in files:
    abf = pyabf.ABF(f)

    # les alle sweeps inn i en matrise
    t = abf.sweepX * 1000                      # seconds -> ms
    sweeps = np.empty((abf.sweepCount, len(t))) # empty array for all sweeps
    for i in range(abf.sweepCount):
        abf.setSweep(i)
        sweeps[i] = abf.sweepY 

    # removing the baseline
    sweeps = sweeps - sweeps[:, t < BASELINE].mean(axis=1, keepdims=True)
    sweeps = butter_lowpass(sweeps, abf.dataRate, cutoff=500, order=4)
    mean = sweeps.mean(axis=0)

    # find stimulus artefact
    stim = t[np.argmax(np.abs(mean)) + 1]

    # y-grenser fra responsen etter artefakten, ikke fra artefakten selv
    after = (t > stim + 2) & (t < T1)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, sweep in enumerate(sweeps):
        ax.plot(t, sweep, lw=0.6, label=f"Sweep {i+1}")
    ax.axvline(stim, color="C3", lw=0.8, ls="--", alpha=0.6)  # stimulus
    plt.legend(fontsize=6, loc="upper right")
    navn = os.path.basename(f).replace(".abf", "")
    ax.set_title(navn, fontsize=9)
    ax.set_xlim(T0, T1)
    ax.set_ylim(-0.45, 0.45)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Field potential (mV)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    output = os.path.join(plots_dir, navn + ".png")
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved: {output}")