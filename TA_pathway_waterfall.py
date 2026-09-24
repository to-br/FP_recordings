import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pyabf
from scipy.signal import butter, sosfiltfilt

# ---- settings ----------------------------------------------------
folder = "22092026"
T0, T1 = 200, 250          # time window (ms) for plotting
BASELINE = 200             # baseline = average of signal before this time (ms)
OFFSET = 0.2               # vertical spacing between stacked traces (mV)
SCALE_MV = 0.1             # amplitude scale bar (mV)
SCALE_MS = 10              # time scale bar (ms)
# -----------------------------------------------------------------------


def butter_lowpass(signal, sampling_rate, cutoff=500, order=4):
    sos = butter(N=order, Wn=cutoff, btype="lowpass", fs=sampling_rate, output="sos")
    return sosfiltfilt(sos, signal)


def stim_time_ms(abf):
    """Stimulus pulse time = onset of the shortest step epoch in the command waveform."""
    ep = abf.sweepEpochs
    durations = [p2 - p1 for p1, p2 in zip(ep.p1s, ep.p2s)]
    idx = durations.index(min(durations))
    return ep.p1s[idx] / abf.dataRate * 1000


def mean_sweep(files):
    """Load + baseline-correct + filter all sweeps from a list of abf files, return grand mean."""
    all_sweeps = []
    t = None
    rate = None
    for f in files:
        abf = pyabf.ABF(f)
        t = abf.sweepX * 1000  # s -> ms
        rate = abf.dataRate
        sweeps = np.empty((abf.sweepCount, len(t)))
        for i in range(abf.sweepCount):
            abf.setSweep(i)
            sweeps[i] = abf.sweepY
        sweeps = sweeps - sweeps[:, t < BASELINE].mean(axis=1, keepdims=True)
        all_sweeps.append(sweeps)
    all_sweeps = np.vstack(all_sweeps)
    mean = butter_lowpass(all_sweeps, rate, cutoff=500, order=4).mean(axis=0)
    return t, mean


# ---- gather files per location -----------------------------------
files = sorted(glob.glob(os.path.join(folder, "*TA_stim*.abf")))
if not files:
    raise SystemExit(f"Fant ingen TA_stim .abf i {os.path.abspath(folder)}")

by_loc = {}
for f in files:
    m = re.search(r"loc(\d+)", os.path.basename(f))
    if m:
        loc = int(m.group(1))
    elif "slm" in os.path.basename(f):
        loc = 1  # slm-filene er loc1, bare uten "loc" i navnet
    else:
        continue
    by_loc.setdefault(loc, []).append(f)

locations = sorted(by_loc)
print(f"Fant {len(locations)} lokasjoner: {locations}")

# ---- build waterfall ------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 7))
cmap = plt.get_cmap("winter")  # blue -> green
colors = cmap(np.linspace(0, 1, len(locations)))

stim = stim_time_ms(pyabf.ABF(files[0]))
ymin = None
for i, loc in enumerate(locations):
    t, mean = mean_sweep(by_loc[loc])
    offset_trace = mean + i * OFFSET
    window = (t > T0) & (t < T1)
    trace_min = offset_trace[window].min()
    ymin = trace_min if ymin is None else min(ymin, trace_min)
    ax.plot(t, offset_trace, color=colors[i], lw=1.0)

ax.axvline(stim, color="C3", lw=1.2, ls="--", alpha=0.7)

# scale bar (bottom-left, in data coords), placed below the lowest trace
x0 = T0 + 2
y0 = ymin - 0.1
ax.plot([x0, x0 + SCALE_MS], [y0, y0], color="black", lw=1.5)
ax.plot([x0, x0], [y0, y0 + SCALE_MV], color="black", lw=1.5)
ax.text(x0 + SCALE_MS / 2, y0 - 0.02, f"{SCALE_MS} ms", ha="center", va="top", fontsize=8)
ax.text(x0 - 2, y0 + SCALE_MV / 2, f"{SCALE_MV} mV", ha="right", va="center", fontsize=8)

ax.set_xlim(T0, T1)
ax.set_title("TA stimulation - laminar profile", fontsize=10)
ax.axis("off")
fig.tight_layout()

plots_dir = os.path.join(folder, "plots")
os.makedirs(plots_dir, exist_ok=True)
output = os.path.join(plots_dir, "TA_stim_waterfall.png")
fig.savefig(output, dpi=200)
plt.close(fig)
print(f"Saved: {output}")
