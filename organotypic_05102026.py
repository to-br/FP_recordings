import numpy as np
import matplotlib.pyplot as plt
import pyabf
from pathlib import Path

from fp_analysis import stim_time_s, preprocess, mean_trace

# ---- settings ----------------------------------------------------
DATE = "05102026"
FOLDER = Path(DATE)
PLOTS = FOLDER / "plots"
# Organotypic slice: no clean early fEPSP flank to put a slope window on, so the
# response is quantified as the negative and positive peak (relative to baseline)
# within this window after the stimulus. Starts after the artifact/fibre volley.
PEAK_WINDOW_S = (0.003, 0.030)
PLOT_WINDOW_S = (-0.005, 0.040)
# -----------------------------------------------------------------------


def peaks(time, signal, stim):
    mask = (time >= stim + PEAK_WINDOW_S[0]) & (time <= stim + PEAK_WINDOW_S[1])
    return signal[mask].min(), signal[mask].max()


def plot_file(file_path, name, label):
    """Raw + filtered sweeps for one file; returns per-sweep (neg_peak, pos_peak)."""
    abf = pyabf.ABF(str(file_path))
    stim = stim_time_s(abf)
    fig, ax = plt.subplots(2, 1, figsize=(5, 6))
    neg, pos = [], []
    raw_no_artifact = []

    for sweep_number in abf.sweepList:
        abf.setSweep(sweepNumber=sweep_number, channel=0)
        time = abf.sweepX
        signal = abf.sweepY
        signal_filtered = preprocess(time, signal, stim, abf.dataRate)

        n, p = peaks(time, signal_filtered, stim)
        neg.append(n)
        pos.append(p)

        no_artifact_mask = (time < stim) | (time > stim + 0.003)
        raw_no_artifact.append(signal[no_artifact_mask])

        mask = (time >= stim + PLOT_WINDOW_S[0]) & (time <= stim + PLOT_WINDOW_S[1])
        ax[0].plot(time, signal, alpha=0.3, lw=0.6, label=f"Sweep {sweep_number}")
        ax[1].plot((time[mask] - stim) * 1000, signal_filtered[mask], alpha=0.7, lw=0.8,
                   label=f"Sweep {sweep_number}")

    raw_no_artifact = np.concatenate(raw_no_artifact)
    pad = 0.1 * (raw_no_artifact.max() - raw_no_artifact.min())
    ax[0].set_ylim(raw_no_artifact.min() - pad, raw_no_artifact.max() + pad)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel(f"Signal ({abf.adcUnits[0]})")
    ax[0].set_title(f"{label} - raw signal", fontsize=9)
    ax[0].legend(loc="upper right", fontsize=6)

    ax[1].axvspan(PEAK_WINDOW_S[0] * 1000, PEAK_WINDOW_S[1] * 1000, color="grey", alpha=0.15, lw=0)
    ax[1].axhline(0, color="k", lw=0.4)
    ax[1].set_xlabel("Time after stimulus (ms)")
    ax[1].set_ylabel(f"Signal ({abf.adcUnits[0]})")
    ax[1].set_title(f"neg peak {np.mean(neg):.3f} / pos peak {np.mean(pos):.3f} {abf.adcUnits[0]}", fontsize=8)
    ax[1].legend(loc="upper right", fontsize=6)

    fig.suptitle(f"{DATE} / {label}", fontsize=10, y=0.98)
    fig.tight_layout()
    out_dir = PLOTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{file_path.stem}.png", dpi=300)
    plt.close(fig)
    return np.array(neg), np.array(pos)


# ==========================================================================
# SR I/O series (intensities from lab notes, recorded in this order)
# ==========================================================================
sr_io_intensity = {
    "00": 100, "01": 75, "02": 125, "03": 25, "04": 50,
    "05": 150, "06": 175, "07": 200, "08": 225, "09": 250, "10": 275, "11": 300,
}

rows = []
for file_path in sorted(FOLDER.glob(f"{DATE}_slice1_sr_io_*.abf")):
    rec_short = f"{int(file_path.stem[-4:]):02d}"
    current = sr_io_intensity[rec_short]
    neg, pos = plot_file(file_path, "sr_io", f"sr_io rec {rec_short} ({current} uA)")
    rows.append((rec_short, current, neg.mean(), neg.std(), pos.mean(), pos.std()))
    print(f"  rec {rec_short} {current:>3} uA: neg {neg.mean():7.3f} +/- {neg.std():.3f}   "
          f"pos {pos.mean():6.3f} +/- {pos.std():.3f} mV")

rows.sort(key=lambda r: r[1])
x = [r[1] for r in rows]
fig, ax = plt.subplots(figsize=(5, 4))
ax.errorbar(x, [r[2] for r in rows], yerr=[r[3] for r in rows], marker="o", capsize=3, label="negative peak")
ax.errorbar(x, [r[4] for r in rows], yerr=[r[5] for r in rows], marker="s", capsize=3, label="positive peak")
for r in rows:
    ax.annotate(r[0], (r[1], r[4]), textcoords="offset points", xytext=(3, 4), fontsize=6, color="C1")
ax.axhline(0, color="k", lw=0.4)
ax.set_xlabel("Stimulus intensity (uA)")
ax.set_ylabel("Peak amplitude (mV)")
ax.set_title(f"I/O - sr_io - {DATE} ({PEAK_WINDOW_S[0]*1000:.0f}-{PEAK_WINDOW_S[1]*1000:.0f} ms)", fontsize=9)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "sr_io" / "io_curve.png", dpi=300)
plt.close(fig)

# Mean traces of the whole series overlaid, coloured by intensity
fig, ax = plt.subplots(figsize=(6, 4))
cmap = plt.get_cmap("viridis")
for file_path in sorted(FOLDER.glob(f"{DATE}_slice1_sr_io_*.abf"),
                        key=lambda p: sr_io_intensity[f"{int(p.stem[-4:]):02d}"]):
    rec_short = f"{int(file_path.stem[-4:]):02d}"
    current = sr_io_intensity[rec_short]
    abf = pyabf.ABF(str(file_path))
    t, s, stim = mean_trace(abf, abf.sweepList)
    m = (t >= stim + PLOT_WINDOW_S[0]) & (t <= stim + PLOT_WINDOW_S[1])
    ax.plot((t[m] - stim) * 1000, s[m], lw=0.9, color=cmap(current / 300), label=f"{current} uA (rec {rec_short})")
ax.axhline(0, color="k", lw=0.4)
ax.set_xlabel("Time after stimulus (ms)")
ax.set_ylabel("Signal (mV)")
ax.set_title(f"sr_io mean traces - {DATE}", fontsize=9)
ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(PLOTS / "sr_io" / "overlay.png", dpi=300)
plt.close(fig)
print(f"[sr_io] lagret i {PLOTS / 'sr_io'}")

# ==========================================================================
# Single recordings: sr at 100 uA, and stim locations loc2-loc5 at 200 uA
# ==========================================================================
plot_file(FOLDER / f"{DATE}_slice1_sr_100uA_0000.abf", "sr_100uA", "sr 100 uA")

fig, ax = plt.subplots(figsize=(6, 4))
for file_path in sorted(FOLDER.glob(f"{DATE}_slice1_loc*_200uA_*.abf")):
    loc = file_path.stem.split("_")[2]
    neg, pos = plot_file(file_path, "locations", f"{loc} 200 uA")
    print(f"  {loc}: neg {neg.mean():7.3f}   pos {pos.mean():6.3f} mV")
    abf = pyabf.ABF(str(file_path))
    t, s, stim = mean_trace(abf, abf.sweepList)
    m = (t >= stim + PLOT_WINDOW_S[0]) & (t <= stim + PLOT_WINDOW_S[1])
    ax.plot((t[m] - stim) * 1000, s[m], lw=0.9, label=loc)
ax.axhline(0, color="k", lw=0.4)
ax.set_xlabel("Time after stimulus (ms)")
ax.set_ylabel("Signal (mV)")
ax.set_title(f"Stim locations, 200 uA - {DATE}", fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(PLOTS / "locations" / "overlay.png", dpi=300)
plt.close(fig)
print(f"[locations] lagret i {PLOTS / 'locations'}")
