import numpy as np
import matplotlib.pyplot as plt
import pyabf
from pathlib import Path

from fp_analysis import stim_time_s, preprocess, mean_trace, find_slope_window, compute_slope

# ---- settings ----------------------------------------------------
DATE = "22092026"
FOLDER = Path(DATE)
# -----------------------------------------------------------------------


def analyse_series(name, file_pattern, rec_to_intensity, ref_rec, fepsp_search_ms, exclude_recs=()):
    """Run the per-sweep slope analysis + I/O curve for one file series.

    The slope window is fixed: defined once on the mean response of ref_rec (within
    fepsp_search_ms after the stimulus, which must start after the fibre volley) and
    then applied to every sweep and intensity in the series."""
    plots_dir = FOLDER / "plots" / name
    plots_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(FOLDER.glob(file_pattern))
    if not files:
        print(f"[{name}] Fant ingen filer for mønster {file_pattern}")
        return
    print(f"[{name}] Fant {len(files)} filer")

    ref_file = next(p for p in files if f"{int(p.stem[-4:]):02d}" == ref_rec)
    ref_abf = pyabf.ABF(str(ref_file))
    ref_time, ref_signal, ref_stim = mean_trace(ref_abf, ref_abf.sweepList)
    slope_window = find_slope_window(ref_time, ref_signal, ref_stim,
                                     search_s=(fepsp_search_ms[0] / 1000, fepsp_search_ms[1] / 1000))
    if slope_window is None:
        print(f"[{name}] Fant ikke fEPSP-flanke i referansen {ref_file.name} - sjekk ref_rec / fepsp_search_ms")
        return
    print(f"[{name}] Fast slope-vindu fra rec {ref_rec} ({rec_to_intensity[ref_rec]} uA): "
          f"{slope_window[0] * 1000:.2f}-{slope_window[1] * 1000:.2f} ms etter stimulus")

    slope_by_rec = {}
    slope_sd_by_rec = {}

    for file_path in files:
        rec = file_path.stem[-4:]
        rec_short = f"{int(rec):02d}"
        if rec_short not in rec_to_intensity:
            print(f"  {file_path.name}: ukjent rec-nummer, hopper over")
            continue

        try:
            abf = pyabf.ABF(str(file_path))
        except Exception as e:
            print(f"  {file_path.name}: KORRUPT/uleselig ({e}) - hopper over")
            continue

        stim = stim_time_s(abf)
        fig, ax = plt.subplots(2, 1, figsize=(5, 6))
        slopes = []
        raw_no_artifact = []

        for sweep_number in abf.sweepList:
            abf.setSweep(sweepNumber=sweep_number, channel=0)
            time = abf.sweepX
            signal = abf.sweepY

            signal_filtered = preprocess(time, signal, stim, abf.dataRate)

            signal_length = 50 * 1e-3

            mask = (time >= stim - signal_length / 4) & (time <= stim + 0.08)
            time_window = time[mask]
            signal_window = signal_filtered[mask]

            slope = compute_slope(time, signal_filtered, stim, slope_window)
            slopes.append(slope)

            # exclude the artifact itself (stim to stim+3ms) when figuring out sensible y-limits
            no_artifact_mask = (time < stim) | (time > stim + 0.003)
            raw_no_artifact.append(signal[no_artifact_mask])

            ax[0].plot(time, signal, alpha=0.3, lw=0.6, label=f"Sweep {sweep_number}")
            ax[1].plot(time_window, signal_window, alpha=0.7, lw=0.8, label=f"Sweep {sweep_number}")

        raw_no_artifact = np.concatenate(raw_no_artifact)
        pad = 0.1 * (raw_no_artifact.max() - raw_no_artifact.min())
        ax[0].set_ylim(raw_no_artifact.min() - pad, raw_no_artifact.max() + pad)

        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel(f"Signal ({abf.adcUnits[0]})")
        ax[0].set_title(f"{name} {rec_short} - raw signal")
        ax[0].legend(loc="upper right", fontsize=6)

        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel(f"Signal ({abf.adcUnits[0]})")
        if len(time_window):
            ax[1].set_xlim(time_window[0], time_window[-1])
        ax[1].axvspan(stim + slope_window[0], stim + slope_window[1], color="grey", alpha=0.2, lw=0)
        ax[1].legend(loc="upper right", fontsize=6)

        if np.all(np.isnan(slopes)):
            print(f"  {file_path.name}: ingen gyldig slope i noen sweep - ekskludert fra I/O-kurve")
            mean_slope = np.nan
        else:
            mean_slope = np.nanmean(slopes)
            sd_slope = np.nanstd(slopes)
            if rec_short not in exclude_recs:
                slope_by_rec[rec_short] = mean_slope
                slope_sd_by_rec[rec_short] = sd_slope

        current = rec_to_intensity[rec_short]
        ax[1].set_title(f"{current} uA (mean slope: {mean_slope:.4f} mV/s)", fontsize=8)
        fig.suptitle(f"{name} / {DATE} / rec {rec_short}", fontsize=10, y=0.98)
        fig.tight_layout()
        fig.savefig(plots_dir / f"all_sweeps_{rec_short}.png", dpi=300)
        plt.close(fig)

    if not slope_by_rec:
        print(f"[{name}] Ingen gyldige punkter for I/O-kurve")
        return

    sorted_recs = sorted(slope_by_rec, key=lambda r: rec_to_intensity[r])
    x = [rec_to_intensity[r] for r in sorted_recs]
    y = [np.abs(slope_by_rec[r]) for r in sorted_recs]
    yerr = [slope_sd_by_rec[r] for r in sorted_recs]

    plt.figure()
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
    plt.xlabel("Stimulus Intensity (uA)")
    plt.ylabel("|Slope| (mV/s)")
    mean_slope_global = np.nanmean(list(slope_by_rec.values()))
    plt.title(f"I/O Curve - {name} - Mean Slope: {mean_slope_global:.4f} mV/s")
    plt.grid()
    plt.tight_layout()
    plt.savefig(plots_dir / "io_curve.png", dpi=300)
    plt.close()
    print(f"[{name}] I/O-kurve lagret: {plots_dir / 'io_curve.png'}")


def analyse_paired_pulse(name, file_pattern, rec_to_intensity, isi_s):
    """Diagnostic paired-pulse plot: overlay pulse 1 and pulse 2 responses, report P2/P1 amplitude ratio."""
    plots_dir = FOLDER / "plots" / name
    plots_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(FOLDER.glob(file_pattern))
    print(f"[{name}] Fant {len(files)} filer")

    for file_path in files:
        rec = file_path.stem[-4:]
        rec_short = f"{int(rec):02d}"
        current = rec_to_intensity.get(rec_short, "ukjent")

        try:
            abf = pyabf.ABF(str(file_path))
        except Exception as e:
            print(f"  {file_path.name}: KORRUPT/uleselig ({e}) - hopper over")
            continue

        stim1 = stim_time_s(abf)
        stim2 = stim1 + isi_s

        fig, ax = plt.subplots(figsize=(5, 4))
        p1_amps, p2_amps = [], []
        for sweep_number in abf.sweepList:
            abf.setSweep(sweepNumber=sweep_number, channel=0)
            time = abf.sweepX
            signal = abf.sweepY
            signal_filtered = preprocess(time, signal, (stim1, stim2), abf.dataRate)

            mask = (time >= stim1 - 0.005) & (time <= stim2 + 0.03)
            ax.plot(time[mask] * 1000, signal_filtered[mask], lw=0.7, alpha=0.6)

            m1 = (time > stim1 + 0.001) & (time < stim1 + isi_s - 0.002)
            m2 = (time > stim2 + 0.001) & (time < stim2 + 0.025)
            if m1.any():
                p1_amps.append(signal_filtered[m1].min())
            if m2.any():
                p2_amps.append(signal_filtered[m2].min())

        for s in (stim1, stim2):
            ax.axvline(s * 1000, color="C3", lw=0.8, ls="--", alpha=0.6)

        ppr = np.nanmean(p2_amps) / np.nanmean(p1_amps) if p1_amps and p2_amps else np.nan
        ax.set_title(f"{name} {rec_short} ({current} uA) - PPR={ppr:.2f}", fontsize=9)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(f"Signal ({abf.adcUnits[0]})")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(plots_dir / f"pp_{rec_short}.png", dpi=200)
        plt.close(fig)
        print(f"  {file_path.name}: P1={np.nanmean(p1_amps):.3f}  P2={np.nanmean(p2_amps):.3f}  PPR={ppr:.2f}")


# ==========================================================================
# sr_0000-0009: two stim locations in one series.
#   0000-0002: original location, threshold search (25, 10, 5 uA)
#   0003:      original location, 50 uA, no response (dead tissue) -> excluded from IO curve
#   0004-0009: new location after moving electrode, clean IO ramp (50-175 uA)
# ==========================================================================
sr_intensity = {
    "00": 25, "01": 10, "02": 5, "03": 50,
    "04": 50, "05": 75, "06": 100, "07": 125, "08": 150, "09": 175,
}
analyse_series("sr", "22092026_cell1_sr_[0-9][0-9][0-9][0-9].abf", sr_intensity,
               ref_rec="07", fepsp_search_ms=(4.5, 20.0), exclude_recs=("00", "01", "02", "03"))

# ==========================================================================
sp_stimloc3_intensity = {"00": 100, "01": 125, "02": 150, "03": 250}
analyse_series("sp_stimloc3", "22092026_cell1_sp_stimloc3_*.abf", sp_stimloc3_intensity,
               ref_rec="01", fepsp_search_ms=(5.0, 20.0))

# ==========================================================================
sr_loc4_intensity = {
    "00": 25, "01": 10, "02": 20, "03": 50,
    "04": 75, "05": 100, "06": 125, "07": 150,
}
analyse_series("sr_loc4", "22092026_cell1_sr_loc4_*.abf", sr_loc4_intensity,
               ref_rec="05", fepsp_search_ms=(4.0, 15.0))

# ==========================================================================
# Paired pulse - diagnostic only (no facilitation reported in lab notes)
pp50_intensity = {"00": "?", "01": 75, "02": 50, "03": 25, "04": 10}
analyse_paired_pulse("pp_50ms", "22092026_cell1_sr_stimloc3_paired_pulse_50ms_*.abf", pp50_intensity, isi_s=0.050)

pp25_intensity = {"05": 25, "06": 25}
analyse_paired_pulse("pp_25ms", "22092026_cell1_sr_stimloc3_paired_pulse_25ms_*.abf", pp25_intensity, isi_s=0.025)
