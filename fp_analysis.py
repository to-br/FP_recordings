"""Shared field-potential analysis helpers used by the acute_slices_*.py scripts.

fEPSP slope is measured in a FIXED time window (relative to the stimulus) that is
defined once on a reference response and then applied unchanged to every sweep and
every stimulus intensity, so the same part of the response is measured throughout.
Population spike amplitude is a separate measure (tangent method), never mixed into
the slope series.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

FILTER_CUTOFF_HZ = 1000
ARTIFACT_BLANK_S = 0.0008              # stimulus artifact is over ~0.5 ms after stim
BASELINE_WINDOW_S = (-0.0055, -0.0005)  # pre-stimulus baseline, relative to stim


def butter_lowpass(signal, sampling_rate, cutoff=FILTER_CUTOFF_HZ, order=4, highpass=None):
    """Zero-phase Butterworth filter. Low-pass at `cutoff`; if `highpass` (Hz) is given,
    band-pass from `highpass` to `cutoff` instead."""
    if highpass is None:
        sos = butter(N=order, Wn=cutoff, btype="lowpass", fs=sampling_rate, output="sos")
    else:
        sos = butter(N=order, Wn=[highpass, cutoff], btype="bandpass", fs=sampling_rate, output="sos")
    return sosfiltfilt(sos, signal, axis=-1)


def stim_time_s(abf):
    """Stimulus pulse time (s) = onset of the shortest step epoch in the command waveform."""
    ep = abf.sweepEpochs
    durations = [p2 - p1 for p1, p2 in zip(ep.p1s, ep.p2s)]
    idx = durations.index(min(durations))
    return ep.p1s[idx] / abf.dataRate


def blank_artifact(time, signal, stim_time, duration=ARTIFACT_BLANK_S):
    """Replace the stimulus artifact with a straight line, so the low-pass filter
    doesn't smear the (huge) artifact out over the first milliseconds of the response."""
    signal = signal.copy()
    blank = (time >= stim_time) & (time <= stim_time + duration)
    signal[blank] = np.interp(time[blank], time[~blank], signal[~blank])
    return signal


def preprocess(time, signal, stim_times, sampling_rate, cutoff=FILTER_CUTOFF_HZ, highpass=None):
    """Blank artifact(s), low-pass (or band-pass, if `highpass` is given) filter, and subtract the pre-stimulus baseline
    (measured before the first stimulus)."""
    stim_times = np.atleast_1d(stim_times)
    for stim in stim_times:
        signal = blank_artifact(time, signal, stim)
    signal = butter_lowpass(signal, sampling_rate, cutoff=cutoff, highpass=highpass)
    base_start, base_end = BASELINE_WINDOW_S
    baseline_mask = (time >= stim_times[0] + base_start) & (time < stim_times[0] + base_end)
    return signal - np.mean(signal[baseline_mask])


def mean_trace(abf, sweeps, cutoff=FILTER_CUTOFF_HZ):
    """Average of the preprocessed sweeps in one file (used as reference response)."""
    stim = stim_time_s(abf)
    traces = []
    for sweep_number in sweeps:
        abf.setSweep(sweepNumber=sweep_number, channel=0)
        traces.append(preprocess(abf.sweepX, abf.sweepY, stim, abf.dataRate, cutoff))
    return abf.sweepX, np.mean(traces, axis=0), stim


def find_slope_window(time, signal, stim_time, search_s, polarity=-1, low_frac=0.2, high_frac=0.8):
    """Define the fixed slope window on a REFERENCE response.

    Within search_s (start, end) after the stimulus, finds the fEPSP peak (negative
    for polarity=-1, positive for +1), the response onset (the point closest to
    baseline before the peak, i.e. after the fibre volley), and the first crossings
    of low_frac and high_frac of the peak amplitude going forward from the onset.
    Returns (t_low, t_high) in seconds RELATIVE TO THE STIMULUS, or None.
    """
    mask = (time >= stim_time + search_s[0]) & (time <= stim_time + search_s[1])
    seg_time = time[mask]
    working = signal[mask] * polarity  # fEPSP is now a positive peak

    peak_idx = np.argmax(working)
    if peak_idx < 1 or working[peak_idx] <= 0:
        return None
    onset_idx = np.argmin(working[:peak_idx + 1])
    rising = working[onset_idx:peak_idx + 1]

    low_candidates = np.where(rising >= low_frac * working[peak_idx])[0]
    high_candidates = np.where(rising >= high_frac * working[peak_idx])[0]
    if len(low_candidates) == 0 or len(high_candidates) == 0:
        return None
    idx_low = onset_idx + low_candidates[0]
    idx_high = onset_idx + high_candidates[0]
    if idx_high - idx_low < 2:
        return None
    return seg_time[idx_low] - stim_time, seg_time[idx_high] - stim_time


def compute_slope(time, signal, stim_time, window):
    """Least-squares slope (signal units per second, original sign) in the fixed
    window (t_low, t_high) relative to the stimulus."""
    fit_mask = (time >= stim_time + window[0]) & (time <= stim_time + window[1])
    if fit_mask.sum() < 2:
        return np.nan
    slope, intercept = np.polyfit(time[fit_mask], signal[fit_mask], 1)
    return slope


def pop_spike_amplitude(time, signal, stim_time, search_s):
    """Population spike amplitude by the tangent method: the vertical distance from
    the spike trough to the line joining the positive peaks on either side of it.
    Returns (amplitude, (t_pre, t_trough, t_post)) with times absolute, or (nan, None)
    if there is no trough with a positive peak on both sides inside search_s."""
    mask = (time >= stim_time + search_s[0]) & (time <= stim_time + search_s[1])
    seg_time = time[mask]
    seg = signal[mask]

    trough = np.argmin(seg)
    if trough == 0 or trough == len(seg) - 1:
        return np.nan, None
    pre = np.argmax(seg[:trough])
    post = trough + np.argmax(seg[trough:])
    # a "peak" on the window edge is just the signal still rising/falling, not a real peak
    if pre == 0 or post == len(seg) - 1:
        return np.nan, None

    tangent_at_trough = np.interp(seg_time[trough], [seg_time[pre], seg_time[post]], [seg[pre], seg[post]])
    amplitude = tangent_at_trough - seg[trough]
    return amplitude, (seg_time[pre], seg_time[trough], seg_time[post])
