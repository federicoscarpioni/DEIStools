"""
08 -- A first lock-in estimation.

The classical instrument, run offline. Each tone is multiplied by a cosine
and a sine at its own frequency, and the products are low-pass filtered; what
survives is that tone's slowly varying phasor, and V/I is Z(f,t).

Same transistor measurement as examples 04-07. The interesting part is not
the demodulation, which is textbook, but the choice of integrator -- and the
fact that a one-period boxcar lock-in turns out to *be* the STFT, reported at
every sample instead of once per window.

Run:  python examples/08_lock_in_first_run.py [--save]
"""
import numpy as np
import matplotlib.pyplot as plt

from deistools.processing.lock_in_amp import (
    integrator_length, integrator_kernel, tone_rejection, valid_mask,
    lia_eis, lia_impedance_ls)
from deistools.processing.stft_functions import estimate_impedance
from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor

from _plotting import parse_args, finish, show_all, overview

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# The measurement to analyse (see example 01)
# --------------------------------------------------------------------------
Tp = 20.0
fs = 1000.0
P = 20

harmonics = np.array([1, 3, 5, 9, 13, 17, 27, 39, 59, 89, 133, 201, 301])
frequencies = harmonics / Tp
amplitudes = np.full(frequencies.size, 0.01)
phases = np.random.default_rng(3).uniform(0, 2 * np.pi, frequencies.size)

N = int(fs * Tp) * P
T = N / fs

sim = simulate_r_rc_transistor(
    T, 1.0 / fs, frequencies, amplitudes, phases,
    R_s=10.0, R_p=30.0, C=1e-3,
    V_gate_start=1.645, V_gate_rate=(2.455 - 1.645) / T, R_T_max=70.0,
    noise_std_V=1e-4, seed=1)

time = sim["time_s"]
voltage = sim["voltage_V"]
current = sim["current_A"]
Z_true = sim["Z_true"]


# --------------------------------------------------------------------------
# Designing the integrator -- this is the whole method
# --------------------------------------------------------------------------
# After mixing at f_k, every *other* tone lands at the difference frequency
# |f_j - f_k|. All the tones here are harmonics of 1/Tp, so every difference
# is a multiple of 1/Tp -- and a boxcar average over a whole number of Tp has
# a spectral null at every one of them.
#
# The window must span whole periods of the BASE frequency, not of the tone
# being demodulated. integrator_length enforces that and refuses anything
# else, because one extra sample detunes the nulls completely.

base_frequency = 1.0 / Tp
samples_window = integrator_length(1.0 / fs, base_frequency, n_periods=1)

print(f"  integrator {samples_window} samples = "
      f"{samples_window / fs:.0f} s = 1 base period")

print("\n  worst rejection of the other 12 tones:")
for n_periods, window in [(1, "boxcar"), (2, "boxcar"),
                          (2, "hann"), (4, "hann")]:
    length = integrator_length(1.0 / fs, base_frequency, n_periods, window)
    kernel = integrator_kernel(length, window)
    # row k holds the rejection of every tone after mixing at f_k, and its
    # k-th entry is that tone itself -- 0 dB by construction. Drop it.
    worst = min(np.delete(tone_rejection(kernel, 1.0 / fs, frequencies, f),
                          k).min()
                for k, f in enumerate(frequencies))
    print(f"    {window:>6}, {n_periods} base period(s)   {worst:>6.0f} dB")

print("\n  The boxcar nulls are exact, so its rejection is limited only by")
print("  arithmetic. Hann rolls off faster far from the tone but has a")
print("  mainlobe twice as wide, which is why one period of it is refused.")


# --------------------------------------------------------------------------
# The estimation
# --------------------------------------------------------------------------
# One pair of FFT convolutions per frequency. The result exists at every
# sample; decimate=200 reports every 0.2 s, which is already far finer than
# the integrator can resolve.

Z, t = lia_eis(time, voltage, current, 1.0 / fs, frequencies,
               samples_window, decimate=200)

print(f"\n  lia_eis -> Z {Z.shape}, reported every "
      f"{t[1] - t[0]:.1f} s over {t[-1]:.0f} s")
print(f"  but the time resolution is the integrator, {samples_window / fs:.0f} s.")
print("  Reporting more densely than that does not resolve more; neighbouring")
print("  instants share most of their window and most of their error.")


# --------------------------------------------------------------------------
# The record edges
# --------------------------------------------------------------------------
# Within half an integrator of either end the window runs past the data, and
# the implementation reflects the signal there. valid_mask marks those.

mask_full = valid_mask(time.size, samples_window)
mask = mask_full[::200][:t.size]
print(f"\n  valid_mask drops {100 * (1 - mask.mean()):.0f}% of the record, "
      f"{samples_window / 2 / fs:.0f} s at each end")


# --------------------------------------------------------------------------
# How good is it?
# --------------------------------------------------------------------------
truth = np.array([np.interp(t, time, Z_true[k].real)
                  + 1j * np.interp(t, time, Z_true[k].imag)
                  for k in range(frequencies.size)])
error = np.abs(Z - truth) / np.abs(truth)

print(f"\n  relative error against the known truth:")
print(f"    all instants       median {np.median(error):.2e}, "
      f"worst {np.max(error):.2e}")
print(f"    valid instants     median {np.median(error[:, mask]):.2e}, "
      f"worst {np.max(error[:, mask]):.2e}")


# --------------------------------------------------------------------------
# It is the STFT, evaluated everywhere
# --------------------------------------------------------------------------
# A boxcar over exactly one base period and a rectangular STFT over the same
# window are the same estimator. STFT reports one value per window; the
# lock-in reports at every sample, so it agrees at the window centres.

Z_stft, _, _, t_stft = estimate_impedance(
    voltage, current, frequencies, Tp, 1.0 / fs, window_fun="rectangular")

centres = np.searchsorted(t, t_stft)
centres = centres[centres < t.size]
agreement = np.abs(Z[:, centres] - Z_stft[:, :centres.size]) \
    / np.abs(Z_stft[:, :centres.size])

truth_stft = np.array([np.interp(t_stft, time, Z_true[k].real)
                       + 1j * np.interp(t_stft, time, Z_true[k].imag)
                       for k in range(frequencies.size)])
error_stft = np.abs(Z_stft - truth_stft) / np.abs(truth_stft)

print(f"\n  against STFT at the {centres.size} window centres:")
print(f"    max relative difference {np.max(agreement):.1e}")
print(f"    STFT median error over its own instants  {np.median(error_stft):.2e}")
print(f"    lock-in median error over all instants   {np.median(error):.2e}")
print("\n  The lock-in is worse on the median, and that is not a defect: it")
print("  also reports *between* the window centres, where more of the")
print("  variation has happened inside the window. STFT is this estimator")
print("  decimated to the points where it looks best.")


# --------------------------------------------------------------------------
# Local least squares -- removing the within-window bias
# --------------------------------------------------------------------------
# Instead of averaging over the window, fit a cosine, a sine, an offset and a
# local slope in it. The slope is the point: it follows the drift inside the
# window rather than smearing it.

stride = 4000                       # one instant every 4 s; one LS solve each
Z_ls = np.vstack([
    lia_impedance_ls(time, voltage, current, 1.0 / fs, f, samples_window,
                     stride=stride)[0]
    for f in frequencies])
t_ls = time[::stride]

truth_ls = np.array([np.interp(t_ls, time, Z_true[k].real)
                     + 1j * np.interp(t_ls, time, Z_true[k].imag)
                     for k in range(frequencies.size)])
error_ls = np.abs(Z_ls - truth_ls) / np.abs(truth_ls)

mask_ls = valid_mask(time.size, samples_window)[::stride][:t_ls.size]

print(f"\n  demodulator comparison, valid instants only:")
print(f"    mixing, 1-period boxcar        {np.median(error[:, mask]):.2e}")
print(f"    local least squares, same window  {np.median(error_ls[:, mask_ls]):.2e}")
print("\n  Same data, same window. The difference is entirely the local slope")
print("  term, which is the within-window averaging bias that limits plain")
print("  mixing on a drifting cell. It costs one least-squares solve per")
print("  reported instant, so it runs on a decimated grid.")


# --------------------------------------------------------------------------
# Figure 1: the standard view -- signals, spectra, estimate
# --------------------------------------------------------------------------
fig1 = overview(time, voltage, current, fs, frequencies, Z=Z[:, mask],
                t_Z=t[mask], Z_true=truth[:, mask])
finish(fig1, "08_lock_in_overview", args.save,
       f"lock-in with a {samples_window / fs:.0f} s boxcar integrator")


# --------------------------------------------------------------------------
# Figure 2: the integrator, the tracking, and the two demodulators
# --------------------------------------------------------------------------
fig2, axs = plt.subplots(1, 3, figsize=(15, 4.6))

offsets = np.linspace(0, 10 * base_frequency, 2000)
for n_periods, window, colour in [(1, "boxcar", "C0"), (2, "boxcar", "C1"),
                                  (2, "hann", "C2")]:
    length = integrator_length(1.0 / fs, base_frequency, n_periods, window)
    kernel = integrator_kernel(length, window)
    n = np.arange(kernel.size)
    response = np.array([np.abs(np.sum(kernel * np.exp(-2j * np.pi * df * n / fs)))
                         for df in offsets])
    axs[0].semilogy(offsets / base_frequency, np.maximum(response, 1e-8),
                    color=colour, lw=1.2, label=f"{window}, {n_periods}")
for h in range(1, 11):
    axs[0].axvline(h, color="C3", alpha=0.25, lw=0.8)
axs[0].set(xlabel="offset / base frequency", ylabel="response",
           title="the nulls land on the other tones", ylim=(1e-8, 2))
axs[0].legend(fontsize=8)

for i, colour in zip((0, 6, 12), ("C0", "C1", "C2")):
    axs[1].plot(time, np.abs(Z_true[i]), color=colour, lw=2, alpha=0.35)
    axs[1].plot(t[mask], np.abs(Z[i][mask]), "-", lw=1.2, color=colour,
                label=f"{frequencies[i]:.2f} Hz")
axs[1].set(xlabel="time / s", ylabel=r"$|Z|$ / $\Omega$",
           title="continuous tracking (pale line: truth)")
axs[1].legend(fontsize=8)

axs[2].loglog(frequencies, np.median(error[:, mask], axis=1), "-o", ms=4,
              color="C0", label="mixing")
axs[2].loglog(frequencies, np.median(error_ls[:, mask_ls], axis=1), "-s", ms=4,
              color="C2", label="local least squares")
axs[2].loglog(frequencies, np.median(error_stft, axis=1), "-^", ms=4,
              color="C3", alpha=0.6, label="STFT")
axs[2].set(xlabel="frequency / Hz", ylabel="median relative error",
           title="error against frequency")
axs[2].grid(True, which="both", alpha=0.25)
axs[2].legend(fontsize=8)

finish(fig2, "08_lock_in_result", args.save,
       "the integrator, the tracking, and the two demodulators")

print("\n  The lock-in is the familiar instrument and the one that gives a")
print("  continuous Z(f,t). Its resolution is the integrator length, the same")
print("  floor STFT has. What it adds over STFT is the local-least-squares")
print("  demodulator, which is the only way here to remove the within-window")
print("  bias without assuming a global model the way BLTVA does.")

show_all(args.save)
