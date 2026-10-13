"""
10 -- STFT: choosing the window length.

Cut the record into consecutive windows, take the DFT of each, divide voltage
by current at the excited bins. One spectrum per window, stamped at its
midpoint.

The window length is the only parameter, and it fixes everything: the time
resolution, how much noise is averaged away, and how much the impedance is
smeared by varying within a window.

Run:  python examples/10_stft_window_length.py [--save]
"""
import numpy as np
import matplotlib.pyplot as plt

from deistools.processing.stft_functions import estimate_impedance
from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor

from _plotting import parse_args, finish, show_all, overview, comparison

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# Excitation and circuit (see example 01)
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

R_s, R_p, C, R_T_max = 10.0, 30.0, 1e-3, 70.0
V_gate_start, V_gate_rate = 1.645, (2.455 - 1.645) / T


def simulate(noise_std_V):
    return simulate_r_rc_transistor(
        T, 1.0 / fs, frequencies, amplitudes, phases,
        R_s=R_s, R_p=R_p, C=C,
        V_gate_start=V_gate_start, V_gate_rate=V_gate_rate, R_T_max=R_T_max,
        noise_std_V=noise_std_V, seed=1)


sim = simulate(1e-4)
time = sim["time_s"]
Z_true = sim["Z_true"]


def truth_at(times):
    """Ground truth resampled onto the instants an estimator reports at."""
    return np.array([np.interp(times, time, Z_true[k].real)
                     + 1j * np.interp(times, time, Z_true[k].imag)
                     for k in range(frequencies.size)])


# --------------------------------------------------------------------------
# One call, written out
# --------------------------------------------------------------------------
# 'period' is the window length in SECONDS. It must hold a whole number of
# multisine periods: a rectangular window over a fractional period leaks, and
# leakage in the response looks exactly like time variation.

Z, voltage_avg, current_avg, t = estimate_impedance(
    sim["voltage_V"], sim["current_A"], frequencies,
    Tp,                     # window length = one multisine period
    1.0 / fs,
    window_fun="rectangular",
)

print(f"  one window per period -> Z {Z.shape}, reported at "
      f"t = {t[0]:.0f}, {t[1]:.0f} ... {t[-1]:.0f} s")


# --------------------------------------------------------------------------
# Window length against ground truth
# --------------------------------------------------------------------------
windows = [1, 2, 4, 5, 10]        # in multisine periods; must divide P
results = []

print("\n  window    length   estimates   median error   worst error")
for w in windows:
    Z_w, _, _, t_w = estimate_impedance(
        sim["voltage_V"], sim["current_A"], frequencies,
        w * Tp, 1.0 / fs, window_fun="rectangular")
    truth_w = truth_at(t_w)
    err = np.abs(Z_w - truth_w) / np.abs(truth_w)
    results.append((f"{w} x Tp", Z_w, t_w, truth_w))
    print(f"  {w:>3} x Tp   {w * Tp:>5.0f} s   {Z_w.shape[1]:>9d}   "
          f"{np.median(err):>12.2e}   {np.max(err):>11.2e}")

print("\n  Shorter is better here: each estimate is an average over whatever")
print(f"  the cell did during the window, and R_ct changes by a factor of "
      f"{sim['R_ct_t'][0] / sim['R_ct_t'][-1]:.1f}")
print(f"  across the record. One period ({Tp:.0f} s) is the floor -- the lowest")
print("  tone would not fit in anything shorter.")

fig1 = comparison(frequencies, results, n_instants=3)
finish(fig1, "10_stft_windows", args.save,
       "STFT: window length against ground truth")


# --------------------------------------------------------------------------
# The trade is with noise
# --------------------------------------------------------------------------
# A short window tracks better but averages less, so the ranking above is not
# a general result: it depends on how much noise there is.

noise_levels = [1e-4, 3e-3, 1e-2, 3e-2]
curves = []

print("\n     noise" + "".join(f"{f'{w} x Tp':>12}" for w in windows) + "    best")
for noise in noise_levels:
    sim_n = simulate(noise)
    errors = []
    for w in windows:
        Z_w, _, _, t_w = estimate_impedance(
            sim_n["voltage_V"], sim_n["current_A"], frequencies,
            w * Tp, 1.0 / fs, window_fun="rectangular")
        truth_w = truth_at(t_w)
        errors.append(np.median(np.abs(Z_w - truth_w) / np.abs(truth_w)))
    curves.append(errors)
    best = windows[int(np.argmin(errors))]
    print(f"  {noise:>8.0e}" + "".join(f"{e:>12.2e}" for e in errors)
          + f"   {best} x Tp")

print("\n  At low noise the shortest window wins. Raise the noise and the")
print("  optimum moves out, as averaging starts to pay for the bias it costs.")

fig2, ax2 = plt.subplots(figsize=(7.5, 5))
for errors, noise in zip(curves, noise_levels):
    ax2.loglog(windows, errors, "-o", label=f"noise {noise:.0e} V")
    ax2.plot(windows[int(np.argmin(errors))], min(errors), "*", ms=16,
             color="black", zorder=5)
ax2.set(xlabel="window length / multisine periods",
        ylabel="median relative error", title="optimum marked with a star")
ax2.grid(True, which="both", alpha=0.25)
ax2.legend(fontsize=8)
finish(fig2, "10_stft_noise_trade", args.save,
       "STFT: the window length trade depends on the noise")


# --------------------------------------------------------------------------
# The best setting, in full
# --------------------------------------------------------------------------
Z_best, _, _, t_best = estimate_impedance(
    sim["voltage_V"], sim["current_A"], frequencies,
    Tp, 1.0 / fs, window_fun="rectangular")

fig3 = overview(time, sim["voltage_V"], sim["current_A"], fs, frequencies,
                Z=Z_best, t_Z=t_best, Z_true=truth_at(t_best))
finish(fig3, "10_stft_best", args.save,
       "STFT at one period per window: record, spectra, estimate")

show_all(args.save)
