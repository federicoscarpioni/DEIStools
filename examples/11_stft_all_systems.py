"""
11 -- STFT on all four systems.

The same estimator, at its best window for each system, against each system's
own ground truth. The floor it reaches is set by something different in every
case.

Run:  python examples/11_stft_all_systems.py [--save]
"""
import numpy as np
import matplotlib.pyplot as plt

from deistools.processing.stft_functions import estimate_impedance
from deistools.simulation.models.r_rc import simulate_r_rc
from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor
from deistools.simulation.models.legendre import simulate_legendre
from deistools.simulation.models.block_oriented import (
    hallemans_2021_system, simulate_block_oriented, best_linear_approximation,
    frozen_impedance)

from _plotting import parse_args, finish, show_all, overview

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# The excitation, shared by all four systems
# --------------------------------------------------------------------------
Tp = 20.0
fs = 1000.0

harmonics = np.array([1, 3, 5, 9, 13, 17, 27, 39, 59, 89, 133, 201, 301])
frequencies = harmonics / Tp
amplitudes = np.full(frequencies.size, 0.01)
phases = np.random.default_rng(3).uniform(0, 2 * np.pi, frequencies.size)

R_s, R_p, C, R_T_max = 10.0, 30.0, 1e-3, 70.0


# --------------------------------------------------------------------------
# System 1: stationary. Six periods is plenty -- nothing varies.
# --------------------------------------------------------------------------
P1 = 6
N1 = int(fs * Tp) * P1
sim1 = simulate_r_rc(N1 / fs, 1.0 / fs, frequencies, amplitudes, phases,
                     R_s=R_s, R_p=R_p, C=C, noise_std_V=1e-4, seed=1)
Z_true1 = np.repeat(sim1["Z_true"], N1, axis=1)      # constant in time


# --------------------------------------------------------------------------
# System 2: transistor-gated, 20 periods
# --------------------------------------------------------------------------
P2 = 20
N2 = int(fs * Tp) * P2
T2 = N2 / fs
sim2 = simulate_r_rc_transistor(
    T2, 1.0 / fs, frequencies, amplitudes, phases,
    R_s=R_s, R_p=R_p, C=C,
    V_gate_start=1.645, V_gate_rate=(2.455 - 1.645) / T2, R_T_max=R_T_max,
    noise_std_V=1e-4, seed=1)
Z_true2 = sim2["Z_true"]


# --------------------------------------------------------------------------
# System 3: prescribed Legendre expansion, 20 periods
# --------------------------------------------------------------------------
order = 3
a = np.zeros((frequencies.size, order + 1), dtype=complex)
a[:, 0] = R_s + R_p / (1 + 1j * 2 * np.pi * frequencies * R_p * C)
for p in range(1, order + 1):
    a[:, p] = (0.35 / p) * a[:, 0]

sim3 = simulate_legendre(T2, 1.0 / fs, frequencies, amplitudes, phases, a,
                         drift_coefficients=None, noise_std_V=1e-4, seed=1)
Z_true3 = sim3["Z_true"]


# --------------------------------------------------------------------------
# System 4: block-oriented nonlinear, 20 periods
# --------------------------------------------------------------------------
branches = hallemans_2021_system()
sim4 = simulate_block_oriented(T2, 1.0 / fs, frequencies, amplitudes, phases,
                               branches, trend_coefficients=None,
                               noise_std_V=1e-4, seed=1)
G, _ = best_linear_approximation(branches, frequencies, amplitudes, 1.0 / fs,
                                 Tp, n_realisations=200, seed=7)
Z_true4 = frozen_impedance(G, N2)


systems = [
    ("stationary", sim1, Z_true1, P1),
    ("transistor", sim2, Z_true2, P2),
    ("legendre", sim3, Z_true3, P2),
    ("block_oriented", sim4, Z_true4, P2),
]


def truth_at(sim, Z_true, times):
    """Ground truth resampled onto the instants the estimator reports at."""
    return np.array([np.interp(times, sim["time_s"], Z_true[k].real)
                     + 1j * np.interp(times, sim["time_s"], Z_true[k].imag)
                     for k in range(frequencies.size)])


# --------------------------------------------------------------------------
# Pick the window for each system
# --------------------------------------------------------------------------
# The window must divide the record into whole multisine periods, so the
# candidates are the divisors of P.

print("          system    P          candidates   best window   median error")
chosen = {}
for name, sim, Z_true, P in systems:
    candidates = [w for w in range(1, P + 1) if P % w == 0]
    errors = []
    for w in candidates:
        Z, _, _, t = estimate_impedance(sim["voltage_V"], sim["current_A"],
                                        frequencies, w * Tp, 1.0 / fs)
        truth = truth_at(sim, Z_true, t)
        errors.append(np.median(np.abs(Z - truth) / np.abs(truth)))
    best = candidates[int(np.argmin(errors))]
    chosen[name] = best
    print(f"  {name:>14}  {P:>3}  {str(candidates):>18}   {best:>6} x Tp   "
          f"{min(errors):>12.2e}")

print("\n  One period wins on the three time-varying systems: the excitation is")
print("  strong, so there is little noise to average and the bias from varying")
print("  within a window dominates.")
print("  The stationary system is the exception -- with no variation there is")
print("  no bias to incur, so averaging the whole record is pure gain.")


# --------------------------------------------------------------------------
# The results
# --------------------------------------------------------------------------
estimates = {}
errors_by_system = {}
for name, sim, Z_true, P in systems:
    w = chosen[name]
    Z, _, _, t = estimate_impedance(sim["voltage_V"], sim["current_A"],
                                    frequencies, w * Tp, 1.0 / fs)
    truth = truth_at(sim, Z_true, t)
    estimates[name] = (Z, t, truth)
    errors_by_system[name] = np.abs(Z - truth) / np.abs(truth)

print()
print("          system   window   estimates       median        worst")
for name, sim, Z_true, P in systems:
    err = errors_by_system[name]
    print(f"  {name:>14}   {chosen[name]:>2} x Tp   {err.shape[1]:>9}   "
          f"{np.median(err):>10.2e}   {np.max(err):>10.2e}")

print()
for name, sim, Z_true, P in systems:
    Z, t, truth = estimates[name]
    fig = overview(sim["time_s"], sim["voltage_V"], sim["current_A"], fs,
                   frequencies, Z=Z, t_Z=t, Z_true=truth)
    finish(fig, f"11_stft_{name}", args.save,
           f"STFT on {name} (window {chosen[name]} x Tp)")


# --------------------------------------------------------------------------
# Where the error sits in frequency
# --------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8.5, 5.5))
for name, sim, Z_true, P in systems:
    err = errors_by_system[name]
    ax2.loglog(frequencies, np.median(err, axis=1), "-o", ms=4, label=name)
ax2.set(xlabel="frequency / Hz", ylabel="median relative error",
        title="STFT: error against frequency")
ax2.grid(True, which="both", alpha=0.25)
ax2.legend(fontsize=8)
finish(fig2, "11_stft_error_vs_frequency", args.save)

print("\n  worst tone per system:")
for name, sim, Z_true, P in systems:
    med = np.median(errors_by_system[name], axis=1)
    print(f"  {name:>14}   worst at {frequencies[int(np.argmax(med))]:>6.2f} Hz,"
          f"   best at {frequencies[int(np.argmin(med))]:>6.2f} Hz")

print("\n  Two patterns. On the strongly varying systems the worst tone is the")
print("  lowest: a window holds one or two cycles there, with no averaging.")
print("  On the others the worst tone is the highest, where |Z| is smallest")
print("  and the error simply follows the signal-to-noise ratio.")

show_all(args.save)
