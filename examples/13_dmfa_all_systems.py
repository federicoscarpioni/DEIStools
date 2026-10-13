"""
13 -- DMFA on all four systems.

A small bandwidth/sharpness sweep per system, picking the best by worst-case
error, then the result on each.

Run:  python examples/13_dmfa_all_systems.py [--save]
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, fftfreq

from deistools.processing.dmfa_functions import (
    extract_impedance, max_elaboration_points, elaboration_frequency_axis,
    elaboration_time_axis)
from deistools.processing.passband_filters import create_fermi_dirac_fun
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
f_min = frequencies.min()

R_s, R_p, C, R_T_max = 10.0, 30.0, 1e-3, 70.0


# --------------------------------------------------------------------------
# The four systems (see examples 00-03)
# --------------------------------------------------------------------------
P1 = 6
N1 = int(fs * Tp) * P1
sim1 = simulate_r_rc(N1 / fs, 1.0 / fs, frequencies, amplitudes, phases,
                     R_s=R_s, R_p=R_p, C=C, noise_std_V=1e-4, seed=1)
Z_true1 = np.repeat(sim1["Z_true"], N1, axis=1)

P2 = 20
N2 = int(fs * Tp) * P2
T2 = N2 / fs
sim2 = simulate_r_rc_transistor(
    T2, 1.0 / fs, frequencies, amplitudes, phases,
    R_s=R_s, R_p=R_p, C=C,
    V_gate_start=1.645, V_gate_rate=(2.455 - 1.645) / T2, R_T_max=R_T_max,
    noise_std_V=1e-4, seed=1)

order = 3
a = np.zeros((frequencies.size, order + 1), dtype=complex)
a[:, 0] = R_s + R_p / (1 + 1j * 2 * np.pi * frequencies * R_p * C)
for p in range(1, order + 1):
    a[:, p] = (0.35 / p) * a[:, 0]
sim3 = simulate_legendre(T2, 1.0 / fs, frequencies, amplitudes, phases, a,
                         drift_coefficients=None, noise_std_V=1e-4, seed=1)

branches = hallemans_2021_system()
sim4 = simulate_block_oriented(T2, 1.0 / fs, frequencies, amplitudes, phases,
                               branches, trend_coefficients=None,
                               noise_std_V=1e-4, seed=1)
G, _ = best_linear_approximation(branches, frequencies, amplitudes, 1.0 / fs,
                                 Tp, n_realisations=200, seed=7)

systems = [
    ("stationary", sim1, Z_true1, N1),
    ("transistor", sim2, sim2["Z_true"], N2),
    ("legendre", sim3, sim3["Z_true"], N2),
    ("block_oriented", sim4, frozen_impedance(G, N2), N2),
]


# --------------------------------------------------------------------------
# DMFA, as in example 12
# --------------------------------------------------------------------------
def run_dmfa(sim, N, bw_factor, sharpness):
    """DMFA at a given bandwidth (in units of f_min) and filter sharpness."""
    T = N / fs
    ft_v = fftshift(fft(sim["voltage_V"])) / N
    ft_i = fftshift(fft(sim["current_A"])) / N
    axis = fftshift(fftfreq(N, 1.0 / fs))
    index_f0 = int(np.argmin(np.abs(axis)))
    indexes = (np.round(frequencies * T).astype(int) + index_f0).tolist()

    bw = bw_factor * f_min
    npts = int(round(16 * bw * T))   # wide enough for the filter to decay
    npts = min(npts - npts % 2, max_elaboration_points(indexes, N))
    filt = create_fermi_dirac_fun(elaboration_frequency_axis(npts, T), 0, bw,
                                  sharpness)
    # report at 0.5/bw, where neighbouring errors are uncorrelated
    keep = max(1, int(round(npts / (2 * bw * T))))
    Z = extract_impedance(ft_v, ft_i, indexes, filt, npts)
    return Z[:, ::keep], elaboration_time_axis(npts, T)[::keep]


def truth_at(sim, Z_true, times):
    """Ground truth resampled onto the instants the estimator reports at."""
    return np.array([np.interp(times, sim["time_s"], Z_true[k].real)
                     + 1j * np.interp(times, sim["time_s"], Z_true[k].imag)
                     for k in range(frequencies.size)])


def score(sim, Z_true, Z_est, t_est):
    """Median and worst error with the ringing record edges discarded."""
    truth = truth_at(sim, Z_true, t_est)
    err = np.abs(Z_est - truth) / np.abs(truth)
    k = max(1, err.shape[1] // 10)
    core = err[:, k:err.shape[1] - k]
    return np.median(core), np.max(core), np.max(err), err


# --------------------------------------------------------------------------
# Choose bandwidth and sharpness per system
# --------------------------------------------------------------------------
bandwidths = [0.5, 1.0, 2.0, 4.0]
sharpnesses = [2, 4, 8, 16]

chosen = {}
print("          system   bandwidth    n       median        worst")
for name, sim, Z_true, N in systems:
    best = None
    for bw_factor in bandwidths:
        for sharpness in sharpnesses:
            Z, t = run_dmfa(sim, N, bw_factor, sharpness)
            median, worst, _, _ = score(sim, Z_true, Z, t)
            if best is None or worst < best[1]:
                best = (median, worst, bw_factor, sharpness)
    median, worst, bw_factor, sharpness = best
    chosen[name] = (bw_factor, sharpness)
    print(f"  {name:>14}   {bw_factor:>5.1f} f_min   {sharpness:>2}   "
          f"{median:>10.2e}   {worst:>10.2e}")

print("\n  The narrowest band wins everywhere: on this grid 2 x f_min already")
print("  reaches the neighbouring tone, so there is no room to widen it. The")
print("  excitation, not the method, is what caps the bandwidth -- and with it")
print("  the time resolution.")


# --------------------------------------------------------------------------
# The results
# --------------------------------------------------------------------------
estimates, errors_by_system = {}, {}
for name, sim, Z_true, N in systems:
    bw_factor, sharpness = chosen[name]
    Z, t = run_dmfa(sim, N, bw_factor, sharpness)
    median, worst_core, worst_all, err = score(sim, Z_true, Z, t)
    estimates[name] = (Z, t, truth_at(sim, Z_true, t))
    errors_by_system[name] = err

print("\n          system   instants   dt / s       median   worst (core)"
      "   worst (all)")
for name, sim, Z_true, N in systems:
    Z, t, _ = estimates[name]
    median, worst_core, worst_all, _ = score(sim, Z_true, Z, t)
    print(f"  {name:>14}   {Z.shape[1]:>8}   {t[1] - t[0]:>6.2f}   "
          f"{median:>10.2e}   {worst_core:>12.2e}   {worst_all:>11.2e}")

print("\n  The last two columns differ only by the record edges. Discarding the")
print("  first and last tenth is not optional -- including them multiplies the")
print("  worst case by several hundred on the time-varying systems.")
print("  The stationary system shows no difference at all: a constant")
print("  impedance occupies one bin, so truncating the band removes nothing")
print("  and there is nothing to ring.")

print()
for name, sim, Z_true, N in systems:
    Z, t, truth = estimates[name]
    bw_factor, sharpness = chosen[name]
    fig = overview(sim["time_s"], sim["voltage_V"], sim["current_A"], fs,
                   frequencies, Z=Z, t_Z=t, Z_true=truth)
    finish(fig, f"13_dmfa_{name}", args.save,
           f"DMFA on {name} (bw = {bw_factor} f_min, n = {sharpness})")


# --------------------------------------------------------------------------
# Where the error sits, in frequency and in time
# --------------------------------------------------------------------------
fig2, axs = plt.subplots(1, 2, figsize=(14, 5.2))
for name, sim, Z_true, N in systems:
    Z, t, _ = estimates[name]
    err = errors_by_system[name]
    axs[0].loglog(frequencies, np.median(err, axis=1), "-o", ms=4, label=name)
    axs[1].semilogy(t / t[-1], np.median(err, axis=0), lw=1.0, label=name)

axs[0].set(xlabel="frequency / Hz", ylabel="median relative error",
           title="against frequency")
axs[0].grid(True, which="both", alpha=0.25)
axs[0].legend(fontsize=8)

axs[1].axvspan(0, 0.1, color="C3", alpha=0.15)
axs[1].axvspan(0.9, 1.0, color="C3", alpha=0.15, label="discarded edges")
axs[1].set(xlabel="position in the record", ylabel="median relative error",
           title="against time")
axs[1].grid(True, which="both", alpha=0.25)
axs[1].legend(fontsize=8)

finish(fig2, "13_dmfa_error_structure", args.save,
       "DMFA: the edges are where it fails")

show_all(args.save)
