"""
15 -- BLTVA on all four systems.

Np and the band width chosen together for each system, within whatever the
record allows, then the result on each.

Run:  python examples/15_bltva_all_systems.py [--save]
"""
import numpy as np
import matplotlib.pyplot as plt

from deistools.processing.bltva import (
    bltva, legendre_basis, select_na, check_identifiability)
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
    ("stationary", sim1, Z_true1, P1, N1),
    ("transistor", sim2, sim2["Z_true"], P2, N2),
    ("legendre", sim3, sim3["Z_true"], P2, N2),
    ("block_oriented", sim4, frozen_impedance(G, N2), P2, N2),
]


def run_bltva(sim, P, N, Np, Nq, band_halfwidth):
    """Fit, and return Z(f,t) at 200 instants plus the full result object."""
    res = bltva(sim["voltage_V"], sim["current_A"], fs, P, harmonics,
                Np=Np, Nq=Nq, Na=None, harmonics="excited",
                band_halfwidth=band_halfwidth)
    idx = np.linspace(0, N - 1, 200).astype(int)
    return res.H_p @ legendre_basis(N, Np)[idx].T, sim["time_s"][idx], res


def truth_at(sim, Z_true, times):
    """Ground truth resampled onto the instants the estimator reports at."""
    return np.array([np.interp(times, sim["time_s"], Z_true[k].real)
                     + 1j * np.interp(times, sim["time_s"], Z_true[k].imag)
                     for k in range(frequencies.size)])


# --------------------------------------------------------------------------
# What each record can afford
# --------------------------------------------------------------------------
# Two separate limits. eq. (49) bounds Nq + 3*Np + Na + 1 below 2P - 1, and
# check_identifiability enforces it. But Algorithm 1 also estimates the noise
# from a sliding band that needs more equations than regressors, and on a
# short record that can bite first.

print("          system    P   eq.(49) allows Np   actually runs Np")
for name, sim, Z_true, P, N in systems:
    allowed, runs = [], []
    for Np in range(1, 15):
        for Nq in (0, 1, 2):
            try:
                check_identifiability(P, Np, Nq, select_na(P, Np))
            except ValueError:
                continue
            allowed.append(Np)
            try:
                run_bltva(sim, P, N, Np, Nq, 50)
                runs.append(Np)
            except ValueError:
                pass
            break
    print(f"  {name:>14}  {P:>3}   {max(allowed):>17}   {max(runs):>16}")

print("\n  They disagree on the six-period record. eq. (49) passes Np = 2 with")
print("  Nq = 2 there, and the noise band then refuses it with exactly as many")
print("  regressors as equations. eq. (49) is necessary, not sufficient -- if")
print("  BLTVA refuses a setting the inequality admits, this is usually why.")
print("  Either way the budget is fixed by P, before any processing happens.")


# --------------------------------------------------------------------------
# Choose Np and the band width together, as example 14 showed one must
# --------------------------------------------------------------------------
np_candidates = [1, 2, 3, 4, 6, 8]
band_candidates = [20, 50, 100, 200, 400]

chosen = {}
print("\n          system   best Np   Nq   band   Na       median        worst")
for name, sim, Z_true, P, N in systems:
    best = None
    for Np in np_candidates:
        for band in band_candidates:
            for Nq in (2, 1, 0):          # the largest Nq that will run
                try:
                    Z, t, _ = run_bltva(sim, P, N, Np, Nq, band)
                except ValueError:
                    continue
                truth = truth_at(sim, Z_true, t)
                worst = np.max(np.abs(Z - truth) / np.abs(truth))
                median = np.median(np.abs(Z - truth) / np.abs(truth))
                if best is None or worst < best[0]:
                    best = (worst, median, Np, Nq, band)
                break
    worst, median, Np, Nq, band = best
    chosen[name] = (Np, Nq, band)
    print(f"  {name:>14}   {Np:>7}   {Nq:>2}   {band:>4}   {select_na(P, Np):>2}   "
          f"{median:>10.2e}   {worst:>10.2e}")

print("\n  The time-varying systems want a high Np. The stationary one cannot")
print("  have it and does not need it -- nothing varies, so Np = 1 is already")
print("  more model than the system contains.")


# --------------------------------------------------------------------------
# The results, and the uncertainty
# --------------------------------------------------------------------------
# Alone among the three estimators, BLTVA reports an uncertainty. It covers
# NOISE ONLY, so comparing it with the error actually made says whether the
# measurement or the model is the limiting factor.

print("\n          system       median   predicted noise   ratio")
estimates = {}
for name, sim, Z_true, P, N in systems:
    Np, Nq, band = chosen[name]
    Z, t, res = run_bltva(sim, P, N, Np, Nq, band)
    truth = truth_at(sim, Z_true, t)
    err = np.abs(Z - truth) / np.abs(truth)
    predicted = np.median(res.std_H_p[:, 0] / np.abs(res.H_p[:, 0]))
    estimates[name] = (Z, t, truth, err, res)
    print(f"  {name:>14}   {np.median(err):>10.2e}   {predicted:>15.2e}   "
          f"{np.median(err) / predicted:>5.1f}")

print("\n  A ratio near one means the error is pure noise. Three of the four are")
print("  within a factor of two, because those systems have no model error to")
print("  have: nothing varies, or the Legendre model is exactly right, or the")
print("  excess is real distortion.")
print("  The transistor is the outlier, and the uncertainty is working rather")
print("  than failing: its Z(f,t) is a rational function of time, not a")
print("  polynomial, so a Legendre series approximates it and leaves a residue")
print("  far above the noise. If your error is far larger than the reported")
print("  sigma, more averaging will not help -- the model is the limit.")

print()
for name, sim, Z_true, P, N in systems:
    Z, t, truth, err, res = estimates[name]
    Np, Nq, band = chosen[name]
    fig = overview(sim["time_s"], sim["voltage_V"], sim["current_A"], fs,
                   frequencies, Z=Z, t_Z=t, Z_true=truth)
    finish(fig, f"15_bltva_{name}", args.save,
           f"BLTVA on {name} (Np = {Np}, Nq = {Nq}, band = {band})")

fig2, ax2 = plt.subplots(figsize=(8.5, 5.5))
for name, sim, Z_true, P, N in systems:
    Z, t, truth, err, res = estimates[name]
    line, = ax2.loglog(frequencies, np.median(err, axis=1), "-o", ms=4,
                       label=f"{name} (actual)")
    ax2.loglog(frequencies, res.std_H_p[:, 0] / np.abs(res.H_p[:, 0]), "--",
               lw=1, alpha=0.7, color=line.get_color(),
               label=f"{name} (predicted)")
ax2.set(xlabel="frequency / Hz", ylabel="relative error",
        title="BLTVA: predicted noise against actual error")
ax2.grid(True, which="both", alpha=0.25)
ax2.legend(fontsize=7, ncol=2)
finish(fig2, "15_bltva_uncertainty", args.save)

show_all(args.save)
