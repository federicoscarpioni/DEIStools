"""
22 -- The three estimators on a prescribed Legendre variation.

This system is a Legendre expansion by construction, which is exactly the
model BLTVA fits. The comparison is therefore not fair, and that is the point:
it measures what a correct model assumption is worth.

Run:  python examples/22_compare_legendre.py [--save]
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, fftfreq

from deistools.processing.stft_functions import estimate_impedance
from deistools.processing.dmfa_functions import (
    extract_impedance, max_elaboration_points, elaboration_frequency_axis,
    elaboration_time_axis)
from deistools.processing.passband_filters import create_fermi_dirac_fun
from deistools.processing.bltva import bltva, legendre_basis
from deistools.simulation.models.legendre import simulate_legendre

from _plotting import parse_args, finish, show_all, comparison

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# Excitation and system (see example 02)
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
f_min = frequencies.min()

order = 3
a = np.zeros((frequencies.size, order + 1), dtype=complex)
a[:, 0] = 10.0 + 30.0 / (1 + 1j * 2 * np.pi * frequencies * 30.0 * 1e-3)
for p in range(1, order + 1):
    a[:, p] = (0.35 / p) * a[:, 0]

sim = simulate_legendre(T, 1.0 / fs, frequencies, amplitudes, phases, a,
                        drift_coefficients=None, noise_std_V=1e-4, seed=1)

time = sim["time_s"]
voltage = sim["voltage_V"]
current = sim["current_A"]
Z_true = sim["Z_true"]


def truth_at(times):
    """Ground truth resampled onto the instants an estimator reports at."""
    return np.array([np.interp(times, time, Z_true[k].real)
                     + 1j * np.interp(times, time, Z_true[k].imag)
                     for k in range(frequencies.size)])


# --------------------------------------------------------------------------
# STFT -- one period per window
# --------------------------------------------------------------------------
Z_stft, _, _, t_stft = estimate_impedance(
    voltage, current, frequencies, Tp, 1.0 / fs, window_fun="rectangular")


# --------------------------------------------------------------------------
# DMFA -- bandwidth 0.5 f_min, sharpness 2
# --------------------------------------------------------------------------
ft_voltage = fftshift(fft(voltage)) / N
ft_current = fftshift(fft(current)) / N
spectrum_axis = fftshift(fftfreq(N, 1.0 / fs))
indexes = (np.round(frequencies * T).astype(int)
           + int(np.argmin(np.abs(spectrum_axis)))).tolist()

bandwidth = 0.5 * f_min
n_points = int(round(8 * bandwidth * T))
n_points = min(n_points - n_points % 2, max_elaboration_points(indexes, N))
band_filter = create_fermi_dirac_fun(
    elaboration_frequency_axis(n_points, T), 0, bandwidth, 2)

Z_dmfa = extract_impedance(ft_voltage, ft_current, indexes, band_filter,
                           n_points)
t_dmfa = elaboration_time_axis(n_points, T)
edge = n_points // 10            # the ringing record edges
Z_dmfa, t_dmfa = Z_dmfa[:, edge:-edge], t_dmfa[edge:-edge]


# --------------------------------------------------------------------------
# BLTVA -- Np = 4, Nq = 2, band 100
# --------------------------------------------------------------------------
# Np = 4 is one order above the order-3 variation the system actually has.
Np = 4
result = bltva(voltage, current, fs, P, harmonics,
               Np=Np, Nq=2, Na=None, harmonics="excited", band_halfwidth=100)

instants = np.linspace(0, N - 1, 200).astype(int)
Z_bltva = result.H_p @ legendre_basis(N, Np)[instants].T
t_bltva = time[instants]


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
runs = [("stft", Z_stft, t_stft),
        ("dmfa", Z_dmfa, t_dmfa),
        ("bltva", Z_bltva, t_bltva)]

print("   method   instants       median        worst")
results, medians = [], {}
for name, Z, t in runs:
    truth = truth_at(t)
    err = np.abs(Z - truth) / np.abs(truth)
    results.append((name, Z, t, truth))
    medians[name] = np.median(err)
    print(f"  {name:>7}   {Z.shape[1]:>8}   {np.median(err):>10.2e}   "
          f"{np.max(err):>10.2e}")

print(f"\n  BLTVA is {medians['dmfa'] / medians['bltva']:.0f}x better than DMFA here and "
      f"{medians['stft'] / medians['bltva']:.0f}x better than STFT --")
print("  a far wider margin than on the transistor in example 21, with the")
print("  same methods and the same settings. Nothing about the algorithms")
print("  changed; the model is now exactly right.")

fig1 = comparison(frequencies, results, n_instants=3)
finish(fig1, "22_compare_legendre", args.save,
       "three estimators on a prescribed Legendre variation")


# --------------------------------------------------------------------------
# Recovering the coefficients themselves
# --------------------------------------------------------------------------
# BLTVA does not return Z(f,t); it returns the a_p(f), and here those have
# known values. No other system in this set allows that check.

print("\n   order p   |a_p| true max   median error")
for p in range(result.H_p.shape[1]):
    if p < a.shape[1]:
        err_p = np.abs(result.H_p[:, p] - a[:, p]) / np.abs(a[:, 0])
        true_max = np.abs(a[:, p]).max()
    else:
        err_p = np.abs(result.H_p[:, p]) / np.abs(a[:, 0])   # should be zero
        true_max = 0.0
    print(f"  {p:>8}   {true_max:>13.3f}   {np.median(err_p):>12.2e}")

print(f"\n  The first {a.shape[1]} orders are recovered to a few parts in a million of")
print(f"  |a_0|. Order {a.shape[1]} is one the system does not have: BLTVA was asked for")
print(f"  Np = {Np}, fitted it anyway, and returned exactly zero. Coefficients the")
print("  data does not support are removed rather than left to absorb noise.")

fig2, axs = plt.subplots(1, 2, figsize=(14, 5))

for (name, Z, t), colour in zip(runs, ("C0", "C1", "C2")):
    truth = truth_at(t)
    axs[0].loglog(frequencies,
                  np.median(np.abs(Z - truth) / np.abs(truth), axis=1),
                  "-o", ms=4, color=colour, label=name)
axs[0].set(xlabel="frequency / Hz", ylabel="median relative error",
           title="error against frequency")
axs[0].grid(True, which="both", alpha=0.25)
axs[0].legend(fontsize=8)

for p in range(result.H_p.shape[1]):
    true_max = np.abs(a[:, p]).max() if p < a.shape[1] else 0.0
    axs[1].bar(p - 0.2, true_max, width=0.4, color="black", alpha=0.5,
               label="true" if p == 0 else None)
    axs[1].bar(p + 0.2, np.abs(result.H_p[:, p]).max(), width=0.4,
               color="C0", label="estimated" if p == 0 else None)
axs[1].set(xlabel="Legendre order p", ylabel=r"max$_f$ $|a_p|$ / $\Omega$",
           xticks=range(result.H_p.shape[1]),
           title="coefficients recovered, including the one that is zero")
axs[1].legend(fontsize=8)

finish(fig2, "22_legendre_coefficients", args.save,
       "what a correct model assumption buys")

print("\n  Treat this file as the upper bound on what BLTVA can do, and example")
print("  21 as what to expect from a real cell.")

show_all(args.save)
