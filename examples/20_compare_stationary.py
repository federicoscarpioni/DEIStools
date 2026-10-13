"""
20 -- The three estimators on a stationary cell.

Nothing varies, so every estimator should return the same constant impedance.
Any time dependence reported here is an artefact of the method.

This is the control: if a method invents variation where there is none,
nothing it says about a system that really varies can be trusted.

Run:  python examples/20_compare_stationary.py [--save]
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
from deistools.simulation.models.r_rc import simulate_r_rc

from _plotting import parse_args, finish, show_all, comparison

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# Excitation and circuit (see example 00)
# --------------------------------------------------------------------------
Tp = 20.0
fs = 1000.0
P = 6

harmonics = np.array([1, 3, 5, 9, 13, 17, 27, 39, 59, 89, 133, 201, 301])
frequencies = harmonics / Tp
amplitudes = np.full(frequencies.size, 0.01)
phases = np.random.default_rng(3).uniform(0, 2 * np.pi, frequencies.size)

N = int(fs * Tp) * P
T = N / fs
f_min = frequencies.min()

sim = simulate_r_rc(T, 1.0 / fs, frequencies, amplitudes, phases,
                    R_s=10.0, R_p=30.0, C=1e-3, noise_std_V=1e-4, seed=1)

voltage = sim["voltage_V"]
current = sim["current_A"]
Z_true = sim["Z_true"]          # (n_freq, 1): constant in time


def truth_at(times):
    """Here the truth is the same at every instant."""
    return np.repeat(Z_true, np.size(times), axis=1)


# --------------------------------------------------------------------------
# STFT -- the whole record in one window, as example 11 selected
# --------------------------------------------------------------------------
Z_stft, _, _, t_stft = estimate_impedance(
    voltage, current, frequencies, P * Tp, 1.0 / fs,
    window_fun="rectangular")


# --------------------------------------------------------------------------
# DMFA -- bandwidth 0.5 f_min, sharpness 4, as example 13 selected
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
    elaboration_frequency_axis(n_points, T), 0, bandwidth, 4)

Z_dmfa = extract_impedance(ft_voltage, ft_current, indexes, band_filter,
                           n_points)
t_dmfa = elaboration_time_axis(n_points, T)


# --------------------------------------------------------------------------
# BLTVA -- Np = 1, Nq = 1, band 20, as example 15 selected
# --------------------------------------------------------------------------
# Six periods leaves almost no coefficient budget: Np = 1 with Nq <= 1 is all
# that will run on this record.
Np = 1
result = bltva(voltage, current, fs, P, harmonics,
               Np=Np, Nq=1, Na=None, harmonics="excited", band_halfwidth=20)

instants = np.linspace(0, N - 1, 200).astype(int)
Z_bltva = result.H_p @ legendre_basis(N, Np)[instants].T
t_bltva = sim["time_s"][instants]


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
runs = [("stft", Z_stft, t_stft),
        ("dmfa", Z_dmfa, t_dmfa),
        ("bltva", Z_bltva, t_bltva)]

print("   method   instants       median        worst")
results = []
for name, Z, t in runs:
    truth = truth_at(t)
    err = np.abs(Z - truth) / np.abs(truth)
    results.append((name, Z, t, truth))
    print(f"  {name:>7}   {Z.shape[1]:>8}   {np.median(err):>10.2e}   "
          f"{np.max(err):>10.2e}")

print("\n  All three agree to within a factor of three, at a few parts in a")
print("  million. On a stationary cell there is nothing to separate them.")

fig1 = comparison(frequencies, results, n_instants=3)
finish(fig1, "20_compare_stationary", args.save,
       "three estimators on a stationary cell")


# --------------------------------------------------------------------------
# Does any of them invent variation?
# --------------------------------------------------------------------------
# The impedance is constant, so any spread over time is entirely spurious.

print("\n   method   instants   spurious spread")
for name, Z, t in runs:
    if Z.shape[1] < 2:
        print(f"  {name:>7}   {Z.shape[1]:>8}   n/a, single estimate")
        continue
    spread = np.std(Z, axis=1) / np.abs(np.mean(Z, axis=1))
    print(f"  {name:>7}   {Z.shape[1]:>8}   {np.median(spread):>15.2e}")

print(f"\n  BLTVA's spread is machine zero, and the reason is worth seeing:")
print(f"  its time-varying coefficient came back as")
print(f"      max |H_1| = {np.abs(result.H_p[:, 1]).max():.1e}   at all {result.H_p.shape[0]} tones")
print("  not merely small. Coefficients the data does not support are removed")
print("  from the fit, so the model concluded by itself that the system is")
print("  stationary. That is a different answer from 'the variation I measured")
print("  was small'.")

fig2, axs = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, (name, Z, t) in zip(axs, runs):
    truth = truth_at(t)
    for i, colour in zip((0, 6, 12), ("C0", "C1", "C2")):
        ax.plot(t, (np.abs(Z[i]) - np.abs(truth[i])) / np.abs(truth[i]),
                "-o", ms=3, color=colour, label=f"{frequencies[i]:.2f} Hz")
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="time / s",
           ylabel=r"$(|Z| - |Z_{true}|)/|Z_{true}|$", title=name)
    ax.legend(fontsize=7)
finish(fig2, "20_stationary_spurious", args.save,
       "any wobble here is invented: the true impedance is constant")

show_all(args.save)
