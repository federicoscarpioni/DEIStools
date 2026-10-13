"""
23 -- The three estimators on a nonlinear system.

A nonlinear system has no transfer function, so "the impedance at time t" is
not a property it possesses. What the estimators are measured against is a
best linear approximation, which depends on the excitation amplitude.

Every assumption that held in 20-22 is weakened here, which makes this the
closest of the four to a real electrochemical measurement.

Run:  python examples/23_compare_block_oriented.py [--save]
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
from deistools.simulation.models.block_oriented import (
    hallemans_2021_system, simulate_block_oriented, best_linear_approximation,
    frozen_impedance)

from _plotting import parse_args, finish, show_all, comparison

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# Excitation and system (see example 03)
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

branches = hallemans_2021_system()
sim = simulate_block_oriented(T, 1.0 / fs, frequencies, amplitudes, phases,
                              branches, trend_coefficients=None,
                              noise_std_V=1e-4, seed=1)

time = sim["time_s"]
voltage = sim["voltage_V"]
current = sim["current_A"]

G, scatter = best_linear_approximation(branches, frequencies, amplitudes,
                                       1.0 / fs, Tp, n_realisations=200,
                                       seed=7)
Z_true = frozen_impedance(G, N)

print(f"  ground truth is a best linear approximation, known to "
      f"{np.abs(scatter / G[:, [0]]).max():.0e} relative")


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
# DMFA -- bandwidth 0.5 f_min, sharpness 8
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
    elaboration_frequency_axis(n_points, T), 0, bandwidth, 8)

Z_dmfa = extract_impedance(ft_voltage, ft_current, indexes, band_filter,
                           n_points)
t_dmfa = elaboration_time_axis(n_points, T)
edge = n_points // 10            # the ringing record edges
Z_dmfa, t_dmfa = Z_dmfa[:, edge:-edge], t_dmfa[edge:-edge]


# --------------------------------------------------------------------------
# BLTVA -- Np = 8, Nq = 2, band 100
# --------------------------------------------------------------------------
# harmonics='all_nl' keeps the unexcited harmonic bins in the fit. On a
# nonlinear system the distortion lands there, so modelling it both improves
# the estimate and makes the distortion variances available below.
Np = 8
result = bltva(voltage, current, fs, P, harmonics,
               Np=Np, Nq=2, Na=None, harmonics="all_nl", band_halfwidth=100)

instants = np.linspace(0, N - 1, 200).astype(int)
Z_bltva = result.H_p @ legendre_basis(N, Np)[instants].T
t_bltva = time[instants]


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
runs = [("stft", Z_stft, t_stft),
        ("dmfa", Z_dmfa, t_dmfa),
        ("bltva", Z_bltva, t_bltva)]

print("\n   method   instants       median        worst")
results, errors = [], {}
for name, Z, t in runs:
    truth = truth_at(t)
    err = np.abs(Z - truth) / np.abs(truth)
    results.append((name, Z, t, truth))
    errors[name] = err
    print(f"  {name:>7}   {Z.shape[1]:>8}   {np.median(err):>10.2e}   "
          f"{np.max(err):>10.2e}")

print("\n  The ordering is the same as everywhere else, but every number is one")
print("  to two orders of magnitude worse than on the linear systems.")

fig1 = comparison(frequencies, results, n_instants=3)
finish(fig1, "23_compare_block_oriented", args.save,
       "three estimators on a nonlinear system")


# --------------------------------------------------------------------------
# Most of the error is not the estimator's fault
# --------------------------------------------------------------------------
# The branch filters are second-order lowpasses near 1 Hz, so the upper tones
# carry almost no signal and relative error is measured against a vanishing
# denominator.

low = frequencies <= 1.5
print("\n   method   f <= 1.5 Hz   f > 1.5 Hz    ratio")
for name, Z, t in runs:
    median_error = np.median(errors[name], axis=1)
    below, above = np.median(median_error[low]), np.median(median_error[~low])
    print(f"  {name:>7}   {below:>11.2e}   {above:>10.2e}   {above / below:>6.0f}")

print(f"\n  |G_0| falls from {np.abs(G[:, 0]).max():.2f} at the resonance to "
      f"{np.abs(G[:, 0]).min():.4f} at the top tone,")
print(f"  a factor of {np.abs(G[:, 0]).max() / np.abs(G[:, 0]).min():.0f}. Choose a frequency grid where the cell responds.")

fig2, axs = plt.subplots(1, 2, figsize=(14, 5))
for (name, Z, t), colour in zip(runs, ("C0", "C1", "C2")):
    axs[0].loglog(frequencies, np.median(errors[name], axis=1), "-o", ms=4,
                  color=colour, label=name)
twin = axs[0].twinx()
twin.loglog(frequencies, np.abs(G[:, 0]), ":", color="0.4", lw=1.5)
twin.set_ylabel(r"$|G_0|$ (grey, dotted)", color="0.4")
axs[0].axvline(1.5, ls="--", color="C3", alpha=0.6)
axs[0].set(xlabel="frequency / Hz", ylabel="median relative error",
           title="error tracks the inverse of the response")
axs[0].legend(fontsize=8, loc="upper left")

V = np.abs(np.fft.rfft(voltage)) / N
excited_bins = np.round(frequencies * T).astype(int)
axs[1].loglog(frequencies, V[excited_bins], "-o", ms=4, color="C0",
              label="excited tones")
axs[1].axhline(1e-4 / np.sqrt(N / 2), ls="--", color="C3",
               label="noise per DFT bin")
axs[1].set(xlabel="frequency / Hz", ylabel=r"$|V|$ / V",
           title="signal against noise, per tone")
axs[1].grid(True, which="both", alpha=0.25)
axs[1].legend(fontsize=8)
finish(fig2, "23_block_response_limit", args.save,
       "why the upper decade is hard here")


# --------------------------------------------------------------------------
# The target moves when the drive moves
# --------------------------------------------------------------------------
# On a linear system, doubling the excitation changes nothing about the
# answer. Here it changes the answer itself.
print()
print("   drive   shift in the ground truth")
shifts = {}
for scale in (0.5, 2.0, 4.0):
    G_s, _ = best_linear_approximation(branches, frequencies,
                                       amplitudes * scale, 1.0 / fs, Tp,
                                       n_realisations=200, seed=7)
    shifts[scale] = np.median(np.abs(G_s[:, 0] - G[:, 0]) / np.abs(G[:, 0]))
    print(f"  x{scale:<5.1f}   {shifts[scale]:>10.2e}")

best_error = np.median(errors["bltva"])
print()
print(f"  Doubling the drive moves the ground truth by {shifts[2.0]:.1e}, against a")
print(f"  best estimator error of {best_error:.1e} -- a factor of {best_error / shifts[2.0]:.0f} apart. At four")
print(f"  times the drive the shift has overtaken it ({shifts[4.0] / best_error:.1f}x the error), because the")
print("  drive dependence grows with amplitude while the estimator error does")
print("  not. Reporting an impedance without the excitation amplitude is")
print("  incomplete whenever the system is not strictly linear.")


# --------------------------------------------------------------------------
# What BLTVA can say that the others cannot
# --------------------------------------------------------------------------
# With harmonics='all_nl' the fit keeps the unexcited harmonic bins, and the
# parity of those bins separates even-order from odd-order distortion.

sndr = np.asarray(result.sndr_db())
print("\n   f / Hz        |H_0|   even dist.    odd dist.   SNDR / dB")
for i in (0, 4, 8, 12):
    even = np.sqrt(result.var_distortion_even[i, 0])
    odd = np.sqrt(result.var_distortion_odd[i, 0])
    print(f"  {frequencies[i]:>7.2f}   {np.abs(result.H_p[i, 0]):>10.4f}   "
          f"{even:>10.2e}   {odd:>10.2e}   {sndr[i]:>9.1f}")

print(f"\n  SNDR is how far the linear part stands above everything that is not")
print(f"  it. It runs {np.nanmin(sndr):.0f} to {np.nanmax(sndr):.0f} dB here, falling where the branch filters")
print("  have rolled off. It is the number to check before trusting an")
print("  impedance from a cell driven hard enough to distort, and neither")
print("  STFT nor DMFA computes it.")

show_all(args.save)
