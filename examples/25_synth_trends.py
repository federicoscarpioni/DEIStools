"""
25 -- Generating trends, and adding them to the Legendre oracle.

There are two ways to put an additive drift on a simulated record, and they
are for different jobs:

  A. ``simulate_legendre(..., drift_coefficients=...)`` -- built into the
     model. The drift is a Legendre polynomial on the record, which is the
     *same basis BLTVA fits*, so a drift made this way is inside the
     estimator's model by construction and a large enough ``Nq`` removes it
     exactly.

  B. ``deistools.simulation.trends`` -- any shape, added after synthesis.
     Use it when the question is what happens to a trend the estimator
     cannot represent.

This script shows both, checks they agree where they overlap, and prices
each shape.

The Legendre system is the right one for this because its time-varying
ground truth is **exact** -- the coefficients are prescribed, not fitted. So
the cost of a trend is a measurement against a known floor rather than a
comparison between two estimates.

Run:  python examples/25_synth_trends.py [--save]
"""
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft, fftshift, fftfreq

from deistools.processing.bltva import bltva, legendre_basis
from deistools.processing.stft_functions import estimate_impedance
from deistools.processing.dmfa_functions import (
    extract_impedance, max_elaboration_points, elaboration_frequency_axis,
    elaboration_time_axis)
from deistools.processing.passband_filters import create_fermi_dirac_fun
from deistools.simulation.models.legendre import simulate_legendre
from deistools.simulation.trends import (
    TRENDS, logit_trend, polynomial_trend, relaxation_trend)

from _plotting import parse_args, finish, show_all, envelope

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# The excitation and the prescribed system (see example 02)
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

R_s, R_p, C = 10.0, 30.0, 1e-3
order = 3
a = np.zeros((frequencies.size, order + 1), dtype=complex)
a[:, 0] = R_s + R_p / (1 + 1j * 2 * np.pi * frequencies * R_p * C)
for p in range(1, order + 1):
    a[:, p] = (0.35 / p) * a[:, 0]


def simulate(drift_coefficients=None):
    return simulate_legendre(
        T, 1.0 / fs, frequencies, amplitudes, phases, a,
        drift_coefficients=drift_coefficients, noise_std_V=1e-4, seed=1)


clean = simulate()
time = clean["time_s"]
current = clean["current_A"]

print(f"  record {T:.0f} s, {N} samples, response {clean['voltage_V'].std():.2f} V rms")


# --------------------------------------------------------------------------
# Route A -- the drift the model makes for you
# --------------------------------------------------------------------------
# drift_coefficients are the weights of the Legendre polynomials on the
# record: index q is the coefficient of L_q. Index 0 is a constant, which
# every estimator discards with the dc bin, so it is left at zero here.

drift_coefficients = [0.0, 0.5, 0.2, 0.1]
built_in = simulate(drift_coefficients=drift_coefficients)

print(f"\n  route A: simulate_legendre(..., drift_coefficients={drift_coefficients})")
print(f"    the drift comes back as sim['drift_V'], "
      f"{np.ptp(built_in['drift_V']):.2f} V p-p")
print(f"    and sim['voltage_V'] already has it added")


# --------------------------------------------------------------------------
# Route B -- the trends module
# --------------------------------------------------------------------------
# Same polynomial, built independently and added by hand. The two agree,
# which is the point: route B is a superset, not a different convention.

rebuilt = polynomial_trend(time, coefficients=drift_coefficients,
                           basis="legendre", remove_mean=False)
agreement = np.max(np.abs((rebuilt - rebuilt.mean())
                          - (built_in["drift_V"] - built_in["drift_V"].mean())))

print(f"\n  route B: polynomial_trend(time, coefficients=..., basis='legendre')")
print(f"    agrees with route A to {agreement:.1e} V once both are centred")
print("    (the offset differs because the model subtracts the sampled mean")
print("     of each basis function, as the estimator does, and the module")
print("     leaves that to remove_mean=)")


# --------------------------------------------------------------------------
# The three standard shapes, at one severity
# --------------------------------------------------------------------------
# peak_to_peak= puts them on equal terms, so what is compared below is the
# shape and not the size.

severity = 4.0 * clean["voltage_V"].std()
shapes = {name: make(time, peak_to_peak=severity) for name, make in TRENDS.items()}

print(f"\n  three standard shapes at {severity:.1f} V p-p:")
for name, values in shapes.items():
    print(f"    {name:<12} max|.| {np.max(np.abs(values)):.2f} V")

print("\n  Adding one is a sum -- nothing about the model changes:")
print("      sim = simulate_legendre(...)")
print("      sim['voltage_V'] = sim['voltage_V'] + trend")


# --------------------------------------------------------------------------
# What each one costs -- and to whom
# --------------------------------------------------------------------------
# The oracle prescribes a_p, so the truth is known exactly rather than
# inferred, and Z(f,t) can be reconstructed on the full time axis.

Z_true = a @ legendre_basis(N, order).T          # (n_freq, N), exact


def bltva_error(voltage, Nq=order):
    """BLTVA carries a trend model of order Nq, so it fits the drift."""
    result = bltva(voltage, current, fs, P, harmonics, Np=order, Nq=Nq,
                   harmonics="excited", band_halfwidth=5 * P)
    return np.median(np.abs(result.H_p[:, :order + 1] - a)
                     / np.abs(a[:, [0]]))


def _against_truth(Z, instants):
    index = np.clip(np.searchsorted(time, instants), 0, N - 1)
    return np.median(np.abs(Z - Z_true[:, index]) / np.abs(Z_true[:, index]))


def stft_error(voltage):
    """STFT divides at the excited bins. No trend model at all."""
    Z, _, _, t_stft = estimate_impedance(voltage, current, frequencies, Tp,
                                         1.0 / fs, window_fun="rectangular")
    return _against_truth(Z, t_stft)


spectrum_axis = fftshift(fftfreq(N, 1.0 / fs))
indexes = (np.round(frequencies * T).astype(int)
           + int(np.argmin(np.abs(spectrum_axis)))).tolist()
bandwidth = 0.5 * frequencies.min()
n_points = int(round(16 * bandwidth * T))
n_points = min(n_points - n_points % 2, max_elaboration_points(indexes, N))
band_filter = create_fermi_dirac_fun(
    elaboration_frequency_axis(n_points, T), 0, bandwidth, 8)
t_dmfa = elaboration_time_axis(n_points, T)
edge = n_points // 10
ft_current = fftshift(fft(current)) / N


def dmfa_error(voltage):
    """DMFA uses the skirts, so it is the most exposed to a drift."""
    Z = extract_impedance(fftshift(fft(voltage)) / N, ft_current, indexes,
                          band_filter, n_points)[:, edge:-edge]
    return _against_truth(Z, t_dmfa[edge:-edge])


severities = (1.0, 10.0, 100.0)
rms = clean["voltage_V"].std()

print(f"\n  median error against the exact truth, by estimator and severity")
print(f"    (trend peak-to-peak as a multiple of the {rms:.2f} V rms response)")
print(f"\n    {'trend':<12} {'p-p/rms':>8}   {'BLTVA':>9} {'STFT':>9} {'DMFA':>9}")
print(f"    {'none':<12} {0:>8.0f}   {bltva_error(clean['voltage_V']):>9.1e} "
      f"{stft_error(clean['voltage_V']):>9.1e} "
      f"{dmfa_error(clean['voltage_V']):>9.1e}")

curves = {"BLTVA": [], "STFT": [], "DMFA": []}
for multiple in severities:
    for name, make in TRENDS.items():
        voltage = clean["voltage_V"] + make(time, peak_to_peak=multiple * rms)
        scores = (bltva_error(voltage), stft_error(voltage),
                  dmfa_error(voltage))
        print(f"    {name:<12} {multiple:>8.0f}   " +
              " ".join(f"{e:>9.1e}" for e in scores))
        for key, value in zip(curves, scores):
            curves[key].append(value)

print("\n  Two things, and the second is the one worth keeping:")
print("    * the SHAPE barely matters. At a given size the three trends cost")
print("      STFT and DMFA almost the same, which is the same fact as in")
print("      example 09: a trend and a tone that share a bin cannot be told")
print("      apart by anything that works bin by bin.")
print("    * WHO is being asked matters enormously. BLTVA holds near its")
print("      floor however large the drift, because Nq fits it jointly with")
print("      the impedance. STFT and DMFA have no trend model, so their")
print("      error grows with the drift, roughly in proportion.")
print("\n  So a trend is not hard in itself. It is hard for an estimator")
print("  that does not model it -- which is what detrending is for, and what")
print("  example 09 measures.")


# --------------------------------------------------------------------------
# Figure 2: who pays for a drift
# --------------------------------------------------------------------------
fig2, axs = plt.subplots(1, 2, figsize=(14, 5))

n_shapes = len(TRENDS)
for (label, values), colour in zip(curves.items(), ("C3", "C0", "C1")):
    grid = np.repeat(severities, n_shapes)
    axs[0].loglog(grid, values, "o", ms=5, color=colour, alpha=0.7,
                  label=label)
    means = [np.mean(values[i * n_shapes:(i + 1) * n_shapes])
             for i in range(len(severities))]
    axs[0].loglog(severities, means, "-", lw=1.2, color=colour)
axs[0].set(xlabel="trend peak-to-peak / response rms",
           ylabel="median error against exact truth",
           title="three shapes per point: the shape hardly shows")
axs[0].legend(fontsize=8)
axs[0].grid(which="both", alpha=0.2)

worst = clean["voltage_V"] + logit_trend(time, peak_to_peak=100.0 * rms)
Z_clean, _, _, t_stft = estimate_impedance(
    clean["voltage_V"], current, frequencies, Tp, 1.0 / fs,
    window_fun="rectangular")
Z_drift, _, _, _ = estimate_impedance(
    worst, current, frequencies, Tp, 1.0 / fs, window_fun="rectangular")
index = np.clip(np.searchsorted(time, t_stft), 0, N - 1)
truth = Z_true[:, index]

axs[1].loglog(frequencies,
              np.median(np.abs(Z_clean - truth) / np.abs(truth), axis=1),
              "--o", ms=4, color="black", label="no trend")
axs[1].loglog(frequencies,
              np.median(np.abs(Z_drift - truth) / np.abs(truth), axis=1),
              "-o", ms=4, color="C2", label="logit, 100x rms")
axs[1].set(xlabel="frequency / Hz", ylabel="median error of STFT",
           title="where a drift lands: the bottom of the grid")
axs[1].legend(fontsize=8)
axs[1].grid(which="both", alpha=0.2)

finish(fig2, "25_trend_cost", args.save,
       "a drift is only hard for an estimator that does not model it")

print("\n  The right-hand panel is the reason detrending exists at all: the")
print("  drift sits in the lowest bins, and so do the slowest tones, so that")
print("  is where an estimator without a trend model loses accuracy.")

show_all(args.save)
