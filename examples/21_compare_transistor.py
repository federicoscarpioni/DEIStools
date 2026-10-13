"""
21 -- The three estimators on a smoothly varying cell.

The realistic case: a circuit whose charge-transfer resistance is swept
through the record, solved as a true time-varying ODE. Nothing about it was
built to suit any of the three methods.

Run:  python examples/21_compare_transistor.py [--save]
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
from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor

from _plotting import parse_args, finish, show_all, comparison

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
f_min = frequencies.min()

sim = simulate_r_rc_transistor(
    T, 1.0 / fs, frequencies, amplitudes, phases,
    R_s=10.0, R_p=30.0, C=1e-3,
    V_gate_start=1.645, V_gate_rate=(2.455 - 1.645) / T, R_T_max=70.0,
    noise_std_V=1e-4, seed=1)

time = sim["time_s"]
voltage = sim["voltage_V"]
current = sim["current_A"]
Z_true = sim["Z_true"]

print(f"  R_ct sweeps {sim['R_ct_t'][0]:.1f} -> {sim['R_ct_t'][-1]:.1f} ohm")
print(f"  frozen-time ground truth accurate to "
      f"{np.std(voltage - sim['voltage_quasistatic_V']) / np.std(voltage):.3%}")


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
# The band and the reporting rate are different concerns, and Npts_elab sets
# both. The band must be wide enough for the filter to have decayed -- 16
# bandwidths puts the truncation at 1e-9, where 8 leaves 2.6e-03. But the
# result must not be reported more densely than 0.5/bw, or neighbouring
# errors are correlated: 84% at the 0.125/bw these examples used to report,
# against 3% at 0.5/bw. So extract wide, then decimate.
n_points = int(round(16 * bandwidth * T))
n_points = min(n_points - n_points % 2, max_elaboration_points(indexes, N))
band_filter = create_fermi_dirac_fun(
    elaboration_frequency_axis(n_points, T), 0, bandwidth, 2)

Z_dmfa = extract_impedance(ft_voltage, ft_current, indexes, band_filter,
                           n_points)
t_dmfa = elaboration_time_axis(n_points, T)

# The band-limited inverse transform rings at the record boundaries, so the
# first and last tenth of a DMFA result is not usable. Discard it, or the
# comparison below is about edge artefacts rather than about the methods.
edge = n_points // 10
Z_dmfa = Z_dmfa[:, edge:-edge]
t_dmfa = t_dmfa[edge:-edge]

# Decimate to 0.5/bw, the spacing at which neighbouring errors stop being
# correlated (see dmfa_functions.independent_time_step).
keep = max(1, int(round(n_points / (2 * bandwidth * T))))
Z_dmfa, t_dmfa = Z_dmfa[:, ::keep], t_dmfa[::keep]


# --------------------------------------------------------------------------
# BLTVA -- Np = 8, Nq = 2, band 50
# --------------------------------------------------------------------------
Np = 8
result = bltva(voltage, current, fs, P, harmonics,
               Np=Np, Nq=2, Na=None, harmonics="excited", band_halfwidth=50)

instants = np.linspace(0, N - 1, 200).astype(int)
Z_bltva = result.H_p @ legendre_basis(N, Np)[instants].T
t_bltva = time[instants]


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
runs = [("stft", Z_stft, t_stft),
        ("dmfa", Z_dmfa, t_dmfa),
        ("bltva", Z_bltva, t_bltva)]

print("\n   method   instants   dt / s       median        worst")
results, medians = [], {}
for name, Z, t in runs:
    truth = truth_at(t)
    err = np.abs(Z - truth) / np.abs(truth)
    results.append((name, Z, t, truth))
    medians[name] = np.median(err)
    print(f"  {name:>7}   {Z.shape[1]:>8}   {np.diff(t).mean():>6.2f}   "
          f"{np.median(err):>10.2e}   {np.max(err):>10.2e}")

print(f"\n  BLTVA is best, by the widest margin on the worst case. DMFA is")
print(f"  close on the median, at an independent spacing of {np.diff(t_dmfa).mean():.0f} s against")
print(f"  STFT's {Tp:.0f} s window -- the same, at this bandwidth. STFT is weakest here")
print("  but needed no assumption about how the system varies.")

fig1 = comparison(frequencies, results, n_instants=3)
finish(fig1, "21_compare_transistor", args.save,
       "three estimators on a smoothly varying cell")


# --------------------------------------------------------------------------
# Tracking, and where each method pays for it
# --------------------------------------------------------------------------
fig2, axs = plt.subplots(1, 2, figsize=(14, 5))

axs[0].plot(time, np.abs(Z_true[0]), color="black", lw=2, alpha=0.5,
            label="ground truth")
for (name, Z, t), colour in zip(runs, ("C0", "C1", "C2")):
    axs[0].plot(t, np.abs(Z[0]), "-o", ms=3, color=colour, label=name)
axs[0].set(xlabel="time / s", ylabel=r"$|Z|$ / $\Omega$",
           title=f"tracking the lowest tone ({frequencies[0]:.2f} Hz)")
axs[0].legend(fontsize=8)

print("\n   method   worst tone    best tone")
for (name, Z, t), colour in zip(runs, ("C0", "C1", "C2")):
    truth = truth_at(t)
    median_error = np.median(np.abs(Z - truth) / np.abs(truth), axis=1)
    axs[1].loglog(frequencies, median_error, "-o", ms=4, color=colour,
                  label=name)
    print(f"  {name:>7}   {frequencies[int(np.argmax(median_error))]:>7.2f} Hz"
          f"   {frequencies[int(np.argmin(median_error))]:>7.2f} Hz")

axs[1].set(xlabel="frequency / Hz", ylabel="median relative error",
           title="error against frequency")
axs[1].grid(True, which="both", alpha=0.25)
axs[1].legend(fontsize=8)
finish(fig2, "21_transistor_tracking", args.save,
       "following the variation, and where each method pays for it")

print("\n  All three are worst at the bottom of the grid. What matters is how")
print("  many cycles of a tone fit inside the time the cell takes to change:")
print(f"  the {frequencies[0]:.2f} Hz tone completes {frequencies[0] * T:.0f} cycles over the whole")
print(f"  variation, the {frequencies[-1]:.1f} Hz tone completes {frequencies[-1] * T:.0f}. The bottom decade of a")
print("  time-varying spectrum is always the least reliable part, whichever")
print("  method produced it.")


# --------------------------------------------------------------------------
# Only one of them reports an uncertainty
# --------------------------------------------------------------------------
predicted = np.median(result.std_H_p[:, 0] / np.abs(result.H_p[:, 0]))
print(f"\n  BLTVA predicted noise   {predicted:.2e}")
print(f"  actual median error     {medians['bltva']:.2e}")
print(f"  ratio                   {medians['bltva'] / predicted:.0f}")
print("\n  The error is far above what noise alone explains, and on this system")
print("  that is correct: Z(f,t) here is a rational function of time, not a")
print("  polynomial, so the Legendre series leaves a residue. The uncertainty")
print("  is reporting that the model, not the measurement, is the limit.")
print("  Neither STFT nor DMFA offers this number at all.")

show_all(args.save)
