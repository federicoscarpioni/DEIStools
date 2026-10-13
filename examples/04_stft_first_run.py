"""
04 -- A first STFT estimation.

One estimation of the transistor-gated cell with the settings example 10
arrives at, to show how the function is called and what it gives back.

Run:  python examples/04_stft_first_run.py [--save]
"""
import numpy as np
import matplotlib.pyplot as plt

from deistools.processing.stft_functions import estimate_impedance
from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor

from _plotting import parse_args, finish, show_all, overview, time_colours

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# The measurement to analyse (see example 01)
# --------------------------------------------------------------------------
Tp = 20.0          # period of the multisine [s]
fs = 1000.0        # sampling rate [Hz]
P = 20             # number of periods measured

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
Z_true = sim["Z_true"]        # frozen-time impedance, for checking


# --------------------------------------------------------------------------
# The estimation
# --------------------------------------------------------------------------
# The record is cut into consecutive windows; each gives one spectrum, taken
# at the excited bins and stamped at the window's midpoint.
#
# 'period' is the window length in SECONDS -- not samples, not multisine
# periods. It must hold a whole number of periods of every tone, or the
# rectangular window leaks and the leakage looks like time variation.

window_length = 1 * Tp        # one multisine period per window

Z, voltage_avg, current_avg, t = estimate_impedance(
    voltage,                  # measured response
    current,                  # measured excitation
    frequencies,              # the excited frequencies [Hz]
    window_length,            # window length [s]
    1.0 / fs,                 # sampling time [s]
    window_fun="rectangular",
)


# --------------------------------------------------------------------------
# What came back
# --------------------------------------------------------------------------
print("  estimate_impedance returns four arrays:")
print(f"    Z             {str(Z.shape):<12} complex, (n_freq, n_windows)")
print(f"    voltage_avg   {str(voltage_avg.shape):<12} mean voltage in each window")
print(f"    current_avg   {str(current_avg.shape):<12} mean current in each window")
print(f"    t             {str(t.shape):<12} midpoint of each window [s]")

print(f"\n  {Z.shape[1]} windows of {window_length:.0f} s, reported at "
      f"t = {t[0]:.0f}, {t[1]:.0f} ... {t[-1]:.0f} s")
print(f"  time resolution is the window length: {window_length:.0f} s")

print(f"\n  the impedance at the first and last window, lowest tone "
      f"({frequencies[0]:.2f} Hz):")
print(f"    t = {t[0]:>5.0f} s   Z = {Z[0, 0].real:7.2f} {Z[0, 0].imag:+7.2f}j ohm")
print(f"    t = {t[-1]:>5.0f} s   Z = {Z[0, -1].real:7.2f} {Z[0, -1].imag:+7.2f}j ohm")

# voltage_avg and current_avg are the DC level in each window. They carry no
# impedance information, but a drifting voltage_avg is a direct warning that
# the cell is doing something besides responding to the excitation.
print(f"\n  mean voltage per window: {voltage_avg.min():+.4f} to "
      f"{voltage_avg.max():+.4f} V  (no trend in this circuit)")


# --------------------------------------------------------------------------
# How good is it?
# --------------------------------------------------------------------------
# The simulator gives the true impedance at every sample, so it can be
# resampled onto the instants the estimator reports at.
truth = np.array([np.interp(t, time, Z_true[k].real)
                  + 1j * np.interp(t, time, Z_true[k].imag)
                  for k in range(frequencies.size)])
error = np.abs(Z - truth) / np.abs(truth)

print(f"\n  relative error against the known truth:")
print(f"    median {np.median(error):.2e}, worst {np.max(error):.2e}")
print(f"    worst tone is {frequencies[int(np.argmax(np.median(error, axis=1)))]:.2f} Hz")


# --------------------------------------------------------------------------
# Figure 1: the standard view -- signals, spectra, estimate
# --------------------------------------------------------------------------
fig1 = overview(time, voltage, current, fs, frequencies, Z=Z, t_Z=t,
                Z_true=truth)
finish(fig1, "04_stft_overview", args.save,
       f"STFT with a {window_length:.0f} s window")


# --------------------------------------------------------------------------
# Figure 2: the estimate as a function of time
# --------------------------------------------------------------------------
fig2, axs = plt.subplots(1, 3, figsize=(15, 4.6))

for i, colour in zip((0, 6, 12), ("C0", "C1", "C2")):
    axs[0].plot(time, np.abs(Z_true[i]), color=colour, lw=2, alpha=0.35)
    axs[0].plot(t, np.abs(Z[i]), "o-", ms=4, color=colour,
                label=f"{frequencies[i]:.2f} Hz")
axs[0].set(xlabel="time / s", ylabel=r"$|Z|$ / $\Omega$",
           title="magnitude (pale line: truth)")
axs[0].legend(fontsize=8)

for i, colour in zip((0, 6, 12), ("C0", "C1", "C2")):
    axs[1].plot(time, np.degrees(np.angle(Z_true[i])), color=colour, lw=2,
                alpha=0.35)
    axs[1].plot(t, np.degrees(np.angle(Z[i])), "o-", ms=4, color=colour)
axs[1].set(xlabel="time / s", ylabel=r"$\angle Z$ / deg", title="phase")

axs[2].loglog(frequencies, np.median(error, axis=1), "-o", ms=4, color="C3")
axs[2].set(xlabel="frequency / Hz", ylabel="median relative error",
           title="error against frequency")
axs[2].grid(True, which="both", alpha=0.25)

finish(fig2, "04_stft_result", args.save,
       "what one STFT estimation gives you")

print("\n  STFT has one parameter and makes no assumption about how the system")
print("  varies. It gives no uncertainty on its output, and cannot resolve")
print("  time more finely than one multisine period. Example 10 looks at what")
print("  the window length choice costs.")

show_all(args.save)
