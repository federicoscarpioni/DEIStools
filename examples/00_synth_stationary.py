"""
00 -- Simulating a stationary cell.

A time-invariant R_s + (R_p || C) driven by a multisine current. The impedance
is known analytically, so this is the reference case: anything time-dependent
an estimator reports here is its own artefact.

Run:  python examples/00_synth_stationary.py [--save]
"""
import numpy as np
import matplotlib.pyplot as plt

from deistools.simulation.models.r_rc import simulate_r_rc, impedance

from _plotting import parse_args, finish, show_all, overview

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# The excitation
# --------------------------------------------------------------------------
# Odd harmonics on a quasi-logarithmic grid. Log spacing because that is what
# electrochemistry uses; odd harmonics because even-order distortion then
# falls on bins no tone occupies, where it can be seen rather than guessed at.
# The same grid is used by every example in this folder.

Tp = 20.0          # period of the multisine [s]
fs = 1000.0        # sampling rate [Hz]
P = 6              # number of periods measured

harmonics = np.array([1, 3, 5, 9, 13, 17, 27, 39, 59, 89, 133, 201, 301])
frequencies = harmonics / Tp                       # 0.05 ... 15.05 Hz
amplitudes = np.full(frequencies.size, 0.01)       # 10 mA per tone
phases = np.random.default_rng(3).uniform(0, 2 * np.pi, frequencies.size)

N = int(fs * Tp) * P        # total samples
T = N / fs                  # total duration [s]


# --------------------------------------------------------------------------
# The circuit
# --------------------------------------------------------------------------
R_s = 10.0         # series resistance [ohm]
R_p = 30.0         # charge-transfer resistance [ohm]
C = 1e-3           # double-layer capacitance [F]

f_apex = 1.0 / (2 * np.pi * R_p * C)     # top of the Nyquist arc [Hz]


# --------------------------------------------------------------------------
# Simulate
# --------------------------------------------------------------------------
sim = simulate_r_rc(
    T, 1.0 / fs,                       # duration [s], sampling time [s]
    frequencies, amplitudes, phases,   # the multisine
    R_s=R_s, R_p=R_p, C=C,             # the circuit
    noise_std_V=1e-4,                  # measurement noise [V rms]
    seed=1,
)

time = sim["time_s"]
voltage = sim["voltage_V"]
current = sim["current_A"]

# Z_true comes back as (n_freq, 1) here: a stationary system has one
# impedance, not one per sample.
Z_true = sim["Z_true"]

print(f"  {frequencies.size} tones, {frequencies.min():.2f} - {frequencies.max():.2f} Hz")
print(f"  {P} periods of {Tp:.0f} s = {T:.0f} s, {N} samples at {fs:.0f} Hz")
print(f"  tau = R_p*C = {R_p * C * 1e3:.0f} ms, arc apex at {f_apex:.2f} Hz")
print(f"  |Z| = {np.abs(Z_true).min():.1f} - {np.abs(Z_true).max():.1f} ohm")


# --------------------------------------------------------------------------
# Figure 1: the record, its spectra, and the impedance
# --------------------------------------------------------------------------
fig = overview(time, voltage, current, fs, frequencies,
               Z=np.repeat(Z_true, 2, axis=1), t_Z=[time[0], time[-1]])
finish(fig, "00_stationary_overview", args.save,
       "stationary R+(R||C): record, spectra, impedance")


# --------------------------------------------------------------------------
# Figure 2: the analytic impedance, continuous and sampled
# --------------------------------------------------------------------------
f_fine = np.geomspace(frequencies.min() / 2, frequencies.max() * 2, 400)
Z_fine = impedance(f_fine, R_s, R_p, C)[:, 0]
Z0 = Z_true[:, 0]

fig2, axs = plt.subplots(1, 3, figsize=(15, 4.6))

axs[0].plot(Z_fine.real, -Z_fine.imag, "-", color="0.7", lw=1.2,
            label="continuous")
axs[0].plot(Z0.real, -Z0.imag, "o", color="C0", ms=6, label="excited tones")
axs[0].set(xlabel=r"Re$(Z)$ / $\Omega$", ylabel=r"$-$Im$(Z)$ / $\Omega$",
           title="Nyquist")
axs[0].set_aspect("equal", adjustable="datalim")
axs[0].legend(fontsize=8)

axs[1].loglog(f_fine, np.abs(Z_fine), color="0.7", lw=1.2)
axs[1].loglog(frequencies, np.abs(Z0), "o", color="C0", ms=5)
axs[1].set(xlabel="frequency / Hz", ylabel=r"$|Z|$ / $\Omega$",
           title="magnitude")
axs[1].grid(True, which="both", alpha=0.25)

axs[2].semilogx(f_fine, np.degrees(np.angle(Z_fine)), color="0.7", lw=1.2)
axs[2].semilogx(frequencies, np.degrees(np.angle(Z0)), "o", color="C0", ms=5)
axs[2].axvline(f_apex, ls=":", color="C3", label=f"apex {f_apex:.2f} Hz")
axs[2].set(xlabel="frequency / Hz", ylabel=r"$\angle Z$ / deg", title="phase")
axs[2].legend(fontsize=8)
axs[2].grid(True, which="both", alpha=0.25)

finish(fig2, "00_stationary_impedance", args.save,
       "the ground truth, continuous and sampled")

show_all(args.save)
