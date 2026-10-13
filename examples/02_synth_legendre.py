"""
02 -- Simulating a prescribed time variation: the Legendre system.

Here the time-varying impedance is written down directly rather than derived
from a circuit:

    Z(f, t) = sum_p  a_p(f) * b_p(t),    b_p(t) the Legendre polynomials

This is the only system in the package whose time-varying ground truth is
exact, which makes it the right one for measuring an estimator's floor.

Run:  python examples/02_synth_legendre.py [--save]
"""
import numpy as np
import matplotlib.pyplot as plt

from deistools.processing.bltva import legendre_basis
from deistools.simulation.models.legendre import simulate_legendre

from _plotting import parse_args, finish, show_all, overview, time_colours

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# The excitation
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


# --------------------------------------------------------------------------
# The coefficients we prescribe
# --------------------------------------------------------------------------
# a_0 is an ordinary R_s + (R_p || C) arc -- the time-averaged impedance.
# The higher orders are made proportional to it, so the arc stays a
# recognisable arc at every instant while its size changes.

R_s, R_p, C = 10.0, 30.0, 1e-3
order = 3          # highest Legendre order present in the system
decay = 0.35       # how strong the variation is relative to a_0

a = np.zeros((frequencies.size, order + 1), dtype=complex)
a[:, 0] = R_s + R_p / (1 + 1j * 2 * np.pi * frequencies * R_p * C)
for p in range(1, order + 1):
    a[:, p] = (decay / p) * a[:, 0]


# --------------------------------------------------------------------------
# Simulate
# --------------------------------------------------------------------------
sim = simulate_legendre(
    T, 1.0 / fs,
    frequencies, amplitudes, phases,
    a,                           # (n_freq, order+1): this is the whole model
    drift_coefficients=None,     # no additive voltage trend
    noise_std_V=1e-4,
    seed=1,
)

time = sim["time_s"]
voltage = sim["voltage_V"]
current = sim["current_A"]
Z_true = sim["Z_true"]

# The basis the estimator uses, imported from the package so that simulator
# and estimator cannot disagree about a convention. Order 0 is constant;
# every higher order has zero mean, which is what makes a_0 the time average.
basis = legendre_basis(N, order)

print(f"  Legendre expansion of order {order}, {frequencies.size} tones")
print(f"  |a_0| = {np.abs(a[:, 0]).min():.1f} - {np.abs(a[:, 0]).max():.1f} ohm"
      f"   (the time-averaged arc)")
print(f"  |a_1| / |a_0| = {decay:.2f}, falling as 1/p for higher orders")
print(f"  |Z| at the lowest tone runs "
      f"{np.abs(Z_true[0]).min():.1f} - {np.abs(Z_true[0]).max():.1f} ohm")
print(f"  frozen-time synthesis is exact here: "
      f"max|Z_true - a @ b.T| = {np.abs(Z_true - a @ basis.T).max():.1e}")


# --------------------------------------------------------------------------
# Figure 1: the record, its spectra, and the impedance
# --------------------------------------------------------------------------
fig1 = overview(time, voltage, current, fs, frequencies, Z=Z_true, t_Z=time)
finish(fig1, "02_legendre_overview", args.save,
       "prescribed Legendre system: record, spectra, impedance")


# --------------------------------------------------------------------------
# Figure 2: the basis, the impedance over time, and |Z| at three tones
# --------------------------------------------------------------------------
fig2, axs = plt.subplots(1, 3, figsize=(15, 4.6))

for p, colour in zip(range(order + 1),
                     plt.get_cmap("viridis")(np.linspace(0, 0.85, order + 1))):
    axs[0].plot(time, basis[:, p], color=colour, label=f"$b_{p}(t)$")
axs[0].set(xlabel="time / s", ylabel="basis value", title="the Legendre basis")
axs[0].legend(fontsize=8)

picks = np.linspace(0, N - 1, 9).astype(int)
for colour, j in zip(time_colours(picks.size), picks):
    axs[1].plot(Z_true[:, j].real, -Z_true[:, j].imag, "-o", ms=3, color=colour)
axs[1].set(xlabel=r"Re$(Z)$ / $\Omega$", ylabel=r"$-$Im$(Z)$ / $\Omega$",
           title="Z(f,t), blue to red in time")
axs[1].set_aspect("equal", adjustable="datalim")

for i, colour in zip((0, frequencies.size // 2, frequencies.size - 1),
                     ("C0", "C1", "C2")):
    axs[2].plot(time, np.abs(Z_true[i]), color=colour,
                label=f"{frequencies[i]:.2f} Hz")
axs[2].set(xlabel="time / s", ylabel=r"$|Z|$ / $\Omega$",
           title="magnitude against time")
axs[2].legend(fontsize=8)

finish(fig2, "02_legendre_structure", args.save,
       "a variation written down rather than derived")

show_all(args.save)
