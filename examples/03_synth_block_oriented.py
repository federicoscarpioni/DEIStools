"""
03 -- Simulating a nonlinear system: the block-oriented model.

Three branches, each a static nonlinearity wrapped around a linear filter,
weighted by the Legendre basis. A nonlinear system has no transfer function,
so the target is the best linear approximation (BLA): the answer a linear
model converges to under this particular excitation, at this amplitude.

Run:  python examples/03_synth_block_oriented.py [--save]
"""
import numpy as np
import matplotlib.pyplot as plt

from deistools.simulation.models.block_oriented import (
    hallemans_2021_system, simulate_block_oriented, best_linear_approximation,
    frozen_impedance)

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
# The system
# --------------------------------------------------------------------------
# Branch p is multiplied by b_p(t), so three branches means Np = 2. Each
# nonlinearity is weak -- a few percent of tanh, or a low-order polynomial --
# which is the regime where a linear approximation is worth making at all.

branches = hallemans_2021_system()

print("  branches:")
for p, branch in enumerate(branches):
    print(f"    {p}  {branch.name:<24} "
          f"f_n = {branch.filter.wn / (2 * np.pi):.1f} Hz, "
          f"zeta = {branch.filter.zeta:.2f}, gain = {branch.filter.dc_gain:.1f}")


# --------------------------------------------------------------------------
# Simulate
# --------------------------------------------------------------------------
sim = simulate_block_oriented(
    T, 1.0 / fs,
    frequencies, amplitudes, phases,
    branches,
    trend_coefficients=None,
    noise_std_V=1e-4,
    seed=1,
)

time = sim["time_s"]
voltage = sim["voltage_V"]
current = sim["current_A"]

# H_linear is the branch filters alone. For a nonlinear system that is NOT
# the quantity an estimator recovers.
H_linear = sim["H_linear"]


# --------------------------------------------------------------------------
# The ground truth: a best linear approximation
# --------------------------------------------------------------------------
# Averaging over random-phase realisations makes the distortion, which is
# uncorrelated with the input, average away while the linear part does not.
# The BLA depends on the nonlinearity and the input amplitude spectrum, not
# on how many periods were measured, so one period is enough to compute it.

G, scatter = best_linear_approximation(
    branches, frequencies, amplitudes, 1.0 / fs, Tp,
    n_realisations=200, seed=7)

Z_true = frozen_impedance(G, N)          # Z(f,t) = sum_p G_p(f) b_p(t)

print(f"\n  BLA known to {np.abs(scatter / G[:, [0]]).max():.0e} relative "
      f"(standard error over 200 realisations)")
print("  how far each branch's BLA sits from its linear part:")
for p in range(G.shape[1]):
    gap = np.abs(G[:, p] - H_linear[:, p]) / np.abs(G[:, p])
    print(f"    branch {p}: {np.median(gap):.1e}")
print("  constant across frequency, because the nonlinearities are static")


# --------------------------------------------------------------------------
# The BLA moves when the drive moves
# --------------------------------------------------------------------------
# On a linear system doubling the excitation changes nothing. Here it changes
# the answer itself, which is why results are comparable only at fixed drive.

print("\n  drive    shift in the BLA relative to the nominal amplitude")
for scale in (0.5, 1.0, 2.0, 4.0):
    G_s, _ = best_linear_approximation(
        branches, frequencies, amplitudes * scale, 1.0 / fs, Tp,
        n_realisations=200, seed=7)
    shift = np.abs(G_s[:, 0] - G[:, 0]) / np.abs(G[:, 0])
    print(f"    x{scale:<5.1f}  {np.median(shift):.2e}")


# --------------------------------------------------------------------------
# Figure 1: the branches
# --------------------------------------------------------------------------
f_fine = np.geomspace(frequencies.min() / 2, frequencies.max() * 2, 500)
probe = np.linspace(-2.0, 2.0, 41)

fig1, axs = plt.subplots(1, 3, figsize=(15, 4.6))

for p, branch in enumerate(branches):
    axs[0].loglog(f_fine, np.abs(branch.filter.response(f_fine)),
                  label=f"$H_{p}$")
axs[0].set(xlabel="frequency / Hz", ylabel=r"$|H_p|$",
           title="the branch filters")
axs[0].legend(fontsize=8)
axs[0].grid(True, which="both", alpha=0.25)

x = np.linspace(-4, 4, 400)
for p, branch in enumerate(branches):
    for fn, style, tag in ((branch.f1, "-", "f1, before filter"),
                           (branch.f2, "--", "f2, after filter")):
        if not np.allclose(fn(probe), probe):        # skip identity blocks
            axs[1].plot(x, fn(x) - x, style, color=f"C{p}", lw=1.2,
                        label=f"branch {p}: {tag}")
axs[1].set(xlabel="input", ylabel="f(x) - x",
           title="the nonlinearities (departure from linear)")
axs[1].legend(fontsize=7)

for p in range(G.shape[1]):
    axs[2].loglog(frequencies, np.abs(G[:, p]), "-o", ms=4, label=f"$G_{p}$")
axs[2].set(xlabel="frequency / Hz", ylabel=r"$|G_p|$",
           title="best linear approximation per branch")
axs[2].legend(fontsize=8)
axs[2].grid(True, which="both", alpha=0.25)

finish(fig1, "03_block_branches", args.save,
       "three nonlinear branches and their best linear approximations")


# --------------------------------------------------------------------------
# Figure 2: the record, its spectra, and the impedance
# --------------------------------------------------------------------------
fig2 = overview(time, voltage, current, fs, frequencies, Z=Z_true, t_Z=time)
finish(fig2, "03_block_overview", args.save,
       "block-oriented system: record, spectra, impedance")


# --------------------------------------------------------------------------
# Figure 3: where the distortion lands
# --------------------------------------------------------------------------
# The grid uses odd harmonics with gaps, so most harmonic bins are unexcited
# and the distortion falls where it can be measured instead of adding
# silently to a tone that matters.

V = np.abs(np.fft.rfft(voltage)) / N
f_axis = np.fft.rfftfreq(N, 1.0 / fs)
excited_bins = np.round(frequencies * T).astype(int)

all_harmonic_bins = np.arange(1, int(harmonics.max() * 1.5)) * P
unexcited = np.setdiff1d(all_harmonic_bins, excited_bins)
odd_bins = unexcited[(unexcited // P) % 2 == 1]
even_bins = unexcited[(unexcited // P) % 2 == 0]

print(f"\n  excited tones          {excited_bins.size}")
print(f"  unexcited odd bins     {odd_bins.size}  "
      f"(median {np.median(V[odd_bins]) / np.median(V[excited_bins]):.1e} of signal)")
print(f"  unexcited even bins    {even_bins.size}  "
      f"(median {np.median(V[even_bins]) / np.median(V[excited_bins]):.1e} of signal)")

fig3, axs3 = plt.subplots(1, 2, figsize=(13, 4.8))

axs3[0].loglog(f_axis[1:], V[1:], color="0.75", lw=0.5)
axs3[0].loglog(f_axis[excited_bins], V[excited_bins], "o", ms=5, color="C3",
               label="excited")
axs3[0].loglog(f_axis[odd_bins], V[odd_bins], "v", ms=3, color="C0",
               label="odd, unexcited")
axs3[0].loglog(f_axis[even_bins], V[even_bins], "^", ms=3, color="C2",
               label="even, unexcited")
axs3[0].set(xlabel="frequency / Hz", ylabel=r"$|V|$ / V",
            title="where the distortion lands")
axs3[0].set_xlim(frequencies.min() * 0.5, frequencies.max() * 1.6)
axs3[0].set_ylim(bottom=V[excited_bins].max() * 1e-8)
axs3[0].legend(fontsize=7)

picks = np.linspace(0, N - 1, 9).astype(int)
for colour, j in zip(time_colours(picks.size), picks):
    axs3[1].plot(Z_true[:, j].real, -Z_true[:, j].imag, "-o", ms=3,
                 color=colour)
axs3[1].set(xlabel=r"Re$(Z)$ / $\Omega$", ylabel=r"$-$Im$(Z)$ / $\Omega$",
            title="the BLA over time, blue to red")
axs3[1].set_aspect("equal", adjustable="datalim")

finish(fig3, "03_block_distortion", args.save,
       "distortion in the spectrum, and the target it leaves behind")

show_all(args.save)
