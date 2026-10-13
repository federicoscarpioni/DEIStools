"""
14 -- BLTVA: Np, Nq and the band width.

BLTVA assumes a form for the variation,

    Z(f, t) = sum_p  H_p(f) * b_p(t)

with b_p the Legendre polynomials, and fits the coefficients H_p directly in
the frequency domain. Instead of cutting the record or filtering a band it
uses the skirts -- the spread that time variation puts around each tone.

    Np     order of the Legendre expansion
    Nq     order of the trend polynomial at DC
    band   DFT bins used either side of each tone

Run:  python examples/14_bltva_parameters.py [--save] [--no-widgets]
"""
import numpy as np
import matplotlib.pyplot as plt

from deistools.processing.bltva import (
    bltva, legendre_basis, select_na, check_identifiability,
    visualize_bltva_fit)
from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor

from _plotting import parse_args, finish, show_all, overview, comparison

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

sim = simulate_r_rc_transistor(
    T, 1.0 / fs, frequencies, amplitudes, phases,
    R_s=10.0, R_p=30.0, C=1e-3,
    V_gate_start=1.645, V_gate_rate=(2.455 - 1.645) / T, R_T_max=70.0,
    noise_std_V=1e-4, seed=1)

time = sim["time_s"]
Z_true = sim["Z_true"]


# --------------------------------------------------------------------------
# One call, written out
# --------------------------------------------------------------------------
# Note that bltva takes HARMONIC NUMBERS, not frequencies, and the number of
# periods P. Na defaults to None, which applies the package's closed-form
# rule; there is no reason to choose it by hand.

result = bltva(
    sim["voltage_V"], sim["current_A"], fs, P, harmonics,
    Np=4,                      # order of the Legendre expansion
    Nq=2,                      # order of the trend polynomial
    Na=None,                   # transient order: chosen automatically
    harmonics="excited",       # fit only the excited bins
    band_halfwidth=100,        # DFT bins either side of each tone
)

print(f"  H_p {result.H_p.shape}   -- (n_freq, Np+1) coefficients, not Z(f,t)")
print(f"  H_p[:, 0] is the time-averaged impedance: "
      f"{np.abs(result.H_p[:, 0]).min():.1f} - "
      f"{np.abs(result.H_p[:, 0]).max():.1f} ohm")
print(f"  Na chosen automatically: {select_na(P, 4)}")
print(f"  worst condition number {result.worst_condition():.1e}, "
      f"worst band residual {result.worst_residual():.1e}")

# Z(f,t) is reconstructed from the coefficients. Evaluating at 200 instants
# rather than all N samples loses nothing -- it is a smooth polynomial.
instants = np.linspace(0, N - 1, 200).astype(int)
Z = result.H_p @ legendre_basis(N, 4)[instants].T
t = time[instants]
print(f"  Z = H_p @ legendre_basis(N, Np).T  ->  {Z.shape}")


# --------------------------------------------------------------------------
# A shorthand, so the sweeps below stay readable
# --------------------------------------------------------------------------
def run_bltva(voltage, Np, Nq, band_halfwidth):
    """Fit, and return Z(f,t) at 200 instants plus the full result object."""
    res = bltva(voltage, sim["current_A"], fs, P, harmonics,
                Np=Np, Nq=Nq, Na=None, harmonics="excited",
                band_halfwidth=band_halfwidth)
    idx = np.linspace(0, N - 1, 200).astype(int)
    return res.H_p @ legendre_basis(N, Np)[idx].T, time[idx], res


def truth_at(times):
    """Ground truth resampled onto the instants the estimator reports at."""
    return np.array([np.interp(times, time, Z_true[k].real)
                     + 1j * np.interp(times, time, Z_true[k].imag)
                     for k in range(frequencies.size)])


def error_of(Z_est, t_est):
    truth = truth_at(t_est)
    return np.abs(Z_est - truth) / np.abs(truth)


# --------------------------------------------------------------------------
# Np -- the order of the expansion
# --------------------------------------------------------------------------
# The three orders are not independent. eq. (49) of the 2021 paper bounds
# them jointly, and check_identifiability enforces it.

np_values = [1, 2, 3, 4, 6, 8]
results = []

print("\n    Np   Na   parameters       median        worst")
for Np in np_values:
    Na = select_na(P, Np)
    n_parameters = check_identifiability(P, Np, 2, Na)   # raises if too many
    Z_p, t_p, _ = run_bltva(sim["voltage_V"], Np, 2, 100)
    err = error_of(Z_p, t_p)
    results.append((f"Np = {Np}", Z_p, t_p, truth_at(t_p)))
    print(f"  {Np:>4}   {Na:>2}   {n_parameters:>10}   "
          f"{np.median(err):>10.2e}   {np.max(err):>10.2e}")

print("\n  The error falls by almost two orders of magnitude up to Np = 4 and")
print("  then flattens. R_ct(t) is a straight ramp, but Z depends on it")
print("  through R_ct/(1 + jwR_ct C), so Z(f,t) is a smooth rational function")
print("  of time that a cubic or quartic captures and a straight line cannot.")
print("  Over-specifying is cheap; under-specifying is a hard floor.")

fig1 = comparison(frequencies, results[:4], n_instants=3)
finish(fig1, "14_bltva_np", args.save,
       "BLTVA: the order of the Legendre expansion")


# --------------------------------------------------------------------------
# Nq -- and why it needs a trend to do anything
# --------------------------------------------------------------------------
print("\n    Nq       median        worst     (no trend in the record)")
for Nq in [0, 1, 2, 3, 4, 6]:
    Z_q, t_q, _ = run_bltva(sim["voltage_V"], 4, Nq, 100)
    err = error_of(Z_q, t_q)
    print(f"  {Nq:>4}   {np.median(err):>10.2e}   {np.max(err):>10.2e}")

print("\n  Nothing moves, and that is correct: this circuit has no trend. The")
print("  gate ramp changes the impedance, which is signal, and leaves the DC")
print("  level of the voltage alone.")

# A real cell charges, ages or relaxes while it is measured, and the voltage
# wanders for reasons that are not the impedance. Add that disturbance -- an
# order-3 polynomial, 5 V peak to peak -- and leave Z_true untouched.
trend_order = 3
trend = legendre_basis(N, trend_order) @ np.array([0.0, 2.0, 1.0, 0.5])
voltage_with_trend = sim["voltage_V"] + trend

print(f"\n  adding an order-{trend_order} trend of {np.ptp(trend):.1f} V peak to peak "
      f"(signal is {sim['voltage_V'].std():.2f} V rms)")
print("\n    Nq       median        worst")
trend_results = []
for Nq in [0, 1, 2, 3, 4, 6]:
    Z_q, t_q, _ = run_bltva(voltage_with_trend, 4, Nq, 100)
    err = error_of(Z_q, t_q)
    trend_results.append((f"Nq = {Nq}", Z_q, t_q, truth_at(t_q)))
    print(f"  {Nq:>4}   {np.median(err):>10.2e}   {np.max(err):>10.2e}")

print(f"\n  Now it matters, and in the worst case rather than the median: a")
print(f"  trend is a disturbance near DC, so it corrupts the lowest tones and")
print(f"  leaves the rest alone. It saturates exactly at Nq = {trend_order}, the order of")
print("  the trend that was added.")

fig2 = comparison(frequencies, trend_results[:4], n_instants=3)
finish(fig2, "14_bltva_nq", args.save,
       "BLTVA: the trend order, on a record that has a trend")


# --------------------------------------------------------------------------
# Band width, and its interaction with Np
# --------------------------------------------------------------------------
# Bands are meant to overlap. The closest pair of tones is 2P = 40 bins
# apart, and a skirt whose centre lies inside the band is fitted explicitly,
# while one outside contributes only a tail that is nearly indistinguishable
# from the transient polynomial.

bands = [50, 100, 200, 400]
np_grid = [3, 4, 6, 8]
worst_grid = np.zeros((len(np_grid), len(bands)))

print("\n  worst relative error")
print("    Np" + "".join(f"{f'band {b}':>12}" for b in bands))
for i, Np in enumerate(np_grid):
    for j, band in enumerate(bands):
        Z_b, t_b, _ = run_bltva(sim["voltage_V"], Np, 2, band)
        worst_grid[i, j] = np.max(error_of(Z_b, t_b))
    print(f"  {Np:>4}" + "".join(f"{v:>12.2e}" for v in worst_grid[i]))

i_best, j_best = np.unravel_index(np.argmin(worst_grid), worst_grid.shape)
best_np, best_band = np_grid[i_best], bands[j_best]

print("\n  Read along the rows: the sign of the effect flips. At low Np a wider")
print("  band helps; at Np = 8 it hurts badly. The two parameters cannot be")
print("  tuned separately.")

# The diagnostics say why.
print("\n   band   bins   skirt centres   parameters   pruned        residual")
for band in bands:
    _, _, res = run_bltva(sim["voltage_V"], 8, 2, band)
    d = res.diagnostics[0]            # the lowest tone's band
    print(f"  {band:>5}   {d['N_w']:>4}   {d['n_centres']:>13}   "
          f"{d['n_theta_full']:>10}   {d['n_unsupported_dropped']:>6}   "
          f"{d['rel_residual']:>13.2e}")

print("\n  More skirt centres fall inside a wider band, and each brings its own")
print("  block of Np+1 regressors. Meanwhile Na is spent out of the same")
print(f"  budget as Np -- {select_na(P, 3)} at Np = 3, only {select_na(P, 8)} at Np = 8 -- and it is Na")
print("  that has to describe the smooth background across the whole band.")
print("  Over 70 bins an order-8 polynomial is ample; over 420 it is not, and")
print("  the residual rises accordingly.")
print("\n  So: raise Np and you must narrow the band.")

fig3, ax3 = plt.subplots(figsize=(7.5, 5))
im = ax3.imshow(np.log10(worst_grid), cmap="viridis_r", aspect="auto")
ax3.set(xticks=range(len(bands)), xticklabels=[str(b) for b in bands],
        yticks=range(len(np_grid)), yticklabels=[f"Np={n}" for n in np_grid],
        xlabel="band halfwidth / bins",
        title="worst relative error over the joint grid")
for i in range(len(np_grid)):
    for j in range(len(bands)):
        ax3.text(j, i, f"{worst_grid[i, j]:.0e}", ha="center", va="center",
                 fontsize=8, color="white")
ax3.plot(j_best, i_best, "*", ms=22, color="red", zorder=5)
plt.colorbar(im, ax=ax3, label=r"$\log_{10}$ worst error")
finish(fig3, "14_bltva_joint", args.save,
       "BLTVA: Np and band width are not independent")


# --------------------------------------------------------------------------
# The best setting, in full
# --------------------------------------------------------------------------
Z_best, t_best, res_best = run_bltva(sim["voltage_V"], best_np, 2, best_band)
err_best = error_of(Z_best, t_best)

print(f"\n  Np = {best_np}, Nq = 2, band = {best_band}, Na = {select_na(P, best_np)}")
print(f"  median {np.median(err_best):.2e}, worst {np.max(err_best):.2e}")

# Alone among the three estimators, BLTVA reports an uncertainty: the noise
# is estimated from bins the excitation never touched and propagated to the
# coefficients.
print("\n   f / Hz        |H_0|     std(H_0)   relative")
for i in (0, 6, 12):
    std = res_best.std_H_p[i, 0]
    magnitude = np.abs(res_best.H_p[i, 0])
    print(f"  {frequencies[i]:>7.2f}   {magnitude:>10.3f}   {std:>10.2e}   "
          f"{std / magnitude:>8.1e}")

fig4 = overview(time, sim["voltage_V"], sim["current_A"], fs, frequencies,
                Z=Z_best, t_Z=t_best, Z_true=truth_at(t_best))
finish(fig4, "14_bltva_best", args.save,
       f"BLTVA at Np = {best_np}, band = {best_band}")


# --------------------------------------------------------------------------
# Looking inside a band
# --------------------------------------------------------------------------
# visualize_bltva_fit takes the same arguments as bltva. For each tone it
# shows the band data against the fitted model, and the coefficients that
# become H_p. A good fit has theta_0 dominant and the higher orders falling
# away; coefficients that grow with order mean the band is being fitted by
# cancellation between large terms.

if args.save or args.no_widgets:
    print("\n  visualize_bltva_fit(response, excitation, fs, P, H_exc, ...)")
    print("  is an interactive slider figure. Run without --save to open it.")
else:
    for tag, Np_, band_ in (("good", best_np, best_band),
                            ("over-specified", 8, 400)):
        print(f"\n  opening the {tag} case (Np = {Np_}, band = {band_}) "
              f"-- close the window to continue")
        visualize_bltva_fit(sim["voltage_V"], sim["current_A"], fs, P,
                            harmonics, Np=Np_, Nq=2, harmonics="excited",
                            band_halfwidth=band_)

show_all(args.save)
