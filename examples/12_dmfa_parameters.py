"""
12 -- DMFA: bandwidth and filter sharpness.

Transform the whole record once; around each excited tone cut out a band with
a passband filter; transform that band back. What comes out is the tone's
complex amplitude against time, and voltage over current is Z(f,t).

Nothing is windowed, so the time resolution is not tied to the multisine
period. It is set by the filter bandwidth instead.

Run:  python examples/12_dmfa_parameters.py [--save] [--no-widgets]
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, fftfreq

from deistools.processing.dmfa_functions import (
    extract_impedance, max_elaboration_points, elaboration_frequency_axis,
    elaboration_time_axis, visualize_dmfa_filtering)
from deistools.processing.passband_filters import create_fermi_dirac_fun
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
f_min = frequencies.min()

sim = simulate_r_rc_transistor(
    T, 1.0 / fs, frequencies, amplitudes, phases,
    R_s=10.0, R_p=30.0, C=1e-3,
    V_gate_start=1.645, V_gate_rate=(2.455 - 1.645) / T, R_T_max=70.0,
    noise_std_V=1e-4, seed=1)

time = sim["time_s"]
Z_true = sim["Z_true"]


def truth_at(times):
    """Ground truth resampled onto the instants an estimator reports at."""
    return np.array([np.interp(times, time, Z_true[k].real)
                     + 1j * np.interp(times, time, Z_true[k].imag)
                     for k in range(frequencies.size)])


# --------------------------------------------------------------------------
# One call, written out in full
# --------------------------------------------------------------------------
# DMFA is assembled from four functions rather than called as one.

ft_voltage = fftshift(fft(sim["voltage_V"])) / N
ft_current = fftshift(fft(sim["current_A"])) / N
spectrum_axis = fftshift(fftfreq(N, 1.0 / fs))
index_f0 = int(np.argmin(np.abs(spectrum_axis)))       # bin of zero frequency
indexes = (np.round(frequencies * T).astype(int) + index_f0).tolist()

bandwidth = 1.0 * f_min      # passband half-width [Hz]
n_points = 160               # DFT bins extracted around each tone

# The filter must be built on the band's LOCAL frequency axis, not on the
# full spectrum axis. The band is n_points bins of width 1/T, so it spans
# n_points/T Hz, while the full axis spans fs.
band_axis = elaboration_frequency_axis(n_points, T)
band_filter = create_fermi_dirac_fun(band_axis, 0, bandwidth, 8)

Z = extract_impedance(ft_voltage, ft_current, indexes, band_filter, n_points)
t = elaboration_time_axis(n_points, T)

print(f"  band spans {n_points / T:.3f} Hz in {n_points} bins of 1/T = {1 / T:.4f} Hz")
print(f"  widest band that fits in the spectrum: "
      f"{max_elaboration_points(indexes, N)} bins")
print(f"  Z {Z.shape}, reported every {t[1] - t[0]:.2f} s")


# --------------------------------------------------------------------------
# A shorthand, so the sweeps below stay readable
# --------------------------------------------------------------------------
def run_dmfa(bw_factor, sharpness):
    """DMFA at a given bandwidth (in units of f_min) and filter sharpness."""
    bw = bw_factor * f_min
    npts = int(round(8 * bw * T))        # band about 8 bandwidths wide
    npts = min(npts - npts % 2, max_elaboration_points(indexes, N))
    axis = elaboration_frequency_axis(npts, T)
    filt = create_fermi_dirac_fun(axis, 0, bw, sharpness)
    return (extract_impedance(ft_voltage, ft_current, indexes, filt, npts),
            elaboration_time_axis(npts, T), filt)


def score(Z_est, t_est):
    """Median and worst relative error, with the record edges discarded.

    The band-limited inverse transform rings at the boundaries, so the first
    and last tenth of a DMFA result is not usable.
    """
    truth = truth_at(t_est)
    err = np.abs(Z_est - truth) / np.abs(truth)
    k = max(1, err.shape[1] // 10)
    core = err[:, k:err.shape[1] - k]
    return np.median(core), np.max(core), np.max(err)


# --------------------------------------------------------------------------
# Bandwidth
# --------------------------------------------------------------------------
# In DFT bins: the record is P periods long, so f_min is exactly P bins. The
# closest pair of tones on this grid sits 2 x f_min apart, which is 2P bins.

bandwidths = [0.5, 1.0, 2.0, 4.0]
results = []

print("\n   bandwidth   instants       median    worst (core)   worst (all)")
for bw_factor in bandwidths:
    Z_b, t_b, _ = run_dmfa(bw_factor, 8)
    median, worst_core, worst_all = score(Z_b, t_b)
    results.append((f"bw = {bw_factor} f_min", Z_b, t_b, truth_at(t_b)))
    print(f"  {bw_factor:>5.1f} f_min   {Z_b.shape[1]:>8}   {median:>10.2e}   "
          f"{worst_core:>13.2e}   {worst_all:>11.2e}")

print("\n  The median hardly moves; the worst case moves by two orders of")
print("  magnitude. A wider band admits more of the neighbouring tone, and")
print("  that appears as a large error at a few instants rather than a small")
print("  error everywhere.")
print("  The last two columns differ only by whether the record edges are")
print("  included -- they are where the band-limited transform rings.")

fig1 = comparison(frequencies, results, n_instants=3)
finish(fig1, "12_dmfa_bandwidth", args.save,
       "DMFA: filter bandwidth against ground truth")


# --------------------------------------------------------------------------
# Sharpness
# --------------------------------------------------------------------------
# n is the Fermi-Dirac exponent: large n approaches a brick wall.

sharpnesses = [2, 4, 8, 16]
worst_grid = np.zeros((len(bandwidths), len(sharpnesses)))

for i, bw_factor in enumerate(bandwidths):
    for j, sharpness in enumerate(sharpnesses):
        Z_b, t_b, _ = run_dmfa(bw_factor, sharpness)
        _, worst_grid[i, j], _ = score(Z_b, t_b)

print("\n  worst relative error, record edges removed:")
print("   bandwidth" + "".join(f"{f'n = {n}':>12}" for n in sharpnesses))
for i, bw_factor in enumerate(bandwidths):
    print(f"  {bw_factor:>5.1f} f_min"
          + "".join(f"{v:>12.2e}" for v in worst_grid[i]))

i_best, j_best = np.unravel_index(np.argmin(worst_grid), worst_grid.shape)
i_worst, j_worst = np.unravel_index(np.argmax(worst_grid), worst_grid.shape)
best_bw, best_n = bandwidths[i_best], sharpnesses[j_best]
worst_bw, worst_n = bandwidths[i_worst], sharpnesses[j_worst]

print(f"\n  best  : bw = {best_bw} f_min, n = {best_n}   "
      f"({worst_grid[i_best, j_best]:.2e})")
print(f"  worst : bw = {worst_bw} f_min, n = {worst_n}   "
      f"({worst_grid[i_worst, j_worst]:.2e})")
print("\n  Both failure modes are in the table, pulling opposite ways:")
print("    wide band + gentle filter   the passband reaches the neighbouring")
print("                                tone, and the leakage dominates")
print("    narrow band + sharp filter  the band is truncated abruptly, so its")
print("                                inverse transform rings in time")
print("  Sharpness has to be matched to the bandwidth, which is why the")
print("  filter has an adjustable n rather than being an ideal passband.")

fig2, axs2 = plt.subplots(1, 2, figsize=(13.5, 5))
local_axis = elaboration_frequency_axis(640, T)
for sharpness in sharpnesses:
    axs2[0].plot(local_axis / f_min,
                 create_fermi_dirac_fun(local_axis, 0, 2.0 * f_min, sharpness),
                 label=f"n = {sharpness}")
axs2[0].axvline(2.0, ls=":", color="C3")
axs2[0].axvline(-2.0, ls=":", color="C3")
axs2[0].annotate("nearest tone", (2.0, 0.55), fontsize=8, color="C3",
                 rotation=90)
axs2[0].set(xlabel=r"offset from the tone / $f_{min}$", ylabel="filter gain",
            title=r"Fermi-Dirac shape at bw = 2 $f_{min}$", xlim=(-6, 6))
axs2[0].legend(fontsize=8)

im = axs2[1].imshow(np.log10(worst_grid), cmap="viridis_r", aspect="auto")
axs2[1].set(xticks=range(len(sharpnesses)),
            xticklabels=[f"n={n}" for n in sharpnesses],
            yticks=range(len(bandwidths)),
            yticklabels=[f"{b}$f_{{min}}$" for b in bandwidths],
            title="worst relative error (edges removed)")
for i in range(len(bandwidths)):
    for j in range(len(sharpnesses)):
        axs2[1].text(j, i, f"{worst_grid[i, j]:.0e}", ha="center", va="center",
                     fontsize=7, color="white")
plt.colorbar(im, ax=axs2[1], label=r"$\log_{10}$ error")
finish(fig2, "12_dmfa_sharpness", args.save,
       "DMFA: the filter, and the bandwidth/sharpness grid")


# --------------------------------------------------------------------------
# The best setting, in full
# --------------------------------------------------------------------------
Z_best, t_best, _ = run_dmfa(best_bw, best_n)
fig3 = overview(time, sim["voltage_V"], sim["current_A"], fs, frequencies,
                Z=Z_best, t_Z=t_best, Z_true=truth_at(t_best))
finish(fig3, "12_dmfa_best", args.save,
       f"DMFA at bw = {best_bw} f_min, n = {best_n}")


# --------------------------------------------------------------------------
# Looking inside the filtering
# --------------------------------------------------------------------------
# A slider over the excited tones, showing the raw band, the filtered band
# and the filter, for voltage and current. The first thing to open when a
# DMFA result looks wrong.

if args.save or args.no_widgets:
    print("\n  visualize_dmfa_filtering(filter, ft_voltage, ft_current,")
    print("                           freq_axis, frequencies, freq_indexes)")
    print("  is an interactive slider figure. Run without --save to open it.")
else:
    for tag, bw_factor, sharpness in (("best", best_bw, best_n),
                                      ("worst", worst_bw, worst_n)):
        _, _, filt = run_dmfa(bw_factor, sharpness)
        print(f"\n  opening the {tag} case (bw = {bw_factor} f_min, "
              f"n = {sharpness}) -- close the window to continue")
        visualize_dmfa_filtering(filt, ft_voltage, ft_current, spectrum_axis,
                                 list(frequencies), indexes)

show_all(args.save)
