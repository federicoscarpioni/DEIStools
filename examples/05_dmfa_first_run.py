"""
05 -- A first DMFA estimation.

One estimation of the transistor-gated cell with the settings example 12
arrives at, to show how the four functions fit together and what they give
back -- including the interactive view of the filtering.

Run:  python examples/05_dmfa_first_run.py [--save] [--no-widgets]
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, fftfreq

from deistools.processing.dmfa_functions import (
    extract_impedance, max_elaboration_points, elaboration_frequency_axis,
    elaboration_time_axis, visualize_dmfa_filtering)
from deistools.processing.passband_filters import create_fermi_dirac_fun
from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor

from _plotting import parse_args, finish, show_all, overview

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# The measurement to analyse (see example 01)
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


# --------------------------------------------------------------------------
# Step 1: transform the whole record, once
# --------------------------------------------------------------------------
# DMFA never cuts the record. It works on the spectrum of the whole thing,
# shifted so that zero frequency sits in the middle.

ft_voltage = fftshift(fft(voltage)) / N
ft_current = fftshift(fft(current)) / N
spectrum_axis = fftshift(fftfreq(N, 1.0 / fs))


# --------------------------------------------------------------------------
# Step 2: find the bin of each excited tone
# --------------------------------------------------------------------------
# A tone at f Hz sits f*T bins from zero frequency, and zero frequency is at
# index N//2 after the shift.

index_f0 = int(np.argmin(np.abs(spectrum_axis)))
indexes = (np.round(frequencies * T).astype(int) + index_f0).tolist()


# --------------------------------------------------------------------------
# Step 3: build the passband filter
# --------------------------------------------------------------------------
# The filter is defined on the band's LOCAL frequency axis, which spans
# n_points bins of width 1/T. Building it on the full spectrum axis instead
# is the one easy mistake -- its bandwidth would be wrong by fs*T/n_points.

bandwidth = 0.5 * f_min       # passband half-width [Hz]
sharpness = 2                 # Fermi-Dirac exponent; large n -> brick wall

n_points = int(round(8 * bandwidth * T))        # band ~8 bandwidths wide
n_points = min(n_points - n_points % 2, max_elaboration_points(indexes, N))

band_axis = elaboration_frequency_axis(n_points, T)
band_filter = create_fermi_dirac_fun(band_axis, 0, bandwidth, sharpness)


# --------------------------------------------------------------------------
# Step 4: extract each tone's envelope and divide
# --------------------------------------------------------------------------
Z = extract_impedance(ft_voltage, ft_current, indexes, band_filter, n_points)
t = elaboration_time_axis(n_points, T)


# --------------------------------------------------------------------------
# What came back
# --------------------------------------------------------------------------
print("  the four steps give:")
print(f"    ft_voltage    {str(ft_voltage.shape):<12} spectrum of the whole record")
print(f"    indexes       {len(indexes):<12} bin of each excited tone")
print(f"    band_filter   {str(band_filter.shape):<12} the passband, on the local axis")
print(f"    Z             {str(Z.shape):<12} complex, (n_freq, n_points)")
print(f"    t             {str(t.shape):<12} the instants it reports at [s]")

print(f"\n  bandwidth {bandwidth:.4f} Hz = {bandwidth / f_min:.1f} x f_min, sharpness n = {sharpness}")
print(f"  band is {n_points} bins of 1/T = {1 / T:.4f} Hz, spanning {n_points / T:.3f} Hz")
print(f"  time step {t[1] - t[0]:.2f} s -- finer than STFT's {Tp:.0f} s, because")
print("  nothing was windowed: the resolution comes from the bandwidth")


# --------------------------------------------------------------------------
# How good is it?
# --------------------------------------------------------------------------
truth = np.array([np.interp(t, time, Z_true[k].real)
                  + 1j * np.interp(t, time, Z_true[k].imag)
                  for k in range(frequencies.size)])
error = np.abs(Z - truth) / np.abs(truth)

# The band-limited inverse transform rings at the record boundaries, so the
# first and last tenth of a DMFA result is not usable. Discard it.
edge = n_points // 10
core = error[:, edge:-edge]
print()
print("  relative error against the known truth:")
print(f"    median {np.median(core):.2e}, worst {np.max(core):.2e}   "
      f"(record edges discarded)")
print(f"    worst {np.max(error):.2e} if the edges are kept -- "
      f"{np.max(error) / np.max(core):.0f}x larger")

print()
print(f"  the impedance at the first and last USABLE instant, lowest "
      f"tone ({frequencies[0]:.2f} Hz):")
for j in (edge, n_points - edge - 1):
    print(f"    t = {t[j]:>5.0f} s   Z = {Z[0, j].real:7.2f} "
          f"{Z[0, j].imag:+7.2f}j ohm     true {truth[0, j].real:7.2f} "
          f"{truth[0, j].imag:+7.2f}j")


# --------------------------------------------------------------------------
# Figure 1: the standard view -- signals, spectra, estimate
# --------------------------------------------------------------------------
fig1 = overview(time, voltage, current, fs, frequencies,
                Z=Z[:, edge:-edge], t_Z=t[edge:-edge],
                Z_true=truth[:, edge:-edge])
finish(fig1, "05_dmfa_overview", args.save,
       f"DMFA at bw = {bandwidth / f_min:.1f} f_min, n = {sharpness}")


# --------------------------------------------------------------------------
# Figure 2: what the filter does to one band
# --------------------------------------------------------------------------
# A still version of what visualize_dmfa_filtering shows with a slider.

tone = 3            # which excited tone to look at
band = range(indexes[tone] - n_points // 2, indexes[tone] + n_points // 2)
offset = spectrum_axis[band] - frequencies[tone]

fig2, axs = plt.subplots(1, 3, figsize=(15, 4.6))

axs[0].semilogy(offset, np.abs(ft_voltage)[band], color="C0", alpha=0.35,
                label="raw band")
axs[0].semilogy(offset, np.abs(ft_voltage)[band] * band_filter, color="C0",
                label="after the filter")
twin = axs[0].twinx()
twin.plot(offset, band_filter, color="orangered", lw=1)
twin.set_ylabel("filter gain", color="orangered")
twin.set_ylim(0, 1.05)
axs[0].set(xlabel=f"offset from {frequencies[tone]:.2f} Hz / Hz",
           ylabel=r"$|V|$ / V", title="voltage band")
axs[0].legend(fontsize=8, loc="lower left")

axs[1].semilogy(offset, np.abs(ft_current)[band], color="C1", alpha=0.35,
                label="raw band")
axs[1].semilogy(offset, np.abs(ft_current)[band] * band_filter, color="C1",
                label="after the filter")
axs[1].set(xlabel=f"offset from {frequencies[tone]:.2f} Hz / Hz",
           ylabel=r"$|I|$ / A", title="current band")
axs[1].legend(fontsize=8, loc="lower left")

axs[2].plot(time, np.abs(Z_true[tone]), color="black", lw=2, alpha=0.4,
            label="truth")
axs[2].plot(t, np.abs(Z[tone]), "-", color="C2", label="estimate")
axs[2].axvspan(t[0], t[edge], color="C3", alpha=0.15)
axs[2].axvspan(t[-edge], t[-1], color="C3", alpha=0.15,
               label="unusable edges")
axs[2].set(xlabel="time / s", ylabel=r"$|Z|$ / $\Omega$",
           title=f"the envelope that comes out ({frequencies[tone]:.2f} Hz)")
axs[2].legend(fontsize=8)

finish(fig2, "05_dmfa_band", args.save,
       "one band: before the filter, after it, and the result")


# --------------------------------------------------------------------------
# The interactive view
# --------------------------------------------------------------------------
# A slider pages through the excited tones, showing the raw band, the
# filtered band and the filter, for voltage and current. The first thing to
# open when a DMFA result looks wrong.

if args.save or args.no_widgets:
    print("\n  visualize_dmfa_filtering(band_filter, ft_voltage, ft_current,")
    print("                           spectrum_axis, frequencies, indexes)")
    print("  is an interactive slider figure. Run without --save to open it.")
else:
    print("\n  opening visualize_dmfa_filtering -- close the window to finish")
    visualize_dmfa_filtering(band_filter, ft_voltage, ft_current,
                             spectrum_axis, list(frequencies), indexes)

show_all(args.save)
