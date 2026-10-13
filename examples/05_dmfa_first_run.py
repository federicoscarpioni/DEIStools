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
    elaboration_time_axis, visualize_dmfa_filtering, pad_record)
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

# The band and the reporting rate are different concerns, and Npts_elab sets
# both. The band must be wide enough for the filter to have decayed -- 16
# bandwidths puts the truncation at 1e-9, where 8 leaves 2.6e-03. But the
# result must not be reported more densely than 0.5/bw, or neighbouring
# errors are correlated: 84% at the 0.125/bw these examples used to report,
# against 3% at 0.5/bw. So extract wide, then decimate.
n_points = int(round(16 * bandwidth * T))
n_points = min(n_points - n_points % 2, max_elaboration_points(indexes, N))

band_axis = elaboration_frequency_axis(n_points, T)
band_filter = create_fermi_dirac_fun(band_axis, 0, bandwidth, sharpness)


# --------------------------------------------------------------------------
# Step 4: extract each tone's envelope and divide
# --------------------------------------------------------------------------
Z = extract_impedance(ft_voltage, ft_current, indexes, band_filter, n_points)
t = elaboration_time_axis(n_points, T)

# Decimate to 0.5/bw, the spacing at which neighbouring errors stop
# being correlated (see independent_time_step).
keep = max(1, int(round(n_points / (2 * bandwidth * T))))
Z, t = Z[:, ::keep], t[::keep]


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
print(f"  time step {t[1] - t[0]:.2f} s, which at this bandwidth is the same as")
print(f"  STFT's {Tp:.0f} s window. Nothing was windowed -- the resolution comes")
print("  from the bandwidth -- but once the output is decimated to the spacing")
print("  where neighbouring errors are independent, there is no free lunch:")
print("  buying finer resolution means a wider band, which example 12 shows")
print("  costs accuracy.")


# --------------------------------------------------------------------------
# How good is it?
# --------------------------------------------------------------------------
truth = np.array([np.interp(t, time, Z_true[k].real)
                  + 1j * np.interp(t, time, Z_true[k].imag)
                  for k in range(frequencies.size)])
error = np.abs(Z - truth) / np.abs(truth)

# The band-limited inverse transform rings at the record boundaries, so the
# first and last tenth of a DMFA result is not usable. Discard it.
edge = max(1, Z.shape[1] // 10)       # of the REPORTED instants
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
for j in (edge, Z.shape[1] - edge - 1):
    print(f"    t = {t[j]:>5.0f} s   Z = {Z[0, j].real:7.2f} "
          f"{Z[0, j].imag:+7.2f}j ohm     true {truth[0, j].real:7.2f} "
          f"{truth[0, j].imag:+7.2f}j")


# --------------------------------------------------------------------------
# The edges are wrap-around, and zero-padding removes most of it
# --------------------------------------------------------------------------
# Multiplying a spectrum by a filter convolves circularly in time: the kernel
# that runs past the end of the record wraps round and picks up the start.
# Here |Z| at the lowest tone goes from about 106 to 43 ohm across the record,
# so each edge is contaminated by a value 2.4x away. Padding before the
# transform makes the convolution linear, and the kernel runs into zeros.

v_pad, i_pad = pad_record(voltage), pad_record(current)
N_pad = v_pad.size
T_pad = N_pad / fs

ft_v_pad = fftshift(fft(v_pad)) / N_pad
ft_i_pad = fftshift(fft(i_pad)) / N_pad
axis_pad = fftshift(fftfreq(N_pad, 1.0 / fs))
indexes_pad = (np.round(frequencies * T_pad).astype(int)
               + int(np.argmin(np.abs(axis_pad)))).tolist()

n_pad_points = int(round(16 * bandwidth * T_pad))
n_pad_points = min(n_pad_points - n_pad_points % 2,
                   max_elaboration_points(indexes_pad, N_pad))
filter_pad = create_fermi_dirac_fun(
    elaboration_frequency_axis(n_pad_points, T_pad), 0, bandwidth, sharpness)

Z_pad = extract_impedance(ft_v_pad, ft_i_pad, indexes_pad, filter_pad,
                          n_pad_points)
t_pad = elaboration_time_axis(n_pad_points, T_pad)
inside = t_pad <= T * 1.0000001                 # drop the padded region
Z_pad, t_pad = Z_pad[:, inside], t_pad[inside]
keep_pad = max(1, int(round(Z_pad.shape[1] / (2 * bandwidth * T))))
Z_pad, t_pad = Z_pad[:, ::keep_pad], t_pad[::keep_pad]

truth_pad = np.array([np.interp(t_pad, time, Z_true[k].real)
                      + 1j * np.interp(t_pad, time, Z_true[k].imag)
                      for k in range(frequencies.size)])
error_pad = np.abs(Z_pad - truth_pad) / np.abs(truth_pad)

print()
print("                     plain        zero-padded")
print(f"  first instant   {np.median(error[:, 0]):.2e}           {np.median(error_pad[:, 0]):.2e}")
print(f"  last instant    {np.median(error[:, -1]):.2e}           {np.median(error_pad[:, -1]):.2e}")
print(f"  worst overall   {np.max(error):.2e}           {np.max(error_pad):.2e}")
print(f"  median          {np.median(error):.2e}           {np.median(error_pad):.2e}")
blend_all = (Z_true[:, 0] + Z_true[:, -1]) / 2
print(f"""
  Why the first instant was 2.6e-01 wrong: multiplying a spectrum by a filter
  convolves CIRCULARLY in time, so the half of the kernel that reaches before
  t = 0 wraps round and lands on the end of the record. The estimate there is
  therefore a blend of the two ends, and it is: across the 13 tones it sits a
  median {np.median(np.abs(Z[:, 0] - blend_all) / np.abs(blend_all)):.1%} from that mean, against {np.median(np.abs(Z[:, 0] - Z_true[:, 0]) / np.abs(Z_true[:, 0])):.0%} from the truth.

  The lowest tone, which has the most skirt and so deviates most, still shows
  the mechanism plainly:

      true Z(0)                {np.abs(Z_true[0, 0]):.1f} ohm
      true Z(T)                {np.abs(Z_true[0, -1]):.1f} ohm
      their mean               {np.abs(blend_all[0]):.1f} ohm
      what plain DMFA reports  {np.abs(Z[0, 0]):.1f} ohm   <- not near 106, near 75

  The worst case improves by more than an order of magnitude and the median
  does not move at all -- as it must not, since the kernel never reaches the
  middle of the record.

  It works better than "assume the signal is zero outside" ought to, because
  the padding attenuates the numerator and the denominator by the same
  factor: the filtered current falls to about half its mid-record value at
  the first instant, and the filtered voltage falls with it. The ratio
  survives what either one alone would not.""")


# --------------------------------------------------------------------------
# Figure: what wrap-around does, and what padding does about it
# --------------------------------------------------------------------------
# Three panels, left to right: the filter kernel against the record it has to
# fit inside; what the unpadded estimate picks up at t = 0; and the error
# against position with and without padding.

kernel = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(band_filter))))
kernel /= kernel.max()
kernel_t = (np.arange(n_points) - n_points // 2) * (T / n_points)
reach = np.abs(kernel_t[kernel > 0.01]).max()     # where it has died to 1%

tone = 0                                     # the lowest tone, worst affected
blend = (Z_true[tone, 0] + Z_true[tone, -1]) / 2

fig_pad, axp = plt.subplots(1, 3, figsize=(15.5, 4.6))

axp[0].plot(kernel_t, kernel, color="C0")
axp[0].axvspan(kernel_t.min(), 0, color="C3", alpha=0.15)
axp[0].annotate("reaches before" + chr(10) + "the record starts",
                (kernel_t.min() * 0.75, 0.55),
                fontsize=8, color="C3", ha="center")
axp[0].set(xlabel="time relative to the estimate / s", ylabel="kernel",
           title=f"the kernel reaches about {reach:.0f} s either side",
           xlim=(-60, 60))

axp[1].plot(time, np.abs(Z_true[tone]), color="black", lw=2, alpha=0.45,
            label="truth")
axp[1].plot(t, np.abs(Z[tone]), "-o", ms=4, color="C3", label="plain DMFA")
axp[1].plot(t_pad, np.abs(Z_pad[tone]), "-o", ms=4, color="C2",
            label="zero-padded")
axp[1].axhline(np.abs(blend), ls="--", color="C1", lw=1.2,
               label=r"$(Z(0)+Z(T))/2$")
axp[1].set(xlabel="time / s", ylabel=r"$|Z|$ / $\Omega$",
           title=f"at {frequencies[tone]:.2f} Hz: the wrap lands on the mean",
           xlim=(-10, 120))
axp[1].legend(fontsize=8)

axp[2].semilogy(t / T, np.median(error, axis=0), "-o", ms=3, color="C3",
                label="plain")
axp[2].semilogy(t_pad / T, np.median(error_pad, axis=0), "-o", ms=3,
                color="C2", label="zero-padded")
axp[2].set(xlabel="position in the record", ylabel="median relative error",
           title="the interior is untouched")
axp[2].grid(True, which="both", alpha=0.25)
axp[2].legend(fontsize=8)

finish(fig_pad, "05_dmfa_padding", args.save,
       "circular wrap-around, and the one-line fix")


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
