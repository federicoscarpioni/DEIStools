"""
09 -- Trend removal, and what each strategy costs.

A real cell drifts. The drift is not what the measurement is for, so the
instinct is to filter it out first. This script shows what the four
strategies actually do -- to the signal, to its spectrum, and to the
estimate -- and why the instinct needs checking.

The record here has 8 periods rather than the 20 used in examples 04-08.
Fewer periods puts the lowest tone at bin 8 instead of bin 20, much closer to
the trend's skirt, which is the situation detrending exists for.

Run:  python examples/09_detrending.py [--save]
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, fftfreq

from deistools.processing.detrending import (
    Differencing, MovingAverage, LowPassRemoval, PolynomialFit)
from deistools.processing.dmfa_functions import (
    extract_impedance, max_elaboration_points, elaboration_frequency_axis,
    elaboration_time_axis)
from deistools.processing.passband_filters import create_fermi_dirac_fun
from deistools.processing.bltva import bltva
from deistools.simulation.models.r_rc import simulate_r_rc
from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor

from _plotting import parse_args, finish, show_all, envelope

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# The measurement, and a drift on top of it
# --------------------------------------------------------------------------
Tp = 20.0
fs = 1000.0
P = 8

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
current = sim["current_A"]
clean = sim["voltage_V"].copy()
Z_true = sim["Z_true"]

x = time / T
trend = 48.0 * (0.5 * x ** 3 - 1.1 * x ** 2 + 0.7 * x)
trend = trend - trend.mean()
trended = clean + trend

print(f"  signal {clean.std():.2f} V rms, drift {np.ptp(trend):.1f} V p-p "
      f"over {T:.0f} s")


# --------------------------------------------------------------------------
# Where the trend lives, and why that is the whole problem
# --------------------------------------------------------------------------
# A drift that unfolds over the whole record has its energy in the lowest few
# DFT bins -- which is exactly where the slowest tones are.

k_exc = harmonics * P
V_clean = fft(clean) / N
V_trend = fft(trend) / N

print(f"\n  trend-to-signal ratio at the lowest excited bins:")
for j in range(5):
    ratio = np.abs(V_trend[k_exc[j]]) / np.abs(V_clean[k_exc[j]])
    flag = "  <- swamped" if ratio > 0.2 else ""
    print(f"    bin {k_exc[j]:>3}  ({frequencies[j]:>5.2f} Hz)   {ratio:>7.3f}{flag}")

print("\n  The trend and the signal share those bins. No filter can separate")
print("  them there: a filter multiplies bin k by F(k), so both are scaled by")
print("  the same number and the ratio above is unchanged -- which the last")
print("  section of this script measures.")


# --------------------------------------------------------------------------
# The four strategies
# --------------------------------------------------------------------------
strategies = [
    ("Differencing(1)",      Differencing(1)),
    ("MovingAverage(20001)", MovingAverage(20_001)),
    ("LowPassRemoval(2, 2)", LowPassRemoval(2.0, 2)),
    ("PolynomialFit(3)",     PolynomialFit(3)),
]

detrended = {name: s.apply(trended) for name, s in strategies}

# C3 is reserved for the drifted record in both figures
colours = ["C0", "C1", "C2", "C4"]


# --------------------------------------------------------------------------
# What each one removes from the excited bins
# --------------------------------------------------------------------------
# To measure damage you need a record where nothing slow is real, so the
# control is the *stationary* cell of example 00, not the transistor: its
# impedance does not move, so anything a strategy changes at an excited bin
# is signal it destroyed.

stationary = simulate_r_rc(T, 1.0 / fs, frequencies, amplitudes, phases,
                           R_s=10.0, R_p=30.0, C=1e-3, noise_std_V=1e-4,
                           seed=1)["voltage_V"]
V_reference = fft(stationary) / N


def excited_bin_change(strategy):
    V_after = fft(strategy.apply(stationary)) / N
    return (np.abs(V_after[k_exc] - V_reference[k_exc])
            / np.abs(V_reference[k_exc]))


print(f"\n  change at the excited bins of a stationary record:")
print(f"    {'strategy':<22} {'lowest tone':>12} {'top tone':>12}")
damage = {}
for name, strategy in strategies:
    change = excited_bin_change(strategy)
    damage[name] = change
    print(f"    {name:<22} {change[0]:>12.1e} {change[-1]:>12.1e}")

print(f"\n  and how it grows with the degree of the polynomial:")
for degree in (1, 3, 8, 12):
    change = excited_bin_change(PolynomialFit(degree))
    print(f"    PolynomialFit({degree:>2})          {change[0]:>12.1e} "
          f"{change[-1]:>12.1e}")

print("\n  Differencing is the outlier, and not because it is careless: it")
print("  attenuates the low bins deliberately, and BLTVA divides that back")
print("  out. Nothing else does, so differencing must never reach DMFA or")
print("  STFT. The polynomial fit is the one that surprises people -- a")
print("  polynomial is not band-limited, so subtracting one takes signal from")
print("  every bin, more than either filter does.")


# --------------------------------------------------------------------------
# The effect on an actual estimate (DMFA)
# --------------------------------------------------------------------------
bandwidth = 0.5 * frequencies.min()
spectrum_axis = fftshift(fftfreq(N, 1.0 / fs))
indexes = (np.round(frequencies * T).astype(int)
           + int(np.argmin(np.abs(spectrum_axis)))).tolist()
n_points = int(round(16 * bandwidth * T))
n_points = min(n_points - n_points % 2, max_elaboration_points(indexes, N))
band_filter = create_fermi_dirac_fun(
    elaboration_frequency_axis(n_points, T), 0, bandwidth, 8)
t_dmfa = elaboration_time_axis(n_points, T)
edge = n_points // 10
ft_current = fftshift(fft(current)) / N

truth = np.array([np.interp(t_dmfa[edge:-edge], time, Z_true[k].real)
                  + 1j * np.interp(t_dmfa[edge:-edge], time, Z_true[k].imag)
                  for k in range(frequencies.size)])


def dmfa_error(voltage):
    """Median relative error at the lowest tone, edges discarded."""
    Z = extract_impedance(fftshift(fft(voltage)) / N, ft_current, indexes,
                          band_filter, n_points)[:, edge:-edge]
    return np.median(np.abs(Z - truth) / np.abs(truth), axis=1)


print(f"\n  DMFA at the lowest tone:")
print(f"    {'strategy':<22} {'trended':>10} {'trend-free':>12}")
print(f"    {'(none)':<22} {dmfa_error(trended)[0]:>10.2e} "
      f"{dmfa_error(clean)[0]:>12.2e}")
dmfa_scores = {}
for name, strategy in strategies:
    with_trend = dmfa_error(strategy.apply(trended))[0]
    without = dmfa_error(strategy.apply(clean))[0]
    dmfa_scores[name] = (with_trend, without)
    print(f"    {name:<22} {with_trend:>10.2e} {without:>12.2e}")

print("\n  Only the polynomial fit helps, and only because it is the one that")
print("  is not a filter. The second column is the price: what the same")
print("  operation costs when there was no trend to remove.")


# --------------------------------------------------------------------------
# BLTVA accepts exactly one of them
# --------------------------------------------------------------------------
# Nq = 2 here, not 5: eq. (49) is Nq + 3Np + Na + 1 < 2P - 1, and with only
# 8 periods that leaves very little room. It is the same budget pressure that
# decides when differencing is worth using at all.

print(f"\n  passed to BLTVA as detrend=:")
for name, strategy in strategies:
    try:
        bltva(trended, current, fs, P, harmonics, Np=2, Nq=2,
              harmonics="excited", band_halfwidth=5 * P, detrend=strategy)
        print(f"    {name:<22} accepted")
    except ValueError as refusal:
        reason = ("its weighting would undo it"
                  if "cannot be used as a BLTVA" in str(refusal)
                  else str(refusal).split(".")[0])
        print(f"    {name:<22} refused -- {reason}")

print("\n  Only differencing is accepted, because only its attenuation is")
print("  something eq. (26) can divide back out. For BLTVA the answer is")
print("  usually no detrending at all, though: raise Nq and let it fit the")
print("  trend jointly with the impedance.")


# --------------------------------------------------------------------------
# Figure 1: the signals
# --------------------------------------------------------------------------
fig1, axs = plt.subplots(1, 2, figsize=(14, 5))

envelope(axs[0], time, trended, color="C3")
axs[0].plot(time, trend, color="black", lw=2, label="the drift")
axs[0].set(xlabel="time / s", ylabel="voltage / V",
           title=f"the record: {np.ptp(trend):.1f} V of drift on "
                 f"{clean.std():.2f} V of signal")
axs[0].legend(fontsize=8, loc="upper left")

# Each trace is scaled to the same visual height so the shapes can be
# compared; the label carries the real amplitude, because that is where the
# four differ most. Differencing comes out 30x smaller -- it attenuates every
# tone, which BLTVA divides back out and nothing else does.
# The first few samples of a differenced record carry the zero-prepend step
# of eq. (12), not signal -- y[0] is the raw voltage, 5000x everything after
# it -- so they are trimmed here exactly as select_differencing_order trims
# them. BLTVA models that step with its boundary regressors instead.
trim = 20
offset = 6.0 * clean.std()
envelope(axs[1], time[trim:], clean[trim:] + 4 * offset, color="black")
axs[1].text(0.02, 4 * offset + 2.2 * clean.std(), "no trend (reference)",
            fontsize=8, color="black")
for n, (name, _) in enumerate(strategies):
    y = detrended[name][trim:]
    level = (3 - n) * offset
    envelope(axs[1], time[trim:], y / y.std() * clean.std() + level,
             color=colours[n])
    axs[1].text(0.02, level + 2.2 * clean.std(),
                f"{name}   amplitude x{y.std() / clean.std():.2f}",
                fontsize=8, color=colours[n])
axs[1].set(xlabel="time / s", ylabel="detrended (scaled, offset)",
           title="after each strategy, each scaled to the same height",
           yticks=[], ylim=(-0.6 * offset, 4.9 * offset))

finish(fig1, "09_detrending_signals", args.save,
       "the drift, and what each strategy leaves behind")


# --------------------------------------------------------------------------
# Figure 2: the spectrum
# --------------------------------------------------------------------------
fig2, axs = plt.subplots(1, 3, figsize=(16, 4.8))

freq_axis = np.fft.rfftfreq(N, 1.0 / fs)


def magnitude(signal):
    return np.abs(np.fft.rfft(signal)) / N


axs[0].loglog(freq_axis[1:], magnitude(trended)[1:], color="C3", lw=0.7,
              label="with drift")
axs[0].loglog(freq_axis[1:], magnitude(clean)[1:], color="black", lw=0.7,
              alpha=0.6, label="no drift")
axs[0].plot(frequencies, magnitude(clean)[k_exc], "o", ms=5, mfc="none",
            color="C2", label="excited tones")
axs[0].set(xlabel="frequency / Hz", ylabel="|V| / V",
           title="the drift is a skirt at the bottom of the spectrum")
axs[0].legend(fontsize=8)
axs[0].grid(True, which="both", alpha=0.2)

bins = np.arange(1, 60)
axs[1].semilogy(bins, magnitude(trended)[bins], color="C3", lw=1.2,
                label="with drift")
axs[1].semilogy(bins, magnitude(clean)[bins], color="black", lw=1.2,
                alpha=0.6, label="no drift")
for n, (name, _) in enumerate(strategies):
    axs[1].semilogy(bins, magnitude(detrended[name])[bins], lw=1.0,
                    color=colours[n], alpha=0.85, label=name)
for k in k_exc[k_exc < 60]:
    axs[1].axvline(k, color="0.4", alpha=0.6, lw=0.8, ls=":")
axs[1].set(xlabel="DFT bin (cycles per record)", ylabel="|V| / V",
           title="the lowest bins: dotted lines are the tones")
axs[1].legend(fontsize=7)
axs[1].grid(True, which="both", alpha=0.2)

for n, (name, _) in enumerate(strategies):
    axs[2].loglog(frequencies, np.maximum(damage[name], 1e-16), "-o", ms=4,
                  color=colours[n], label=name)
axs[2].set(xlabel="frequency / Hz", ylabel="relative change at the tone",
           title="what each removes when there is no trend")
axs[2].legend(fontsize=7)
axs[2].grid(True, which="both", alpha=0.2)

finish(fig2, "09_detrending_spectra", args.save,
       "the drift in the spectrum, and the cost of removing it")

print("\n  Middle panel is the whole story: the drift's skirt rises above the")
print("  lowest tones, and every strategy pulls it down. The right panel is")
print("  the bill for doing so, measured where there was nothing to remove.")

show_all(args.save)
