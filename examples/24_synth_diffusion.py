"""
24 -- Simulating a charging battery: Randles with finite-space diffusion.

The fifth system, and the first with any diffusion in it. A DC current charges
an insertion electrode, so the state of charge drifts through the record and
drags two parameters with it: the diffusion resistance, which tracks the OCV
slope, and the charge-transfer resistance, which tracks the Butler-Volmer
concentration factor.

Two things make it unlike the other four:

* the impedance is **not RC-shaped**. A blocking far boundary gives a
  45-degree Warburg tail that turns up into the intercalation capacitance, so
  |Z| grows without limit towards DC instead of settling on a resistance.
* the trend is a **real charge curve**. Every other drift in this package is
  a low-order polynomial, which is the basis BLTVA already fits; this one is
  a curve that happens to need a polynomial of some order, and the last
  figure prices it. Over this record a cubic still leaves 4% of it, and the
  examples use Nq = 2. Start the charge nearer the OCV inflection at SOC 0.52
  and it gets dearer.

Unlike `r_rc_transistor` this model does not integrate an ODE: both varying
parameters multiply the output of a fixed subsystem, so the frozen-time
synthesis is very nearly what the circuit actually does. What is left is the
double-layer loop, measured during development at 3.5e-3 rms on a 20C record
and found to be first order in the drift rate (see simulation/README.md).
This record is 40x gentler, which puts it near 1e-4 by that law -- an
extrapolation, not a measurement: integrating 2000 s at the step the stiff
modes need is far more expensive than the simulation it would check.

Run:  python examples/24_synth_diffusion.py [--save]
"""
import numpy as np
import matplotlib.pyplot as plt

from deistools.simulation.models.randles_diffusion import (
    simulate_randles_diffusion, default_ocv, default_ocv_slope)

from _plotting import parse_args, finish, show_all, overview

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# The excitation
#
# This is the one system that cannot use the shared 0.05-15 Hz grid of
# examples 00-03. Finite diffusion only shows itself around w*tau_d ~ 1, which
# is 8 mHz for the tau_d below, so the record has to reach into the millihertz
# or there is no diffusion in the picture to estimate. That forces a 200 s
# base period, and ten of those is a 2000 s record -- over which a battery
# charges appreciably. That tension is the physics, not an inconvenience: the
# lower you look, the less stationary the cell is while you look.
# --------------------------------------------------------------------------
Tp = 200.0         # period of the multisine [s]
fs = 50.0          # sampling rate [Hz]
P = 10             # number of periods measured

harmonics = np.array([1, 2, 3, 5, 9, 17, 31, 57, 105, 193, 355, 653, 1201, 2209])
frequencies = harmonics / Tp
amplitudes = np.full(frequencies.size, 0.05)
phases = np.random.default_rng(3).uniform(0, 2 * np.pi, frequencies.size)

N = int(fs * Tp) * P
T = N / fs


# --------------------------------------------------------------------------
# The cell
#
# Cell-scale rather than the lab-circuit scale of 00-03: tens of milliohms,
# a farad of double layer. At 10 ohm and 1 mF the charge curve would be
# microvolts against volts of excitation and the trend would be invisible.
# --------------------------------------------------------------------------
R_s = 0.02         # series resistance [ohm]
C_dl = 1.0         # double-layer capacitance [F]
R_ct_ref = 0.015   # charge-transfer resistance at soc_start [ohm]
R_d_ref = 0.08     # diffusion resistance at soc_start [ohm]
tau_d = 20.0       # L^2/D, the diffusion time constant [s]

soc_start = 0.2    # where the charge begins
c_rate = 0.5       # 1/h, so SOC advances 0.5 * T/3600 over the record


# --------------------------------------------------------------------------
# Simulate
# --------------------------------------------------------------------------
sim = simulate_randles_diffusion(
    T, 1.0 / fs,
    frequencies, amplitudes, phases,
    R_s=R_s, C_dl=C_dl, R_ct_ref=R_ct_ref, R_d_ref=R_d_ref, tau_d=tau_d,
    soc_start=soc_start, c_rate=c_rate,
    include_dc=True,
    noise_std_V=2e-5,
    seed=1,
)

time = sim["time_s"]
voltage = sim["voltage_V"]
current = sim["current_A"]
soc = sim["soc_t"]
R_d = sim["R_d_t"]
R_ct = sim["R_ct_t"]
drift = sim["drift_V"]
Z_true = sim["Z_true"]            # frozen-time impedance, (n_freq, N)

ac = voltage - drift

print(f"  SOC        {soc[0]:.3f} -> {soc[-1]:.3f} at {c_rate}C over {T:.0f} s")
print(f"  capacity   {sim['capacity_As']:.1f} As = {sim['capacity_As']/3.6:.1f} mAh"
      f", implied by tau_d, R_d and the OCV slope")
print(f"  i_dc       {sim['current_dc_A']*1e3:.2f} mA")
print(f"  R_d        {R_d[0]*1e3:.1f} -> {R_d.min()*1e3:.1f} -> {R_d[-1]*1e3:.1f}"
      f" mohm, following the OCV slope through its minimum -- so the"
      f" variation is not monotone")
print(f"  R_ct       {R_ct[0]*1e3:.1f} -> {R_ct[-1]*1e3:.1f} mohm")
print(f"  tones      {frequencies.min()*1e3:.0f} mHz - {frequencies.max():.1f} Hz,"
      f"  w*tau_d = {2*np.pi*frequencies.min()*tau_d:.2f} - "
      f"{2*np.pi*frequencies.max()*tau_d:.0f}")
print(f"  |Z|        {np.abs(Z_true[:, 0]).max()*1e3:.0f} mohm at the bottom tone"
      f" -> {np.abs(Z_true[:, 0]).min()*1e3:.0f} mohm at the top")
print(f"  trend      {np.ptp(drift)*1e3:.1f} mV, against {np.std(ac)*1e3:.1f} mV"
      f" rms of excitation response")


# --------------------------------------------------------------------------
# Figure 1: what drives the variation
# --------------------------------------------------------------------------
soc_sweep = np.linspace(0.02, 0.98, 500)

fig1, axs = plt.subplots(1, 3, figsize=(15, 4.6))

axs[0].plot(soc_sweep, default_ocv(soc_sweep), color="0.6", lw=1.5)
axs[0].axvspan(soc[0], soc[-1], color="C0", alpha=0.25, label="this record")
axs[0].set(xlabel="state of charge", ylabel="OCV / V",
           title="open-circuit voltage")
axs[0].legend(fontsize=8)

axs[1].plot(soc_sweep, np.abs(default_ocv_slope(soc_sweep)), color="0.6", lw=1.5)
axs[1].axvspan(soc[0], soc[-1], color="C0", alpha=0.25)
axs[1].set(xlabel="state of charge", ylabel=r"$|dU/dSOC|$ / V",
           title=r"the slope, which $R_d$ follows", yscale="log")

axs[2].plot(time, R_d * 1e3, color="C3", label=r"$R_d$")
axs[2].plot(time, R_ct * 1e3, color="C2", label=r"$R_{ct}$")
axs[2].set(xlabel="time / s", ylabel=r"resistance / m$\Omega$",
           title="the resulting variation", yscale="log")
axs[2].legend(fontsize=8)

finish(fig1, "24_diffusion_drivers", args.save,
       "the charge curve, its slope, and the parameters they drive")


# --------------------------------------------------------------------------
# Figure 2: the record, its spectra, and the impedance
# --------------------------------------------------------------------------
# The Nyquist plots are the point of comparison with 00-03: a 45-degree line
# rather than a semicircle, turning up at the bottom tone where the blocking
# boundary starts to act as a capacitance. The tiny arc near the origin is
# R_ct against C_dl.
fig2 = overview(time, voltage, current, fs, frequencies, Z=Z_true, t_Z=time)
finish(fig2, "24_diffusion_overview", args.save,
       "charging insertion electrode: record, spectra, impedance")


# --------------------------------------------------------------------------
# Figure 3: the trend that is not a polynomial, and where the variation lives
# --------------------------------------------------------------------------
x = 2 * time / time[-1] - 1          # the interval BLTVA's basis is built on
fig3, axs3 = plt.subplots(1, 2, figsize=(13, 4.6))

axs3[0].plot(time, drift * 1e3, color="black", lw=1.5, label="charge curve")
for order, colour in ((1, "C0"), (3, "C1"), (5, "C2")):
    fit = np.polyval(np.polyfit(x, drift, order), x)
    left = np.std(drift - fit) / np.std(drift)
    axs3[0].plot(time, fit * 1e3, color=colour, lw=1.0, ls="--",
                 label=f"order {order}: {left:.1%} left")
axs3[0].set(xlabel="time / s", ylabel="trend / mV",
            title="what the charge curve costs in trend order")
axs3[0].legend(fontsize=8)

spread = np.abs(Z_true).max(axis=1) / np.abs(Z_true).min(axis=1)
axs3[1].semilogx(frequencies, spread, "-o", ms=4, color="C3")
axs3[1].axhline(1.0, ls=":", color="0.5")
axs3[1].set(xlabel="frequency / Hz", ylabel=r"max$|Z|$ / min$|Z|$ over the record",
            title="the variation is a low-frequency problem")
axs3[1].grid(True, which="both", alpha=0.25)

finish(fig3, "24_diffusion_trend", args.save,
       "the trend, and how the variation is distributed over frequency")

# The left panel is a statement about Nq, not a claim that the curve is
# unfittable: order 5 does absorb it here. The point is that it has to be
# bought, where legendre.py's drift is a polynomial to begin with and costs
# nothing. The order needed grows with how much of the OCV curve the record
# crosses, which is to say with the C rate and the record length -- the same
# two knobs that set how non-stationary the impedance is.

# Worth reading off the right-hand panel: the impedance barely moves at the
# top of the band and moves by several times at the bottom. Every estimator
# here needs a long window to reach a low tone and a short one to keep up with
# the variation, and this system puts those two demands on the same tones.

show_all(args.save)
