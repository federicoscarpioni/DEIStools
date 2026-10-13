"""
01 -- Simulating a time-varying cell: the transistor-gated circuit.

A transistor in series with R_ct is driven by a slow gate ramp, so the
charge-transfer resistance sweeps smoothly through the record. The circuit is
solved as a true time-varying ODE, not by freezing time, and this script
measures how far the two differ.

Run:  python examples/01_synth_transistor.py [--save]
"""
import numpy as np
import matplotlib.pyplot as plt

from deistools.simulation.models.r_rc_transistor import (
    simulate_r_rc_transistor, transistor_r_ct)

from _plotting import parse_args, finish, show_all, overview

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# The excitation
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


# --------------------------------------------------------------------------
# The circuit
# --------------------------------------------------------------------------
R_s = 10.0         # series resistance [ohm]
R_p = 30.0         # R_ct with the transistor fully conducting [ohm]
R_T_max = 70.0     # extra resistance with it fully blocking [ohm]
C = 1e-3           # double-layer capacitance [F]

# The transistor is linear between V_off and V_on and saturates outside.
# The ramp below stays inside that window, from 5% to 95% of the way across,
# so R_ct(t) sweeps smoothly with no clipped segment at either end. The flat
# parts of the characteristic carry no information and would put a kink in
# R_ct(t) that no smooth-variation estimator should be asked to follow.
V_off, V_on = 1.6, 2.5                     # transistor thresholds [V]
V_gate_start = V_off + 0.05 * (V_on - V_off)
V_gate_stop = V_off + 0.95 * (V_on - V_off)
V_gate_rate = (V_gate_stop - V_gate_start) / T       # [V/s]


# --------------------------------------------------------------------------
# Simulate
# --------------------------------------------------------------------------
sim = simulate_r_rc_transistor(
    T, 1.0 / fs,
    frequencies, amplitudes, phases,
    R_s=R_s, R_p=R_p, C=C,
    V_gate_start=V_gate_start, V_gate_rate=V_gate_rate, R_T_max=R_T_max,
    noise_std_V=1e-4,
    seed=1,
)

time = sim["time_s"]
voltage = sim["voltage_V"]
current = sim["current_A"]
R_ct = sim["R_ct_t"]              # the time-varying resistance
V_gate = sim["V_gate_t"]          # the ramp driving it
Z_true = sim["Z_true"]            # frozen-time impedance, (n_freq, N)

# voltage_quasistatic_V is the frozen-time synthesis: what the response would
# be if the circuit were always in its instantaneous steady state. The real
# response comes from integrating the ODE, and the difference is the error in
# treating "the impedance at time t" as a meaningful quantity.
residual = voltage - sim["voltage_quasistatic_V"]
quasistatic_error = np.std(residual) / np.std(voltage)

print(f"  gate     {V_gate[0]:.3f} -> {V_gate[-1]:.3f} V over {T:.0f} s")
print(f"  R_ct     {R_ct[0]:.1f} -> {R_ct[-1]:.1f} ohm, smooth and monotone")
print(f"  tau      {R_ct.max() * C * 1e3:.0f} -> {R_ct.min() * C * 1e3:.0f} ms")
print(f"  arc apex {1 / (2 * np.pi * R_ct[0] * C):.2f} -> "
      f"{1 / (2 * np.pi * R_ct[-1] * C):.2f} Hz")
print(f"  frozen-time ground truth is accurate to {quasistatic_error:.3%} rms")


# --------------------------------------------------------------------------
# Figure 1: the transistor, the ramp, and the resulting variation
# --------------------------------------------------------------------------
v_sweep = np.linspace(V_off - 0.3, V_on + 0.3, 400)
r_sweep = transistor_r_ct(v_sweep, R_p, R_T_max)

fig1, axs = plt.subplots(1, 3, figsize=(15, 4.6))

axs[0].plot(v_sweep, r_sweep, color="0.6", lw=1.5)
axs[0].axvspan(V_gate_start, V_gate_stop, color="C0", alpha=0.25,
               label="part of the ramp used")
axs[0].axvline(V_off, ls=":", color="C3")
axs[0].axvline(V_on, ls=":", color="C3")
axs[0].annotate("blocking", (V_off - 0.25, R_p + R_T_max * 0.9), fontsize=8,
                color="C3")
axs[0].annotate("conducting", (V_on - 0.05, R_p + R_T_max * 0.1), fontsize=8,
                color="C3")
axs[0].set(xlabel="gate voltage / V", ylabel=r"$R_{ct}$ / $\Omega$",
           title="transistor characteristic")
axs[0].legend(fontsize=8)

axs[1].plot(time, V_gate, color="C2")
axs[1].set(xlabel="time / s", ylabel="gate voltage / V", title="the drive")

axs[2].plot(time, R_ct, color="C3")
axs[2].set(xlabel="time / s", ylabel=r"$R_{ct}(t)$ / $\Omega$",
           title="the resulting variation")

finish(fig1, "01_transistor_gate", args.save,
       "the gate ramp and what it does to R_ct")


# --------------------------------------------------------------------------
# Figure 2: the record, its spectra, and the impedance
# --------------------------------------------------------------------------
# In the voltage spectrum each tone has grown skirts: time variation spreads
# every line into its neighbouring bins. The current has none, because the
# excitation is exactly periodic. Those skirts are what DMFA and BLTVA use.

fig2 = overview(time, voltage, current, fs, frequencies, Z=Z_true, t_Z=time)
finish(fig2, "01_transistor_overview", args.save,
       "transistor-gated cell: record, spectra, impedance")


# --------------------------------------------------------------------------
# Figure 3: ODE solution against frozen-time synthesis
# --------------------------------------------------------------------------
two_periods = slice(0, int(2 * Tp * fs))

fig3, axs3 = plt.subplots(1, 2, figsize=(13, 4.6))
axs3[0].plot(time[two_periods], voltage[two_periods], color="C0", lw=0.8,
             label="ODE solution")
axs3[0].plot(time[two_periods], sim["voltage_quasistatic_V"][two_periods],
             color="C1", lw=0.8, ls="--", label="frozen-time synthesis")
axs3[0].set(xlabel="time / s", ylabel="voltage / V", title="first two periods")
axs3[0].legend(fontsize=8)

axs3[1].plot(time, residual, color="C3", lw=0.5)
axs3[1].set(xlabel="time / s", ylabel="difference / V",
            title=f"ODE minus frozen time ({quasistatic_error:.3%} rms)")

finish(fig3, "01_transistor_quasistatic", args.save,
       "how good is the frozen-time ground truth?")

show_all(args.save)
