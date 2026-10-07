"""
Simulate a non-stationary R + (R+T)||C circuit whose impedance drifts
over time due to a transistor-controlled charge-transfer resistance,
and visualise the time-domain signals and the evolving Nyquist plot.
"""
import numpy as np
import matplotlib.pyplot as plt

from deistools.simulation import simulate_r_rc_transistor
from deistools.visualise import plot_technique, plot_impedance_set

# --- Circuit parameters ---
R_s = 10          # series resistance [Ohm]
R_p = 30          # base parallel resistance [Ohm]
C = 1e-3          # capacitance [F]
R_T_max = 70      # maximum transistor resistance [Ohm]

# Gate voltage ramp — sweeps through the transistor transition region
V_gate_start = 1.2    # V (below V_off = 1.6 => transistor fully blocking)
V_gate_rate = 0.2     # V/s (ramp crosses V_on = 2.5 within the measurement)

# --- Excitation signal ---
frequencies = np.array([1.0, 5.0, 10.0, 50.0, 100.0])   # Hz
amplitudes = np.ones_like(frequencies) * 0.01              # A
phases = np.zeros_like(frequencies)                        # rad

# --- Acquisition settings ---
duration_s = 10.0
sampling_time_s = 1e-4

# --- Run simulation ---
result = simulate_r_rc_transistor(
    duration_s=duration_s,
    sampling_time_s=sampling_time_s,
    frequencies=frequencies,
    amplitudes=amplitudes,
    phases=phases,
    R_s=R_s,
    R_p=R_p,
    C=C,
    V_gate_start=V_gate_start,
    V_gate_rate=V_gate_rate,
    R_T_max=R_T_max,
    noise_std_V=1e-4,
    noise_std_A=1e-5,
    seed=42,
)

# --- Visualise time-domain signals ---
plot_technique(result["voltage_V"], result["current_A"], result["time_s"])

# --- Visualise evolving impedance (Nyquist) ---
# Subsample the time axis so the interactive slider has a manageable number of
# frames while still showing the impedance drift across the measurement.
Z = result["Z_true"]
step = max(1, Z.shape[1] // 200)
plot_impedance_set(Z[:, ::step])

plt.show()
