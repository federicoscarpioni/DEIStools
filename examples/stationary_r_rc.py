"""
Simulate a stationary R+RC circuit excited by a multi-frequency current signal
and visualise the time-domain signals and the Nyquist impedance plot.
"""
import numpy as np
import matplotlib.pyplot as plt

from deistools.simulation import simulate_r_rc
from deistools.visualise import plot_technique, plot_impedance_set

# --- Circuit parameters ---
R_s = 10        # series resistance [Ohm]
R_p = 50        # parallel resistance [Ohm]
C = 1e-3        # capacitance [F]

# --- Excitation signal ---
frequencies = np.array([1.0, 5.0, 10.0, 50.0, 100.0])   # Hz
amplitudes = np.ones_like(frequencies) * 0.01              # A
phases = np.zeros_like(frequencies)                        # rad

# --- Acquisition settings ---
duration_s = 5.0
sampling_time_s = 1e-4

# --- Run simulation ---
result = simulate_r_rc(
    duration_s=duration_s,
    sampling_time_s=sampling_time_s,
    frequencies=frequencies,
    amplitudes=amplitudes,
    phases=phases,
    R_s=R_s,
    R_p=R_p,
    C=C,
    noise_std_V=1e-4,
    noise_std_A=1e-5,
    seed=42,
)

# --- Visualise time-domain signals ---
plot_technique(result["voltage_V"], result["current_A"], result["time_s"])

# --- Visualise impedance (Nyquist) ---
# Z_true has shape (n_freq, N); plot_impedance_set expects (n_freq, n_time).
# For a stationary circuit every column is identical, so we can plot any single
# snapshot or a few columns to verify stationarity.
Z = result["Z_true"]
plot_impedance_set(Z[:, ::1000])  # subsample time axis for the slider

plt.show()
