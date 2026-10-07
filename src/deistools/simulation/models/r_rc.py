"""
Stationary R + RC circuit (series resistance plus a parallel RC element).

    Z(f) = R_s + R_p / (1 + j * 2*pi*f * R_p * C)

All parameters are scalars (time-invariant). Use this as a stationary
reference baseline when benchmarking time-varying estimation algorithms.
"""
import numpy as np
from ..engine import simulate


def impedance(f, R_s, R_p, C):
    f = np.atleast_1d(f)[:, None]
    Z = R_s + R_p / (1 + 1j * 2 * np.pi * f * R_p * C)
    return Z


def simulate_r_rc(duration_s, sampling_time_s, frequencies, amplitudes, phases,
                  R_s, R_p, C, noise_std_V=0.0, noise_std_A=0.0, seed=None):
    def impedance_fn(f, t):
        return impedance(f, R_s, R_p, C)

    result = simulate(duration_s, sampling_time_s, frequencies, amplitudes, phases,
                      impedance_fn, noise_std_V, noise_std_A, seed)
    result.update({'R_s': R_s, 'R_p': R_p, 'C': C})
    return result
