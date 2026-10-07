"""
Non-stationary R + (R+T)||C circuit.

A transistor T is placed in series with R_p inside the parallel branch:

    R_s ---+--- R_p --- T ---+
           |                 |
           +------- C -------+

The transistor is modelled as a voltage-controlled resistance R_T(V_gate):

    V_gate >= V_on  : R_T = 0         (fully conducting, short circuit)
    V_off < V_gate < V_on : R_T varies linearly from R_T_max to 0
    V_gate <= V_off : R_T = R_T_max   (fully blocking)

By driving V_gate with a linear ramp, the effective charge-transfer resistance
R_ct(t) = R_p + R_T(V_gate(t)) changes continuously over the measurement,
producing a physically motivated non-stationary system.

Impedance at each instant:

    Z(f, t) = R_s + R_ct(t) / (1 + j * 2*pi*f * R_ct(t) * C)
"""
import numpy as np
from ..engine import simulate


V_OFF_DEFAULT = 1.6   # transistor threshold voltage [V] — fully blocking below this
V_ON_DEFAULT  = 2.5   # transistor saturation voltage [V] — fully conducting above this


def transistor_r_ct(V_gate, R_p, R_T_max, V_off=V_OFF_DEFAULT, V_on=V_ON_DEFAULT):
    alpha = np.clip((V_gate - V_off) / (V_on - V_off), 0.0, 1.0)
    R_T = R_T_max * (1.0 - alpha)
    return R_p + R_T


def impedance(f, R_s, R_ct_t, C):
    f = np.atleast_1d(f)[:, None]
    Z = R_s + R_ct_t / (1 + 1j * 2 * np.pi * f * R_ct_t * C)
    return Z


def simulate_r_rc_transistor(duration_s, sampling_time_s, frequencies, amplitudes, phases,
                              R_s, R_p, C, V_gate_start, V_gate_rate, R_T_max,
                              V_off=V_OFF_DEFAULT, V_on=V_ON_DEFAULT,
                              noise_std_V=0.0, noise_std_A=0.0, seed=None):
    N = int(round(duration_s / sampling_time_s))
    t = np.arange(N) * sampling_time_s
    V_gate_t = V_gate_start + V_gate_rate * t
    R_ct_t = transistor_r_ct(V_gate_t, R_p, R_T_max, V_off, V_on)

    def impedance_fn(f, t):
        return impedance(f, R_s, R_ct_t, C)

    result = simulate(duration_s, sampling_time_s, frequencies, amplitudes, phases,
                      impedance_fn, noise_std_V, noise_std_A, seed)
    result.update({
        'R_s': R_s,
        'R_p': R_p,
        'C': C,
        'V_gate_t': V_gate_t,
        'R_ct_t': R_ct_t,
    })
    return result
