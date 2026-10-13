"""
Non-stationary R + (R_ct(t) || C) circuit, solved as a true time-varying ODE.

A transistor T is placed in series with R_p inside the parallel branch:

    R_s ---+--- R_p --- T ---+
           |                 |
           +------- C -------+

The transistor is modelled as a voltage-controlled resistance R_T(V_gate):

    V_gate >= V_on        : R_T = 0           (fully conducting, short circuit)
    V_off < V_gate < V_on : R_T varies linearly from R_T_max to 0
    V_gate <= V_off       : R_T = R_T_max     (fully blocking)

Driving V_gate with a linear ramp makes the effective charge-transfer
resistance R_ct(t) = R_p + R_T(V_gate(t)) change continuously, producing a
physically motivated non-stationary system. This is the circuit of
Breugelmans et al., Electrochim. Acta 76 (2012) 375, Fig. 7 -- the only model
in this package that can be built on the bench as well as simulated.

Frozen-time impedance versus true response
------------------------------------------
The *frozen-time* (quasi-static) impedance

    Z(f, t) = R_s + R_ct(t) / (1 + j*2*pi*f * R_ct(t) * C)

is the ground truth an estimator is supposed to recover, and it is returned as
``Z_true``. It is **not**, however, the response the circuit actually produces.
Because Z depends nonlinearly on R_ct(t), the quasi-static response is only an
approximation of the governing equation, valid when R_ct varies slowly
compared with the local time constant tau = R_ct*C. At realistic gate-ramp
rates the two differ by on the order of 1% rms -- far above the numerical
precision of the estimators under test, so the distinction matters.

The voltage returned here is therefore obtained by integrating the circuit
equation. Under galvanostatic excitation, with v_C the voltage across the
parallel branch:

    i(t) = v_C/R_ct(t) + C * dv_C/dt
    =>  dv_C/dt = (i(t) - v_C/R_ct(t)) / C
        v(t)    = R_s*i(t) + v_C(t)

a scalar linear ODE with a time-varying coefficient, integrated with
fixed-step RK4. ``voltage_quasistatic_V`` carries the frozen-time synthesis
alongside it so the modelling error is directly observable.

Contrast with ``legendre.py``: there the frozen-time synthesis is algebraically
*exact*, because the basis functions are real and the impedance is linear in
them. Here it is not, which is precisely what makes this model useful for
testing estimators against a system that does not satisfy their assumptions.
"""
import warnings

import numpy as np

from ..engine import generate_multisine, synthesize_voltage


V_OFF_DEFAULT = 1.6   # transistor threshold voltage [V] — fully blocking below this
V_ON_DEFAULT  = 2.5   # transistor saturation voltage [V] — fully conducting above this


def transistor_r_ct(V_gate, R_p, R_T_max, V_off=V_OFF_DEFAULT, V_on=V_ON_DEFAULT):
    alpha = np.clip((V_gate - V_off) / (V_on - V_off), 0.0, 1.0)
    R_T = R_T_max * (1.0 - alpha)
    return R_p + R_T


def impedance(f, R_s, R_ct_t, C):
    """Frozen-time impedance, shape (n_freq, N) when R_ct_t is an array."""
    f = np.atleast_1d(f)[:, None]
    Z = R_s + R_ct_t / (1 + 1j * 2 * np.pi * f * R_ct_t * C)
    return Z


def _parallel_impedance(f, R_ct, C):
    """Impedance of the R_ct || C branch alone (excludes R_s)."""
    return R_ct / (1 + 1j * 2 * np.pi * np.asarray(f) * R_ct * C)


def _integrate_parallel_branch(dt, current, R_ct, current_half, R_ct_half, C, v_C0):
    """Fixed-step RK4 for dv_C/dt = (i(t) - v_C/R_ct(t)) / C.

    ``current``/``R_ct`` are sampled on the output grid; ``current_half``/
    ``R_ct_half`` on the midpoints t_n + dt/2, which RK4 needs. Evaluating
    them analytically rather than interpolating keeps the method fourth order.
    """
    N = current.size
    v_C = np.empty(N)
    v_C[0] = v_C0

    g = 1.0 / R_ct
    g_half = 1.0 / R_ct_half

    for n in range(N - 1):
        v = v_C[n]
        k1 = (current[n] - v * g[n]) / C
        k2 = (current_half[n] - (v + 0.5 * dt * k1) * g_half[n]) / C
        k3 = (current_half[n] - (v + 0.5 * dt * k2) * g_half[n]) / C
        k4 = (current[n + 1] - (v + dt * k3) * g[n + 1]) / C
        v_C[n + 1] = v + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    return v_C


def simulate_r_rc_transistor(duration_s, sampling_time_s, frequencies, amplitudes, phases,
                              R_s, R_p, C, V_gate_start, V_gate_rate, R_T_max,
                              V_off=V_OFF_DEFAULT, V_on=V_ON_DEFAULT,
                              noise_std_V=0.0, noise_std_A=0.0, seed=None):
    """Simulate the transistor-gated circuit by integrating its ODE.

    Returns a dict with the usual ``time_s``, ``voltage_V``, ``current_A``,
    ``Z_true`` and ``frequencies_Hz``, plus:

    ``voltage_quasistatic_V``
        The frozen-time synthesis, for comparison. ``voltage_V`` minus this is
        the error incurred by assuming quasi-static behaviour.
    ``R_ct_t``, ``V_gate_t``
        The time-varying charge-transfer resistance and the gate ramp driving it.

    ``Z_true`` is the frozen-time impedance at each excited frequency and each
    sample, which is the quantity an estimator should recover.
    """
    rng = np.random.default_rng(seed)

    N = int(round(duration_s / sampling_time_s))
    dt = sampling_time_s
    t = np.arange(N) * dt
    frequencies = np.asarray(frequencies)
    amplitudes = np.asarray(amplitudes)
    phases = np.asarray(phases)

    def gate(tq):
        return V_gate_start + V_gate_rate * tq

    def r_ct(tq):
        return transistor_r_ct(gate(tq), R_p, R_T_max, V_off, V_on)

    V_gate_t = gate(t)
    R_ct_t = r_ct(t)

    tau_min = float(np.min(R_ct_t) * C)
    if dt > 0.1 * tau_min:
        warnings.warn(
            f"sampling_time_s={dt:.3g} is not small against the fastest circuit "
            f"time constant tau={tau_min:.3g}; the RK4 integration may be "
            "inaccurate. Reduce sampling_time_s.")

    # Noise-free current drives the circuit; measurement noise is added after.
    current_A = generate_multisine(t, frequencies, amplitudes, phases)
    current_half = generate_multisine(t + 0.5 * dt, frequencies, amplitudes, phases)
    R_ct_half = r_ct(t + 0.5 * dt)

    # Start from the quasi-static state, i.e. as if the circuit had already
    # been running. Starting from v_C = 0 would add a ~5*tau startup transient
    # that has nothing to do with the time variation under study.
    Zp0 = _parallel_impedance(frequencies, R_ct_t[0], C)
    v_C0 = float(np.sum(amplitudes * np.real(Zp0 * np.exp(1j * phases))))

    v_C = _integrate_parallel_branch(dt, current_A, R_ct_t,
                                     current_half, R_ct_half, C, v_C0)
    voltage_V = R_s * current_A + v_C

    Z_true = impedance(frequencies, R_s, R_ct_t, C)
    voltage_quasistatic_V = synthesize_voltage(t, frequencies, amplitudes, phases, Z_true)

    if noise_std_V > 0:
        voltage_V = voltage_V + rng.normal(0.0, noise_std_V, N)
    if noise_std_A > 0:
        current_A = current_A + rng.normal(0.0, noise_std_A, N)

    return {
        'time_s': t,
        'voltage_V': voltage_V,
        'current_A': current_A,
        'Z_true': Z_true,
        'frequencies_Hz': frequencies,
        'voltage_quasistatic_V': voltage_quasistatic_V,
        'R_s': R_s,
        'R_p': R_p,
        'C': C,
        'V_gate_t': V_gate_t,
        'R_ct_t': R_ct_t,
    }
