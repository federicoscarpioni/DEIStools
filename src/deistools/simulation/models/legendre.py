"""
Legendre polynomial oracle model for non-stationary transfer function estimation.

The transfer function at each excited frequency f_k is parameterised as a sum of
Legendre polynomials in time up to order P:

    H(f_k, t) = sum_{p=0}^{P} a_p(f_k) * L_p(2*t/T - 1)

where T is the total window length and L_p is the p-th Legendre polynomial.

An additive voltage drift is modelled separately as its own Legendre expansion:

    v_drift(t) = sum_{q=0}^{Q} b_q * L_q(2*t/T - 1)

This is exactly the signal model that BLTVA assumes, so the algorithm should
recover the coefficients a_p and b_q exactly (up to noise). Use this model as
an oracle test: if BLTVA fails to recover known coefficients, there is a bug.

The voltage response is:

    v(t) = v_drift(t) + sum_k |H(f_k,t)| * A_k * cos(2*pi*f_k*t + phi_k + angle(H(f_k,t)))

Specifying coefficients
-----------------------
impedance_coefficients : array_like, shape (n_freq, P+1), complex
    a[k, p] is the coefficient of L_p for frequency f_k.
    a[k, 0] sets the mean impedance at f_k over the window.
    a[k, 1] sets a linear variation, a[k, 2] a quadratic one.

drift_coefficients : array_like, shape (Q+1,), real, optional
    b[q] is the coefficient of L_q for the additive drift.
    Set to None for no drift (default).
"""
import numpy as np
from numpy.polynomial.legendre import legvander
from ..engine import simulate as _simulate


def legendre_impedance(f, t, T_window, impedance_coefficients):
    tau = 2 * t / T_window - 1
    B = legvander(tau, impedance_coefficients.shape[1] - 1)   # (N, P+1)
    Z = impedance_coefficients @ B.T                          # (n_freq, N)
    return Z


def legendre_drift(t, T_window, drift_coefficients):
    tau = 2 * t / T_window - 1
    B = legvander(tau, len(drift_coefficients) - 1)   # (N, Q+1)
    return B @ np.asarray(drift_coefficients, dtype=float)


def simulate_legendre(duration_s, sampling_time_s, frequencies, amplitudes, phases,
                      impedance_coefficients, drift_coefficients=None,
                      noise_std_V=0.0, noise_std_A=0.0, seed=None):
    impedance_coefficients = np.asarray(impedance_coefficients, dtype=complex)
    T_window = duration_s

    def impedance_fn(f, t):
        return legendre_impedance(f, t, T_window, impedance_coefficients)

    result = _simulate(duration_s, sampling_time_s, frequencies, amplitudes, phases,
                       impedance_fn, noise_std_V, noise_std_A, seed)

    if drift_coefficients is not None:
        t = result['time_s']
        drift_V = legendre_drift(t, T_window, drift_coefficients)
        result['voltage_V'] += drift_V
        result['drift_V'] = drift_V
    else:
        result['drift_V'] = np.zeros_like(result['time_s'])

    result['impedance_coefficients'] = impedance_coefficients
    result['drift_coefficients'] = drift_coefficients
    return result
