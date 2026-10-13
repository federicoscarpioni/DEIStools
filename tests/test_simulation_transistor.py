"""Tests for the transistor-gated circuit solved as a true time-varying ODE.

This model exists to provide a system that does *not* satisfy the estimators'
assumptions. Because Z depends nonlinearly on R_ct(t), the frozen-time
synthesis is only an approximation here -- unlike ``legendre.py``, where it is
algebraically exact. Measuring that gap is the point, so the simulation itself
has to be right first.

Three properties pin the integration down:

* a **static gate** makes the circuit LTI, where the steady-state response has
  a closed form, so the solver can be checked against exact analysis;
* **halving the gate ramp rate halves** the departure from quasi-static, which
  is the expected first-order scaling in the variation rate;
* the solver converges at **fourth order** in the step size, as RK4 should.
"""
import warnings

import numpy as np
import pytest
from numpy.fft import fft

from deistools.simulation.models.r_rc_transistor import (
    simulate_r_rc_transistor, transistor_r_ct,
    V_OFF_DEFAULT, V_ON_DEFAULT,
)

R_S, R_P, C, R_T_MAX = 10.0, 30.0, 1e-3, 70.0
FREQS = np.array([1.0, 5.0, 10.0, 50.0, 100.0])
AMPS = 0.01 * np.ones_like(FREQS)
PHASES = np.random.default_rng(0).uniform(0, 2 * np.pi, FREQS.size)
DT = 1e-4
DURATION = 10.0


def _simulate(V_gate_start, V_gate_rate, duration=DURATION, dt=DT, **kw):
    return simulate_r_rc_transistor(
        duration, dt, FREQS, AMPS, PHASES,
        R_S, R_P, C, V_gate_start, V_gate_rate, R_T_MAX, **kw)


def _fft_eis(sim, duration=DURATION):
    """Impedance by FFT-EIS at the excited bins (exact: integer periods)."""
    n = sim['voltage_V'].size
    V = fft(sim['voltage_V']) / n
    I = fft(sim['current_A']) / n
    k = np.round(FREQS * duration).astype(int)
    return V[k] / I[k]


# --------------------------------------------------------------------------- #
# Correctness of the integration
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("V_gate", [1.0, 2.0, 3.0])
def test_static_gate_reproduces_analytic_lti_impedance(V_gate):
    """With the gate held fixed the circuit is time-invariant, so FFT-EIS on
    the integrated response must return the analytic R + R_ct||C impedance.

    This is the strongest available check on the solver: an exact closed form
    to compare against, with no modelling approximation in between. It also
    confirms the initial condition is the true steady state -- otherwise a
    startup transient would leak into the spectrum.
    """
    sim = _simulate(V_gate, 0.0)

    R_ct = transistor_r_ct(V_gate, R_P, R_T_MAX)
    Z_analytic = R_S + R_ct / (1 + 1j * 2 * np.pi * FREQS * R_ct * C)

    err = np.abs(_fft_eis(sim) - Z_analytic) / np.abs(Z_analytic)
    assert err.max() < 1e-8, f"max relative error {err.max():.3e}"


def test_integration_is_fourth_order_in_step_size():
    """RK4 must show fourth-order convergence under step refinement.

    All steps are compared at t = 1.0 s exactly. Comparing the final *sample*
    instead would compare different instants, since the grids differ.
    """
    vals = {}
    for dt in (4e-4, 2e-4, 1e-4, 5e-5):
        sim = _simulate(1.2, 0.2, duration=2.0, dt=dt)
        idx = int(round(1.0 / dt))
        assert abs(sim['time_s'][idx] - 1.0) < 1e-12
        vals[dt] = sim['voltage_V'][idx]

    ref = vals[5e-5]
    errors = [abs(vals[dt] - ref) for dt in (4e-4, 2e-4, 1e-4)]
    orders = [np.log2(a / b) for a, b in zip(errors, errors[1:])]

    assert all(o > 3.8 for o in orders), f"observed orders {orders}"


def test_static_gate_gives_constant_r_ct():
    sim = _simulate(2.0, 0.0)
    assert np.allclose(sim['R_ct_t'], sim['R_ct_t'][0])


# --------------------------------------------------------------------------- #
# Departure from the frozen-time model
# --------------------------------------------------------------------------- #

def _quasistatic_error(sim, settle_s=1.0):
    m = sim['time_s'] > settle_s
    v, vq = sim['voltage_V'], sim['voltage_quasistatic_V']
    return float(np.std(v[m] - vq[m]) / np.std(v[m]))


def test_quasistatic_error_scales_linearly_with_variation_rate():
    """Halving the gate ramp rate must halve the departure from quasi-static.

    The gate is kept strictly inside the transistor's transition window for
    the whole record, so R_ct genuinely varies at every rate. Letting the ramp
    run outside that window would freeze R_ct and collapse the test to the
    LTI case, where the error is zero for an unrelated reason.
    """
    rates = (0.08, 0.04, 0.02, 0.01)
    errors = []
    for rate in rates:
        sim = _simulate(1.65, rate)
        assert sim['V_gate_t'].min() > V_OFF_DEFAULT
        assert sim['V_gate_t'].max() < V_ON_DEFAULT
        errors.append(_quasistatic_error(sim))

    for coarse, fine in zip(errors, errors[1:]):
        assert 1.8 < coarse / fine < 2.2, f"error ratios {errors}"


def test_quasistatic_model_error_is_significant_at_realistic_rates():
    """At a realistic ramp the frozen-time synthesis is wrong by ~1% rms.

    This is the justification for integrating the ODE at all: the gap is two
    to three orders of magnitude above the precision of the estimators under
    test, so using the quasi-static response as data would make estimator
    bias and simulation bias indistinguishable.
    """
    sim = _simulate(1.2, 0.2)
    err = _quasistatic_error(sim)

    assert err > 1e-3, (
        f"quasi-static error {err:.2e} is small enough that the ODE would be "
        "unnecessary; the test parameters no longer exercise this tier")
    assert err < 5e-2, f"quasi-static error {err:.2e} is implausibly large"


def test_z_true_is_the_frozen_time_impedance():
    """Z_true must stay the quasi-static impedance -- it is the quantity an
    estimator is asked to recover, not the response the circuit produces."""
    sim = _simulate(1.2, 0.2)
    R_ct = sim['R_ct_t']
    expected = R_S + R_ct / (1 + 1j * 2 * np.pi * FREQS[:, None] * R_ct * C)

    assert sim['Z_true'].shape == (FREQS.size, sim['time_s'].size)
    assert np.allclose(sim['Z_true'], expected)


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #

def test_returns_expected_keys_and_shapes():
    sim = _simulate(1.2, 0.2)
    n = int(round(DURATION / DT))

    for key in ('time_s', 'voltage_V', 'current_A', 'voltage_quasistatic_V',
                'V_gate_t', 'R_ct_t'):
        assert sim[key].shape == (n,), f"{key} has shape {sim[key].shape}"
    assert sim['Z_true'].shape == (FREQS.size, n)
    assert np.allclose(sim['frequencies_Hz'], FREQS)


def test_warns_when_step_is_coarse_against_the_time_constant():
    """dt must stay well below tau = R_ct*C or RK4 loses accuracy."""
    with pytest.warns(UserWarning, match="time constant"):
        # gate crosses fully open, so tau_min = R_p*C = 0.03 s against dt = 5 ms
        _simulate(1.2, 5.0, duration=0.5, dt=5e-3)


def test_no_warning_at_a_sane_step():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _simulate(1.2, 0.2)


def test_measurement_noise_does_not_drive_the_circuit():
    """noise_std_A models a noisy ammeter, not a noisy source, so the
    integrated voltage must be unchanged by it."""
    clean = _simulate(1.2, 0.2, seed=0)
    noisy = _simulate(1.2, 0.2, noise_std_A=1e-4, seed=0)

    assert np.array_equal(clean['voltage_V'], noisy['voltage_V'])
    assert not np.array_equal(clean['current_A'], noisy['current_A'])


def test_voltage_noise_is_additive_and_reproducible():
    a = _simulate(1.2, 0.2, noise_std_V=1e-3, seed=7)
    b = _simulate(1.2, 0.2, noise_std_V=1e-3, seed=7)
    c = _simulate(1.2, 0.2, noise_std_V=1e-3, seed=8)

    assert np.array_equal(a['voltage_V'], b['voltage_V'])
    assert not np.array_equal(a['voltage_V'], c['voltage_V'])
