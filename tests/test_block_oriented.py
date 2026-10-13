"""Block-oriented nonlinear time-varying simulation, Hallemans TIM 2021 §II/VII.

This is the only model here that produces genuine nonlinear distortion, and
so the only one that can give Section V a ground truth. The other two
prescribe an impedance and synthesise a response from it, which makes the
answer true by construction.

The ground truth has to be *computed*. The best linear approximation of a
nonlinear block is not its linear part -- the static nonlinearities move it --
so :func:`best_linear_approximation` averages ``Y(k)/U(k)`` over many random
phase realisations, exactly as the paper obtains the true BLAs it validates
against in Fig. 10.

That gives the sharpest test in the suite: BLTVA should land on the Monte
Carlo BLA, and *not* on the linear part, for any branch whose nonlinearity is
odd.
"""
import warnings

import numpy as np
import pytest

from deistools.processing.bltva import bltva, legendre_basis
from deistools.simulation import (
    simulate_block_oriented, best_linear_approximation, frozen_impedance,
    Branch, SecondOrderLowpass, hallemans_2021_system, simulate_legendre,
)
from tests.conftest import odd_linear_grid

FS, TP, P = 20.0, 100.0, 8
N, DT = int(FS * TP) * P, 1.0 / FS
T = N / FS
H = np.array([1, 3, 5, 9, 15, 25])
FREQS = H / TP
AMPS = np.full(H.size, 1.0)
PHASES = np.random.default_rng(0).uniform(0, 2 * np.pi, H.size)


def _linear_branches():
    return [Branch(SecondOrderLowpass.from_hz(0.7, 0.15, 1.0)),
            Branch(SecondOrderLowpass.from_hz(1.0, 0.10, 0.2)),
            Branch(SecondOrderLowpass.from_hz(1.5, 0.15, 0.2))]


# --------------------------------------------------------------------------- #
# The filter
# --------------------------------------------------------------------------- #

def test_second_order_filter_has_the_stated_gain_and_resonance():
    H_f = SecondOrderLowpass.from_hz(1.0, 0.1, 0.2)
    assert H_f.response(0.0).real == pytest.approx(0.2)
    # at the natural frequency the response peaks at dc_gain / (2*zeta)
    assert abs(H_f.response(1.0)) == pytest.approx(0.2 / (2 * 0.1))


def test_filter_response_is_conjugate_symmetric():
    """Required, or the filtered signal would not come back real."""
    H_f = SecondOrderLowpass.from_hz(0.7, 0.15, 1.0)
    f = np.array([0.1, 0.7, 3.0])
    assert np.allclose(H_f.response(-f), np.conj(H_f.response(f)))


def test_from_hz_and_rad_per_second_differ_by_two_pi():
    """The paper writes its natural frequencies in Hz while calling them
    ``wn``, so the constructor makes the choice explicit."""
    assert SecondOrderLowpass.from_hz(0.7, 0.15).wn == pytest.approx(2 * np.pi * 0.7)


# --------------------------------------------------------------------------- #
# The simulator
# --------------------------------------------------------------------------- #

def test_linear_branch_matches_analytic_steady_state():
    """DFT-domain filtering is exact for a periodic input, so a linear branch
    must reproduce the closed-form steady-state response."""
    branch = Branch(SecondOrderLowpass.from_hz(0.7, 0.15, 1.0))
    sim = simulate_block_oriented(T, DT, FREQS, AMPS, PHASES, [branch])

    t, Hv = sim['time_s'], branch.filter.response(FREQS)
    analytic = sum(a * abs(h) * np.cos(2 * np.pi * f * t + p + np.angle(h))
                   for f, a, p, h in zip(FREQS, AMPS, PHASES, Hv))

    assert np.abs(sim['voltage_V'] - analytic).max() < 1e-10


def test_a_linear_block_system_is_the_legendre_model():
    """With every branch linear, this model *is* ``legendre.py`` with
    ``a_p(f) = H_p(j2*pi*f)``.

    The two tiers are then consistent by construction rather than by
    coincidence, which is what lets results from one be compared with the
    other.
    """
    branches = _linear_branches()
    block = simulate_block_oriented(T, DT, FREQS, AMPS, PHASES, branches)

    a = np.stack([br.filter.response(FREQS) for br in branches], axis=1)
    legendre = simulate_legendre(T, DT, FREQS, AMPS, PHASES, a,
                                 drift_coefficients=None)

    assert np.abs(block['voltage_V'] - legendre['voltage_V']).max() < 1e-10


def test_trend_is_added_as_a_legendre_series():
    coeffs = [0.0, 5.0, 3.0]
    sim = simulate_block_oriented(T, DT, FREQS, AMPS, PHASES,
                                  _linear_branches()[:1],
                                  trend_coefficients=coeffs)
    expected = legendre_basis(N, len(coeffs) - 1) @ np.asarray(coeffs)
    assert np.allclose(sim['drift_V'], expected)


def test_branch_linearity_is_reported():
    system = hallemans_2021_system()
    assert all(not br.is_linear for br in system)
    assert Branch(SecondOrderLowpass.from_hz(1.0, 0.1)).is_linear


# --------------------------------------------------------------------------- #
# The best linear approximation
# --------------------------------------------------------------------------- #

def test_bla_of_a_linear_branch_is_exactly_its_filter():
    branch = Branch(SecondOrderLowpass.from_hz(0.7, 0.15, 1.0))
    G, _ = best_linear_approximation([branch], FREQS, AMPS, DT, T,
                                     n_realisations=10, seed=0)
    expected = branch.filter.response(FREQS)
    assert np.abs(G[:, 0] - expected).max() / np.abs(expected).max() < 1e-12


def test_bla_of_a_purely_even_nonlinearity_is_unshifted():
    """A square-law term generates only even-order products, which land on
    even harmonics and never on the excited odd ones.

    This reproduces the paper's own observation about branch p = 1 of its
    simulation: its distortion is purely even and therefore "introduce[s] no
    uncertainty on the BLA".
    """
    branch = Branch(SecondOrderLowpass.from_hz(1.0, 0.10, 0.2),
                    f1=lambda x: x + 0.08 * x ** 2)
    G, _ = best_linear_approximation([branch], FREQS, AMPS, DT, T,
                                     n_realisations=10, seed=0)
    expected = branch.filter.response(FREQS)
    assert np.abs(G[:, 0] - expected).max() / np.abs(expected).max() < 1e-12


def test_bla_of_an_odd_nonlinearity_is_shifted():
    """A cube-law term folds back onto the fundamental, so the BLA is *not*
    the linear part. This is why the ground truth has to be measured."""
    branch = Branch(SecondOrderLowpass.from_hz(1.5, 0.15, 0.2),
                    f2=lambda x: x + 0.15 * x ** 3)
    G, _ = best_linear_approximation([branch], FREQS, AMPS, DT, T,
                                     n_realisations=40, seed=0)
    linear = branch.filter.response(FREQS)
    shift = np.abs(G[:, 0] - linear) / np.abs(linear)
    assert np.median(shift) > 1e-2, f"expected a visible shift, got {np.median(shift):.2e}"


def test_monte_carlo_bla_converges():
    """The reported standard error must fall as realisations are added,
    roughly as 1/sqrt(n)."""
    system = hallemans_2021_system()
    _, few = best_linear_approximation(system, FREQS, AMPS, DT, T,
                                       n_realisations=16, seed=0)
    _, many = best_linear_approximation(system, FREQS, AMPS, DT, T,
                                        n_realisations=64, seed=0)
    ratio = np.median(np.abs(few[:, 2])) / np.median(np.abs(many[:, 2]))
    assert 1.2 < ratio < 4.0, f"standard error scaled by {ratio:.2f}, expected ~2"


# --------------------------------------------------------------------------- #
# What BLTVA should return
# --------------------------------------------------------------------------- #

def test_bltva_recovers_the_bla_rather_than_the_linear_part():
    """The sharpest statement available about what BLTVA estimates.

    For a branch with an odd nonlinearity the best linear approximation and
    the underlying linear filter are different objects, and BLTVA must land on
    the former. Measured here: 1.7e-03 from the BLA against 8.4e-03 from the
    linear part on branch 0, and 1.2e-02 against 3.2e-02 on branch 2.

    Branch 1 is excluded because its nonlinearity is purely even, so the two
    targets coincide and the test would be vacuous.
    """
    fs, tp, p = 20.0, 100.0, 20
    n, dt = int(fs * tp) * p, 1.0 / 20.0
    h = odd_linear_grid(h_max=59, L=5, seed=0)
    freqs, amps = h / tp, np.full(h.size, 1.0)
    system = hallemans_2021_system()

    G, _ = best_linear_approximation(system, freqs, amps, dt, n * dt,
                                     n_realisations=60, seed=0)
    phases = np.random.default_rng(7).uniform(0, 2 * np.pi, freqs.size)
    sim = simulate_block_oriented(n * dt, dt, freqs, amps, phases, system,
                                  trend_coefficients=[0.0, 5.0, 3.0],
                                  noise_std_V=0.005, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = bltva(sim['voltage_V'], sim['current_A'], fs, p, h,
                  Np=2, Nq=5, harmonics="all_nl", band_halfwidth=5 * p)

    scale = np.abs(G[:, 0])
    for branch in (0, 2):
        to_bla = np.median(np.abs(r.H_p[:, branch] - G[:, branch]) / scale)
        to_linear = np.median(
            np.abs(r.H_p[:, branch] - sim['H_linear'][:, branch]) / scale)
        assert to_bla < to_linear / 2, (
            f"branch {branch}: {to_bla:.2e} from the BLA but "
            f"{to_linear:.2e} from the linear part -- expected much closer "
            "to the BLA")


def test_distortion_is_detected_on_the_paper_system():
    """Section V applied to a system with known nonlinear blocks.

    The paper reports distortion at least 20 dB below the linear part, which
    is the regime where a linear model is worth making; the SNDR here should
    land in the same range rather than at either extreme.
    """
    fs, tp, p = 20.0, 100.0, 20
    n, dt = int(fs * tp) * p, 1.0 / 20.0
    h = odd_linear_grid(h_max=59, L=5, seed=0)
    freqs, amps = h / tp, np.full(h.size, 1.0)
    phases = np.random.default_rng(7).uniform(0, 2 * np.pi, freqs.size)
    sim = simulate_block_oriented(n * dt, dt, freqs, amps, phases,
                                  hallemans_2021_system(),
                                  noise_std_V=0.005, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = bltva(sim['voltage_V'], sim['current_A'], fs, p, h,
                  Np=2, Nq=5, harmonics="all_nl", band_halfwidth=5 * p)

    assert r.var_distortion_odd is not None
    sndr = np.median(r.sndr_db())
    assert 15.0 < sndr < 45.0, f"SNDR {sndr:.1f} dB is outside the plausible range"


def test_frozen_impedance_matches_the_legendre_reconstruction():
    G = np.array([[10 + 0j, 3 + 0j, 1 + 0j], [5 - 1j, 1.5 + 0j, 0.5 + 0j]])
    Z = frozen_impedance(G, N)

    assert Z.shape == (2, N)
    assert np.allclose(Z, G @ legendre_basis(N, 2).T)
    # b_0 = 1 and higher orders are mean-removed, so the time average is G_0
    assert np.allclose(Z.mean(axis=1), G[:, 0])
