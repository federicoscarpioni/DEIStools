"""Nonlinear distortion variances, Hallemans TIM 2021 Section V.

    sigma2_Ys,class,p(k) = max(|Ys,class,0(k)|^2 - sigma2_V(k), 0)   p = 0
                         = |Ys,class,p(k)|^2                         p >= 1   eq. (69)

read at the harmonics that carry no excitation (eq. 70, 71), and folded into
the uncertainty through eq. (79).

The claim being tested is the one that justifies the whole odd-random-phase
construction: **even-order and odd-order distortion can be told apart**.
Because the excitation occupies only odd harmonics, even-order distortion
lands on the even ones, where nothing competes with it, and odd-order
distortion shows up on the odd harmonics the random omission left empty. Feed
the estimator a response with a known square-law or cube-law term and the
corresponding class must light up while the other stays at the floor.

This needs ``harmonics="all_nl"``: with ``"excited"`` there are no regressors
at unexcited harmonics and the distortion is unobservable by construction.
"""
import warnings

import numpy as np
import pytest

from deistools.processing.bltva import bltva
from deistools.simulation.models.legendre import simulate_legendre
from tests.conftest import GRIDS, rrc_legendre_coefficients

FS, TP, P = 50.0, 20.0, 20
N = int(FS * TP) * P
GRID = GRIDS["odd_linear_L5"]
FREQS = GRID / TP
FIT = dict(Np=2, Nq=5, harmonics="all_nl", band_halfwidth=5 * P)


def _linear_response(noise=1e-3, seed=1):
    a = rrc_legendre_coefficients(FREQS, variation=(0.30, 0.10))
    phases = np.random.default_rng(0).uniform(0, 2 * np.pi, GRID.size)
    return simulate_legendre(N / FS, 1 / FS, FREQS, np.full(GRID.size, 0.01),
                             phases, a, drift_coefficients=None,
                             noise_std_V=noise, noise_std_A=0.0, seed=seed)


def _run(response, sim, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bltva(response, sim['current_A'], FS, P, GRID, **{**FIT, **kw})


def _totals(r):
    return (np.nansum(r.var_distortion_even, axis=1),
            np.nansum(r.var_distortion_odd, axis=1))


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

def test_the_grid_excites_only_odd_harmonics():
    """The premise of the separation: without it, even-order distortion would
    fall on excited bins and be indistinguishable from the response."""
    assert np.all(GRID % 2 == 1)


def test_a_linear_system_shows_no_distortion():
    sim = _linear_response()
    even, odd = _totals(_run(sim['voltage_V'], sim))

    assert np.allclose(even, 0.0)
    assert np.allclose(odd, 0.0)


def test_square_law_distortion_appears_in_the_even_class():
    """y + a*y^2 generates even-order products, which land on even harmonics."""
    sim = _linear_response()
    v = sim['voltage_V']
    r = _run(v + 0.05 * v ** 2 / v.std(), sim)
    even, odd = _totals(r)

    assert np.median(even) > 1e-5, f"even distortion not detected: {np.median(even):.2e}"
    assert np.median(even) > 100 * max(np.median(odd), 1e-12), (
        f"even/odd separation failed: even={np.median(even):.2e} "
        f"odd={np.median(odd):.2e}")


def test_cube_law_distortion_appears_in_the_odd_class():
    """y + a*y^3 generates odd-order products, which land on the odd harmonics
    the random omission left unexcited."""
    sim = _linear_response()
    v = sim['voltage_V']
    r = _run(v + 0.05 * v ** 3 / v.std() ** 2, sim)
    even, odd = _totals(r)

    assert np.median(odd) > 1e-5, f"odd distortion not detected: {np.median(odd):.2e}"
    assert np.median(odd) > 100 * max(np.median(even), 1e-12), (
        f"odd/even separation failed: odd={np.median(odd):.2e} "
        f"even={np.median(even):.2e}")


@pytest.mark.parametrize("alpha", [0.02, 0.05, 0.10])
def test_distortion_grows_with_the_nonlinearity(alpha):
    sim = _linear_response()
    v = sim['voltage_V']
    weak = np.median(_totals(_run(v + 0.01 * v ** 3 / v.std() ** 2, sim))[1])
    strong = np.median(_totals(_run(v + alpha * v ** 3 / v.std() ** 2, sim))[1])

    assert strong > weak, f"alpha={alpha} gave {strong:.2e} against {weak:.2e}"


def test_the_bla_moves_with_the_nonlinearity():
    """The best linear approximation of a nonlinear system is not the
    underlying linear system, and it is not supposed to be.

    Adding a cube-law term feeds third-order products back onto the
    fundamental, so H_0 shifts -- 6.8% at alpha = 0.02, 51% at alpha = 0.15
    here. That shift is the quantity Section V exists to quantify, which is
    why the method reports a distortion variance rather than pretending the
    estimate is unaffected.
    """
    sim = _linear_response()
    v = sim['voltage_V']
    a = rrc_legendre_coefficients(FREQS, variation=(0.30, 0.10))
    clean = _run(v, sim)

    shifts = []
    for alpha in (0.02, 0.05, 0.15):
        r = _run(v + alpha * v ** 3 / v.std() ** 2, sim)
        shifts.append(np.median(
            np.abs(r.H_p[:, 0] - clean.H_p[:, 0]) / np.abs(a[:, 0])))

    assert shifts[0] < shifts[1] < shifts[2]
    assert shifts[0] > 1e-3, "a 2% cube-law term should move the BLA measurably"


def test_distortion_uncertainty_tracks_the_actual_bias():
    """eq. (79) must be the right size, not merely nonzero.

    The displacement of H_0 caused by the nonlinearity, divided by the
    predicted distortion standard deviation, is constant across nonlinearity
    strengths -- measured at 2.83 for every alpha tried. The error bar
    therefore scales with the bias it is describing, and the bias sits inside
    about three standard deviations, which is what a variance estimate should
    give for a single random-phase realisation.
    """
    sim = _linear_response()
    v = sim['voltage_V']
    clean = _run(v, sim)

    ratios = []
    for alpha in (0.02, 0.05, 0.08, 0.15):
        r = _run(v + alpha * v ** 3 / v.std() ** 2, sim)
        shift = np.abs(r.H_p[:, 0] - clean.H_p[:, 0])
        dist_std = np.sqrt(np.maximum(
            np.einsum("mpp->mp", r.cov_H_p_distortion).real, 0.0))[:, 0]
        ratios.append(np.median(shift / np.maximum(dist_std, 1e-30)))

    ratios = np.array(ratios)
    assert np.all(ratios < 5.0), f"bias far outside the error bar: {ratios}"
    assert np.ptp(ratios) / ratios.mean() < 0.2, (
        f"uncertainty does not scale with the bias: {ratios}")


# --------------------------------------------------------------------------- #
# Signal-to-nonlinear-distortion ratio
# --------------------------------------------------------------------------- #

def test_sndr_falls_as_the_nonlinearity_grows():
    """The number Section V says to use when deciding whether a linear model
    is acceptable."""
    sim = _linear_response()
    v = sim['voltage_V']
    ratios = [np.median(_run(v + a * v ** 3 / v.std() ** 2, sim).sndr_db())
              for a in (0.02, 0.05, 0.15)]

    assert ratios[0] > ratios[1] > ratios[2], f"SNDR not monotone: {ratios}"
    assert 0 < ratios[-1] < 60


def test_sndr_requires_unexcited_harmonics():
    sim = _linear_response()
    r = _run(sim['voltage_V'], sim, harmonics="excited")

    assert r.var_distortion_odd is None
    with pytest.raises(ValueError, match="no distortion estimate"):
        r.sndr_db()


# --------------------------------------------------------------------------- #
# Contribution to the uncertainty, eq. (79)
# --------------------------------------------------------------------------- #

def test_distortion_covariance_is_hermitian_psd():
    """eq. (79) subtracts a noise term and can overshoot; the result is
    projected back to the positive semidefinite cone."""
    sim = _linear_response()
    v = sim['voltage_V']
    r = _run(v + 0.05 * v ** 3 / v.std() ** 2, sim)

    for C in r.cov_H_p_distortion:
        assert np.allclose(C, C.conj().T)
        w = np.linalg.eigvalsh(C)
        # relative tolerance: the projection is exact only to the precision of
        # the eigendecomposition, and these matrices have entries of order 10
        assert w.min() > -1e-12 * max(abs(w).max(), 1.0)


def test_total_uncertainty_is_noise_plus_distortion():
    """eq. (76)."""
    sim = _linear_response()
    v = sim['voltage_V']
    r = _run(v + 0.05 * v ** 3 / v.std() ** 2, sim)

    assert np.allclose(r.cov_H_p, r.cov_H_p_noise + r.cov_H_p_distortion)
    assert np.all(r.std_H_p >= np.sqrt(
        np.maximum(np.einsum("mpp->mp", r.cov_H_p_noise).real, 0.0)) - 1e-18)


def test_distortion_inflates_the_error_bars():
    """A nonlinear system is less certainly described by a linear model, and
    the uncertainty must say so."""
    sim = _linear_response()
    v = sim['voltage_V']

    clean = _run(v, sim)
    distorted = _run(v + 0.08 * v ** 3 / v.std() ** 2, sim)

    assert np.median(distorted.std_H_p[:, 0]) > np.median(clean.std_H_p[:, 0])


def test_total_uncertainty_falls_back_to_noise_without_distortion():
    sim = _linear_response()
    r = _run(sim['voltage_V'], sim, harmonics="excited")

    assert r.cov_H_p_distortion is None
    assert np.array_equal(r.cov_H_p, r.cov_H_p_noise)
