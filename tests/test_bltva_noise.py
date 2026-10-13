"""Tests for Algorithm 1, the noise variance estimator.

Hallemans TIM 2021 Section III. The testable property is sharp: for white
output noise of standard deviation ``s``, the 1/N-scaled DFT has

    Var(V(k)) = s**2 / N

so the estimator has a known target rather than a self-consistency check. It
must hit that target across decades of noise, without being inflated by the
deterministic content it is supposed to fit away -- the skirts of a
time-varying system and a dominant drift.

Note on eq. (53)
----------------
The published rank threshold reads "sigma_{i+1}/sigma_1 < a snr", which taken
literally truncates at m = 1 always. ``test_rank_threshold_is_unbiased``
pins the reading actually implemented, 1/(a*snr), by asserting the property
that distinguishes it: freedom from bias. The literal reading fails it by a
factor of 2 to 6.
"""
import warnings

import numpy as np
import pytest

from deistools.processing.bltva import (
    estimate_noise_variance, _noise_band_bins, _noise_regressors,
    legendre_basis, legendre_spectra,
)
from tests.conftest import GRIDS, ORACLE_P

GRID = GRIDS["quasi_log_15"]
FIT = dict(Np=2, Nq=5, Na=6)


@pytest.fixture
def noisy(make_oracle):
    """Oracle dataset with a given white output-noise level."""
    def _make(noise, drift=None, seed=0):
        return make_oracle(grid="quasi_log_15", variation=(0.30, 0.10),
                           drift=drift, noise_std_V=noise, seed=seed)
    return _make


def _estimate(o, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimate_noise_variance(o.voltage, o.current, o.fs, o.P, o.H_exc,
                                       **{**FIT, **kw})


def _expected(noise, N):
    return noise ** 2 / N


# --------------------------------------------------------------------------- #
# Recovery of a known noise floor
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("noise", [1e-2, 1e-3, 1e-4])
def test_recovers_white_noise_floor(noisy, noise):
    """The mean estimate must match s**2/N at every noise level."""
    o = noisy(noise)
    ne = _estimate(o)

    ratio = ne.sigma2.mean() / _expected(noise, o.N)
    assert 0.85 < ratio < 1.15, (
        f"noise variance off by {ratio:.3f}x at s={noise:.0e}")


def test_scales_quadratically_with_noise_amplitude(noisy):
    """Ten times the noise amplitude must give a hundred times the variance."""
    a = _estimate(noisy(1e-4)).sigma2.mean()
    b = _estimate(noisy(1e-3)).sigma2.mean()

    assert 85 < b / a < 115, f"variance scaled by {b / a:.1f}x, expected ~100x"


def test_rank_threshold_is_unbiased(noisy):
    """Bias must not track the noise level.

    This is what separates the implemented reading of eq. (53) from the
    literal one: under the literal threshold the estimate is inflated by a
    factor that grows as the noise falls (1.9x, 4.0x, 5.8x over these three
    levels), because too little deterministic content is removed.
    """
    ratios = []
    for s in (1e-2, 1e-3, 1e-4):
        o = noisy(s)
        ratios.append(_estimate(o).sigma2.mean() / _expected(s, o.N))

    assert max(ratios) / min(ratios) < 1.25, (
        f"bias varies with noise level: ratios {ratios}")


def test_dominant_drift_does_not_inflate_the_estimate(noisy):
    """A trend is deterministic content the band model absorbs, not noise."""
    o = noisy(1e-3, drift=[0.5, 0.20, 0.05])
    ratio = _estimate(o).sigma2.mean() / _expected(1e-3, o.N)

    assert 0.85 < ratio < 1.25, f"drift inflated the noise estimate by {ratio:.3f}x"


def test_collapses_on_noiseless_data(noisy):
    """With no noise the estimate must fall far below any real noise floor."""
    o = noisy(0.0)
    ne = _estimate(o)

    assert np.median(ne.sigma2) < _expected(1e-6, o.N), (
        f"median {np.median(ne.sigma2):.2e} is not negligible")


@pytest.mark.parametrize("dof", [3, 5, 10, 20])
def test_unbiased_across_degrees_of_freedom(noisy, dof):
    o = noisy(1e-3)
    ratio = _estimate(o, dof=dof).sigma2.mean() / _expected(1e-3, o.N)
    assert 0.85 < ratio < 1.15, f"dof={dof} gives ratio {ratio:.3f}"


# --------------------------------------------------------------------------- #
# Band construction
# --------------------------------------------------------------------------- #

def test_band_excludes_harmonic_bins():
    """eq. (38): K_nl is removed, because at a harmonic the model cannot
    separate noise from dynamics."""
    P = 20
    K_w = _noise_band_bins(k_c=100, n=13, N=100_000, P=P)

    assert K_w.size > 0
    assert np.all(K_w % P != 0), "a harmonic bin survived in the noise band"
    assert np.all(K_w >= 1) and np.all(K_w < 50_000)


def test_band_halfwidth_follows_eq39(noisy):
    """n = min(ceil((dof + n_theta_max + 2)/2), P) with n_theta_max = Nq+3Np+Na+1."""
    o = noisy(1e-3)
    for dof in (3, 5, 10):
        ne = _estimate(o, dof=dof)
        n_theta_max = FIT["Nq"] + 3 * FIT["Np"] + FIT["Na"] + 1
        expected = min(int(np.ceil((dof + n_theta_max + 2) / 2)), ORACLE_P)
        assert ne.n == expected, f"dof={dof}: n={ne.n}, expected {expected}"


def test_hyperbola_regressors_are_finite():
    """The 1/(k-k')^p columns are only safe because the harmonics were removed
    from the band; a surviving harmonic would divide by zero."""
    N, P = 100_000, 20
    B = legendre_spectra(legendre_basis(N, 5))
    K_w = _noise_band_bins(k_c=200, n=13, N=N, P=P)
    K = _noise_regressors(K_w, 200, N, P, B, Np=2, Nq=5, Na=6)

    assert np.all(np.isfinite(K))
    assert K.shape[0] == K_w.size


def test_raises_when_band_cannot_support_the_model(noisy):
    """Too few periods for the requested orders must fail loudly.

    eq. (49) catches this before the band is even built, so the message names
    the condition rather than reporting a shape mismatch downstream.
    """
    o = noisy(1e-3)
    with pytest.raises(ValueError, match=r"eq\. \(49\)"):
        estimate_noise_variance(o.voltage, o.current, o.fs, 2, o.H_exc,
                                Np=8, Nq=8, Na=10)


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #

def test_covers_the_excited_range(noisy):
    o = noisy(1e-3)
    ne = _estimate(o)
    assert ne.k.min() <= o.P * o.H_exc.min()
    assert ne.k.max() >= o.P * o.H_exc.max()
    assert ne.sigma2.shape == ne.k.shape == ne.m.shape
    assert np.all(ne.sigma2 >= 0)


def test_at_interpolates_between_band_centres(noisy):
    o = noisy(1e-3)
    ne = _estimate(o)

    assert np.allclose(ne.at(ne.k), ne.sigma2)

    mid = 0.5 * (ne.k[:-1] + ne.k[1:])
    interp = ne.at(mid)
    lo = np.minimum(ne.sigma2[:-1], ne.sigma2[1:])
    hi = np.maximum(ne.sigma2[:-1], ne.sigma2[1:])
    assert np.all((interp >= lo - 1e-30) & (interp <= hi + 1e-30))


def test_step_subsampling_preserves_the_level(noisy):
    """Subsampling the sweep is a cost optimisation, not a change of answer."""
    o = noisy(1e-3)
    coarse = _estimate(o).sigma2.mean()
    fine = _estimate(o, step=3).sigma2.mean()

    assert 0.85 < coarse / fine < 1.15
