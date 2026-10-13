"""Uncertainty of the BLTVA, Hallemans TIM 2021 Section VI.

    C_theta = K^+ C_V K^+H                        eq. (72)
    C_V     = diag(sigma2_V(k_1) ... )            eq. (73)
    C_G     = C_theta[target] / |U(k_e)|^2        eq. (76), (84)
    sigma2(jw, t) = b(t) C_G b(t)^T               eq. (87)

Only the noise contribution is implemented. The distortion term of eq. (79)
needs the nonlinear distortion coefficients of Section V, which this package
does not estimate, so ``cov_H_p`` is a lower bound on the total uncertainty
whenever the system is driven nonlinearly. On the linear simulations used
here the two coincide.

The central test is the paper's own validation method (Figs. 4 and 10):
compare the predicted standard deviation against the sample standard
deviation over many noise realisations. Everything else is a property check.
"""
import warnings

import numpy as np
import pytest

from deistools.processing.bltva import bltva, estimate_noise_variance
from tests.conftest import GRIDS, rrc_legendre_coefficients
from deistools.simulation.models.legendre import simulate_legendre

# A deliberately small record: the Monte Carlo test runs the estimator dozens
# of times, and the property under test does not depend on record length.
MC_FS, MC_TP, MC_P = 50.0, 20.0, 20
MC_N = int(MC_FS * MC_TP) * MC_P
MC_GRID = GRIDS["odd_linear_L5"]
MC_FREQS = MC_GRID / MC_TP
FIT = dict(Np=2, Nq=5, harmonics="excited", band_halfwidth=5 * MC_P)


def _realisation(noise, seed):
    a = rrc_legendre_coefficients(MC_FREQS, variation=(0.30, 0.10))
    phases = np.random.default_rng(0).uniform(0, 2 * np.pi, MC_GRID.size)
    sim = simulate_legendre(MC_N / MC_FS, 1 / MC_FS, MC_FREQS,
                            np.full(MC_GRID.size, 0.01), phases, a,
                            drift_coefficients=None, noise_std_V=noise,
                            noise_std_A=0.0, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bltva(sim['voltage_V'], sim['current_A'], MC_FS, MC_P,
                     MC_GRID, **FIT)


# --------------------------------------------------------------------------- #
# Monte Carlo validation
# --------------------------------------------------------------------------- #

def test_predicted_std_matches_the_sample_std_over_realisations():
    """The paper's validation: predicted variance against sample variance.

    With ``runs`` realisations the sample standard deviation itself carries a
    relative uncertainty of about ``1/sqrt(2*(runs-1))``, so the tolerance is
    set to accommodate that rather than to demand exactness.
    """
    runs, noise = 40, 2e-3
    estimates = []
    predicted = None
    for r in range(runs):
        res = _realisation(noise, seed=1000 + r)
        estimates.append(res.H_p)
        if predicted is None:
            predicted = res.std_H_p.copy()

    sample = np.asarray(estimates).std(axis=0, ddof=1)
    ratio = predicted / sample

    assert 0.85 < np.median(ratio) < 1.15, f"median ratio {np.median(ratio):.3f}"
    assert ratio.min() > 0.6, f"worst underestimate {ratio.min():.3f}"
    assert ratio.max() < 1.6, f"worst overestimate {ratio.max():.3f}"


def test_uncertainty_scales_with_the_noise_amplitude():
    """sigma scales linearly with the noise standard deviation, eq. (72)-(73)."""
    low = _realisation(1e-3, seed=7).std_H_p
    high = _realisation(1e-2, seed=7).std_H_p

    ratio = np.median(high / low)
    assert 8.0 < ratio < 12.0, f"sigma scaled by {ratio:.2f}x for 10x the noise"


def test_uncertainty_collapses_without_noise():
    noisy = _realisation(1e-3, seed=11).std_H_p
    clean = _realisation(0.0, seed=11).std_H_p
    assert clean.max() < noisy.min() / 10


# --------------------------------------------------------------------------- #
# Structure of the covariance
# --------------------------------------------------------------------------- #

def test_covariance_is_hermitian_and_positive_semidefinite(make_oracle):
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-3)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=2, Nq=5, harmonics="excited", band_halfwidth=5 * o.P)

    assert r.cov_H_p.shape == (o.H_exc.size, 3, 3)
    for C in r.cov_H_p:
        assert np.allclose(C, C.conj().T), "covariance is not Hermitian"
        assert np.linalg.eigvalsh(C).min() > -1e-18, "covariance is not PSD"


def test_pruned_coefficients_carry_no_uncertainty(make_oracle):
    """Algorithm 2 sets unsupported coefficients to exactly zero; a coefficient
    that was never estimated has no variance of its own."""
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-4)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=6, Nq=5, harmonics="excited", band_halfwidth=5 * o.P)

    pruned = r.H_p == 0
    assert pruned.any(), "nothing was pruned, so this proves nothing"
    assert np.all(r.std_H_p[pruned] == 0.0)


def test_std_is_the_square_root_of_the_diagonal(make_oracle):
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-3)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=2, Nq=5, harmonics="excited", band_halfwidth=5 * o.P)

    expected = np.sqrt(np.einsum("mpp->mp", r.cov_H_p).real)
    assert np.allclose(r.std_H_p, expected)


# --------------------------------------------------------------------------- #
# Uncertainty over time, eq. (87)
# --------------------------------------------------------------------------- #

def test_variance_over_time_has_the_right_shape(make_oracle):
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-3)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=2, Nq=5, harmonics="excited", band_halfwidth=5 * o.P)

    idx = np.arange(0, o.N, 1000)
    assert r.variance_over_time(idx).shape == (o.H_exc.size, idx.size)
    assert np.all(r.variance_over_time(idx) >= 0)
    assert np.allclose(r.std_over_time(idx) ** 2, r.variance_over_time(idx))


def test_uncertainty_is_largest_at_the_edges_of_the_record(make_oracle):
    """|L_p| peaks at the ends of the window, so eq. (87) makes the estimate
    least certain there.

    This is why the paper plots its time-varying impedances over
    [T/20, T - T/20] rather than the full record (Fig. 8).
    """
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-3)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=2, Nq=5, harmonics="excited", band_halfwidth=5 * o.P)

    idx = np.array([0, o.N // 2, o.N - 1])
    var = r.variance_over_time(idx)

    assert np.all(var[:, 0] > var[:, 1]), "start should be less certain than centre"
    assert np.all(var[:, 2] > var[:, 1]), "end should be less certain than centre"


def test_time_average_variance_relates_to_h0(make_oracle):
    """b_0 = 1 and the higher orders are mean-removed, so at the midpoint of
    the record the variance is dominated by the H_0 term."""
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-3)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=2, Nq=5, harmonics="excited", band_halfwidth=5 * o.P)

    mid = r.variance_over_time(np.array([o.N // 2]))[:, 0]
    var_h0 = r.std_H_p[:, 0] ** 2
    assert np.all(mid < 3 * var_h0)


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #

def test_no_uncertainty_without_a_noise_estimate(make_oracle):
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-3)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Np=2, Nq=5,
              harmonics="excited", band_halfwidth=5 * o.P, prune=False)

    assert r.cov_H_p_noise is None
    for access in (lambda: r.cov_H_p, lambda: r.std_H_p, r.variance_over_time):
        with pytest.raises(ValueError, match="no uncertainty available"):
            access()


def test_uncertainty_available_without_pruning(make_oracle):
    """prune and uncertainty are independent: an explicit noise estimate gives
    error bars on an unpruned fit."""
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ne = estimate_noise_variance(o.voltage, o.current, o.fs, o.P, o.H_exc,
                                     Np=2, Nq=5)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Np=2, Nq=5,
              harmonics="excited", band_halfwidth=5 * o.P,
              noise=ne, prune=False)

    assert r.cov_H_p is not None
    assert all(d["n_unsupported_dropped"] == 0 for d in r.diagnostics)
    assert np.all(r.std_H_p > 0)
