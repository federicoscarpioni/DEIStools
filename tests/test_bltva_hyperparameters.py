"""Hyperparameter selection and the identifiability bound.

Two rules from the papers, previously neither implemented nor enforced:

* ``select_na`` -- TIM 2022 eq. (22), ``Na = floor(3P/4) - Np - d + 1``.
* ``check_identifiability`` -- TIM 2021 eq. (49), ``Nq + 3*Np + Na + 1 < 2P - 1``.

The module used to warn on ``Np >= floor(P/3)`` instead, a condition that
appears in none of the three papers. It was simultaneously too strict (it
fires at P=20, Np=6, where eq. (49) has ample headroom) and too permissive
(it ignores Nq and Na entirely, so it passed the old stationary test
configuration that eq. (49) rejects).
"""
import numpy as np
import pytest

from deistools.processing.bltva import bltva, select_na, check_identifiability
from deistools.processing import MultiFrequencyAnalysis
from tests.conftest import ORACLE_P

FIT = dict(Nq=5, harmonics="excited", band_halfwidth=5 * ORACLE_P)


# --------------------------------------------------------------------------- #
# select_na, eq. (22)
# --------------------------------------------------------------------------- #

def test_select_na_matches_the_published_example():
    """TIM 2022 Section VI: P=20, Np=4, d=3 is stated to give Na=9."""
    assert select_na(P=20, Np=4, d=3) == 9


@pytest.mark.parametrize("P,Np,d", [(20, 2, 0), (40, 10, 0), (40, 10, 1),
                                    (29, 8, 2), (20, 4, 3)])
def test_select_na_follows_the_formula(P, Np, d):
    assert select_na(P, Np, d) == int(np.floor(3 * P / 4)) - Np - d + 1


def test_select_na_without_differencing_drops_the_d_term():
    assert select_na(P=20, Np=2) == select_na(P=20, Np=2, d=0)
    assert select_na(P=20, Np=2) == 14


def test_select_na_raises_when_it_would_go_negative():
    with pytest.raises(ValueError, match=r"eq\. \(22\)"):
        select_na(P=4, Np=10)


# --------------------------------------------------------------------------- #
# check_identifiability, eq. (49)
# --------------------------------------------------------------------------- #

def test_check_identifiability_returns_n_theta_max():
    assert check_identifiability(P=20, Np=2, Nq=5, Na=6) == 5 + 6 + 6 + 1


def test_check_identifiability_accepts_the_boundary_from_below():
    """n_theta_max = 2P - 2 passes; 2P - 1 does not."""
    P = 20
    Na = (2 * P - 2) - (5 + 3 * 2 + 1)          # make n_theta_max exactly 2P-2
    assert check_identifiability(P, Np=2, Nq=5, Na=Na) == 2 * P - 2
    with pytest.raises(ValueError, match=r"eq\. \(49\)"):
        check_identifiability(P, Np=2, Nq=5, Na=Na + 1)


def test_bltva_rejects_an_unidentifiable_configuration(make_oracle):
    o = make_oracle(grid="quasi_log_15")
    with pytest.raises(ValueError, match=r"eq\. \(49\)"):
        bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=10, Nq=5, Na=10, **{k: v for k, v in FIT.items() if k != "Nq"})


def test_old_stationary_configuration_is_now_rejected():
    """The configuration the stationary tests used before this work --
    P=10, Np=4, Nq=4, Na=6 -- gives n_theta_max = 23 against 2P-1 = 19.

    It passed only because a stationary signal is degenerate: with no time
    variation almost any hyperparameter choice recovers Z. The old
    ``Np < floor(P/3)`` warning did fire, but as a warning it was ignored.
    """
    with pytest.raises(ValueError, match=r"eq\. \(49\)"):
        check_identifiability(P=10, Np=4, Nq=4, Na=6)


def test_no_warning_is_emitted_for_a_valid_configuration(make_oracle):
    """The removed Np < floor(P/3) rule used to fire at P=20, Np=6, which
    eq. (49) accepts comfortably (n_theta_max = 30 < 39)."""
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Np=6, Na=6, **FIT)


# --------------------------------------------------------------------------- #
# Na=None wiring
# --------------------------------------------------------------------------- #

def test_auto_na_matches_an_explicit_equivalent(make_oracle):
    """Na=None must behave exactly as passing select_na(P, Np)."""
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-4)

    auto = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Np=2, Na=None, **FIT)
    explicit = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Np=2,
                     Na=select_na(o.P, 2), **FIT)

    assert np.allclose(auto.H_p, explicit.H_p)


def test_auto_na_recovers_the_oracle(make_oracle):
    """The automatically chosen order must estimate as well as the hand-picked
    one it replaces. Na=14 here against the old default of 6."""
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-4)

    auto = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Np=2, Na=None, **FIT)
    err = o.relative_error(auto.H_p)

    assert select_na(o.P, 2) == 14
    assert err[:, 0].max() < 1e-4, f"H_0 error {err[:, 0].max():.2e}"
    assert err[:, 1:].max() < 5e-4, f"H_p>=1 error {err[:, 1:].max():.2e}"


def test_mfa_exposes_the_selectors():
    assert MultiFrequencyAnalysis.select_na(20, 4, 3) == 9
    assert MultiFrequencyAnalysis.check_identifiability(20, 2, 5, 6) == 18
