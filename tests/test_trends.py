"""Tests for the standard additive drifts.

Two of the three have an external anchor and are checked against it: the
relaxation trend against the closed form of Hallemans 2022 eq. (31)-(32), and
the Legendre polynomial against the simulation package's own
``legendre_drift``. The logit has no anchor, so what is tested is the
behaviour the module docstring claims for it.
"""
import numpy as np
import pytest

from deistools.simulation.trends import (
    HARD_CUBIC, TRENDS, logit_trend, polynomial_trend, relaxation_trend,
    trend_by_name,
)
from deistools.simulation.models.legendre import legendre_drift

FS, DURATION = 1000.0, 160.0
N = int(FS * DURATION)
TIME = np.arange(N) / FS


def _legendre_residue(values, order, duration=DURATION, t=None):
    """Residue left by a least-squares Legendre fit, relative to the trend."""
    from numpy.polynomial.legendre import legvander
    t = TIME if t is None else t
    basis = legvander(2.0 * t / duration - 1.0, order)
    coefficients, *_ = np.linalg.lstsq(basis, values, rcond=None)
    return float(np.std(values - basis @ coefficients) / np.std(values))


# --------------------------------------------------------------------------- #
# Anchored against the paper and against the package's own basis
# --------------------------------------------------------------------------- #

def test_relaxation_matches_the_closed_form_of_the_2022_paper():
    """eq. (31)-(32): step response of K/(s^2 + c1 s + c0), evaluated at 2t.

    This is the form that lived in ``tests/test_detrending.py`` before the
    module existed, so agreement has to be exact rather than close.
    """
    K, c1, c0 = 5.92e-4, 1.25e-3, 9.87e-6
    natural = np.sqrt(c0)
    zeta = c1 / (2 * natural)
    ringing = natural * np.sqrt(1 - zeta ** 2)
    tt = 2.0 * TIME
    expected = (K / c0) * (
        1 - np.exp(-zeta * natural * tt)
        * (np.cos(ringing * tt)
           + zeta / np.sqrt(1 - zeta ** 2) * np.sin(ringing * tt)))

    assert np.array_equal(relaxation_trend(TIME, remove_mean=False), expected)


def test_legendre_basis_agrees_with_the_simulation_model():
    """``basis="legendre"`` must be the same polynomial the oracle drifts by,
    so that a trend built here is inside BLTVA's Nq block by construction.

    ``legendre_drift`` subtracts the sampled mean of each basis function, as
    the estimator does; this module leaves that to ``remove_mean``. So the two
    agree once both are centred, and the small difference before that is
    exactly that mean.
    """
    coefficients = [0.0, 0.5, 0.2, 0.1]
    theirs = legendre_drift(TIME, DURATION, coefficients)
    mine = polynomial_trend(TIME, coefficients=coefficients, basis="legendre",
                            remove_mean=False)

    centred = lambda v: v - v.mean()
    assert np.allclose(centred(mine), centred(theirs), atol=1e-12)
    assert not np.allclose(mine, theirs, atol=1e-12), "the mean should differ"


def test_the_hard_cubic_reproduces_what_example_09_used_to_carry():
    """The cubic was inline in the example before this module existed. It is
    not from a paper -- which is why it is a named constant with a docstring
    saying so -- but it has to stay reproducible or the example's printed
    numbers change."""
    x = TIME / DURATION
    inline = 48.0 * (0.5 * x ** 3 - 1.1 * x ** 2 + 0.7 * x)
    inline -= inline.mean()

    rebuilt = polynomial_trend(TIME, coefficients=HARD_CUBIC,
                               basis="monomial", scale=48.0)
    assert np.allclose(rebuilt, inline, atol=1e-10)
    assert np.ptp(rebuilt) == pytest.approx(6.62, abs=0.01)


# --------------------------------------------------------------------------- #
# The three shapes really are different in kind
# --------------------------------------------------------------------------- #

def test_a_polynomial_trend_is_removed_exactly_at_its_own_order():
    cubic = polynomial_trend(TIME, coefficients=HARD_CUBIC, basis="monomial",
                             scale=48.0)
    assert _legendre_residue(cubic, 3) < 1e-12
    assert _legendre_residue(cubic, 1) > 0.1, "but not at a lower order"


def test_the_logit_resists_a_polynomial_fit_at_every_order():
    values = logit_trend(TIME, peak_to_peak=6.6)
    residues = [_legendre_residue(values, q) for q in (3, 8, 15, 25)]
    assert residues == sorted(residues, reverse=True), residues
    assert residues[0] > 1e-2, residues
    assert residues[-1] > 1e-8, "should still not be at round-off by order 25"


def test_the_logit_is_as_hard_on_a_short_record_as_on_a_long_one():
    """Its shape is fixed in normalised time, so its difficulty cannot be
    changed by accident -- unlike the relaxation trend below."""
    short_t = np.arange(16_000) / 100.0
    long_t = np.arange(200_000) / 100.0
    short = _legendre_residue(logit_trend(short_t, duration=160.0), 8,
                              duration=160.0, t=short_t)
    long = _legendre_residue(logit_trend(long_t, duration=2000.0), 8,
                             duration=2000.0, t=long_t)
    assert short == pytest.approx(long, rel=0.05), (short, long)


def test_the_relaxation_trend_is_only_hard_on_a_long_enough_record():
    """The trap the module docstring warns about, as a number.

    The published shape rings with a period near 1000 s. A record that sees
    only its smooth leading edge is fitted by a modest polynomial; one that
    sees a cycle or more is not. Nine orders of magnitude between them, from
    the same trend.
    """
    residues = {}
    for duration in (160.0, 2000.0):
        t = np.arange(int(duration * 10)) / 10.0
        residues[duration] = _legendre_residue(
            relaxation_trend(t, duration=duration), 8, duration=duration, t=t)

    assert residues[160.0] < 1e-8, residues
    assert residues[2000.0] > 1e-3, residues
    assert residues[2000.0] / residues[160.0] > 1e6


# --------------------------------------------------------------------------- #
# Parameters really are parameters
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_the_polynomial_order_follows_the_coefficients(order):
    coefficients = np.zeros(order + 1)
    coefficients[order] = 1.0
    values = polynomial_trend(TIME, coefficients=coefficients)
    assert _legendre_residue(values, order) < 1e-12
    if order > 1:
        assert _legendre_residue(values, order - 1) > 1e-3


@pytest.mark.parametrize("zeta", [0.2, 0.9, 1.0, 1.001, 2.0, 5.0])
def test_every_damping_regime_is_finite(zeta):
    """The closed form usually quoted divides by sqrt(1 - zeta^2), which is
    NaN at and above critical damping. Since the point of the module is that
    the parameters can be changed, all three branches are implemented."""
    natural = np.sqrt(9.87e-6)
    values = relaxation_trend(TIME, damping=2 * zeta * natural,
                              remove_mean=False)
    assert np.all(np.isfinite(values))
    assert np.all(np.diff(values) > 0) if zeta >= 1.0 else True


def test_the_damping_branches_join_up_at_critical():
    """A discontinuity at zeta = 1 would mean one of the three branches is
    wrong, and nothing else would reveal it."""
    natural = np.sqrt(9.87e-6)
    below, at, above = [
        relaxation_trend(TIME, damping=2 * z * natural, remove_mean=False)
        for z in (1.0 - 1e-6, 1.0, 1.0 + 1e-6)]
    assert np.allclose(below, at, rtol=1e-4)
    assert np.allclose(above, at, rtol=1e-4)


def test_the_logit_window_controls_the_severity():
    narrow = logit_trend(TIME, soc_start=0.2, soc_end=0.8, remove_mean=False)
    default = logit_trend(TIME, remove_mean=False)
    wide = logit_trend(TIME, soc_start=0.01, soc_end=0.99, remove_mean=False)
    assert np.ptp(narrow) < np.ptp(default) < np.ptp(wide)
    assert np.all(np.diff(default) > 0), "charging should rise"
    assert np.all(np.diff(
        logit_trend(TIME, soc_start=0.95, soc_end=0.05)) < 0), "discharge falls"


@pytest.mark.parametrize("soc", [0.0, 1.0, -0.1, 1.5])
def test_a_state_of_charge_outside_the_open_interval_is_refused(soc):
    with pytest.raises(ValueError, match="strictly inside"):
        logit_trend(TIME, soc_start=soc)


# --------------------------------------------------------------------------- #
# The shared conventions
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(TRENDS))
def test_every_trend_honours_the_shared_conventions(name):
    make = TRENDS[name]
    assert np.isclose(make(TIME).mean(), 0.0, atol=1e-12), "remove_mean"
    assert not np.isclose(make(TIME, remove_mean=False).mean(), 0.0)
    assert np.ptp(make(TIME, peak_to_peak=6.6)) == pytest.approx(6.6)
    assert make(TIME).shape == TIME.shape
    assert np.all(np.isfinite(make(TIME)))


def test_duration_is_inferred_as_the_record_and_not_the_last_sample():
    """``t[-1]`` is one sample short of ``N*dt``. The difference is O(1/N) and
    irrelevant to the shape, but it breaks an exact comparison against a
    figure computed the package's way."""
    explicit = polynomial_trend(TIME, duration=DURATION)
    inferred = polynomial_trend(TIME)
    assert np.array_equal(explicit, inferred)

    off_by_one = polynomial_trend(TIME, duration=TIME[-1])
    assert not np.array_equal(off_by_one, inferred)
    assert np.allclose(off_by_one, inferred, atol=1e-3)


def test_trend_by_name_lists_the_options_when_it_fails():
    with pytest.raises(ValueError, match="logit, polynomial, relaxation"):
        trend_by_name("exponential", TIME)
    assert np.array_equal(trend_by_name("logit", TIME), logit_trend(TIME))


def test_a_flat_trend_cannot_be_rescaled():
    with pytest.raises(ValueError, match="flat"):
        polynomial_trend(TIME, coefficients=[3.0], peak_to_peak=1.0)
