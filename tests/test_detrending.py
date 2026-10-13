"""Tests for trend removal, Hallemans et al. IEEE TIM 71 (2022) 6501711.

The paper gives two theorems that pin the implementation exactly, so most of
this file checks against those rather than against itself:

* **Theorem 1**, eq. (12): ``Y_d(k) = F_d(k) Y(k) + L_d(k)``. This ties
  ``apply``, ``frequency_response`` and ``boundary_coefficients`` together --
  if any one of them is wrong the identity breaks.
* **Theorem 2**: differencing a degree-``d`` polynomial ``d`` times leaves
  *only* boundary effects, a polynomial of degree ``d-1`` in ``z_k^-1``.
  This is why a polynomial trend vanishes and why the boundary regressors
  are needed.

The headline behavioural test reproduces the comparison of the paper's
Section VI: against a trend no finite polynomial captures, differencing
recovers the low frequencies that parametric trend removal loses.
"""
import warnings

import numpy as np
import pytest

from deistools.processing.bltva import (
    bltva, build_regressors, estimate_noise_variance, legendre_basis,
    legendre_spectra, select_na,
)
from deistools.processing.detrending import (
    Detrending, DiagonalDetrending, Differencing, LowPassRemoval,
    MovingAverage, PolynomialFit, check_response, select_differencing_order,
)
from tests.conftest import GRIDS, ORACLE_P

FIT = dict(Np=2, harmonics="excited", band_halfwidth=5 * ORACLE_P)


def hallemans_2022_trend(t):
    """The trend of 2022 eq. (31)-(32): step response of

        C(s) = 5.92e-4 / (s^2 + 1.25e-3 s + 9.87e-6)

    evaluated at 2t. The paper chooses it precisely because it "cannot be
    exactly captured by a finite number of polynomials", which is what makes
    parametric trend removal struggle with it.
    """
    K, c1, c0 = 5.92e-4, 1.25e-3, 9.87e-6
    wn = np.sqrt(c0)
    zeta = c1 / (2 * wn)
    wd = wn * np.sqrt(1 - zeta ** 2)
    tt = 2.0 * np.asarray(t)
    return (K / c0) * (1 - np.exp(-zeta * wn * tt) *
                       (np.cos(wd * tt) + zeta / np.sqrt(1 - zeta ** 2) * np.sin(wd * tt)))


@pytest.fixture
def trended(make_oracle):
    """Oracle dataset with the 2022 paper's dominant non-polynomial trend."""
    def _make(noise=5e-3, scale=1.0):
        o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10),
                        noise_std_V=noise)
        o.sim['voltage_V'] = o.voltage + scale * hallemans_2022_trend(o.sim['time_s'])
        return o
    return _make


# --------------------------------------------------------------------------- #
# Theorems
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_theorem_1_relates_differenced_and_original_spectra(d):
    """eq. (12): Y_d(k) = F_d(k) Y(k) + L_d(k)."""
    rng = np.random.default_rng(0)
    N = 4096
    t = np.arange(N)
    y = 3.0 + 0.01 * t + np.sin(2 * np.pi * 7 * t / N) + rng.normal(0, 0.1, N)

    det = Differencing(d)
    Y_d = np.fft.fft(det.apply(y)) / N
    Y = np.fft.fft(y) / N

    k = np.arange(N)
    F = det.frequency_response(k, N)
    lam = det.boundary_coefficients(y)
    L = np.exp(-2j * np.pi * np.outer(k, np.arange(d)) / N) @ lam / N

    err = np.abs(Y_d - (F * Y + L)).max() / np.abs(Y_d).max()
    assert err < 1e-10, f"Theorem 1 violated by {err:.2e}"


@pytest.mark.parametrize("d", [1, 2, 3])
def test_theorem_2_polynomial_trend_leaves_only_boundary_terms(d):
    """A degree-d polynomial, differenced d times, has a DFT at k != 0 that is
    exactly a degree-(d-1) polynomial in z_k^-1.

    This is the mechanism by which the trend disappears: F_d(k)*P(k) and the
    boundary term L_d(k) cancel, leaving only the boundary contribution.
    """
    N = 2048
    n = np.arange(N)
    rng = np.random.default_rng(1)
    coeffs = rng.uniform(1.0, 3.0, d + 1)
    p = sum(c * (n / N) ** m for m, c in enumerate(coeffs))

    P_d = np.fft.fft(Differencing(d).apply(p)) / N

    k = np.arange(1, N // 2)                       # k != 0
    basis = np.exp(-2j * np.pi * np.outer(k, np.arange(d)) / N)
    resid = P_d[k] - basis @ np.linalg.lstsq(basis, P_d[k], rcond=None)[0]

    rel = np.linalg.norm(resid) / np.linalg.norm(P_d[k])
    assert rel < 1e-8, f"residual beyond the boundary basis: {rel:.2e}"


# --------------------------------------------------------------------------- #
# Differencing mechanics
# --------------------------------------------------------------------------- #

def test_apply_composes():
    rng = np.random.default_rng(2)
    y = rng.normal(size=500)
    assert np.allclose(Differencing(3).apply(y),
                       Differencing(1).apply(Differencing(2).apply(y)))


def test_first_difference_is_the_plain_difference_with_zero_prefix():
    y = np.array([2.0, 5.0, 9.0, 14.0])
    assert np.allclose(Differencing(1).apply(y), [2.0, 3.0, 4.0, 5.0])


def test_frequency_response_is_a_highpass_vanishing_at_dc():
    N = 1024
    F = Differencing(1).frequency_response(np.arange(N // 2), N)
    assert F[0] == pytest.approx(0.0)
    assert np.all(np.diff(np.abs(F)) > 0), "|F| must increase with k up to Nyquist"


@pytest.mark.parametrize("d,expected", [(1, 0), (2, 1), (3, 2), (5, 4)])
def test_extra_regressor_count_follows_eq21(d, expected):
    """eq. (21) sums m = 1..d-1: the m = 0 constant is already spanned by the
    transient polynomial, so including it would be redundant."""
    det = Differencing(d)
    assert det.n_extra_params == expected
    assert det.extra_regressors(np.arange(1, 50), 1024).shape == (49, expected)


def test_order_zero_is_rejected():
    with pytest.raises(ValueError, match="order must be >= 1"):
        Differencing(0)


def test_detrending_is_abstract():
    """A strategy that cannot report its frequency response cannot be used:
    the weighting of eq. (26) would be impossible."""
    with pytest.raises(TypeError):
        Detrending()


# --------------------------------------------------------------------------- #
# Order selection, eq. (11)
# --------------------------------------------------------------------------- #

def test_select_order_finds_a_trend(trended):
    o = trended()
    d = select_differencing_order(o.voltage, o.fs, o.freqs.max())
    assert 1 <= d <= 3, f"selected d={d} for a strongly trended record"


def test_select_order_returns_zero_for_a_white_series():
    """With nothing to remove, any differencing raises the variance."""
    y = np.random.default_rng(3).normal(size=20_000)
    assert select_differencing_order(y, fs=100.0, f_max=40.0) == 0


def test_select_order_is_biased_by_a_low_pass_response(make_oracle):
    """eq. (11) assumes 'trend plus random series'. A strongly low-pass
    response violates that: its variance is dominated by the lowest excited
    frequencies, which differencing attenuates, so the criterion recommends
    differencing a record that has no trend.

    Recorded rather than worked around -- it is a property of the published
    rule, and the reason ``select_differencing_order`` is documented as a
    starting point rather than an answer.
    """
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-3)
    assert select_differencing_order(o.voltage, o.fs, o.freqs.max()) >= 1


def test_select_order_responds_to_an_added_trend(trended, make_oracle):
    clean = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=5e-3)
    o = trended()
    d_clean = select_differencing_order(clean.voltage, clean.fs, clean.freqs.max())
    d_trend = select_differencing_order(o.voltage, o.fs, o.freqs.max())
    assert d_trend >= d_clean


def test_select_order_needs_enough_samples():
    with pytest.raises(ValueError, match="too few"):
        select_differencing_order(np.arange(10.0), fs=1000.0, f_max=1.0)


# --------------------------------------------------------------------------- #
# Integration with the estimator
# --------------------------------------------------------------------------- #

def test_differencing_beats_parametric_removal_at_low_frequency(trended):
    """The result of 2022 Section VI, Fig. 5.

    Against a trend no finite polynomial captures, parametric removal with few
    coefficients loses the lowest frequencies -- the paper reports that it
    "fails at estimating the low frequencies correctly" with Nq = 2 -- while
    differencing recovers them. Measured here: 1.35 relative error against
    7.8e-03 at the lowest excited frequency.
    """
    o = trended()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parametric = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Nq=2, **FIT)
        differenced = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Nq=5,
                            detrend=Differencing(1), **FIT)

    e_par = o.relative_error(parametric.H_p)[0, 0]
    e_dif = o.relative_error(differenced.H_p)[0, 0]

    assert e_dif < 5e-2, f"differencing did not recover the lowest band: {e_dif:.2e}"
    assert e_dif < e_par / 20, (
        f"expected a large gain at low frequency: {e_dif:.2e} vs {e_par:.2e}")


def test_detrending_leaves_high_frequencies_alone(trended):
    """A trend is a low-frequency phenomenon, so the upper bands must be the
    same with and without detrending."""
    o = trended()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Nq=5, **FIT)
        diffed = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Nq=5,
                       detrend=Differencing(1), **FIT)

    top = slice(len(o.H_exc) // 2, None)
    assert np.allclose(plain.H_p[top], diffed.H_p[top], rtol=1e-2)


def test_detrending_drops_the_trend_block_and_adds_boundary_columns(trended):
    """eq. (21): with the trend removed rather than modelled, the drift block
    goes and the filter's boundary terms take its place."""
    import deistools.processing.bltva as M

    o = trended()
    det = Differencing(3)
    Na = select_na(o.P, 2, d=det.order)
    args = (o.voltage, o.current, o.fs, o.P, o.H_exc, 2, 5, Na, "excited")

    _, _, _, k_exc, centres, _, B, V, _ = M._prepare(*args, det)
    N = o.N
    K_w = M._drop_ill_weighted_bins(
        M._band_bins(k_exc, 3, N, 5 * o.P), int(k_exc[3]), N, det)
    _, cmap = M.build_regressors(K_w, int(k_exc[3]), [], B, 2, 5, Na, det)

    assert "drift" not in cmap
    assert cmap["boundary"].stop - cmap["boundary"].start == det.n_extra_params


def test_na_shrinks_with_the_differencing_order():
    """eq. (22): Na = floor(3P/4) - Np - d + 1."""
    assert select_na(20, 2, d=0) - select_na(20, 2, d=3) == 3


def test_noise_floor_is_preserved_through_the_weighting(make_oracle):
    """eq. (20): differencing shapes signal and noise identically, so the
    weighted problem sees the same noise variance as the undifferenced one."""
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-3)
    expected = 1e-3 ** 2 / o.N

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ne = estimate_noise_variance(o.voltage, o.current, o.fs, o.P, o.H_exc,
                                     Np=2, Nq=5, detrend=Differencing(1))

    ratio = np.median(ne.sigma2) / expected
    assert 0.5 < ratio < 2.0, f"noise floor shifted by {ratio:.3f}x under weighting"


def test_band_floor_keeps_only_well_weighted_bins():
    """|F| ~ k^d near DC, so requiring |F(k)| >= tol*|F(k_ref)| keeps
    k >= k_ref * tol**(1/d)."""
    import deistools.processing.bltva as M

    N, k_ref = 100_000, 20
    K_w = np.arange(1, 121)
    for d in (1, 2):
        kept = M._drop_ill_weighted_bins(K_w, k_ref, N, Differencing(d))
        expected_min = k_ref * M.MIN_WEIGHT_RATIO ** (1.0 / d)
        assert kept[0] == pytest.approx(np.ceil(expected_min), abs=1)
        assert kept[-1] == K_w[-1], "the upper end must be untouched"


def test_band_floor_refuses_when_nothing_usable_remains():
    """If the weighting is singular across the whole band, say so rather than
    returning amplified garbage."""
    import deistools.processing.bltva as M

    with pytest.raises(ValueError, match="singular this close to DC"):
        M._drop_ill_weighted_bins(np.arange(1, 4), 1000, 100_000, Differencing(2))


def test_band_floor_is_inactive_without_detrending():
    import deistools.processing.bltva as M

    K_w = np.arange(1, 121)
    assert np.array_equal(M._drop_ill_weighted_bins(K_w, 20, 100_000, None), K_w)


def test_no_detrending_is_unchanged(make_oracle):
    """detrend=None must be bit-identical to not passing it at all."""
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-4)
    a = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Nq=5, **FIT)
    b = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Nq=5, detrend=None, **FIT)
    assert np.array_equal(a.H_p, b.H_p)


# --------------------------------------------------------------------------- #
# The protocol contract, for every strategy that claims a response
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("strategy", [
    Differencing(1), Differencing(2), Differencing(4),
    MovingAverage(101), MovingAverage(401),
    LowPassRemoval(3.0, 2), LowPassRemoval(5.0, 4),
])
def test_apply_matches_the_declared_frequency_response(strategy):
    """Step 1 of the protocol: ``F(k)`` must describe what ``apply`` did.

    If it does not, BLTVA's weighting compensates for the wrong thing and the
    result is biased with nothing to warn the user.
    """
    report = check_response(strategy, N=4096)
    assert report["exact"], f"{strategy!r} residual {report['residual']:.2e}"
    assert report["imaginary"] < 1e-9, "the detrended record is not real"


def test_the_contract_check_catches_a_lying_response():
    """A strategy whose response does not match its own ``apply`` must fail.

    Guards the guard: without this, ``check_response`` returning ~1e-15 for
    everything would be indistinguishable from it not working.
    """
    class Liar(MovingAverage):
        """Declares a boxcar response and then differences instead."""
        def apply(self, y):
            return np.diff(np.asarray(y, dtype=float), prepend=0.0)

    assert not check_response(Liar(101), N=4096)["exact"]

    class Unscaled(MovingAverage):
        """Applies the right shape with the wrong gain."""
        def apply(self, y):
            return 2.0 * super().apply(y)

    assert not check_response(Unscaled(101), N=4096)["exact"]


def test_a_boxcar_must_be_odd_to_have_no_phase():
    with pytest.raises(ValueError, match="odd"):
        MovingAverage(400)
    with pytest.raises(ValueError, match=">= 3"):
        MovingAverage(1)


def test_low_pass_removal_rejects_nonsense():
    with pytest.raises(ValueError, match="positive"):
        LowPassRemoval(0.0)
    with pytest.raises(ValueError, match=">= 1"):
        LowPassRemoval(3.0, order=0)


# --------------------------------------------------------------------------- #
# Why only differencing may be used as a BLTVA weighting factor
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("strategy", [
    MovingAverage(401), LowPassRemoval(3.0, 2), PolynomialFit(3),
])
def test_bltva_refuses_the_strategies_its_weighting_would_undo(strategy, trended):
    o = trended()
    with pytest.raises(ValueError, match="cannot be used as a BLTVA"):
        bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, detrend=strategy, **FIT)


def test_a_diagonal_filter_cannot_change_the_trend_to_signal_ratio():
    """The structural reason the refusal above is not merely conservative.

    A filter applied by circular convolution multiplies bin ``k`` by ``F(k)``,
    so a trend and a signal that share that bin are scaled by the same number
    and their ratio is untouched -- for *any* diagonal filter, whatever its
    shape. This is the same fact that stops a filter rescuing the DMFA tones
    that sit in the dc skirt.
    """
    N = 8192
    t = np.linspace(0.0, 1.0, N)
    rng = np.random.default_rng(0)
    signal = np.sin(2 * np.pi * 40 * t) + 0.1 * rng.normal(size=N)
    trend = 30.0 * (t ** 3 - 0.4 * t)

    bins = np.array([8, 16, 40, 100])
    S, T = np.fft.fft(signal), np.fft.fft(trend)
    before = np.abs(T[bins]) / np.abs(S[bins])

    for strategy in (MovingAverage(201), LowPassRemoval(2.0, 3)):
        F = strategy.frequency_response(np.arange(N), N)
        after = np.abs((F * T)[bins]) / np.abs((F * S)[bins])
        assert np.allclose(after, before, rtol=1e-12), strategy


def test_the_weighting_cancels_a_zero_phase_response_exactly():
    """eq. (26) with a real ``F`` leaves the skirt regressors unchanged.

    ``W^(1/2) = diag(1/|F|)`` and the skirts carry a factor ``F``, so for a
    real, zero-phase response the weighted design matrix is bit-identical to
    the undetrended one. Nothing about the trend was removed; only the trend
    block of eq. (58) was, which is why the refusal exists.
    """
    N, P, Np, Nq, Na = 4096, 20, 2, 5, 8
    k_e = P * 7
    K_w = np.arange(k_e - 5 * P, k_e + 5 * P + 1)
    B = legendre_spectra(legendre_basis(N, max(Np, Nq)))
    strategy = MovingAverage(401)

    plain, cp = build_regressors(K_w, k_e, np.array([k_e]), B, Np, Nq, Na, None)
    shaped, cs = build_regressors(K_w, k_e, np.array([k_e]), B, Np, Nq, Na,
                                  strategy)
    w = np.abs(strategy.frequency_response(K_w, N))

    for block in ("target", f"h{k_e}"):
        a, c = plain[:, cp[block]], shaped[:, cs[block]]
        assert np.allclose(c / w[:, None], a, atol=1e-14), block

    # and the trend block is the one thing that went missing
    assert "drift" in cp and "drift" not in cs


def test_differencing_is_not_diagonal_which_is_why_it_works():
    """The complement of the test above.

    Differencing annihilates a polynomial trend in the *time domain*, which no
    multiplication in frequency can do. Its response is also complex, so
    ``F/|F|`` is a phase ramp the magnitude weighting does not cancel.
    """
    N = 4096
    t = np.arange(N)
    polynomial = 3.0 + 2.0 * t - 0.5 * t ** 2

    # exact annihilation in time, away from the zero-prepend boundary
    differenced = Differencing(3).apply(polynomial)
    assert np.allclose(differenced[3:], 0.0, atol=1e-6 * np.abs(polynomial).max())

    # no diagonal filter can do that: it would need F(k) = 0 at every bin
    assert not np.allclose(
        MovingAverage(401).apply(polynomial)[500:-500], 0.0,
        atol=1e-6 * np.abs(polynomial).max())

    # and F is complex, so the weighting leaves a phase behind
    F = Differencing(1).frequency_response(np.arange(1, N // 2), N)
    assert np.max(np.abs(F / np.abs(F) - 1.0)) > 0.5


# --------------------------------------------------------------------------- #
# The diagonal strategies, for the estimators that have no weighting
# --------------------------------------------------------------------------- #

def test_the_response_at_a_tone_is_not_the_diagnostic():
    """Why :meth:`band_response` exists and reports ripple.

    A boxcar's response passes exactly through 1 at bins where the Dirichlet
    kernel nulls, so ``|F|`` at the tone can be perfect while the band either
    side -- which DMFA reconstructs the time variation from -- swings by 3x.
    """
    report = MovingAverage(20001).band_response(8, 4, 160_000)
    assert report["centre"] == pytest.approx(1.0, abs=1e-3)
    assert report["ripple"] > 3.0
    assert report["maximum"] > 1.0, "the Dirichlet kernel overshoots"

    # the Butterworth magnitude is monotone and never exceeds one
    smooth = LowPassRemoval(2.0, 2).band_response(8, 4, 160_000)
    assert smooth["ripple"] < 1.1
    assert smooth["maximum"] <= 1.0


def test_differencing_would_destroy_a_dmfa_estimate():
    """The warning in the module docstring, as a number.

    BLTVA divides the attenuation back out; DMFA and STFT cannot, so the
    lowest tone does not survive differencing at all.
    """
    N, k_min = 160_000, 8
    assert abs(Differencing(1).frequency_response(
        np.array([k_min]), N)[0]) < 1e-3


# --------------------------------------------------------------------------- #
# The polynomial fit: the one with no frequency response
# --------------------------------------------------------------------------- #

def test_polynomial_fit_removes_a_polynomial_trend_exactly():
    """Exact on a trend of its own degree, and only then.

    The second half is the distortion in miniature: with a signal present the
    projection takes some of the signal too, because the fit has no way to
    tell the two apart. That is uncorrectable, and it is why BLTVA fits this
    basis jointly with the impedance instead of subtracting it first.
    """
    N = 2000
    t = np.linspace(-1.0, 1.0, N)
    trend = 4.0 - 3.0 * t + 2.0 * t ** 2 - t ** 3

    assert np.allclose(PolynomialFit(3).apply(trend), 0.0, atol=1e-9)
    assert np.std(PolynomialFit(2).apply(trend)) > 0.1, "a degree short"

    signal = np.sin(2 * np.pi * 37 * np.arange(N) / N)
    cleaned = PolynomialFit(3).apply(signal + trend)
    assert not np.allclose(cleaned, signal, atol=1e-8)
    assert np.std(cleaned - signal) / np.std(signal) < 0.2


def test_polynomial_fit_has_no_frequency_response():
    """A projection is not a convolution, so step 2 cannot be carried out.

    The error must say where to go instead, because the alternative is already
    in the estimator.
    """
    with pytest.raises(NotImplementedError, match="Nq"):
        PolynomialFit(3).frequency_response(np.arange(10), 100)


def test_polynomial_fit_distorts_a_record_with_no_trend():
    """The cost the class docstring quotes, in its mechanism.

    Subtracting a fitted polynomial from a trend-free record removes signal,
    and the amount grows with the degree. Nothing corrects it -- which is the
    reason BLTVA fits this basis rather than subtracting it.
    """
    N = 20_000
    rng = np.random.default_rng(0)
    signal = np.zeros(N)
    for h in (1, 3, 9, 27):
        signal += np.sin(2 * np.pi * h * np.arange(N) / N * 8
                         + rng.uniform(0, 2 * np.pi))

    removed = [np.std(signal - PolynomialFit(q).apply(signal))
               for q in (1, 3, 8, 12)]
    assert removed == sorted(removed), "distortion must grow with the degree"
    assert removed[-1] > 2 * removed[0]


def test_nq_is_the_supported_route_for_a_polynomial_trend(make_oracle):
    """What the refusal points at must actually work.

    A polynomial drift is recovered by the ``Nq`` block without any
    detrending, which is the whole reason ``PolynomialFit`` is refused rather
    than wired in.
    """
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10),
                    drift=np.array([0.0, 0.5, 0.2, 0.1]), noise_std_V=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Nq=5, **FIT)
    assert np.median(o.relative_error(result.H_p)) < 1e-2
