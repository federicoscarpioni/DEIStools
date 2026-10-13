"""
BLTVA — Band-Limited Time-Varying Analysis estimator.

Estimates the time-varying transfer function H(w,t) between an excitation signal and a
response signal measured under (odd random phase) multisine excitation, following

    Hallemans et al., Electrochim. Acta 466 (2023) 142939, eq. (71)-(81)
    Hallemans et al., IEEE TIM 70 (2021) 6503814, eq. (55), (58), Algorithm 2
    Breugelmans et al., Electrochim. Acta 76 (2012) 375, eq. (1)-(14)   [LTV-only variant]

Model, per excited bin k_e, over a local band K_w:

    V(k) = sum_p     theta_p(k_e) B_p(k - k_e)          # target skirt
         + sum_{k'}  sum_p theta_p(k') B_p(k - k')      # other harmonics in band
         + sum_q     theta_q B_q(k)                     # drift skirt at DC
         + sum_r     alpha_r L_r(xi(k))                 # transients + out-of-band

    H_p(w_ke) = theta_p(k_e) / U(k_e)
    H(w_ke, t) = sum_p H_p(w_ke) b_p(t)

with the scaled DFT X(k) = (1/N) sum_n x(n) exp(-j2 pi k n / N) and Legendre basis
b_p(t) = L_p(2t/T - 1).

Naming is deliberately excitation/response rather than current/voltage: in galvanostatic
mode pass (excitation=i, response=v) and read the result as an impedance; in potentiostatic
mode pass (excitation=v, response=i) and read it as an admittance.

Included: the band-limited least-squares estimate (TIM 2021 eq. 55-58), noise
variance estimation (Algorithm 1), regressor selection (Algorithm 2),
nonlinear distortion variances (Section V), uncertainty from both noise and
distortion (Section VI), and trend removal by differencing (TIM 2022), via
:mod:`deistools.processing.detrending`.

Distortion estimation needs ``harmonics="all_nl"`` and an excitation that
leaves harmonics unexcited -- an odd random phase multisine. With
``harmonics="excited"`` there are no regressors away from the excited bins,
so the distortion is unobservable and ``cov_H_p`` falls back to the noise
term alone. The excitation must also carry *uniformly random* phases, which
no inspection of the data can confirm; see the precondition note on
:class:`BLTVAResult`.

Designing such an excitation is out of scope here -- this module takes
``H_exc`` as given, whatever produced it. The companion ``multisine``
package has ``Multisine.generate_orp`` for it, but nothing in DEIStools
depends on that, by design: a user with their own waveform designer should
not be shut out.

Still outstanding: the local band is narrower than eq. (55) prescribes; see
``_band_bins``.

Dependencies: numpy, matplotlib.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.legendre import legvander
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

__all__ = [
    "BLTVAResult", "bltva", "legendre_basis", "legendre_spectra",
    "visualize_bltva_fit", "NoiseEstimate", "estimate_noise_variance",
    "select_na", "check_identifiability",
]


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class BLTVAResult:
    """Output of :func:`bltva`.

    Attributes
    ----------
    f_exc : (M,) float
        Excited frequencies in Hz.
    t : (N,) float
        Time axis in s.
    H_p : (M, Np+1) complex
        Basis coefficients of the transfer function at each excited frequency.
        ``H_p[:, 0]`` is the best stationary (time-averaged) transfer function.
    H_wt : (M, N) complex
        Time-varying transfer function, reconstructed on the full time axis.
    diagnostics : list of dict
        One entry per excited frequency: band width, unknowns before and after
        pruning, how many regressors each pruning step removed, the noise
        variance used as threshold, condition number, rank, relative residual.

    Notes
    -----
    **Precondition on the distortion results.** ``var_distortion_even``,
    ``var_distortion_odd``, ``cov_H_p_distortion`` and :meth:`sndr_db` are
    only meaningful for an odd random phase excitation. That means three
    things at once:

    1. only odd harmonics excited, so even-order distortion lands on empty
       even harmonics;
    2. some odd harmonics left unexcited, so odd-order distortion has
       somewhere to appear;
    3. phases drawn uniformly at random, which is what makes
       ``E{exp(j*phi)} = 0`` and the best linear approximation well defined.

    The first two are visible in ``H_exc`` and are enforced: a grid with no
    unexcited odd harmonic in band yields ``None`` and :meth:`sndr_db`
    raises. **The third cannot be checked from a single record and is the
    caller's responsibility.** Phases chosen for a low crest factor --
    Schroeder, or a best-of-N random search -- violate it, and the distortion
    variances then come out *wrong rather than absent*, with nothing to
    indicate it. If you optimised the crest factor, do not read these
    attributes.

    The drift is deliberately not reported. In Hallemans TIM 2021 ``theta_q``
    is a nuisance parameter: eq. (58) carries the trend so that it does not
    bias ``theta_p(k_e)``, and eq. (68) extracts only ``G_p``. Within a single
    band the trend's 1/k tail is nearly collinear with the transient
    polynomial, so ``theta_q`` is not individually identifiable there. An
    earlier version of this module returned it anyway, reconstructed from the
    lowest band alone, and it was wrong by ~50% of the drift range with a 26%
    imaginary part. Recovering the trend belongs to a detrending step, which
    knows what it removed.
    """

    f_exc: np.ndarray
    t: np.ndarray
    H_p: np.ndarray
    H_wt: np.ndarray
    cov_H_p_noise: np.ndarray | None = None
    cov_H_p_distortion: np.ndarray | None = None
    var_distortion_even: np.ndarray | None = None
    var_distortion_odd: np.ndarray | None = None
    response_power: np.ndarray | None = None
    diagnostics: list = field(default_factory=list)
    _basis: np.ndarray | None = field(default=None, repr=False)

    def worst_condition(self) -> float:
        return max(d["cond"] for d in self.diagnostics)

    def worst_residual(self) -> float:
        return max(d["rel_residual"] for d in self.diagnostics)

    @property
    def cov_H_p(self) -> np.ndarray:
        """Total covariance of ``H_p``, eq. (76): noise plus odd distortion.

        Falls back to the noise term alone when distortion was not estimated,
        i.e. when ``harmonics="excited"`` left no unexcited harmonic to read
        it from.
        """
        if self.cov_H_p_noise is None:
            raise ValueError(
                "no uncertainty available: bltva was called with prune=False "
                "and no noise estimate, so C_V is unknown")
        if self.cov_H_p_distortion is None:
            return self.cov_H_p_noise
        return self.cov_H_p_noise + self.cov_H_p_distortion

    @property
    def std_H_p(self) -> np.ndarray:
        """Total standard deviation of each ``H_p``, shape ``(M, Np+1)``."""
        diag = np.einsum("mpp->mp", self.cov_H_p).real
        return np.sqrt(np.maximum(diag, 0.0))

    def sndr_db(self) -> np.ndarray:
        """Signal-to-nonlinear-distortion ratio at each excited frequency, dB.

        ``|Y(k_e)|^2`` against the total distortion variance, which is what
        Section V says to compare in order to "decide whether the linear
        approximation of the system is acceptable". Large is good: the
        measurement examples in the paper report a minimum around 30 dB and
        treat a linear model as justified there.

        Requires an odd random phase excitation -- see the precondition on
        :attr:`var_distortion_odd`.
        """
        if self.var_distortion_odd is None:
            raise ValueError(
                'no distortion estimate: run with harmonics="all_nl" and an '
                "excitation that leaves harmonics unexcited")
        total = np.nansum(self.var_distortion_even, axis=1) + \
            np.nansum(self.var_distortion_odd, axis=1)
        with np.errstate(divide="ignore"):
            return 10.0 * np.log10(self.response_power / total)

    def variance_over_time(self, t_index=None) -> np.ndarray:
        """eq. (87): sigma^2(jw_k, t) = b(t) C_G(jw_k) b(t)^T.

        The uncertainty of the time-varying transfer function, shape
        ``(M, len(t_index))`` -- by default ``(M, N)`` over the whole record.
        Pass ``t_index`` to evaluate on a subset, since the full array is
        ``M x N``.

        The variance is time-dependent even though ``C_G`` is not: the
        Legendre basis weights the coefficients differently at each instant,
        and the higher orders grow towards the edges of the window. That is
        why estimates are least certain at the start and end of a record.
        """
        cov = self.cov_H_p                       # raises when unavailable
        b = self._basis if t_index is None else self._basis[t_index]
        # real by construction: b is real and C_G is Hermitian
        var = np.einsum("tp,mpq,tq->mt", b, cov, b).real
        return np.maximum(var, 0.0)

    def std_over_time(self, t_index=None) -> np.ndarray:
        """Standard deviation of H(w, t); the square root of eq. (87)."""
        return np.sqrt(self.variance_over_time(t_index))


# --------------------------------------------------------------------------- #
# Basis functions
# --------------------------------------------------------------------------- #

def legendre_basis(N: int, order: int) -> np.ndarray:
    """Rescaled Legendre basis b_p(t) = L_p(2t/T - 1), sampled on N points.

    Returns a real array of shape ``(N, order+1)``.

    The sample mean is removed from every p >= 1. On a finite sampled window the
    continuous property (1/T) int b_p dt = 0 does not hold exactly, so without this
    every time-varying regressor leaks into the DC bin.
    """
    x = 2.0 * np.arange(N) / N - 1.0
    b = legvander(x, order)                       # (N, order+1)
    b[:, 1:] -= b[:, 1:].mean(axis=0, keepdims=True)
    return b


def legendre_spectra(b: np.ndarray) -> np.ndarray:
    """Scaled DFT of the basis functions: B_p(k) = FFT(b_p)/N.

    Kept at full length N. The skirts are two-sided about their own centre, so a
    column centred at harmonic k' is indexed as ``B[(K_w - k') % N, p]``.
    """
    return np.fft.fft(b, axis=0) / b.shape[0]


# --------------------------------------------------------------------------- #
# Band construction
# --------------------------------------------------------------------------- #

def _excited_at(k_exc: np.ndarray, i: int) -> int:
    """``k_exc[i]``, reflected about either end when the index runs off.

    The bands at the edges of the grid have no outer neighbour to measure a
    margin against. Reflecting keeps their width and shape comparable with the
    interior bands instead of truncating them, which would make the first and
    last estimates quietly different from the rest.
    """
    M = k_exc.size
    if 0 <= i < M:
        return int(k_exc[i])
    if i < 0:
        return int(2 * k_exc[0] - k_exc[min(-i, M - 1)])
    return int(2 * k_exc[-1] - k_exc[max(2 * (M - 1) - i, 0)])


def _band_bins(k_exc: np.ndarray, e: int, N: int, halfwidth: int | None) -> np.ndarray:
    """DFT bins of the local band centred on ``k_exc[e]``.

    ``halfwidth=None`` builds the band of Hallemans TIM 2021 eq. (55)::

        K_w = {k_e^- - dk^-, ..., k_e^-, ..., k_e, ..., k_e^+, ..., k_e^+ + dk^+}

    which spans the three consecutive excited frequencies ``k_e^-, k_e, k_e^+``
    and extends half the distance towards the next excited bin beyond each of
    the outer two. Only the centre frequency's coefficients are kept; the band
    then slides to the next excited bin.

    The width matters because of how skirts are accounted for. A skirt whose
    centre lies inside the band gets its own ``theta_p(k')`` block (eq. 57-58);
    one whose centre lies outside contributes only a tail, which is smooth and
    monotone and therefore nearly collinear with the transient polynomial. A
    narrow band leaves the *immediate* neighbours in that second category,
    where the fit cannot separate their tails from the transient and the
    ambiguity reaches the target coefficients. The eq. (55) band parameterises
    both neighbours explicitly, so the nearest unmodelled skirt sits two
    spacings away and is flat enough for the polynomial to absorb honestly.

    ``halfwidth`` overrides the rule with a fixed number of bins, which is
    more robust on grids whose spacing changes abruptly (see
    ``test_neighbour_rule_degrades_on_abrupt_spacing_change``) and avoids
    enormous bands at the top of a sparse quasi-log grid.
    """
    k_e = int(k_exc[e])
    if halfwidth is not None:
        lo, hi = k_e - halfwidth, k_e + halfwidth
    elif k_exc.size == 1:
        raise ValueError("halfwidth must be given when only one frequency is excited")
    else:
        inner_lo, inner_hi = _excited_at(k_exc, e - 1), _excited_at(k_exc, e + 1)
        outer_lo, outer_hi = _excited_at(k_exc, e - 2), _excited_at(k_exc, e + 2)
        lo = inner_lo - (inner_lo - outer_lo) // 2
        hi = inner_hi + (outer_hi - inner_hi) // 2
    lo = max(1, lo)                 # bin 0 carries the DC offset of the response
    hi = min(N // 2 - 1, hi)        # positive frequencies only
    if hi <= lo:
        raise ValueError(f"empty band at excited bin {k_e}")
    return np.arange(lo, hi + 1)


#: Warn when an eq. (55) band spans more than this many harmonics.
MAX_HARMONICS_PER_BAND = 40


def _warn_if_bands_are_huge(k_exc, N, halfwidth, P):
    """Warn before fitting when the eq. (55) rule produces unusable bands.

    The rule scales with the local excited-frequency spacing, which on a
    sparse quasi-log grid grows by orders of magnitude towards the top. The
    widest band of a 15-tone, 3-decade grid reaches 20321 bins and, with
    ``harmonics="all_nl"``, 3071 regressors -- 332 s against 0.6 s for a fixed
    half-width, and no better an answer (1.4e-05 against 1.3e-05).

    Checked up front rather than per band so the warning arrives before the
    time is spent rather than after.
    """
    if halfwidth is not None or k_exc.size < 2:
        return
    widths = [_band_bins(k_exc, e, N, None).size for e in range(k_exc.size)]
    worst = max(widths)
    if worst > MAX_HARMONICS_PER_BAND * P:
        warnings.warn(
            f"the eq. (55) band reaches {worst} bins (~{worst // P} harmonics) "
            f"at the top of this grid, which is slow and brings in many "
            f"nuisance regressors for no gain. Pass band_halfwidth (a few "
            f"multiples of P, e.g. {5 * P}) to bound it.",
            stacklevel=3)


def build_regressors(K_w, k_e, centres, B, Np, Nq, Na, detrend=None):
    """TIM 2021 eq. (58), or TIM 2022 eq. (21) when detrending is in use.

    Without detrending: target skirt, other harmonics in band, trend block at
    DC, transient polynomial. The trend block is present in every band, as the
    paper prescribes; bands where it carries nothing are handled by the pruning
    of Algorithm 2 step 6 on statistical grounds, not by excluding it up front.

    With detrending, two things change (2022 eq. 21). The skirts are multiplied
    by the filter response ``F(k)``, because the data they model has been
    through that filter. And the trend block disappears -- the trend was
    removed rather than modelled, which is the entire point -- replaced by the
    filter's own boundary regressors.
    """
    N = B.shape[0]
    blocks, cmap, j = [], {}, 0

    def add(name, block):
        nonlocal j
        cmap[name] = slice(j, j + block.shape[1])
        blocks.append(block)
        j += block.shape[1]

    F = detrend.frequency_response(K_w, N)[:, None] if detrend is not None else 1.0

    add("target", F * B[(K_w - k_e) % N, : Np + 1])
    for kp in centres:
        add(f"h{kp}", F * B[(K_w - kp) % N, : Np + 1])

    if detrend is None:
        # q = 0 is identically zero off DC; its coefficient is the mean, V[0].
        if Nq >= 1:
            add("drift", B[K_w, 1 : Nq + 1])
    elif detrend.n_extra_params:
        add("boundary", detrend.extra_regressors(K_w, N))

    xi = 2.0 * (K_w - K_w[0]) / (K_w[-1] - K_w[0]) - 1.0
    add("transient", legvander(xi, Na).astype(complex))
    return np.hstack(blocks), cmap


#: Smallest |F(k)| a band bin may have, as a fraction of |F| at the band
#: centre. See :func:`_drop_ill_weighted_bins`.
MIN_WEIGHT_RATIO = 0.25


def _drop_ill_weighted_bins(K_w, k_ref, N, detrend, tol=MIN_WEIGHT_RATIO):
    """Discard band bins where the detrending weight blows up.

    The weighting 1/|F(k)| restores the noise exactly (eq. 20), but it also
    restores whatever the filter failed to remove. Differencing only cancels a
    trend *exactly* when the trend is a polynomial of degree <= d (Theorem 2);
    a real trend is merely close to one, and the leftover is amplified by
    1/|F(k)|, which diverges as k -> 0.

    Measured on the 2022 paper's own trend (the step response of eq. 32, which
    no finite polynomial captures) with d = 1: in the weighted spectrum the
    residual trend exceeds the signal by 77x at bin 1 and 43x at bin 3, falls
    to 2.6x by bin 10 and is negligible past bin 20. A band reaching down to
    bin 1 fitted with 38% residual error; the same band floored at bin 5 fitted
    with 0.02%.

    The cut is made on |F(k)| rather than on k so that it scales with the
    differencing order: since |F| ~ k^d near DC, requiring
    |F(k)| >= tol*|F(k_ref)| keeps k >= k_ref * tol**(1/d).
    """
    if detrend is None:
        return K_w
    F = np.abs(detrend.frequency_response(K_w, N))
    F_ref = abs(detrend.frequency_response(np.array([k_ref]), N)[0])
    kept = K_w[F >= tol * F_ref]
    if kept.size < 2:
        raise ValueError(
            f"detrending leaves only {kept.size} usable bins in the band at "
            f"bin {k_ref}: the weighting 1/|F| is singular this close to DC. "
            "Excite a higher lowest harmonic, or difference less.")
    return kept


def _weights(K_w, N, detrend):
    """eq. (26): W^(1/2) = diag(1/|F(k)|), as a row scaling.

    Differencing attenuates the low frequencies and amplifies the high ones.
    It does so to the signal and the noise alike, so weighting by 1/|F|
    restores a homoscedastic problem and no SNR is lost (eq. 20).
    """
    if detrend is None:
        return None
    w = np.abs(detrend.frequency_response(K_w, N))
    w[w == 0.0] = np.finfo(float).tiny       # bin 0 only, which bands exclude
    return w


# --------------------------------------------------------------------------- #
# Algorithm 1 — noise variance estimation
#
# Hallemans TIM 2021 Section III. A sliding band runs over the spectrum; in
# each band the deterministic content is fitted and the residual is taken as
# noise. Two details distinguish this from the signal model of eq. (58):
#
#   * the bins of K_nl are *removed* from the band (eq. 38), because at a
#     harmonic the model cannot separate noise from dynamics;
#   * the skirts are modelled by hyperbolas theta_p/(k-k')^p rather than by
#     the exact Legendre spectra B_p (eq. 42). For noise estimation only the
#     deterministic content has to be absorbed, not identified, and the
#     hyperbolas are cheaper and better conditioned on a narrow band.
#
# The variance comes from a rank-truncated residual (eq. 46-48): only the m
# leading left-singular directions of the regression matrix are projected out,
# so directions that carry no signal above the noise are not spent.
# --------------------------------------------------------------------------- #

@dataclass
class NoiseEstimate:
    """Noise variance of the scaled DFT, estimated on a grid of DFT bins.

    Attributes
    ----------
    k : (K,) int
        Bins at which the variance was estimated.
    sigma2 : (K,) float
        Estimated variance of ``V(k)``, the 1/N-scaled DFT. For white output
        noise of standard deviation ``s`` in the time domain this is ``s**2/N``.
    n : int
        Half-width of the sliding band, eq. (39).
    dof : int
        Target degrees of freedom left after the fit.
    m : (K,) int
        Number of singular directions projected out in each band.
    """

    k: np.ndarray
    sigma2: np.ndarray
    n: int
    dof: int
    m: np.ndarray

    def at(self, k):
        """Variance interpolated onto arbitrary bins.

        sigma2_V(k) is a smooth function of frequency (it is filtered
        band-limited white noise, eq. (19)), so linear interpolation between
        band centres is adequate and lets the sweep be subsampled.
        """
        return np.interp(np.asarray(k, dtype=float), self.k, self.sigma2)


def _noise_band_bins(k_c, n, N, P):
    """eq. (38): 2n+1 bins centred on k_c, with the harmonics K_nl removed."""
    bins = np.arange(max(1, k_c - n), min(N // 2, k_c + n + 1))
    return bins[bins % P != 0]


def _noise_regressors(K_w, k_c, N, P, B, Np, Nq, Na, detrend=None):
    """eq. (42): hyperbolic skirts at nearby harmonics, trend skirt, transient.

    ``K_nl,w`` is the set of harmonics within +/- P of the band centre
    (eq. 41), so at most three contribute. The hyperbolas never blow up
    because the harmonics themselves were removed from ``K_w``.

    Under detrending the same substitutions apply as in the signal model: the
    skirts carry the filter response, the trend block goes away, and the
    filter's boundary terms are added. The caller then applies the weighting,
    so the residual that becomes the noise estimate is homoscedastic.
    """
    lo = max(0, ((k_c - P) // P) * P)
    centres = [kp for kp in range(lo, min(N // 2, k_c + P + 1), P)
               if abs(kp - k_c) <= P]

    F = detrend.frequency_response(K_w, N)[:, None] if detrend is not None else 1.0

    blocks = []
    for kp in centres:
        delta = (K_w - kp).astype(float)
        blocks.append(F * np.stack([delta ** (-p) for p in range(1, Np + 1)], axis=1))
    if detrend is None:
        if Nq >= 1:
            blocks.append(B[K_w, 1 : Nq + 1])
    elif detrend.n_extra_params:
        blocks.append(detrend.extra_regressors(K_w, N))
    xi = 2.0 * (K_w - K_w[0]) / (K_w[-1] - K_w[0]) - 1.0
    blocks.append(legvander(xi, Na))

    return np.hstack(blocks).astype(complex)


def _truncated_residual_variance(Y_w, U, m):
    """eq. (46)-(48): variance from the residual after removing m directions."""
    Um = U[:, :m]
    eps = Y_w - Um @ (Um.conj().T @ Y_w)
    return float(np.real(np.vdot(eps, eps)) / (Y_w.size - m))


def _noise_variance_in_band(V, k_c, n, N, P, B, Np, Nq, Na, safety, detrend=None):
    """Algorithm 1 at one band centre. Returns (sigma2, m)."""
    K_w = _noise_band_bins(k_c, n, N, P)
    K_w = _drop_ill_weighted_bins(K_w, k_c, N, detrend)
    Y_w = V[K_w]
    K = _noise_regressors(K_w, k_c, N, P, B, Np, Nq, Na, detrend)
    w = _weights(K_w, N, detrend)
    if w is not None:
        K = K / w[:, None]
        Y_w = Y_w / w

    n_w = K_w.size
    if n_w <= K.shape[1]:
        raise ValueError(
            f"noise band at bin {k_c}: {n_w} equations for {K.shape[1]} "
            "regressors. Lower Np/Nq/Na or raise dof.")

    U, s, _ = np.linalg.svd(K, full_matrices=False)
    ratio = s / s[0]

    def _rank(tol):
        return int(np.clip(np.count_nonzero(ratio >= tol), 1, n_w - 1))

    # Step 1: discard only numerically null directions, eq. (50)-(51).
    l = _rank(1e-10)
    sigma2 = _truncated_residual_variance(Y_w, U, l)

    # Step 2: re-truncate at the noise floor, eq. (52)-(54).
    #
    # The published eq. (53) prints the threshold as "a snr". Taken literally
    # that always yields m = 1, since sigma_{i+1}/sigma_1 never exceeds 1 while
    # a*snr >= 10. Of the two readable alternatives only 1/(a*snr) is unbiased:
    # on a known white floor it recovers the variance to within 3% over three
    # decades of noise, where a/snr overestimates by 1.9x to 5.8x and leaves
    # deterministic content in the residual. It is also the sense in which `a`
    # is a *safety* factor -- it lowers the threshold, so more directions are
    # removed and less signal survives into the noise estimate.
    if sigma2 > 0:
        snr = np.sqrt(np.real(np.vdot(Y_w, Y_w)) / (n_w * sigma2))
        if snr > 0:
            m = _rank(1.0 / (safety * snr))
            sigma2 = _truncated_residual_variance(Y_w, U, m)
            return sigma2, m
    return sigma2, l


def estimate_noise_variance(response, excitation, fs, P, H_exc, *,
                            Np=6, Nq=5, Na=None, dof=5, safety=10.0,
                            k_max=None, step=None, detrend=None):
    """Estimate the noise variance spectrum (Hallemans TIM 2021, Algorithm 1).

    Parameters
    ----------
    response, excitation, fs, P, H_exc
        As for :func:`bltva`.
    Np, Nq, Na : int
        Upper bounds on the model orders, used to size the sliding band.
    dof : int
        Degrees of freedom left after the fit in each band, eq. (39).
        Larger means a wider band, so a smoother but less local estimate.
    safety : float
        Safety factor ``a`` in the rank threshold, eq. (53).
    k_max : int or None
        Highest bin to cover. Defaults to a margin past the top excited bin,
        which is enough for every band BLTVA will use.
    step : int or None
        Spacing of band centres. Defaults to the band half-width ``n``, so
        consecutive bands abut. The paper sweeps every bin; because the
        variance is smooth in frequency this subsamples and interpolates,
        which is far cheaper and indistinguishable in practice.

    Returns
    -------
    NoiseEstimate
    """
    if Na is None:
        Na = select_na(P, Np, d=getattr(detrend, 'order', 0))
    N, _T, _t, k_exc, _c, _b, B, V, _U = _prepare(
        response, excitation, fs, P, H_exc, Np, Nq, Na, "all_nl", detrend)

    n_theta_max = Nq + 3 * Np + Na + 1
    n = min(int(np.ceil((dof + n_theta_max + 2) / 2)), P)
    if 2 * n + 1 <= n_theta_max:
        raise ValueError(
            f"P={P} is too small for Np={Np}, Nq={Nq}, Na={Na}: the noise band "
            f"({2 * n + 1} bins) cannot support {n_theta_max} regressors. "
            "Measure more periods or lower the model orders.")

    if k_max is None:
        k_max = min(N // 2 - 1, int(k_exc.max()) + 2 * P + n)
    if step is None:
        step = max(1, n)

    centres = np.arange(max(1, n), k_max + 1, step)
    sigma2 = np.empty(centres.size)
    m_used = np.empty(centres.size, dtype=int)
    for i, k_c in enumerate(centres):
        sigma2[i], m_used[i] = _noise_variance_in_band(
            V, int(k_c), n, N, P, B, Np, Nq, Na, safety, detrend)

    return NoiseEstimate(k=centres, sigma2=sigma2, n=n, dof=dof, m=m_used)


# --------------------------------------------------------------------------- #
# Shared setup and per-band fit (used by both the estimator and its visualizer)
# --------------------------------------------------------------------------- #

def select_na(P, Np, d=0):
    """Transient polynomial order, Hallemans TIM 2022 eq. (22).

        Na = floor(3P/4) - Np - d + 1

    ``d`` is the differencing order used for trend removal; it is 0 while no
    detrending is applied, which reduces the rule to ``floor(3P/4) - Np + 1``.
    Verified against the 2022 simulation example: P = 20, Np = 4, d = 3 gives
    Na = 9, as the paper states.

    The order is an upper bound. Algorithm 2 step 3 prunes transient columns
    that the skirts already span, so choosing generously costs little.
    """
    Na = int(np.floor(3 * P / 4)) - Np - d + 1
    if Na < 0:
        raise ValueError(
            f"eq. (22) gives Na={Na} for P={P}, Np={Np}, d={d}. Measure more "
            "periods or lower Np.")
    return Na


def check_identifiability(P, Np, Nq, Na):
    """Hallemans TIM 2021 eq. (49): Nq + 3*Np + Na + 1 < 2P - 1.

    A band holds at most three harmonics of K_nl, so the largest number of
    parameters any band has to support is ``n_theta_max = Nq + 3*Np + Na + 1``.
    The band itself spans at most 2P+1 bins, of which up to three are removed,
    so there must be fewer parameters than that for the fit to be determined.

    Returns ``n_theta_max``. Raises when the condition fails -- as the paper
    puts it, at that point one should redo the measurement with more periods.
    """
    n_theta_max = Nq + 3 * Np + Na + 1
    if n_theta_max >= 2 * P - 1:
        raise ValueError(
            f"eq. (49) violated: Nq + 3*Np + Na + 1 = {n_theta_max} is not "
            f"below 2P - 1 = {2 * P - 1} for P={P}, Np={Np}, Nq={Nq}, Na={Na}. "
            "There are more parameters than a band can support. Measure more "
            "periods, or lower the model orders.")
    return n_theta_max


def _prepare(response, excitation, fs, P, H_exc, Np, Nq, Na, harmonics,
             detrend=None):
    """Validate inputs and precompute the arrays shared by every band fit.

    Returns N, T, t, k_exc, all_centres, b, B, V, U.
    """
    response = np.asarray(response, dtype=float)
    excitation = np.asarray(excitation, dtype=float)
    if response.shape != excitation.shape or response.ndim != 1:
        raise ValueError("response and excitation must be 1-D arrays of equal length")

    N = response.size
    if N % P:
        raise ValueError(
            f"N={N} is not an integer number of periods of P={P}. Trim the record "
            "to P*(fs*Tp) samples before calling; a non-integer record leaks and the "
            "leakage is indistinguishable from time variation."
        )
    if harmonics not in ("all_nl", "excited"):
        raise ValueError("harmonics must be 'all_nl' or 'excited'")

    T = N / fs
    t = np.arange(N) / fs

    H_exc = np.asarray(H_exc, dtype=int)
    k_exc = P * H_exc                                   # excited DFT bins, 0-based
    if k_exc.max() >= N // 2:
        raise ValueError("an excited harmonic lies at or above Nyquist")
    check_identifiability(P, Np, Nq, Na)

    b = legendre_basis(N, max(Np, Nq))                  # (N, max+1) real
    B = legendre_spectra(b)                             # (N, max+1) complex
    if detrend is not None:
        response = detrend.apply(response)
    V = np.fft.fft(response) / N
    U = np.fft.fft(excitation) / N

    all_centres = k_exc if harmonics == "excited" else np.arange(0, N // 2, P)

    return N, T, t, k_exc, all_centres, b, B, V, U


def _solve_band(K, V_w, rcond):
    """eq. (64)/(67): theta = K^+ Y_w, the Moore-Penrose minimum-norm solution.

    Deliberately *not* column-equilibrated. Equilibration flatters the reported
    condition number but changes which solution the pseudo-inverse selects: it
    minimises ||theta * scale|| rather than ||theta||, so columns of tiny norm
    -- the drift skirt in a high-frequency band, where B_q(k) ~ 1/k -- are free
    to carry enormous coefficients that cancel against the transient
    polynomial. The fitted values stay correct, but each column then looks
    individually significant to the contribution test of eq. (66), and the
    pruning of step 6 can never fire. Measured on one such band, equilibration
    gave |theta_drift| = 3.8e+10 and pruned nothing; the plain pseudo-inverse
    gave 1.1e-10 and pruned all five drift columns, with identical residual
    and identical H_0.

    Equilibration is still used for rank and condition diagnostics, where
    scale genuinely should not count.
    """
    theta, _, rank, _ = np.linalg.lstsq(K, V_w, rcond=rcond)
    return theta, int(rank)


def _equilibrated_svd(K):
    scale = np.linalg.norm(K, axis=0)
    scale[scale == 0.0] = 1.0
    return np.linalg.svd(K / scale, compute_uv=False)


def _numerical_rank(K, tol=1e-10):
    s = _equilibrated_svd(K)
    return int(np.count_nonzero(s / s[0] >= tol)) if s.size else 0


def _condition_number(K):
    s = _equilibrated_svd(K)
    return float(s[0] / s[-1]) if s.size and s[-1] > 0 else np.inf


def _prune_dependent_transients(K, cmap, keep):
    """Algorithm 2 step 3: drop transient polynomials that the skirts already span.

    The highest-order transient column is removed while doing so leaves the
    numerical rank unchanged -- i.e. while it is linearly dependent on the
    columns already present. The first column whose removal *does* drop the
    rank is independent, so it is kept and the loop stops.
    """
    cols = list(range(cmap["transient"].start, cmap["transient"].stop))
    while len(cols) > 1:
        rank_now = _numerical_rank(K[:, keep])
        trial = keep.copy()
        trial[cols[-1]] = False
        if _numerical_rank(K[:, trial]) < rank_now:
            break
        keep = trial
        cols.pop()
    return keep


def _prune_unsupported(K, cmap, keep, theta, sigma2_n):
    """Algorithm 2 step 6: drop regressors whose contribution is below the noise.

    eq. (66): a_r = max_i |theta_r|^2 |K[i,r]|^2, the peak contribution of
    regressor r to the modelled spectrum. Regressors with a_r < 3*sigma_n^2
    are not supported by the data at 95% confidence and are removed.

    The transient block is exempt -- it was already reduced in step 3, and it
    models the out-of-band behaviour that has to stay to keep the fit honest.
    """
    contribution = np.abs(theta) ** 2 * (np.abs(K) ** 2).max(axis=0)
    prunable = keep.copy()
    prunable[cmap["transient"]] = False

    drop = prunable & (contribution < 3.0 * sigma2_n)
    if drop.all():                      # never strip the model entirely
        return keep, 0
    return keep & ~drop, int(drop.sum())


def _parameter_covariance(K, keep, sigma2_bins):
    """eq. (72): C_theta = K^+ C_V K^+H, with C_V = diag(sigma2_V(k)), eq. (73).

    Returned at full width with zeros for pruned regressors: a coefficient
    that Algorithm 2 set to zero was not estimated, so it carries no
    uncertainty of its own.
    """
    if sigma2_bins is None:
        return None
    Kp = np.linalg.pinv(K[:, keep])
    C_kept = (Kp * sigma2_bins) @ Kp.conj().T

    n = K.shape[1]
    C = np.zeros((n, n), dtype=complex)
    idx = np.flatnonzero(keep)
    C[np.ix_(idx, idx)] = C_kept
    return C


def _fit_band(e, k_exc, N, band_halfwidth, all_centres, B, Np, Nq, Na,
              V, rcond, noise=None, detrend=None, prune=True):
    """Fit one excited frequency (Hallemans TIM 2021, Algorithm 2).

    Shared by the estimator and ``visualize_bltva_fit`` so both see the same
    model. ``sigma2_n`` is the mean noise variance in the band, from
    Algorithm 1; passing ``None`` skips the statistical pruning of step 6 and
    leaves a plain least-squares fit.

    ``theta`` is returned at full length, with zeros where a regressor was
    pruned, so ``cmap`` indexes it unchanged (eq. 68: coefficients that were
    not identified are set to zero).
    """
    k_e = int(k_exc[e])
    K_w = _band_bins(k_exc, e, N, band_halfwidth)
    K_w = _drop_ill_weighted_bins(K_w, k_e, N, detrend)
    centres = [int(kp) for kp in all_centres
               if K_w[0] <= kp <= K_w[-1] and kp != k_e]

    K, cmap = build_regressors(K_w, k_e, centres, B, Np, Nq, Na, detrend)
    n_theta = K.shape[1]
    if n_theta >= K_w.size:
        raise ValueError(
            f"band at bin {k_e}: {n_theta} unknowns for {K_w.size} equations. "
            "Widen the band, lower Np/Na, or use harmonics='excited'."
        )

    V_w = V[K_w]

    # eq. (27): solve the weighted system Phi = W^(1/2) K against W^(1/2) Y.
    # theta is unchanged by a row scaling, so nothing has to be undone after.
    w = _weights(K_w, N, detrend)
    if w is not None:
        K = K / w[:, None]
        V_w = V_w / w

    # eq. (73): the noise variance at each bin of the band. The weighting
    # above leaves it unchanged (eq. 20), so the same values apply either way.
    sigma2_bins = noise.at(K_w) if noise is not None else None
    sigma2_n = float(sigma2_bins.mean()) if sigma2_bins is not None else None

    keep = np.ones(n_theta, dtype=bool)

    keep = _prune_dependent_transients(K, cmap, keep)
    n_transient_dropped = int(n_theta - keep.sum())

    theta = np.zeros(n_theta, dtype=complex)
    theta[keep], rank = _solve_band(K[:, keep], V_w, rcond)

    n_unsupported = 0
    if prune and sigma2_n is not None and sigma2_n > 0:
        keep, n_unsupported = _prune_unsupported(K, cmap, keep, theta, sigma2_n)
        if n_unsupported:
            theta = np.zeros(n_theta, dtype=complex)
            theta[keep], rank = _solve_band(K[:, keep], V_w, rcond)

    cond = _condition_number(K[:, keep])
    fit = K[:, keep] @ theta[keep]
    residual = V_w - fit

    cov = _parameter_covariance(K, keep, sigma2_bins)

    # Algorithm 2 step 9: the coefficients fitted at the other harmonics in the
    # band. At a harmonic that is not excited U(k') = 0, so eq. (60)/(61)
    # reduce to theta_p(k') = Y_s,p(k') -- the distortion coefficients are
    # read off directly, with no further work.
    harmonic_theta = {int(kp): theta[cmap[f"h{kp}"]].copy() for kp in centres}

    return {
        "k_e": k_e, "K_w": K_w, "centres": centres, "K": K, "cmap": cmap,
        "V_w": V_w, "theta": theta, "fit": fit, "residual": residual,
        "keep": keep, "cond": cond, "rank": rank, "cov": cov,
        "sigma2_n": sigma2_n, "sigma2_bins": sigma2_bins,
        "harmonic_theta": harmonic_theta,
        "n_theta": int(keep.sum()), "n_theta_full": int(n_theta),
        "n_transient_dropped": n_transient_dropped,
        "n_unsupported_dropped": n_unsupported,
    }


# --------------------------------------------------------------------------- #
# Section V — nonlinear distortion variances
# --------------------------------------------------------------------------- #

def _neighbouring_harmonics(k_e, candidates):
    """Closest members of ``candidates`` below and above ``k_e``."""
    below = [k for k in candidates if k < k_e]
    above = [k for k in candidates if k > k_e]
    return (max(below) if below else None,
            min(above) if above else None)


def _distortion_variances(fit, P, k_exc_set, sigma2_n, Np):
    """eq. (69)-(71): even and odd distortion variances at an excited bin.

    A harmonic ``k' = m*P`` is an even-order contribution when ``m`` is even
    and an odd-order one when ``m`` is odd (eq. 32-33). The excited bins are
    always a subset of the odd ones, which is why an odd random phase
    multisine can separate the two: even distortion lands on harmonics that
    carry no excitation, and the odd harmonics left empty by the random
    omission expose the odd distortion.

    At an excited bin the distortion cannot be read directly -- the signal is
    there -- so eq. (71) averages the two nearest odd harmonics that are *not*
    excited. The same construction is used here for the even class, which the
    paper states only at the even harmonics themselves.

    eq. (69) subtracts the noise variance from the p = 0 term, since the raw
    squared coefficient contains both; the higher orders are left alone.
    """
    harmonic = fit["harmonic_theta"]
    odd_free = [k for k in harmonic if (k // P) % 2 == 1 and k not in k_exc_set]
    even_free = [k for k in harmonic if (k // P) % 2 == 0]

    out = {}
    for name, pool in (("odd", odd_free), ("even", even_free)):
        lo, hi = _neighbouring_harmonics(fit["k_e"], pool)
        sides = [np.abs(harmonic[k]) ** 2 for k in (lo, hi) if k is not None]
        if not sides:
            out[name] = np.full(Np + 1, np.nan)
            continue
        var = np.mean(sides, axis=0)
        if sigma2_n is not None:
            var = var.copy()
            var[0] = max(var[0] - sigma2_n, 0.0)     # eq. (69), p = 0 only
        out[name] = var
    return out["even"], out["odd"], odd_free


def _distortion_covariance(fit, B, N, Np, odd_free, sigma2_bins):
    """eq. (79): covariance of G_p due to odd nonlinear distortion.

    The raw outer products of the distortion coefficients at the nearest
    unexcited odd harmonics, minus the noise covariance of having estimated
    those same coefficients, so that only the distortion remains.

    The subtraction can overshoot and leave the result slightly indefinite.
    Negative eigenvalues are clipped, which keeps it a usable covariance; the
    paper applies the same idea to the scalar variances in eq. (69).
    """
    k_e, K_w = fit["k_e"], fit["K_w"]
    lo, hi = _neighbouring_harmonics(k_e, odd_free)
    terms = []
    for k in (lo, hi):
        if k is None:
            continue
        y = fit["harmonic_theta"][k][:, None]            # (Np+1, 1)
        Bk = B[(K_w - k) % N, : Np + 1]                  # eq. (80)/(81)
        Bp = np.linalg.pinv(Bk)
        terms.append(y @ y.conj().T - (Bp * sigma2_bins) @ Bp.conj().T)
    if not terms:
        return None

    C = sum(terms) / len(terms)
    w, Q = np.linalg.eigh((C + C.conj().T) / 2)
    return (Q * np.maximum(w, 0.0)) @ Q.conj().T


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #

def bltva(response, excitation, fs, P, H_exc, *,
          Np=6, Nq=5, Na=None, harmonics="all_nl",
          band_halfwidth=None, rcond=None, noise=None, prune=True,
          detrend=None):
    """Estimate the time-varying transfer function H(w_k, t).

    Parameters
    ----------
    response, excitation : (N,) real
        Measured time series. Galvanostatic: response=voltage, excitation=current.
    fs : float
        Sampling frequency [Hz].
    P : int
        Number of measured multisine periods. Must divide N exactly.
    H_exc : sequence of int
        Excited harmonic numbers of the multisine (h_m, so f_m = h_m/T_p).
    Np : int
        Order of the Legendre expansion of the transfer function. Controls
        time resolution: the variation is resolved on a scale of about T/Np.
    Nq : int
        Order of the Legendre expansion of the drift.
    Na : int or None
        Order of the transient / out-of-band polynomial. None selects it from
        eq. (22) via :func:`select_na`. All three orders are jointly bounded
        by eq. (49), which :func:`check_identifiability` enforces.
    harmonics : {"all_nl", "excited"}
        "all_nl"  -> regressors at every harmonic {0, P, 2P, ...} in band. This is the
                     BLTVA of Hallemans TIM 2021; needed when odd nonlinear distortion
                     is present, since its skirts sit on non-excited harmonics.
        "excited" -> regressors only at excited bins. The LTV estimator of
                     Breugelmans 2012 eq. (11). Far fewer unknowns; correct when the
                     response is linear.
    band_halfwidth : int or None
        Band half-width in DFT bins. None uses the half-distance-to-neighbours rule.
        For sparse log-spaced multisines, a few multiples of P (e.g. 5*P) is better.
    rcond : float or None
        Singular-value truncation passed to ``np.linalg.lstsq``.
    noise : NoiseEstimate or None
        Noise variance spectrum driving the pruning threshold of Algorithm 2
        step 6. Computed internally when ``prune`` is true and none is given;
        pass one to reuse it across calls, since it is the expensive part.
    prune : bool
        Run the statistical regressor selection of Algorithm 2 step 6. With it
        off the result is a plain least-squares fit of the full eq. (58) model,
        which overfits noise into the high Legendre orders.

    Returns
    -------
    BLTVAResult
    """
    if Na is None:
        Na = select_na(P, Np, d=getattr(detrend, 'order', 0))
    N, T, t, k_exc, all_centres, b, B, V, U = _prepare(
        response, excitation, fs, P, H_exc, Np, Nq, Na, harmonics, detrend)
    _warn_if_bands_are_huge(k_exc, N, band_halfwidth, P)

    if prune and noise is None:
        noise = estimate_noise_variance(response, excitation, fs, P, H_exc,
                                        Np=Np, Nq=Nq, Na=Na, detrend=detrend)

    # --- sweep the excited frequencies -------------------------------------
    M = k_exc.size
    H_p = np.zeros((M, Np + 1), dtype=complex)
    has_cov = noise is not None
    cov_noise = np.zeros((M, Np + 1, Np + 1), dtype=complex) if has_cov else None
    cov_dist = np.zeros((M, Np + 1, Np + 1), dtype=complex) if has_cov else None
    var_even = np.full((M, Np + 1), np.nan)
    var_odd = np.full((M, Np + 1), np.nan)
    Y_exc = np.zeros(M)
    any_distortion = False
    k_exc_set = {int(k) for k in k_exc}
    diagnostics = []

    for e in range(M):
        fit = _fit_band(e, k_exc, N, band_halfwidth, all_centres, B,
                        Np, Nq, Na, V, rcond, noise, detrend, prune)
        cmap, theta, K_w = fit["cmap"], fit["theta"], fit["K_w"]
        gain = np.abs(U[fit["k_e"]]) ** 2

        H_p[e] = theta[cmap["target"]] / U[fit["k_e"]]
        Y_exc[e] = np.abs(V[fit["k_e"]]) ** 2

        if has_cov:
            # eq. (84): dividing theta by U(k_e) divides the covariance by
            # |U(k_e)|^2.
            tgt = cmap["target"]
            cov_noise[e] = fit["cov"][tgt, tgt] / gain

        # Section V needs regressors at the unexcited harmonics, which only
        # harmonics="all_nl" provides.
        if harmonics == "all_nl":
            even, odd, odd_free = _distortion_variances(
                fit, P, k_exc_set, fit["sigma2_n"], Np)
            var_even[e], var_odd[e] = even, odd
            if odd_free:
                any_distortion = True
                if has_cov:
                    C = _distortion_covariance(
                        fit, B, N, Np, odd_free, fit["sigma2_bins"])
                    if C is not None:
                        cov_dist[e] = C / gain

        diagnostics.append({
            "k_e": fit["k_e"],
            "f_exc": fit["k_e"] / T,
            "N_w": int(K_w.size),
            "n_theta": fit["n_theta"],
            "n_theta_full": fit["n_theta_full"],
            "n_centres": len(fit["centres"]),
            "n_transient_dropped": fit["n_transient_dropped"],
            "n_unsupported_dropped": fit["n_unsupported_dropped"],
            "sigma2_n": fit["sigma2_n"],
            "cond": fit["cond"],
            "rank": fit["rank"],
            "rel_residual": float(np.linalg.norm(fit["residual"]) / np.linalg.norm(fit["V_w"])),
        })

    # --- reconstruct over time ---------------------------------------------
    H_wt = H_p @ b[:, : Np + 1].T                       # (M, N)

    if not any_distortion:
        var_even = var_odd = None
        cov_dist = None

    return BLTVAResult(
        f_exc=k_exc / T, t=t, H_p=H_p, H_wt=H_wt,
        cov_H_p_noise=cov_noise, cov_H_p_distortion=cov_dist,
        var_distortion_even=var_even, var_distortion_odd=var_odd,
        response_power=Y_exc, diagnostics=diagnostics, _basis=b[:, : Np + 1],
    )


# --------------------------------------------------------------------------- #
# Visual check of the per-band least-squares fit
# --------------------------------------------------------------------------- #

def visualize_bltva_fit(response, excitation, fs, P, H_exc, *,
                       Np=6, Nq=5, Na=None, harmonics="all_nl",
                       band_halfwidth=None, rcond=None, noise=None, prune=True,
                       detrend=None):
    """Slider-based visual check of the band-limited least-squares fit.

    For each excited frequency (selected with a slider, as in
    :func:`deistools.processing.dmfa_functions.visualize_dmfa_filtering`),
    shows:

    - top: the local DFT band data ``|V(k)|`` against the least-squares model
      ``|K @ theta|``, restricted to the band actually used for that
      frequency.
    - bottom: the fitted target-block Legendre coefficients ``|theta_p|``
      (the ones that become ``H_p``), which exposes overfitting or
      ill-conditioning as unreasonably large high-order coefficients.

    Parameters are identical to :func:`bltva`, so the same call used
    to produce a result can be reused here to inspect it.

    Returns
    -------
    fig, axs
    """
    if Na is None:
        Na = select_na(P, Np, d=getattr(detrend, 'order', 0))
    N, T, _t, k_exc, all_centres, _b, B, V, U = _prepare(
        response, excitation, fs, P, H_exc, Np, Nq, Na, harmonics, detrend)
    _warn_if_bands_are_huge(k_exc, N, band_halfwidth, P)
    M = k_exc.size

    if prune and noise is None:
        noise = estimate_noise_variance(response, excitation, fs, P, H_exc,
                                        Np=Np, Nq=Nq, Na=Na, detrend=detrend)

    def plot_band(e):
        fit = _fit_band(e, k_exc, N, band_halfwidth, all_centres, B,
                        Np, Nq, Na, V, rcond, noise, detrend, prune)
        f_band = fit["K_w"] / T
        f_exc = fit["k_e"] / T
        theta_p = fit["theta"][fit["cmap"]["target"]]

        axs[0].plot(f_band, np.abs(fit["V_w"]), '-o', color='C0', label='Data')
        axs[0].plot(f_band, np.abs(fit["fit"]), 'x', color='C1', label='Fit')
        axs[0].axvline(f_exc, color='green', linestyle='--')
        axs[0].set_xlabel('Frequency / Hz')
        axs[0].set_ylabel('|V(k)|')
        axs[0].set_title(
            f'Excited frequency {f_exc:.4g} Hz   '
            f'cond={fit["cond"]:.2e}   '
            f'rel. residual={np.linalg.norm(fit["residual"]) / np.linalg.norm(fit["V_w"]):.2e}   '
            f'kept {fit["n_theta"]}/{fit["n_theta_full"]} regressors'
        )
        axs[0].legend()

        axs[1].bar(np.arange(theta_p.size), np.abs(theta_p), color='C0')
        axs[1].set_yscale('log')
        axs[1].set_xlabel('Legendre order p')
        axs[1].set_ylabel(r'$|\theta_p|$ (target block)')

    def update(val):
        for ax in axs:
            ax.clear()
        plot_band(val)
        fig.canvas.draw_idle()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.subplots_adjust(bottom=0.25)

    sliders_freq_ax = fig.add_axes([0.20, 0.15, 0.60, 0.03])
    slider_frequency = Slider(
        ax=sliders_freq_ax,
        label="Excited freq. index",
        valmin=0,
        valmax=M - 1,
        valinit=0,
        valstep=1,
    )

    plot_band(slider_frequency.val)
    slider_frequency.on_changed(update)
    plt.show()

    return fig, axs


# --------------------------------------------------------------------------- #
# Minimal self-test: LTI ground truth, no drift.
# The estimator must return H_0 = V(k_e)/U(k_e) exactly and H_p = 0 for p >= 1.
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    fs, Tp, P = 200.0, 180.0, 30
    N = int(fs * Tp) * P
    T = N / fs
    t = np.arange(N) / fs

    H_exc = np.array([1, 3, 5, 9, 13, 21, 33, 51, 81, 129])
    A = 0.05 / H_exc ** 0.4
    phi = rng.uniform(0, 2 * np.pi, H_exc.size)

    def Z_true(f):                                   # simple R + R||C
        w = 2 * np.pi * f
        return 0.02 + 0.01 / (1 + 1j * w * 0.5)

    exc = np.zeros(N)
    resp = np.zeros(N)
    for h, a, p in zip(H_exc, A, phi):
        f = h / Tp
        z = Z_true(f)
        exc += a * np.cos(2 * np.pi * f * t + p)
        resp += a * np.abs(z) * np.cos(2 * np.pi * f * t + p + np.angle(z))

    r = bltva(resp, exc, fs, P, H_exc,
              Np=4, Nq=4, Na=6, harmonics="excited", band_halfwidth=5 * P)

    Z_ref = Z_true(r.f_exc)
    err0 = np.abs(r.H_p[:, 0] - Z_ref) / np.abs(Z_ref)
    leak = np.abs(r.H_p[:, 1:]).max(axis=1) / np.abs(r.H_p[:, 0])

    print(f"max relative error on H_0 : {err0.max():.3e}   (target < 1e-10)")
    print(f"max spurious |H_p|/|H_0|  : {leak.max():.3e}   (target < 1e-10)")
    print(f"worst cond(K)             : {r.worst_condition():.3e}")
    print(f"worst relative residual   : {r.worst_residual():.3e}")

    # Visual check of the per-band fit on the same signal, ground truth is LTI so
    # the fit should overlay the data exactly and theta_p>=1 should sit at the
    # numerical noise floor.
    visualize_bltva_fit(resp, exc, fs, P, H_exc,
                        Np=4, Nq=4, Na=6, harmonics="excited", band_halfwidth=5 * P)