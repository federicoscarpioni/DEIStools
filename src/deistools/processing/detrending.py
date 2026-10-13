"""
Trend removal for BLTVA, following Hallemans et al., IEEE TIM 71 (2022) 6501711.

Why this is not a preprocessing step
------------------------------------
A trend cannot simply be subtracted before estimation. Whatever filter removes
it also shapes the signal of interest and the noise, and the estimator has to
know about that shaping or it will report a biased transfer function with
wrong uncertainties. Removing a trend therefore changes **four** things at
once (2022 paper, eq. 18-27):

1. every skirt regressor is multiplied by the filter response ``F(k)``   (eq. 21)
2. the cost function is weighted by ``W = (F F^H)^-1``                   (eq. 26)
3. the filter's boundary effects add their own regressors               (eq. 21)
4. the transient order ``Na`` shrinks by the order of the filter        (eq. 22)

A strategy here is consequently not "a function that returns a cleaned
signal". It is the bundle of those four things, which is what the protocol
below encodes. A strategy that cannot report its ``frequency_response`` cannot
be used safely, because step 2 would be impossible.

The weighting is what makes differencing free: it attenuates low frequencies
and amplifies high ones, but it does the same to the signal and the noise, so
the signal-to-noise ratio at every bin is unchanged (eq. 20). A naive
low-pass-and-subtract has no such guarantee -- it would destroy exactly the
low-frequency impedance information the measurement is for -- which is why
``frequency_response`` is mandatory rather than optional.

Why only differencing works here
--------------------------------
Of the strategies in this module, **only differencing may be passed to
BLTVA**. That is not an implementation gap; it follows from what the
weighting does, and the reason is worth stating because it is easy to assume
the opposite. (Differencing is also not usually the best option -- plain
``Nq``, with no detrending at all, beats it unless eq. (49) caps ``Nq`` too
low. See :class:`Differencing`.)

A filter applied by circular convolution is *diagonal* in frequency: it
multiplies bin ``k`` by ``F(k)`` and nothing else. Trend and signal that
share a bin are scaled by the same number, so

    |T(k)| / |S(k)|   is unchanged, for every k and every diagonal filter

-- measured here as bit-identical before and after. The filter cannot
separate them, because at a single bin there is nothing to separate them
with. Step 2 then undoes even the attenuation: for a real, zero-phase
response ``F/|F| = 1``, so the weighted system is *identical* to the one
built from the raw record (verified to 5.6e-17 on the skirt regressors).
What survives is only the damage: the trend block of eq. (58) has been
deleted, the trend is still in the data, and nothing models it. Measured on
the 2022 paper's own trend, with ``F(k) = 1`` exactly -- a filter that does
nothing at all, which the protocol happily accepts:

    ==========================================  ==========
    configuration                               error at the lowest tone
    ==========================================  ==========
    no trend, Nq = 5                              2.2e-04
    trend, Nq = 5, no detrending                  1.6e-02
    trend, ``F(k) = 1`` "detrending"              2.6e+00
    ==========================================  ==========

A moving average does score well in that test (1.8e-04), which looks like a
refutation until one varies the averaging length: ``L = 101``, ``401`` and
``1601`` all return 1.8e-04. An answer that does not depend on the averaging
length is not coming from the averaging. It comes from the ``1/|F(k)|``
weighting applied to the *transient* block, which eq. (21) does not multiply
by ``F``: since ``F ~ k^2`` near DC for any boxcar, the weighting hands the
transient polynomial a ``k^-2`` envelope, and that reshaped basis absorbs the
trend skirt. The trend removal is an accident of the reparametrisation, and
a Butterworth low-pass -- same protocol, different ``1/|F|`` shape -- scores
2.6e-02, worse than no detrending at all.

Differencing escapes all of this because it is **not** diagonal. The
zero-prepend of eq. (12) makes ``Delta^d`` annihilate a degree-``d-1``
polynomial *identically in the time domain*, which no multiplication in
frequency can do, and leaves a boundary term that step 3 models exactly. Its
``F(k) = (1 - z_k^-1)^d`` is also complex, so ``F/|F|`` is a phase ramp that
the magnitude weighting does not cancel.

The non-circular route is closed for the others by eq. (49) rather than by
principle: a causal length-``L`` moving average with a zero-padded history
would annihilate a constant exactly, but its boundary term needs ``L-1``
regressors -- 400 columns for ``L = 401``, against the ``2P-1 = 39`` a band
can support.

So the other three strategies are provided for the estimators that have no
weighting to undo, and for studying what naive detrending costs. They refuse
to be used as a BLTVA weighting factor, and :func:`check_response` is the
test that decides which category a new strategy falls into.

Which strategy for which estimator
----------------------------------
Detrending is an argument to the BLTVA call rather than state on
:class:`~deistools.processing.mfa.MultiFrequencyAnalysis`, because only BLTVA
can compensate for the shaping. The other estimators have no weighting at
all, so for them whatever a filter removes is simply gone, and the choice
inverts:

    =======================  =====================  =====================
    strategy                 BLTVA                  DMFA / STFT
    =======================  =====================  =====================
    :class:`Differencing`    the only one that      **destroys the
                             works                  signal**
    :class:`MovingAverage`   refused                at best 1.7x, and not
                                                    consistently
    :class:`LowPassRemoval`  refused                no help on an
                                                    overlapping trend
    :class:`PolynomialFit`   refused; use ``Nq``    the only one that
                                                    reliably helps
    =======================  =====================  =====================

The two columns are exact complements, and one fact explains both. A diagonal
filter cannot separate trend from signal inside a shared bin, so neither
diagonal strategy helps anywhere. What remains is whether the estimator can
undo an attenuation: BLTVA can, which is what lets differencing trade the
whole low-frequency band away and get it back, so only differencing works
there. DMFA and STFT cannot, so only the non-diagonal strategy -- the
polynomial fit -- is any use, and it pays in distortion instead.

**Never pass differencing to DMFA or STFT.** ``Differencing(1)`` attenuates
the lowest excited bin of a 160000-sample record to ``|F| = 3e-04``; the
measured DMFA error at that tone is 1.00e+00, i.e. the tone is gone. In BLTVA
the same attenuation costs nothing, because step 2 divides it back out. This
is the single most dangerous thing in this module, and the reason
``frequency_response`` is mandatory: it is the only way to see the 3e-04
coming.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from math import comb

import numpy as np
from numpy.polynomial.legendre import legvander

__all__ = [
    "Detrending", "Differencing", "DiagonalDetrending", "MovingAverage",
    "LowPassRemoval", "PolynomialFit", "select_differencing_order",
    "check_response", "require_bltva_compatible",
]


class Detrending(ABC):
    """A trend-removal strategy and everything the estimator needs about it."""

    #: Order of the filter, fed to ``select_na`` as ``d`` (2022 eq. 22).
    order: int = 0

    #: Whether this strategy may be passed to BLTVA as ``detrend``. False for
    #: every strategy whose effect the eq. (26) weighting undoes, which is all
    #: of the diagonal ones -- see the module docstring and
    #: :func:`require_bltva_compatible`. Such a strategy is still usable on its
    #: own, and with the estimators that apply no weighting.
    bltva_compatible: bool = True

    @abstractmethod
    def apply(self, y: np.ndarray) -> np.ndarray:
        """Return the detrended time series, same length as ``y``."""

    @abstractmethod
    def frequency_response(self, k: np.ndarray, N: int) -> np.ndarray:
        """``F(k)``, the filter's response at DFT bins ``k`` of an N-point record."""

    def extra_regressors(self, k: np.ndarray, N: int) -> np.ndarray:
        """Columns modelling the filter's boundary effects, shape ``(len(k), n)``.

        Empty by default: a filter with no startup discontinuity adds nothing.
        """
        return np.zeros((len(k), 0), dtype=complex)

    @property
    def n_extra_params(self) -> int:
        return 0


class Differencing(Detrending):
    """``d``-th order differencing, 2022 paper Section III.

        y_d(n) = Delta^d y~(n),   y~(n) = 0 for n < 0

    Differencing suppresses a smooth trend without modelling it, which is the
    point of the 2022 paper: a trend that no finite polynomial captures well
    cannot be removed parametrically, and trying inflates the variance of the
    low-frequency estimates.

    What it buys is **columns, not accuracy**. The strategy to beat is no
    detrending at all with a generous ``Nq``, and on the paper's own
    non-polynomial trend that usually wins -- relative error at the lowest
    excited tone:

        ======  ====================  =================  ==============
        Np      Nq cap from eq. (49)  best Nq            Differencing(1)
        ======  ====================  =================  ==============
        2       17                    1.4e-04 (Nq=7)     8.2e-03
        4       13                    2.2e-04 (Nq=11)    5.5e-04
        6       9                     1.2e-04 (Nq=8)     7.1e-02
        8       5                     1.9e+00 (Nq=4)     4.6e-02
        ======  ====================  =================  ==============

    The paper's comparison is the ``Nq = 2`` case, where parametric removal
    does fail (1.35 against 7.8e-03). Raise ``Nq`` and it reverses, because a
    joint fit does not distort and a larger basis models the trend better.

    The last row is where differencing earns its place. Eq. (49),
    ``Nq + 3*Np + Na + 1 < 2P - 1``, means a high ``Np`` eats the budget: at
    ``Np = 8`` the cap is ``Nq = 5`` and the trend is then unmodellable
    (1.9e+00 at ``Nq = 4``, 5.4e+00 at ``Nq = 3``), while differencing
    replaces the whole drift block with ``d-1`` boundary columns -- 15 per
    band instead of 20 -- and reaches 4.6e-02. That configuration is not
    otherwise broken; on a trend-free record it gives 2.0e-04.

    So reach for ``Nq`` first, and for differencing when eq. (49) will not
    allow the trend order the record needs.

    Its response is a high-pass, eq. (12b)::

        F_d(k) = (1 - z_k^-1)^d,     z_k = exp(j*2*pi*k/N)

    Because the samples at negative indices are replaced by zeros rather than
    their true values, differencing leaves a boundary term: a polynomial of
    degree ``d-1`` in ``z_k^-1`` (Theorem 2). Those are the extra regressors.
    The ``m = 0`` term is omitted because a constant is already spanned by the
    transient polynomial, which is why eq. (21) sums from ``m = 1``.
    """

    def __init__(self, order: int = 1):
        if order < 1:
            raise ValueError(f"differencing order must be >= 1, got {order}")
        self.order = int(order)

    def __repr__(self):
        return f"Differencing(order={self.order})"

    def apply(self, y):
        y = np.asarray(y, dtype=float)
        for _ in range(self.order):
            y = np.diff(y, prepend=0.0)
        return y

    def frequency_response(self, k, N):
        z_inv = np.exp(-2j * np.pi * np.asarray(k) / N)
        return (1.0 - z_inv) ** self.order

    def extra_regressors(self, k, N):
        k = np.asarray(k)
        m = np.arange(1, self.order)                      # eq. (21): m = 1..d-1
        if m.size == 0:
            return np.zeros((k.size, 0), dtype=complex)
        return np.exp(-2j * np.pi * np.outer(k, m) / N)

    @property
    def n_extra_params(self):
        return max(self.order - 1, 0)

    def boundary_coefficients(self, y):
        """eq. (12c): the exact lambda_m, from the last ``d`` samples of ``y``.

        Not needed for estimation -- the gammas are fitted -- but it makes the
        boundary term checkable against theory.
        """
        y = np.asarray(y, dtype=float)
        d = self.order
        return np.array([
            sum((-1) ** (r + m + 1) * comb(d, r + m) * y[-r]
                for r in range(1, d - m + 1))
            for m in range(d)
        ])


def select_differencing_order(y, fs, f_max, max_order=5):
    """Pick ``d`` by minimising the variance of the differenced series, eq. (11).

        d = argmin_{d >= 0} var(Delta^d y_bar(n))

    ``y_bar`` is ``y`` subsampled by ``floor(fs / (2*f_max))``, which the paper
    requires so that the series looks random to the criterion rather than
    oversampled and smooth. Differencing too few times leaves trend; too many
    amplifies the high-frequency end and the variance rises again, so the
    minimum is the balance point.

    Returns 0 when no differencing helps, meaning the record carries no trend
    worth removing.

    Caveat: the criterion assumes the record is "a trend plus a random time
    series", which the paper justifies by the random phases of the multisine.
    A strongly low-pass response breaks that assumption -- its variance is
    dominated by the lowest excited frequencies, which differencing attenuates
    -- so the rule can return ``d >= 1`` on a record with no trend at all.
    Treat the result as a starting point and check it against the spectrum
    rather than applying it blindly.
    """
    y = np.asarray(y, dtype=float)
    q = max(1, int(np.floor(fs / (2.0 * f_max))))
    series = y[::q]
    if series.size <= max_order + 1:
        raise ValueError(
            f"subsampling by {q} leaves only {series.size} samples, too few to "
            f"difference {max_order} times")

    variances = []
    for d in range(max_order + 1):
        if d:
            series = np.diff(series, prepend=0.0)
        # the first d samples carry the zero-padding step, not signal
        variances.append(float(np.var(series[d:])))
    return int(np.argmin(variances))


# --------------------------------------------------------------------------- #
# Diagonal strategies
#
# Everything below is applied as a circular convolution, i.e. by multiplying
# the DFT by a real, symmetric, zero-phase response. That makes step 1 of the
# protocol exact by construction and step 3 empty, which is why these classes
# satisfy the interface -- and, as the module docstring shows, exactly why the
# eq. (26) weighting cancels them. They are marked unusable with BLTVA.
# --------------------------------------------------------------------------- #

class DiagonalDetrending(Detrending):
    """Trend removal by a zero-phase response, applied by FFT.

    A subclass supplies :meth:`frequency_response` only; ``apply`` is then
    ``ifft(F * fft(y))``, which makes ``Y_d(k) = F(k) Y(k)`` hold exactly and
    contributes no boundary regressors.

    Two consequences follow from being diagonal, and both matter more than the
    convenience:

    * the response must be symmetric, ``F(k) = F(N-k)``, or the detrended
      record is not real. :func:`check_response` checks this.
    * the record is treated as periodic. A trend makes ``y[N-1]`` differ
      sharply from ``y[0]``, and the circular filter sees that jump as signal,
      so the lowest bins carry a residual the filter put there itself. There
      is no diagonal fix: a window would break step 1.
    """

    bltva_compatible = False

    def apply(self, y):
        y = np.asarray(y, dtype=float)
        N = y.size
        F = self.frequency_response(np.arange(N), N)
        return np.fft.ifft(F * np.fft.fft(y)).real

    def band_response(self, k_centre, half_width, N):
        """``|F|`` across the skirt an estimator actually uses around a tone.

        The response *at* a tone is not the diagnostic, and reporting it alone
        is actively misleading. DMFA reconstructs the time variation from the
        bins either side of each tone, so what biases the envelope is how much
        ``|F|`` *varies* over that band, not its value at the centre.
        Measured on ``MovingAverage(20001)`` with ``N = 160000``, a DMFA band
        of +-4 bins around bin 8: ``|F|`` at the tone is 1.0000, while across
        the band it runs from 0.363 to 1.214 -- a 3.3x ripple, and not even
        monotone, because the Dirichlet kernel overshoots. Nothing divides the
        estimate by this again, so all of it is uncorrected bias.

        For STFT, which uses only the centre bin, ``half_width = 0`` gives the
        relevant number.

        Parameters
        ----------
        k_centre : int
            The tone's DFT bin, ``P * h`` for harmonic ``h``.
        half_width : int
            Half-width in bins of the band the estimator uses. For DMFA this
            is ``round(bandwidth * T)``.
        N : int
            Record length in samples.

        Returns
        -------
        dict with ``centre``, ``minimum``, ``maximum`` and ``ripple``
        (``maximum / minimum``, 1.0 for a perfectly flat band).
        """
        k = np.arange(k_centre - half_width, k_centre + half_width + 1)
        F = np.abs(self.frequency_response(k, N))
        lo = float(F.min())
        return {"centre": float(abs(self.frequency_response(
                    np.array([k_centre]), N)[0])),
                "minimum": lo,
                "maximum": float(F.max()),
                "ripple": float(F.max() / lo) if lo > 0 else float("inf")}


class MovingAverage(DiagonalDetrending):
    """Subtract a centred moving average of ``length`` samples.

        y_d(n) = y(n) - (1/L) * sum_{m=-M}^{M} y(n-m),   L = 2M+1

    The response is one minus the Dirichlet kernel, which is real and
    symmetric because the window is centred::

        F(k) = 1 - sin(pi k L / N) / (L sin(pi k / N))

    ``length`` must be odd, so that the window has a true centre and the
    response carries no phase. Near DC ``F(k) ~ (L^2 - 1) (pi k / N)^2 / 6``:
    a double zero, so the constant bin is annihilated and everything else is
    merely attenuated.

    Choosing ``length``: a boxcar is a poor filter and this is where it shows.
    Its response is the Dirichlet kernel, so ``F`` has sidelobes, overshoots
    above 1, and is not monotone. Checking ``|F|`` at the tone is therefore
    worthless -- ``MovingAverage(20001)`` on a 160000-sample record has
    ``|F| = 1.00005`` at bin 8 and swings from 0.363 to 1.214 over the +-4
    bins either side. Use :meth:`band_response` and judge by ``ripple``.

    Measured cost on DMFA with that band, at the lowest tone, against a clean
    baseline of 4.1e-03:

        ====================  ==========  ==========
        record                no filter   with it
        ====================  ==========  ==========
        no trend              4.1e-03     1.9e-02
        trended, L = 20001    8.7e-02     5.0e-02
        trended, L = 40001    8.7e-02     9.7e-02
        ====================  ==========  ==========

    Read those three rows together before concluding anything. ``L = 20001``
    *does* improve the trended record, by 1.7x -- per-bin invariance does not
    forbid it, because a filter can still reweight which bins dominate a band
    fit even though it cannot change the ratio within any one of them. But
    ``L = 40001`` makes it worse, so the direction is not even consistent
    across settings, and the win is bought by raising the trend-free floor
    4.5x, from 4.1e-03 to 1.9e-02. Nothing is gained that
    :class:`PolynomialFit` does not gain better (7.9e-03 on the same record).

    The boxcar is here for completeness. :class:`LowPassRemoval` is strictly
    the better shape.
    """

    def __init__(self, length: int):
        length = int(length)
        if length < 3:
            raise ValueError(f"moving-average length must be >= 3, got {length}")
        if length % 2 == 0:
            raise ValueError(
                f"moving-average length must be odd so the window is centred "
                f"and the response has no phase, got {length}. Use "
                f"{length + 1}.")
        self.length = length
        self.order = 2                       # double zero at DC

    def __repr__(self):
        return f"MovingAverage(length={self.length})"

    def frequency_response(self, k, N):
        x = np.pi * np.asarray(k, dtype=float) / N
        s = np.sin(x)
        at_dc = np.abs(s) < 1e-15
        safe = np.where(at_dc, 1.0, s)
        dirichlet = np.where(at_dc, 1.0,
                             np.sin(self.length * x) / (self.length * safe))
        return (1.0 - dirichlet).astype(complex)


class LowPassRemoval(DiagonalDetrending):
    """Subtract the low-pass content: high-pass with a Butterworth magnitude.

        F(k) = r^n / sqrt(1 + r^(2n)),      r = |f_k| / f_c

    The filter is defined by this magnitude and applied as such, so it is
    zero-phase and has no startup transient -- unlike the same Butterworth run
    as a time-domain recursion, which would have both and would then need
    boundary regressors it cannot afford.

    The cutoff must stay well below the slowest excited bin,
    ``P * min(H_exc)``; :meth:`band_response` reports the cost there. Unlike
    a boxcar, the Butterworth magnitude is monotone and never exceeds 1, so a
    cutoff a few times below the slowest tone gives a genuinely flat band:
    measured at ``cutoff = 2`` over +-4 bins around bin 8, ripple 1.03x and a
    DMFA error of 5.8e-03 on a trend-free record against a 4.1e-03 baseline.

    It still does not remove a trend that overlaps the tones -- 8.7e-02
    before, 8.8e-02 after at ``cutoff = 2``, and 1.2e-01 at ``cutoff = 4,
    order = 6`` -- for the reason in the module docstring. What it does well
    is remove a trend that is genuinely *slower* than everything excited,
    which is the case the cutoff describes. Whether a given trend is that
    case is an empirical question, and :meth:`band_response` answers it.

    Parameters
    ----------
    cutoff : float
        Half-power frequency in **cycles per record**, which is the DFT bin
        index: bin ``k`` is ``k / T`` Hz for a record of ``T`` seconds.
        Expressing it this way keeps the strategy independent of the sampling
        rate, and it is the only unit ``frequency_response(k, N)`` can
        resolve. A cutoff of 3 means "treat anything slower than three cycles
        over the whole record as trend".
    order : int
        Butterworth order ``n``. ``F ~ r^n`` near DC, so a higher order rolls
        off more sharply and leaves more of the lowest tone intact.
    """

    def __init__(self, cutoff: float, order: int = 2):
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        if order < 1:
            raise ValueError(f"Butterworth order must be >= 1, got {order}")
        self.cutoff = float(cutoff)
        self.order = int(order)

    def __repr__(self):
        return f"LowPassRemoval(cutoff={self.cutoff:g}, order={self.order})"

    def frequency_response(self, k, N):
        k = np.asarray(k, dtype=float) % N
        bins = np.minimum(k, N - k)              # |f|, so F is symmetric
        r = bins / self.cutoff
        rn = r ** self.order
        return (rn / np.sqrt(1.0 + rn ** 2)).astype(complex)


# --------------------------------------------------------------------------- #
# Parametric strategy
# --------------------------------------------------------------------------- #

class PolynomialFit(Detrending):
    """Fit a polynomial of degree ``order`` by least squares and subtract it.

        y_d = (I - b (b^T b)^-1 b^T) y

    with ``b`` the Legendre basis on the record, the same basis BLTVA uses for
    its trend block. Legendre rather than monomial because the monomial
    Vandermonde is hopeless beyond degree ~15 while Legendre stays usable.

    This is the one strategy with **no frequency response at all**, and not
    for want of deriving it. Subtracting a least-squares fit is an orthogonal
    projection: a rank-``(order+1)`` operator whose frequency-domain kernel is
    ``sum_q B_q(k) B_q(k')*``, dense in ``(k, k')``. It mixes bins, so no
    ``F(k)`` exists, step 2 of the protocol cannot be carried out, and
    :meth:`frequency_response` raises.

    That is not a loss, because BLTVA already does polynomial trend removal
    and does it better. The ``Nq`` drift block of eq. (58) is this same
    Legendre basis, *fitted jointly with the impedance instead of subtracted
    first*, so the trend is separated from the signal using the whole band of
    data rather than removed from it by projection, and the distortion
    measured below does not arise. Use ``Nq``, not this.

    Being non-diagonal is also what makes it the **only** strategy here that
    helps DMFA or STFT, which have no trend model of their own and no
    weighting either. On a trended record it takes the lowest tone from
    8.7e-02 to 7.9e-03 where both diagonal filters leave it at 8.8e-02 or
    worse.

    The price is the distortion the user pays silently, measured by applying
    the same subtraction to a record with **no trend**, so that everything
    removed is signal. At the lowest tone, against a clean error of 4.1e-03:

        ======  ========
        degree  DMFA error at the lowest tone
        ======  ========
        1       4.9e-03
        3       7.9e-03
        8       4.0e-02
        12      6.2e-02
        ======  ========

    The distortion grows with the degree and there is nothing to correct it
    with, which is both the reason the 2022 paper reaches for differencing and
    the reason BLTVA fits this basis instead of subtracting it. Keep the
    degree as low as the trend allows.
    """

    bltva_compatible = False

    def __init__(self, order: int = 1):
        if order < 0:
            raise ValueError(f"polynomial order must be >= 0, got {order}")
        self.order = int(order)

    def __repr__(self):
        return f"PolynomialFit(order={self.order})"

    def basis(self, N):
        """The Legendre design matrix on the record, shape ``(N, order+1)``."""
        xi = 2.0 * np.arange(N) / (N - 1) - 1.0
        return legvander(xi, self.order)

    def trend(self, y):
        """The fitted trend itself, which ``apply`` subtracts."""
        y = np.asarray(y, dtype=float)
        b = self.basis(y.size)
        coefficients, *_ = np.linalg.lstsq(b, y, rcond=None)
        return b @ coefficients

    def apply(self, y):
        return np.asarray(y, dtype=float) - self.trend(y)

    def frequency_response(self, k, N):
        raise NotImplementedError(
            "PolynomialFit has no frequency response: subtracting a "
            "least-squares fit is a rank-deficient projection, not a "
            "convolution, so it mixes DFT bins and no F(k) exists. Inside "
            "BLTVA use the Nq drift block instead, which is the same "
            "Legendre basis fitted jointly rather than subtracted -- see the "
            "class docstring.")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def check_response(strategy, N=4096, seed=0):
    """Verify step 1 of the protocol: that ``apply`` is what ``F(k)`` claims.

    The contract a strategy must satisfy is eq. (12) of the 2022 paper,

        DFT(apply(y))  =  F(k) * DFT(y)  +  (something in the span of
                                             ``extra_regressors``)

    for every ``y``. This measures the part of the residual that the boundary
    regressors cannot absorb, relative to the signal, on a random record with
    a quadratic trend. A strategy that satisfies the contract returns ~1e-15;
    anything else is reporting a response that does not describe what it did,
    and an estimator built on it is biased with no warning.

    A constant column joins ``extra_regressors`` in the projection, because
    the estimator has one: eq. (21) omits the ``m = 0`` boundary term on the
    grounds that the transient polynomial already spans a constant, and the
    first column of ``legvander`` is exactly that. Leaving it out would fail
    ``Differencing(1)`` -- whose entire boundary term is that constant -- at
    7.8e-03, which is a gap in the check rather than in the class.

    Also reports whether the detrended record came back real, which requires
    ``F(k) = F(N-k)``.

    Returns
    -------
    dict with keys ``residual`` (the relative residual above), ``imaginary``
    (the largest imaginary part the response would have produced) and
    ``exact`` (whether ``residual`` is at round-off).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, N)
    y = rng.normal(size=N) + 3.0 * t ** 2 + 0.5 * t

    k = np.arange(N)
    F = strategy.frequency_response(k, N)
    target = F * np.fft.fft(y)
    residual = np.fft.fft(strategy.apply(y)) - target

    # what the estimator can legitimately absorb: the boundary block, plus the
    # constant that eq. (21) leaves to the transient polynomial
    absorbable = np.hstack([np.ones((N, 1), dtype=complex),
                            strategy.extra_regressors(k, N)])
    coefficients, *_ = np.linalg.lstsq(absorbable, residual, rcond=None)
    residual = residual - absorbable @ coefficients

    scale = np.max(np.abs(target))
    relative = float(np.max(np.abs(residual)) / scale) if scale else float("inf")
    imaginary = float(np.max(np.abs(np.fft.ifft(F * np.fft.fft(y)).imag)))
    return {"residual": relative, "imaginary": imaginary,
            "exact": relative < 1e-10}


def require_bltva_compatible(strategy):
    """Raise unless ``strategy`` may be used as a BLTVA weighting factor.

    Called by the estimator. The rejected strategies are not broken -- they
    satisfy the protocol -- but the eq. (26) weighting undoes them exactly, so
    passing one deletes the trend block of eq. (58) while leaving the trend in
    the data. The module docstring has the measurements.
    """
    if strategy is None or getattr(strategy, "bltva_compatible", True):
        return
    raise ValueError(
        f"{strategy!r} cannot be used as a BLTVA detrending strategy. It is "
        "diagonal in frequency, so the eq. (26) weighting 1/|F(k)| undoes it "
        "exactly: the trend-to-signal ratio at every bin is unchanged, while "
        "the trend block of eq. (58) has been removed. Measured on the 2022 "
        "paper's trend, the lowest tone goes from 1.6e-02 (no detrending, "
        "Nq=5) to 2.6e+00.\n"
        "  - for a polynomial trend, pass Nq instead: it is the same basis, "
        "fitted jointly rather than subtracted.\n"
        "  - for a trend no polynomial captures, pass Differencing(d), which "
        "cancels it in the time domain rather than by attenuation.\n"
        "  - to study what naive detrending costs, apply the strategy "
        "yourself and call bltva with detrend=None.")
