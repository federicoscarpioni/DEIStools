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

Only differencing is implemented. A moving average or a Butterworth low-pass
has a well-defined ``F(k)`` and contributes no boundary block, so adding one
is a matter of filling in the same four methods.

Use with BLTVA only
-------------------
The other estimators in this package (STFT, DMFA) have no way to compensate
for the shaping, so handing them a detrended signal biases the result with no
warning. Detrending is therefore an argument to the BLTVA call rather than
state on :class:`~deistools.processing.mfa.MultiFrequencyAnalysis`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from math import comb

import numpy as np

__all__ = ["Detrending", "Differencing", "select_differencing_order"]


class Detrending(ABC):
    """A trend-removal strategy and everything the estimator needs about it."""

    #: Order of the filter, fed to ``select_na`` as ``d`` (2022 eq. 22).
    order: int = 0

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
