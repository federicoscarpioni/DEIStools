"""
Best linear approximation per burst, after Pintelon, Louarroudi and Lataire.

The short-time Fourier estimator in :mod:`~deistools.processing.stft_functions`
reads the excited bins of each window and divides. That is enough to get an
impedance and nothing else: a single window gives no way to say whether the
number it produced is limited by measurement noise, by nonlinearity, or by the
cell changing while the window was open.

This module adds that. It keeps STFT's assumption -- the system is whatever it
is inside one window, with no model of how it varies -- and uses the bins STFT
throws away to classify the error.

The one structural fact
-----------------------
Cut a record into windows of ``P`` whole multisine periods. The DFT of a
window has fundamental ``1/(P*Tp)``, so a tone at harmonic ``h`` of the
multisine lands on bin ``P*h`` and **only every P-th bin is a harmonic of the
multisine at all**. A nonlinear system of the PISPO class (period in, same
period out) answers a periodic input with a periodic output, so its distortion
is harmonically related to the input and can land *only* on those bins. That
splits the spectrum of one burst into four classes:

    ========================  ====================================
    bins                      what is there
    ========================  ====================================
    ``k = P*h``, h excited    signal (+ everything below)
    ``k = P*m``, m odd        odd-order distortion + noise + drift
    ``k = P*m``, m even       even-order distortion + noise + drift
    ``k`` not a multiple of P noise + drift
    ========================  ====================================

**This requires ``P >= 2``.** At ``P = 1`` the window fundamental *is* the
multisine fundamental, every bin is a harmonic, and the fourth class is empty.
On the package's 13-tone grid, in band, :func:`classify_bins` counts

    ===  =======  =====  ======  ========
    P    noise    odd    even    excited
    ===  =======  =====  ======  ========
    1    0        168    180     13
    2    301      138    150     13
    4    903      138    150     13
    8    2107     138    150     13
    ===  =======  =====  ======  ========

so the noise class appears only at ``P >= 2`` and then grows with the burst,
while the detection lines stay put. Since one period is exactly the window the
plain STFT prefers, the uncertainty is bought with time resolution and there
is no way around that.

What the noise class actually contains
--------------------------------------
On a time-varying cell the non-harmonic bins are **not** noise alone. The time
variation convolves each excited line into its neighbours (Lataire 2012
eq. 48: a linear-in-time variation has ``B_1(k) ~ 1/k``), so every tone grows a
skirt, and with a dozen tones the skirts cover the band. Measured on the
transistor system, a 40 s burst cut from the usual 400 s record, as a ratio to
the pure-noise floor at the same bins:

    ================  =======
    distance to the   ratio to
    nearest tone      noise floor
    ================  =======
    1-2 bins          7656
    6-10 bins         463
    21-50 bins        71
    51-100 bins       44
    ================  =======

It decays, and it never arrives. So a noise level read off these bins is an
**upper bound**, and on a strongly varying cell a loose one. That is not a
defect of the implementation; separating the two needs a model of the
variation, which is what :mod:`~deistools.processing.bltva` is for and what
this module deliberately does not assume.

:func:`ratio_noise_to_time_variation` says which case you are in, so the bound
can be read correctly rather than trusted blindly.

References
----------
Lataire, Louarroudi and Pintelon, *IEEE TIM* **61** (2012) 2132 -- the
best linear time-invariant approximation of a time-varying system, and the
frequency-correlation test used here.

Pintelon, Louarroudi and Lataire, *IEEE TIM* **61** (2012) 3387 -- the
multiple-input time-invariant equivalent, and separating leakage from time
variation.

Pintelon, Louarroudi and Lataire, *IEEE TIM* **62** (2013) 3361 -- the
periodic-excitation procedure this module follows: which bins carry what, and
why two periods are the minimum.

Pintelon and Schoukens, *System Identification: A Frequency Domain Approach*,
2nd ed., Wiley-IEEE Press (2012) -- the underlying framework.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "BurstBLA", "classify_bins", "ratio_noise_to_time_variation",
    "burst_bla", "bla_eis",
]


#: Ratio of the frequency autocorrelation at lags 0 and 1 for a residual that
#: is pure linear-in-time variation, Lataire 2012 eq. (31). The test statistic
#: is scaled by its reciprocal so that 1 means "time variation".
PI_SQUARED_OVER_SIX = np.pi ** 2 / 6.0


class BurstBLA:
    """What one burst says about itself.

    The three ``std_*`` attributes are **measured levels, not a
    decomposition**: each is what the bins of one class imply for the
    standard deviation of ``impedance``, with nothing subtracted. They are
    reported that way on purpose. Subtracting one from another is how the
    distortion is normally quoted, and it is available from
    :meth:`distortion_odd` and :meth:`distortion_even`, but the subtraction
    has two failure modes that a stored, already-subtracted number would
    hide -- see those methods.

    Attributes
    ----------
    impedance : (M,) complex
        ``V/I`` at the excited bins -- the same number the STFT returns.
    std_noise : (M,) float
        Level at the non-harmonic bins. Noise alone on a time-invariant cell;
        noise *plus the time-variation skirt* otherwise, so an **upper
        bound**. :attr:`r_ntv` says which.
    std_odd_lines, std_even_lines : (M,) float or None
        Level at the unexcited odd and even harmonics. Each contains the
        noise as well as any distortion of that parity. ``None`` when the
        grid leaves no harmonic of that parity free in band.
    r_ntv : float
        Which error dominates the non-harmonic bins. Near 1 means the time
        variation does; well above 1 means noise does. See
        :func:`ratio_noise_to_time_variation`.
    n_bins : dict
        How many bins fell in each class, so a thin estimate is visible.
    """

    __slots__ = ("impedance", "std_noise", "std_odd_lines",
                 "std_even_lines", "r_ntv", "n_bins", "n_local")

    def __init__(self, impedance, std_noise, std_odd_lines,
                 std_even_lines, r_ntv, n_bins, n_local=30):
        self.impedance = impedance
        self.std_noise = std_noise
        self.std_odd_lines = std_odd_lines
        self.std_even_lines = std_even_lines
        self.r_ntv = r_ntv
        self.n_bins = n_bins
        self.n_local = n_local

    def __repr__(self):
        return (f"BurstBLA({self.impedance.size} tones, "
                f"r_ntv={self.r_ntv:.2f}, bins={self.n_bins})")

    def dominant_error(self, tolerance=3.0):
        """``'time variation'`` or ``'noise'``, from :attr:`r_ntv`.

        ``tolerance`` is how far above 1 the ratio must sit before the
        residual is called noise. The statistic is itself noisy on a short
        burst, so a wide margin is the honest default.

        The test needs a residual to work on. On a noiseless simulation the
        non-harmonic bins hold nothing but round-off and the answer is
        meaningless -- measured at 0.86 for a stationary noiseless cell,
        which would read as "time variation" for a system that has none.
        Real records always have noise; simulated ones should be given some.
        """
        return "noise" if self.r_ntv > tolerance else "time variation"

    def _excess(self, level, z=3.0):
        """``sqrt(level^2 - noise^2)``, NaN unless the excess is significant.

        A positive difference is not by itself evidence. Each level is a mean
        of ``n_local`` bin powers, so it carries a relative standard error of
        ``1/sqrt(n_local)``, and their difference about ``sqrt(2/n_local)``.
        The excess is reported only when it clears ``z`` of those, i.e.

            level^2 - noise^2  >  z * sqrt(2/n_local) * noise^2

        Without the threshold, merely requiring a positive difference fires
        on **10 of 13 tones of a linear stationary cell**, where the two
        levels agree to 2.5% and the sign is a coin toss. With it, none do.

        NaN rather than zero because the two mean different things: zero
        would claim the distortion was measured and found absent, while NaN
        says this burst cannot resolve it.
        """
        if level is None:
            return None
        noise_power = self.std_noise ** 2
        excess = level ** 2 - noise_power
        threshold = z * np.sqrt(2.0 / max(self.n_local, 1)) * noise_power
        return np.where(excess > threshold, np.sqrt(np.abs(excess)), np.nan)

    def distortion_odd(self):
        """Odd-order distortion, the noise removed. NaN where unresolvable.

        This is the one that matters for an odd multisine: odd-order
        distortion lands *on* the excited lines as well as the empty odd
        ones, so it biases the impedance, where even-order distortion only
        sits beside it.

        Two things make the result NaN, and both are information:

        * **The burst is too short to resolve it.** With ``n_local`` bins
          averaged, a power estimate carries about ``1/sqrt(n_local)``
          relative error, so when there is no real distortion the difference
          scatters about zero and half the tones come back NaN. Taking
          ``max(excess, 0)`` instead would rectify that scatter into a
          spurious positive level -- measured at 1.3e-05 on a *linear*
          stationary cell, which is exactly the false alarm this avoids.
        * **The cell is varying.** Then the non-harmonic bins carry the
          time-variation skirt and sit *above* the harmonic ones, so the
          subtraction is negative everywhere. Measured on the block-oriented
          system, which is nonlinear *and* time-varying: every tone NaN,
          although the distortion is certainly there. Check :attr:`r_ntv`
          first; if it is near 1, use
          :func:`~deistools.processing.bltva.bltva`, which models the
          variation and reports ``var_distortion_odd`` with it removed.
        """
        return self._excess(self.std_odd_lines)

    def distortion_even(self):
        """Even-order distortion, the noise removed. NaN where unresolvable.

        Same caveats as :meth:`distortion_odd`. Even-order distortion does
        not land on the excited lines of an odd multisine, so it does not
        bias the impedance; it is reported because its presence says
        something about the cell.
        """
        return self._excess(self.std_even_lines)


def classify_bins(n_samples, periods_per_window, harmonics, band_only=True):
    """Which DFT bins of one burst carry what.

    Parameters
    ----------
    n_samples : int
        Length of the burst.
    periods_per_window : int
        Whole multisine periods inside it. Must be >= 2, or the noise class
        is empty -- see the module docstring.
    harmonics : sequence of int
        Excited harmonic numbers of the multisine.
    band_only : bool
        Restrict every class to bins at or below the highest excited tone.
        Above it there is no signal to compare against and the data
        acquisition's own filtering shapes whatever is left.

    Returns
    -------
    dict of boolean masks over ``k = 1 .. n_samples//2 - 1``, with keys
    ``excited``, ``odd_nl``, ``even_nl``, ``noise`` and ``band``, plus
    ``bins`` (the bin indices themselves) and ``k_excited`` (the excited bins
    in the order ``harmonics`` gives them).
    """
    periods = int(periods_per_window)
    if periods < 2:
        raise ValueError(
            f"periods_per_window must be >= 2, got {periods}. At one period "
            "every bin of the burst is a harmonic of the multisine, so "
            "nonlinear distortion and noise land in the same places and "
            "cannot be separated. This is the price of the uncertainty "
            "estimate, paid in time resolution.")

    harmonics = np.asarray(harmonics, dtype=int)
    k = np.arange(1, n_samples // 2)
    is_harmonic = (k % periods == 0)
    order = np.where(is_harmonic, k // periods, -1)

    excited = np.isin(order, harmonics)
    free = is_harmonic & ~excited
    band = (k <= harmonics.max() * periods) if band_only else np.ones_like(k, bool)

    return {
        "bins": k,
        "k_excited": harmonics * periods,
        "excited": excited & band,
        "odd_nl": free & (order % 2 == 1) & band,
        "even_nl": free & (order % 2 == 0) & band,
        "noise": (~is_harmonic) & band,
        "band": band,
    }


def ratio_noise_to_time_variation(spectrum, noise_mask, bins):
    """Is the residual noise, or the cell moving? Lataire 2012 eq. (33).

    The two are told apart by their correlation over frequency rather than
    their size. Stationary noise is uncorrelated between neighbouring DFT
    bins. A time variation is not: it reaches each bin by convolving the
    excited lines with the spectrum of the variation, so neighbours share
    contributions. For a linear-in-time variation the ratio of the
    autocorrelation at lags 0 and 1 works out to exactly ``pi^2/6``
    (eq. 31), so scaling by ``6/pi^2`` puts "pure time variation" at 1:

        R = (6/pi^2) * mean|V(k)|^2 / |mean V(k) conj(V(k+1))|

    Near 1 the residual is time variation; well above 1 it is noise. Only
    *adjacent* bins of ``noise_mask`` are used, since the lag has to be a
    real neighbour.

    Measured on one 80 s burst: 12.7 for a stationary cell with noise, 0.97
    for the block-oriented system and 0.77 for the transistor -- both of which
    vary.

    Returns ``inf`` when fewer than two adjacent pairs survive, which is the
    honest answer rather than a number built from nothing.
    """
    selected = np.flatnonzero(noise_mask)
    if selected.size < 3:
        return float("inf")
    adjacent = selected[np.flatnonzero(np.diff(bins[selected]) == 1)]
    if adjacent.size < 2:
        return float("inf")

    lag0 = np.mean(np.abs(spectrum[adjacent]) ** 2)
    lag1 = np.abs(np.mean(spectrum[adjacent] * np.conj(spectrum[adjacent + 1])))
    if lag1 == 0.0:
        return float("inf")
    return float(lag0 / lag1 / PI_SQUARED_OVER_SIX)


def _local_power(power, mask, bins, centres, n_local):
    """Mean power in the ``n_local`` masked bins nearest each centre.

    Taken locally rather than over the whole band because both the noise and
    the distortion are coloured: a single number for the band would be wrong
    at both ends. Returns ``(power, count)`` per centre.
    """
    available = bins[mask]
    if available.size == 0:
        return (np.full(centres.size, np.nan),
                np.zeros(centres.size, dtype=int))
    values = power[mask]
    out = np.empty(centres.size)
    used = np.empty(centres.size, dtype=int)
    for i, centre in enumerate(centres):
        order = np.argsort(np.abs(available - centre))[:n_local]
        out[i] = values[order].mean()
        used[i] = order.size
    return out, used


def burst_bla(voltage, current, harmonics, periods_per_window, n_local=30):
    """Impedance of one burst, with the error split by where it came from.

    Parameters
    ----------
    voltage, current : (n,) real
        One burst, holding exactly ``periods_per_window`` multisine periods.
    harmonics : sequence of int
        Excited harmonic numbers.
    periods_per_window : int
        Whole multisine periods in the burst, at least 2.
    n_local : int
        How many bins of a class to average for the level at each tone. More
        is steadier and less local; the default resolves the colouring of the
        noise without being noisy itself.

    Returns
    -------
    BurstBLA

    Notes
    -----
    The levels are referred to the impedance by first-order propagation,

        var(Z)/|Z|^2 = var(V)/|V|^2 + var(I)/|I|^2

    which treats the two channels as independently disturbed. That is the
    same convention :func:`~deistools.processing.dmfa_functions.extract_impedance_with_error`
    uses, so the two are comparable.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    if voltage.shape != current.shape or voltage.ndim != 1:
        raise ValueError("voltage and current must be 1-D arrays of equal length")

    n = voltage.size
    classes = classify_bins(n, periods_per_window, harmonics)
    k = classes["bins"]
    k_exc = classes["k_excited"]
    if k_exc.max() >= k[-1]:
        raise ValueError(
            f"the highest excited harmonic sits at bin {k_exc.max()}, at or "
            f"above the burst's Nyquist bin {k[-1] + 1}")

    V = np.fft.fft(voltage)[1:n // 2] / n
    I = np.fft.fft(current)[1:n // 2] / n
    index = k_exc - 1                                 # k starts at 1

    impedance = V[index] / I[index]
    power_v, power_i = np.abs(V) ** 2, np.abs(I) ** 2
    relative = np.abs(impedance) ** 2

    def level(mask):
        """Level at one class of bins, referred to the impedance."""
        if not mask.any():
            return None
        lv, _ = _local_power(power_v, mask, k, k_exc, n_local)
        li, _ = _local_power(power_i, mask, k, k_exc, n_local)
        return np.sqrt(relative * (lv / power_v[index] + li / power_i[index]))

    return BurstBLA(
        impedance=impedance,
        std_noise=level(classes["noise"]),
        std_odd_lines=level(classes["odd_nl"]),
        std_even_lines=level(classes["even_nl"]),
        r_ntv=ratio_noise_to_time_variation(V, classes["noise"], k),
        n_bins={name: int(classes[name].sum())
                for name in ("excited", "odd_nl", "even_nl", "noise")},
        n_local=n_local,
    )


def bla_eis(voltage, current, period, sampling_time, harmonics,
            periods_per_window=4, n_local=30):
    """Sliding bursts over a whole record: STFT, with the error attributed.

    Parameters
    ----------
    voltage, current : (N,) real
        The record.
    period : float
        One multisine period, in **seconds**.
    sampling_time : float
        1/fs.
    harmonics : sequence of int
        Excited harmonic numbers, so that tone ``m`` is at
        ``harmonics[m] / period`` Hz.
    periods_per_window : int
        Multisine periods per burst, at least 2. This is the time resolution
        and it cannot be 1 -- see :func:`classify_bins`.
    n_local : int
        Bins averaged for each level; passed to :func:`burst_bla`.

    Returns
    -------
    (results, t) : list of BurstBLA, (n_bursts,) float
        One entry per burst, stamped at the burst midpoint.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    samples = int(round(period / sampling_time)) * int(periods_per_window)
    if samples < 4:
        raise ValueError("burst shorter than four samples")

    n_bursts = voltage.size // samples
    if n_bursts == 0:
        raise ValueError(
            f"the record holds {voltage.size} samples, short of the {samples} "
            f"a burst of {periods_per_window} periods needs")

    results, centres = [], []
    for b in range(n_bursts):
        sl = slice(b * samples, (b + 1) * samples)
        results.append(burst_bla(voltage[sl], current[sl], harmonics,
                                 periods_per_window, n_local))
        centres.append((b + 0.5) * samples * sampling_time)
    return results, np.asarray(centres)
