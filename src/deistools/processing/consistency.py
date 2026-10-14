r"""
Does the record agree with the numbers that describe it?

Every estimator here is handed a sampling time, a list of excited frequencies
and -- for the windowed ones -- a multisine period, and every one of them
assumes those numbers describe the record they were given. Four quantities are
tied together by one equation,

.. math:: N = P \, T_p \, f_s

the samples in the record, the multisine periods it holds, the period and the
sampling rate. Get one wrong and almost nothing refuses to run. The excited
tones stop landing on DFT bin centres, the leakage that a periodic record was
designed to avoid comes back, and every estimator in the package reports it as
impedance: STFT sees a window that is not a whole number of periods and calls
the leakage time variation, BLA's bin classes point at the wrong bins, BLTVA
fits its Legendre polynomials to a record whose period it has wrong.

The three error modes, and what separates them
----------------------------------------------
The arithmetic alone cannot say *which* number is wrong -- only that they
disagree. The signal can, because a wrong ``fs`` or ``Tp`` moves the spectral
lines while a wrong ``N`` only smears them:

    =========================  ===========================================
    what is wrong              what the spectrum does
    =========================  ===========================================
    ``N`` (record truncated    the lines stay where they are and the
    mid-period)                *nominal* bin positions slide off the grid
                               instead, in proportion to the harmonic
                               number. Measured and nominal still agree
                               with each other; each tone grows a leakage
                               skirt because its bin is no longer a bin.
    ``fs`` (card clock, or a   every line moves, by an amount
    typo)                      *proportional to its harmonic number*,
                               away from a nominal position that is off
                               the grid as well
    ``Tp`` (generator period   the same as a wrong ``fs``: of the two only
    is not the one designed)   the product ``T_p f_s`` is observable from
                               one record, so the report gives both
                               readings and you choose
    ``P`` (miscounted)         nothing -- the record is what it is. The
                               count is checked against the other three,
                               and against how many whole periods the
                               signal turns out to hold
    =========================  ===========================================

So the measurement to make is the *fractional* bin position of each tone,
:math:`k_i`, against its nominal position :math:`h_i N T_s / T_p`. Fitting

.. math:: k_i^{\rm meas} = (1 + e) \, k_i^{\rm nom}

gives one number, the relative timebase error :math:`e`, and the three
readings of it are

.. math:: f_s^{\rm true} = f_s / (1+e), \quad
          T_p^{\rm true} = T_p / (1+e), \quad
          P^{\rm true} = (1+e) P

A record whose only fault is a truncated tail gives :math:`e \ne 0` as well,
and is told apart by :math:`P^{\rm true}` landing on a whole number while
:math:`f_s^{\rm true}` comes out at an implausible value -- which is why the
report prints all three and leaves the choice to the person who knows what
the instrument was set to.

:math:`e` is measured twice, by two routes that fail differently:

* from the whole-record spectrum, by sub-bin interpolation of each peak
  (Jacobsen's estimator). Precise, and blind to nothing except a tone buried
  in noise.
* from the phase of each tone period by period. If the period is right the
  phase is constant across periods; if it is off by a fraction of a sample
  the phase advances linearly, at a rate proportional to the harmonic number.
  This one also responds to the cell genuinely drifting -- but drift is not
  proportional to the harmonic number, and that is the discriminant.

They agree on a timing error and part company on a drifting cell, so the pair
says more than either alone.

What else this looks at
-----------------------
The same pass over the spectrum answers the neighbouring questions cheaply:
whether every listed tone is actually present (a tone with no line above the
local noise floor was not excited, or is not at the frequency it is listed
at), whether anything *not* on the list is comparable to the tones (mains
pickup, a forgotten harmonic, strong distortion), whether the harmonic numbers
are integers at all -- :meth:`~deistools.processing.mfa.MultiFrequencyAnalysis.run_bla`,
``run_bltva`` and ``run_lia`` all round ``f / f.min()`` and will quietly use
the rounded value -- and whether the lowest listed tone really is the
fundamental, which those same three functions assume.

Nothing here raises on a bad record; reporting is the whole job. The one
exception is a sampling time that is not a positive number, which leaves
nothing to report on.

References
----------
Jacobsen and Kootsookos, *IEEE Signal Processing Magazine* **24** (2007) 123
-- the sub-bin frequency estimator used for the peak positions.

Pintelon and Schoukens, *System Identification: A Frequency Domain Approach*,
2nd ed., Wiley-IEEE Press (2012), ch. 2 -- leakage, and why periodic
excitation is measured over a whole number of periods.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

from .stft_functions import ON_BIN_TOLERANCE

__all__ = [
    "Finding", "ConsistencyReport", "check_consistency",
    "visualize_consistency",
]

#: Severity order. ``info`` is worth printing and not worth worrying about.
LEVELS = ("ok", "info", "warn", "fail")

#: A listed tone whose line is weaker than this above the local noise floor is
#: reported as possibly not excited at all.
SNR_WARN_DB = 10.0

#: A line that is not on the tone list is reported once it reaches this
#: fraction of the median excited-tone amplitude.
STRAY_FRACTION = 0.05

#: ``run_bla``, ``run_bltva`` and ``run_lia`` all take the harmonic numbers as
#: ``round(frequencies / frequencies.min())``. Residual above this and the
#: rounding is inventing a grid rather than tidying one.
HARMONIC_TOLERANCE = 0.01

#: How far below the lowest listed tone to look for the true fundamental.
#: ``run_bla`` and friends assume the lowest tone is harmonic 1.
MAX_SUBDIVISION = 16


@dataclass
class Finding:
    """One thing the check noticed, and what to do about it."""

    level: str      #: one of :data:`LEVELS`
    code: str       #: short stable name, for testing against
    message: str    #: the sentence a person reads

    def __str__(self):
        return f"[{self.level.upper():<4}] {self.code}: {self.message}"


@dataclass
class ConsistencyReport:
    """Everything the check measured, and what it concluded.

    The headline is :attr:`worst` and :attr:`findings`; ``print(report)``
    lays out the numbers in the order you need them to decide which of the
    four is wrong. The arrays are kept so that
    :func:`visualize_consistency` can draw them without measuring twice.
    """

    # --- the metadata, as handed in and as implied -------------------------
    n_samples: int
    sampling_time: float
    sampling_frequency: float
    duration: float
    period_claimed: float | None
    period: float                   #: the one the arithmetic below uses
    base_frequency: float
    samples_per_period: float
    periods_in_record: float
    n_periods_claimed: int | None
    frequencies: np.ndarray
    harmonics: np.ndarray           #: ``frequencies * period``, unrounded
    subdivision: int                #: 1 if the lowest tone is the fundamental

    # --- bin positions ----------------------------------------------------
    bin_nominal: np.ndarray         #: fractional, ``f * N * Ts``
    bin_measured: np.ndarray        #: fractional, from the spectrum
    bin_offset: np.ndarray          #: measured minus nominal
    measured_on: np.ndarray         #: 'V' or 'I', whichever tone was cleaner
    at_search_edge: np.ndarray      #: peak hit the window edge: unreliable

    # --- the timebase error, both ways ------------------------------------
    timebase_error: float           #: e, from the whole-record spectrum
    timebase_scatter: float         #: rms residual of that fit, in bins
    timebase_error_folded: float    #: e, from the period-by-period phase
    phase_drift: np.ndarray         #: (n_periods, n_tones) unwrapped, rad
    phase_slope: np.ndarray         #: rad per period, per tone

    # --- levels -----------------------------------------------------------
    amplitude_voltage: np.ndarray
    amplitude_current: np.ndarray
    snr_db_voltage: np.ndarray
    snr_db_current: np.ndarray
    stray_lines: list              #: (frequency, amplitude, fraction, label)

    # --- for the figure ---------------------------------------------------
    bin_frequencies: np.ndarray = field(repr=False, default=None)
    magnitude_voltage: np.ndarray = field(repr=False, default=None)
    magnitude_current: np.ndarray = field(repr=False, default=None)
    voltage: np.ndarray = field(repr=False, default=None)
    current: np.ndarray = field(repr=False, default=None)

    findings: list = field(default_factory=list)

    # ----------------------------------------------------------------------
    @property
    def worst(self) -> str:
        """The highest severity reached, ``'ok'`` if nothing was noticed."""
        if not self.findings:
            return "ok"
        return max(self.findings, key=lambda f: LEVELS.index(f.level)).level

    @property
    def ok(self) -> bool:
        """True when nothing worse than ``info`` came up."""
        return LEVELS.index(self.worst) <= LEVELS.index("info")

    def __getitem__(self, code):
        """The findings with a given code, as a list."""
        return [f for f in self.findings if f.code == code]

    def worst_offset(self):
        """Largest measured-minus-nominal offset, over the trustworthy tones.

        A tone whose peak was found at the edge of its search window is left
        out: that line may not be the one it was looking for, and one mistyped
        frequency should not be allowed to set the headline number.
        """
        trusted = np.isfinite(self.bin_offset) & ~self.at_search_edge
        if not np.any(trusted):
            return np.nan
        return float(np.max(np.abs(self.bin_offset[trusted])))

    def implied(self):
        """The three readings of the measured timebase error.

        Returns ``(fs, period, n_periods)``: what each would have to be for
        the tones to sit where the spectrum says they do, on the assumption
        that the other two are right.
        """
        scale = 1.0 + self.timebase_error
        return (self.sampling_frequency / scale,
                self.period / scale,
                self.periods_in_record * scale)

    def __str__(self):
        return self.summary()

    def summary(self):
        """The numbers, in the order you need them to place the blame."""
        spp, P = self.samples_per_period, self.periods_in_record
        d_spp = spp - round(spp) if np.isfinite(spp) else np.nan
        d_P = P - round(P) if np.isfinite(P) else np.nan
        if self.period_claimed is None:
            claimed = " (from the tone list)"
        else:
            claimed = " (as given)"
        if self.n_periods_claimed is not None:
            P_int = self.n_periods_claimed
        elif np.isfinite(P):
            P_int = int(round(P))
        else:
            P_int = 0

        lines = [f"Multi-frequency record check -- {self.worst.upper()}", ""]
        lines += [
            f"  samples             {self.n_samples}",
            f"  sampling rate       {self.sampling_frequency:.10g} Hz"
            f"          (Ts = {self.sampling_time:.6g} s)",
            f"  duration            {self.duration:.10g} s",
            f"  multisine period    {self.period:.10g} s{claimed}",
            f"  fundamental         {self.base_frequency:.10g} Hz",
            f"  samples per period  {spp:.6f}"
            f"     {_exactness(d_spp, 'sample')}",
            f"  periods in record   {P:.6f}"
            f"     {_exactness(d_P, 'period')}",
            f"  N = P x Tp x fs     {self.n_samples} = {P_int}"
            f" x {self.period:.10g} s x {self.sampling_frequency:.10g} Hz"
            f" = {P_int * self.period * self.sampling_frequency:.10g}",
            "",
            f"  tones               {self.frequencies.size} from"
            f" {self.frequencies.min():.6g} to {self.frequencies.max():.6g} Hz",
            f"  harmonic numbers    {self.harmonics.min():.6g} ..",
            ]
        lines[-1] += (f" {self.harmonics.max():.6g}, integer to"
                      f" {np.max(np.abs(self.harmonics - np.round(self.harmonics))):.2g}")
        residual = np.max(np.abs(self.bin_nominal - np.round(self.bin_nominal)))
        lines.append(f"  nominal bins        {np.min(self.bin_nominal):.6g} .."
                     f" {np.max(self.bin_nominal):.6g}, on grid to"
                     f" {residual:.2g} bins")

        lines += ["", "  measured from the signals"]
        if np.isfinite(self.timebase_error):
            fs_i, Tp_i, P_i = self.implied()
            lines += [
                f"    timebase error    {self.timebase_error:+.3e}"
                f"  (scatter {self.timebase_scatter:.2g} bins,"
                f" phase route {self.timebase_error_folded:+.3e})",
                f"    reads as          fs = {fs_i:.10g} Hz   |"
                f"   Tp = {Tp_i:.10g} s   |   P = {P_i:.6f}",
                f"    worst tone moved  {self.worst_offset():.3g} bins",
            ]
        else:
            lines.append("    timebase error    not measurable")
        for name, snr in (("V", self.snr_db_voltage), ("I", self.snr_db_current)):
            finite = snr[np.isfinite(snr)]
            if finite.size:
                lines.append(f"    tone SNR {name}        {finite.min():.1f} .."
                             f" {finite.max():.1f} dB, median"
                             f" {np.median(finite):.1f} dB")
            else:
                lines.append(f"    tone SNR {name}        noiseless")
        if self.stray_lines:
            lines.append(f"    stray lines       {len(self.stray_lines)}:")
            for f_hz, amp, frac, label in self.stray_lines:
                lines.append(f"                        {f_hz:>12.6g} Hz"
                             f"  {100 * frac:6.1f}% of the tones  {label}")
        else:
            lines.append("    stray lines       none above"
                         f" {100 * STRAY_FRACTION:g}% of the tones")

        lines.append("")
        if self.findings:
            # The verdict first, the evidence after: several of the findings
            # below offer alternatives that only the verdict chooses between.
            headline = [f for f in self.findings
                        if f.code in ("diagnosis", "off_grid_scatter")]
            lines += [f"  {f}" for f in headline]
            lines += [f"  {f}" for f in self.findings if f not in headline]
        else:
            lines.append("  nothing to report")
        return "\n".join(lines)


def _exactness(delta, unit):
    """``'exact'``, or how far from whole the quantity is."""
    if not np.isfinite(delta):
        return "not known"
    if abs(delta) < 1e-9:
        return "exact"
    return f"off by {delta:+.6g} {unit}"


# --------------------------------------------------------------------------- #
# Measuring where the tones actually are
# --------------------------------------------------------------------------- #

def _sub_bin_offset(spectrum, k):
    """Sub-bin position of a peak at bin ``k``, by Jacobsen's estimator.

    For a rectangular window the three bins around a peak fix its fractional
    position to about 1e-3 bins at usable signal-to-noise, which is two to
    three orders of magnitude finer than ``argmax`` alone and is what makes a
    clock error of a few parts per million visible in one record.

    The estimate is clipped to half a bin: beyond that the peak is not the
    one at ``k``, and reporting a larger offset would only move the error
    somewhere harder to see.
    """
    if k < 1 or k > spectrum.size - 2:
        return 0.0
    denominator = 2.0 * spectrum[k] - spectrum[k - 1] - spectrum[k + 1]
    if denominator == 0:
        return 0.0
    offset = np.real((spectrum[k - 1] - spectrum[k + 1]) / denominator)
    if not np.isfinite(offset):
        return 0.0
    return float(np.clip(offset, -0.5, 0.5))


def _search_halfwidth(k_round, n_bins, relative=0.02, floor=10):
    """How far either side of each nominal bin it is safe to look.

    Two bounds. A tone is allowed to have moved by ``relative`` of its own
    frequency -- a 2% clock error is already an absurd instrument, and
    anything larger is not a peak that drifted but a different peak. And the
    window must never reach half way to a neighbouring excited bin, or the
    search can lock onto the neighbour; this is the same bound
    :func:`~deistools.processing.stft_functions.search_freq_indexes` uses,
    for the same reason.
    """
    k_round = np.asarray(k_round, dtype=int)
    want = np.maximum(floor, np.ceil(relative * np.abs(k_round))).astype(int)
    if k_round.size > 1:
        gaps = np.diff(k_round)
        nearest = np.empty(k_round.size, dtype=int)
        nearest[0], nearest[-1] = gaps[0], gaps[-1]
        nearest[1:-1] = np.minimum(gaps[:-1], gaps[1:])
        want = np.minimum(want, np.maximum((nearest - 1) // 2, 1))
    # Never run off either end of the half spectrum.
    room = np.minimum(np.maximum(k_round - 1, 1), n_bins - 2 - k_round)
    return np.clip(np.minimum(want, room), 1, None)


def _measure_peaks(spectrum, centres, halfwidth):
    """Fractional bin of the strongest line near each centre.

    Returns the positions and a flag per tone: the peak landed on the edge of
    its search window, so the real one may be outside it and the position is
    not to be trusted.
    """
    magnitude = np.abs(spectrum)
    position = np.empty(centres.size)
    at_edge = np.zeros(centres.size, dtype=bool)
    for i, (centre, half) in enumerate(zip(centres, halfwidth)):
        low = max(int(centre) - int(half), 0)
        high = min(int(centre) + int(half), magnitude.size - 1)
        if high <= low:
            position[i] = centre
            continue
        peak = low + int(np.argmax(magnitude[low:high + 1]))
        at_edge[i] = (peak == low or peak == high) and high - low > 1
        position[i] = peak + _sub_bin_offset(spectrum, peak)
    return position, at_edge


def _fit_timebase(bin_nominal, bin_measured, weight):
    """Weighted fit of ``k_meas = (1 + e) k_nom`` through the origin.

    Through the origin on purpose. A constant shift of every tone by the same
    number of *bins* is not something a sampling rate, a period or a record
    length can do -- only a proportional move is -- so the model has one
    parameter and the scatter around it is then meaningful: small scatter
    means a timebase error, large scatter means the tones are simply not
    where the list says.
    """
    bin_nominal = np.asarray(bin_nominal, dtype=float)
    bin_measured = np.asarray(bin_measured, dtype=float)
    weight = np.asarray(weight, dtype=float)
    good = np.isfinite(bin_measured) & np.isfinite(weight) & (weight > 0)
    good &= bin_nominal != 0
    if np.count_nonzero(good) == 0:
        return np.nan, np.nan
    k, m, w = bin_nominal[good], bin_measured[good], weight[good]
    denominator = np.sum(w * k * k)
    if denominator == 0:
        return np.nan, np.nan
    error = float(np.sum(w * k * (m - k)) / denominator)
    residual = m - (1.0 + error) * k
    scatter = float(np.sqrt(np.sum(w * residual ** 2) / np.sum(w)))
    return error, scatter


def _noise_floor(magnitude, centres, excited, halfwidth=64, minimum=8):
    """Median magnitude of the unexcited bins around each tone."""
    free = ~excited
    out = np.empty(centres.size)
    fallback = np.median(magnitude[free]) if np.any(free) else 0.0
    for i, centre in enumerate(centres):
        low = max(int(centre) - halfwidth, 1)
        high = min(int(centre) + halfwidth + 1, magnitude.size)
        window = free[low:high]
        if np.count_nonzero(window) >= minimum:
            out[i] = np.median(magnitude[low:high][window])
        else:
            out[i] = fallback
    return out


def _snr_db(amplitude, floor):
    """Amplitude over noise floor in dB; ``inf`` where the floor is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(floor > 0, amplitude / np.where(floor > 0, floor, 1.0),
                         np.inf)
        return 20.0 * np.log10(ratio)


def _stray_lines(magnitude, bin_frequencies, excited, reference, base_frequency,
                 fraction=STRAY_FRACTION, limit=8):
    """Lines off the tone list that are comparable to the tones themselves.

    Threshold is a fraction of the median excited line rather than a multiple
    of the noise floor, because the noise floor is not the question: a line
    thirty times the floor and a thousand times below the tones changes no
    impedance, while one at a tenth of the tones does. Each is labelled with
    what it is most likely to be -- a harmonic of the multisine (so
    distortion, or a tone left off the list) or mains.
    """
    threshold = fraction * reference
    candidate = np.flatnonzero((magnitude > threshold) & (~excited))
    # Keep local maxima only, so one line is reported once and not with its
    # own skirt.
    candidate = candidate[(candidate > 0) & (candidate < magnitude.size - 1)]
    candidate = candidate[(magnitude[candidate] >= magnitude[candidate - 1])
                          & (magnitude[candidate] >= magnitude[candidate + 1])]
    order = candidate[np.argsort(magnitude[candidate])[::-1]]

    out, taken = [], []
    for k in order:
        if any(abs(int(k) - t) <= 3 for t in taken):
            continue
        taken.append(int(k))
        f_hz = float(bin_frequencies[k])
        amplitude = 2.0 * float(magnitude[k])
        label = "unidentified"
        if base_frequency > 0:
            harmonic = f_hz / base_frequency
            if abs(harmonic - round(harmonic)) < 0.05 and round(harmonic) >= 1:
                label = f"harmonic {int(round(harmonic))} of the fundamental"
        for mains in (50.0, 60.0):
            if abs(f_hz - mains) < 0.6 or abs(f_hz - 2 * mains) < 0.6:
                label = f"{mains:.0f} Hz mains or its second harmonic"
        out.append((f_hz, amplitude, float(magnitude[k]) / reference, label))
        if len(out) >= limit:
            break
    return sorted(out)


def _infer_fundamental(frequencies, max_subdivision=MAX_SUBDIVISION,
                       tolerance=HARMONIC_TOLERANCE):
    """The largest fundamental of which every listed tone is a harmonic.

    ``run_bla``, ``run_bltva`` and ``run_lia`` all take the harmonic numbers
    as ``round(frequencies / frequencies.min())``, which is right only when
    the lowest listed tone is harmonic 1. A grid starting at harmonic 3 --
    perfectly ordinary, the low tones cost the most time to measure -- makes
    every one of those three estimators wrong about every harmonic number,
    silently. So the fundamental is searched for rather than assumed: the
    answer is ``f_min / m``, and ``m > 1`` is worth saying out loud.

    Parsimony matters here, because a *single* tone that is simply mistyped
    can be accommodated by a small enough fundamental and would otherwise be
    reported as the whole grid being subdivided. A subdivision is accepted
    only when it brings in at least two tones that ``m = 1`` leaves out; one
    stray tone is left to the harmonic-number check, which is where it
    belongs.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    lowest = frequencies.min()
    if lowest <= 0:
        return lowest, 1

    def fitting(m):
        harmonics = frequencies * m / lowest
        return int(np.count_nonzero(
            np.abs(harmonics - np.round(harmonics)) <= tolerance))

    at_one = fitting(1)
    if at_one == frequencies.size:
        return lowest, 1
    for m in range(2, int(max_subdivision) + 1):
        if fitting(m) == frequencies.size and fitting(m) >= at_one + 2:
            return lowest / m, m
    return lowest, 1


def _folded_phase(signal, samples_per_period, n_periods, harmonics):
    """Phase of each tone, period by period, unwrapped along the periods.

    The record is cut into whole periods and each one transformed on its own.
    If the period is right every segment starts at the same point of the
    multisine and the phase is constant; if the period is wrong by a fraction
    of a sample the phase walks, at a rate proportional to the harmonic
    number. A cell that is genuinely drifting also moves these phases, but
    not in proportion to the harmonic number, which is how the two are told
    apart.
    """
    spp = int(round(samples_per_period))
    periods = int(n_periods)
    harmonics = np.round(harmonics).astype(int)
    usable = (harmonics >= 1) & (harmonics < spp // 2)
    if spp < 8 or periods < 3 or not np.any(usable):
        empty = np.zeros((0, harmonics.size))
        return empty, np.full(harmonics.size, np.nan)

    segments = np.asarray(signal[:periods * spp], dtype=float).reshape(periods, spp)
    spectra = np.fft.rfft(segments, axis=1)
    phase = np.full((periods, harmonics.size), np.nan)
    phase[:, usable] = np.unwrap(np.angle(spectra[:, harmonics[usable]]), axis=0)

    index = np.arange(periods, dtype=float)
    index -= index.mean()
    slope = np.full(harmonics.size, np.nan)
    centred = phase[:, usable] - np.nanmean(phase[:, usable], axis=0)
    slope[usable] = (index @ centred) / np.sum(index ** 2)
    return phase, slope


# --------------------------------------------------------------------------- #
# The check itself
# --------------------------------------------------------------------------- #

def check_consistency(voltage, current, frequencies, sampling_time, *,
                      period=None, n_periods=None,
                      snr_warn_db=SNR_WARN_DB,
                      stray_fraction=STRAY_FRACTION):
    """Check a record against the metadata that is supposed to describe it.

    Parameters
    ----------
    voltage, current : array_like
        The measured record, as handed to the estimators.
    frequencies : array_like
        The excited frequencies in Hz, as handed to the estimators.
    sampling_time : float
        Seconds between samples.
    period : float, optional
        The multisine period in seconds -- the ``period`` argument of
        :meth:`~deistools.processing.mfa.MultiFrequencyAnalysis.run_stft_eis`
        and the ``P`` of ``run_bltva``. Pass it: it is one of the numbers
        being checked, and without it the fundamental has to be inferred from
        the tone list, which can only ever confirm the list is self-consistent.
    n_periods : int, optional
        How many multisine periods you believe the record holds. Checked
        against the other three.
    snr_warn_db : float
        Below this, a listed tone is reported as possibly not excited.
    stray_fraction : float
        A line off the list is reported at this fraction of the median tone.

    Returns
    -------
    :class:`ConsistencyReport`
        ``print(report)`` for the numbers, ``report.findings`` for the
        verdicts, ``report.ok`` for a single boolean, and
        :func:`visualize_consistency` for the figure.

    Notes
    -----
    Cheap enough to run before every estimation: two real FFTs of the record
    and one of each period.
    """
    findings = []
    add = lambda level, code, message: findings.append(Finding(level, code, message))

    sampling_time = float(sampling_time)
    if not np.isfinite(sampling_time) or sampling_time <= 0:
        raise ValueError(f"sampling_time must be a positive number, got "
                         f"{sampling_time!r}; there is nothing to check "
                         "a record against without it")

    voltage = np.asarray(voltage, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()
    frequencies = np.asarray(frequencies, dtype=float).ravel()

    # --- the arrays, before anything is measured from them -----------------
    if voltage.size != current.size:
        shortest = min(voltage.size, current.size)
        add("fail", "arrays",
            f"voltage has {voltage.size} samples and current {current.size}. "
            f"They are two channels of one measurement and must have the same "
            f"length; the first {shortest} of each are checked below, but one "
            f"of the two files is not the one you think it is.")
        voltage, current = voltage[:shortest], current[:shortest]

    cleaned = []
    for name, signal in (("voltage", voltage), ("current", current)):
        finite = np.isfinite(signal)
        bad = int(signal.size - np.count_nonzero(finite))
        if bad:
            add("fail", "not_finite",
                f"{name} has {bad} non-finite samples ({100 * bad / signal.size:.2g}% "
                f"of the record). One NaN makes the whole DFT NaN, so every "
                f"estimator returns NaN at every frequency. For the checks "
                f"below they are replaced by the channel mean, in a copy -- "
                f"the array you passed is not touched.")
            signal = signal.copy()
            signal[~finite] = np.mean(signal[finite]) if np.any(finite) else 0.0
        cleaned.append(signal)
    voltage, current = cleaned

    n_samples = int(voltage.size)
    sampling_frequency = 1.0 / sampling_time
    duration = n_samples * sampling_time
    if n_samples < 16:
        add("fail", "too_short",
            f"{n_samples} samples is too short to check anything.")

    if frequencies.size == 0 or np.any(frequencies <= 0):
        add("fail", "frequencies",
            "the frequency list must hold positive frequencies in Hz; "
            f"got {frequencies}")
        frequencies = frequencies[frequencies > 0] if frequencies.size else frequencies
    if frequencies.size and np.any(np.diff(frequencies) <= 0):
        add("warn", "frequency_order",
            "the frequency list is not strictly increasing. Every estimator "
            "returns its rows in the order of this list, so the impedance "
            "rows will be in that same scrambled order.")

    nyquist = 0.5 * sampling_frequency
    if frequencies.size and frequencies.max() >= nyquist:
        above = frequencies[frequencies >= nyquist]
        add("fail", "nyquist",
            f"{above.size} tone(s) at or above the Nyquist frequency "
            f"{nyquist:.6g} Hz (worst {above.max():.6g} Hz). These are "
            f"aliased: they appear at {sampling_frequency - above.max():.6g} Hz "
            f"and the impedance reported for them is another frequency's.")
    elif frequencies.size and frequencies.max() > 0.4 * sampling_frequency:
        add("info", "nyquist",
            f"the top tone {frequencies.max():.6g} Hz is at "
            f"{100 * frequencies.max() / nyquist:.0f}% of Nyquist, where an "
            f"anti-alias filter has amplitude and phase of its own. It divides "
            f"out of V/I only if both channels went through the same one.")

    # --- the fundamental: is the lowest tone harmonic 1? -------------------
    if frequencies.size:
        base_inferred, subdivision = _infer_fundamental(frequencies)
    else:
        base_inferred, subdivision = np.nan, 1
    if subdivision > 1:
        add("warn", "fundamental",
            f"the listed tones are harmonics of {base_inferred:.6g} Hz, not of "
            f"the lowest listed tone {frequencies.min():.6g} Hz: the lowest one "
            f"is harmonic {subdivision}, not 1. run_bla, run_bltva and run_lia "
            f"all take the harmonic numbers as round(f / f.min()) and will be "
            f"wrong about every one of them. Pass period={1 / base_inferred:.10g} "
            f"and treat their harmonic grids as suspect.")

    # --- the four numbers --------------------------------------------------
    period_claimed = None if period is None else float(period)
    if period_claimed is not None:
        period_used = period_claimed
        if frequencies.size and np.isfinite(base_inferred):
            implied = 1.0 / base_inferred
            ratio = period_used / implied
            if abs(ratio - 1) > 1e-6 and abs(ratio - round(ratio)) > 1e-6:
                add("warn", "period_vs_tones",
                    f"period={period_used:.10g} s, but the tone list is a "
                    f"harmonic grid of {implied:.10g} s. The two disagree by a "
                    f"factor {ratio:.6g}, which is not a whole number of "
                    f"periods either way, so at least one of them is wrong.")
    elif frequencies.size and np.isfinite(base_inferred):
        period_used = 1.0 / base_inferred
        add("info", "period",
            f"no period given, so the tone list was used: it is a harmonic "
            f"grid of {period_used:.10g} s. That makes the period check "
            f"self-referential -- pass the period you set on the generator to "
            f"get a real one.")
    else:
        period_used = np.nan

    base_frequency = 1.0 / period_used if np.isfinite(period_used) else np.nan
    harmonics = frequencies * period_used
    samples_per_period = period_used / sampling_time
    periods_in_record = duration / period_used

    # --- the triangle N = P Tp fs ------------------------------------------
    bin_nominal = frequencies * n_samples * sampling_time
    grid_residual = (float(np.max(np.abs(bin_nominal - np.round(bin_nominal))))
                     if frequencies.size else np.nan)

    if np.isfinite(samples_per_period):
        off = samples_per_period - round(samples_per_period)
        if abs(off) > 1e-9:
            spp_int = int(round(samples_per_period))
            add("fail", "samples_per_period",
                f"one multisine period is {samples_per_period:.6f} samples, not "
                f"a whole number. Every windowed estimator turns the period "
                f"into int(round(period / sampling_time)) = {spp_int} samples, "
                f"so each window is {off:+.4g} samples wrong and the error "
                f"accumulates down the record. For a whole number, either "
                f"fs = {round(samples_per_period) / period_used:.10g} Hz at this "
                f"period, or period = {spp_int * sampling_time:.10g} s at this fs.")

    if np.isfinite(periods_in_record):
        whole = int(round(periods_in_record))
        off = periods_in_record - whole
        if abs(off) > 1e-9 and whole >= 1:
            spp_int = max(int(round(samples_per_period)), 1)
            complete = int(np.floor(periods_in_record + 1e-9))
            past = n_samples - complete * spp_int
            short = (complete + 1) * spp_int - n_samples
            add("fail", "whole_periods",
                f"the record holds {periods_in_record:.6f} multisine periods, "
                f"not a whole number: it runs {past} samples past {complete} "
                f"whole periods, and stops {short} samples short of "
                f"{complete + 1}. The DFT "
                f"of a record cut mid-period leaks, and the leakage is exactly "
                f"what STFT, DMFA and BLTVA report as time variation; "
                f"bltva() refuses the record outright (N % P). Three ways out, "
                f"and only you know which number is the wrong one: keep the "
                f"first {complete * spp_int} samples of {n_samples}, which is "
                f"{complete} whole periods (cut_signal_tail); or fs is really "
                f"{n_samples / (whole * period_used):.10g} Hz; or the period is "
                f"really {n_samples / (whole * sampling_frequency):.10g} s. The "
                f"measured timebase error below says which.")

    if n_periods is not None and np.isfinite(periods_in_record):
        if abs(periods_in_record - n_periods) > 1e-6:
            add("fail", "n_periods",
                f"you say the record holds {n_periods} periods; N, period and fs "
                f"say {periods_in_record:.6f}. BLTVA reads the excited tones at "
                f"bins P*h and treats every bin with k % P == 0 as a multisine "
                f"harmonic, so a wrong P does not degrade the estimate, it reads "
                f"a different set of bins. run_bla's periods_per_window is "
                f"affected the same way.")

    if frequencies.size and np.isfinite(grid_residual):
        if grid_residual > ON_BIN_TOLERANCE:
            worst = int(np.argmax(np.abs(bin_nominal - np.round(bin_nominal))))
            add("fail", "on_grid",
                f"{frequencies[worst]:.6g} Hz sits {grid_residual:.4g} bins from "
                f"a bin centre (bin spacing {1 / duration:.6g} Hz). "
                f"compute_freq_indexes raises on anything past "
                f"{ON_BIN_TOLERANCE} bins, so run_fft_eis and run_dmfa will not "
                f"start; what a tone off a bin centre costs is leakage into its "
                f"neighbours, which is the same error the windowed methods make "
                f"silently.")
        elif grid_residual > 1e-6:
            add("info", "on_grid",
                f"the tones are within {grid_residual:.2g} bins of a centre -- "
                f"close enough for compute_freq_indexes, far enough to leak a "
                f"little. Exact is reachable: it needs f * N * sampling_time to "
                f"be whole for every tone.")

    if frequencies.size > 1:
        rounded = np.round(bin_nominal).astype(int)
        gaps = np.diff(rounded)
        if np.any(gaps == 0):
            share = frequencies[:-1][gaps == 0]
            add("fail", "bin_collision",
                f"{share.size} pair(s) of tones land on the same DFT bin "
                f"(first at {share[0]:.6g} Hz). The impedance reported for both "
                f"is the same number; the record is too short to separate them, "
                f"and needs {1 / np.min(np.diff(frequencies)):.6g} s to do so.")
        elif np.any(gaps == 1):
            add("warn", "bin_collision",
                f"{int(np.count_nonzero(gaps == 1))} pair(s) of tones land on "
                f"adjacent DFT bins. Nothing can be read about either one's "
                f"neighbourhood -- no leakage check, no local noise floor, and "
                f"search_freq_indexes has no room to search.")

    harmonic_residual = (float(np.max(np.abs(harmonics - np.round(harmonics))))
                         if frequencies.size and np.isfinite(period_used) else np.nan)
    if np.isfinite(harmonic_residual) and harmonic_residual > HARMONIC_TOLERANCE:
        worst = int(np.argmax(np.abs(harmonics - np.round(harmonics))))
        add("warn", "harmonics",
            f"harmonic number {harmonics[worst]:.6g} at {frequencies[worst]:.6g} Hz "
            f"is not an integer ({harmonic_residual:.3g} off). run_bla, "
            f"run_bltva and run_lia round it to {round(harmonics[worst])}, and "
            f"then work at a frequency you did not excite.")

    # --- what the signals themselves say -----------------------------------
    spectrum_voltage = np.fft.rfft(voltage) / n_samples
    spectrum_current = np.fft.rfft(current) / n_samples
    magnitude_voltage = np.abs(spectrum_voltage)
    magnitude_current = np.abs(spectrum_current)
    bin_frequencies = np.fft.rfftfreq(n_samples, sampling_time)
    n_bins = bin_frequencies.size

    tiny = np.finfo(float).tiny
    floor_voltage_all = max(float(np.median(magnitude_voltage[1:])), tiny)
    floor_current_all = max(float(np.median(magnitude_current[1:])), tiny)

    if frequencies.size and n_bins > 4:
        centres = np.clip(np.round(bin_nominal).astype(int), 1, n_bins - 2)
        # Which channel shows each tone more clearly. Volts and amps cannot be
        # compared, so the comparison is each channel's line against its own
        # broadband level, which is a ratio and has no units.
        clarity_voltage = magnitude_voltage[centres] / floor_voltage_all
        clarity_current = magnitude_current[centres] / floor_current_all
        use_voltage = clarity_voltage > clarity_current

        halfwidth = _search_halfwidth(centres, n_bins)
        position_voltage, edge_voltage = _measure_peaks(
            spectrum_voltage, centres, halfwidth)
        position_current, edge_current = _measure_peaks(
            spectrum_current, centres, halfwidth)
        bin_measured = np.where(use_voltage, position_voltage, position_current)
        at_search_edge = np.where(use_voltage, edge_voltage, edge_current)
        measured_on = np.where(use_voltage, "V", "I")
        weight = np.clip(np.where(use_voltage, clarity_voltage, clarity_current),
                         0.0, 1e4)

        timebase_error, timebase_scatter = _fit_timebase(
            bin_nominal, bin_measured, np.where(at_search_edge, 0.0, weight))

        # A large error means the first pass was searching off-centre. Recentre
        # on what it found and measure again, narrowly: a wide window around
        # the right place is how a neighbouring line gets picked up instead.
        if (np.isfinite(timebase_error)
                and abs(timebase_error) * np.max(bin_nominal) > 1.0):
            moved_centres = np.clip(
                np.round(bin_nominal * (1.0 + timebase_error)).astype(int),
                1, n_bins - 2)
            narrow = np.clip(_search_halfwidth(moved_centres, n_bins), 1, 4)
            position_voltage, edge_voltage = _measure_peaks(
                spectrum_voltage, moved_centres, narrow)
            position_current, edge_current = _measure_peaks(
                spectrum_current, moved_centres, narrow)
            bin_measured = np.where(use_voltage, position_voltage, position_current)
            at_search_edge = np.where(use_voltage, edge_voltage, edge_current)
            timebase_error, timebase_scatter = _fit_timebase(
                bin_nominal, bin_measured, np.where(at_search_edge, 0.0, weight))

        peak_voltage = np.clip(np.round(position_voltage).astype(int), 0, n_bins - 1)
        peak_current = np.clip(np.round(position_current).astype(int), 0, n_bins - 1)
        excited = np.zeros(n_bins, dtype=bool)
        excited[:3] = True
        for k in np.round(bin_measured).astype(int):
            excited[max(int(k) - 3, 0):int(k) + 4] = True

        floor_voltage = _noise_floor(magnitude_voltage, peak_voltage, excited)
        floor_current = _noise_floor(magnitude_current, peak_current, excited)
        amplitude_voltage = 2.0 * magnitude_voltage[peak_voltage]
        amplitude_current = 2.0 * magnitude_current[peak_current]
        snr_db_voltage = _snr_db(magnitude_voltage[peak_voltage], floor_voltage)
        snr_db_current = _snr_db(magnitude_current[peak_current], floor_current)

        stray_lines = []
        for tag, magnitude, peaks in (("in the voltage", magnitude_voltage, peak_voltage),
                                      ("in the current", magnitude_current, peak_current)):
            reference = float(np.median(magnitude[peaks]))
            if reference <= 0:
                continue
            for f_hz, amp, frac, label in _stray_lines(
                    magnitude, bin_frequencies, excited, reference,
                    base_frequency, fraction=stray_fraction, limit=4):
                stray_lines.append((f_hz, amp, frac, f"{label}, {tag}"))
        stray_lines.sort()
    else:
        bin_measured = np.full(frequencies.shape, np.nan)
        at_search_edge = np.zeros(frequencies.shape, dtype=bool)
        measured_on = np.full(frequencies.shape, "-", dtype="<U1")
        timebase_error = timebase_scatter = np.nan
        amplitude_voltage = np.full(frequencies.shape, np.nan)
        amplitude_current = np.full(frequencies.shape, np.nan)
        snr_db_voltage = np.full(frequencies.shape, np.nan)
        snr_db_current = np.full(frequencies.shape, np.nan)
        stray_lines = []
        use_voltage = np.zeros(frequencies.shape, dtype=bool)

    bin_offset = bin_measured - bin_nominal

    # --- the same error, measured again period by period --------------------
    # On the current, because the current is the excitation (every estimator
    # here is called with the voltage as the response). Its phase moves with
    # the timebase and with nothing else, while the voltage's also moves when
    # the cell drifts -- which is why the two routes below can disagree.
    phase_drift = np.zeros((0, frequencies.size))
    phase_slope = np.full(frequencies.shape, np.nan)
    timebase_error_folded = np.nan
    if (frequencies.size and np.isfinite(samples_per_period)
            and samples_per_period >= 8 and periods_in_record >= 3):
        complete = int(np.floor(periods_in_record + 1e-9))
        phase_drift, phase_slope = _folded_phase(
            current, samples_per_period, complete, harmonics)
        whole_harmonics = np.round(harmonics)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_tone = phase_slope / (2.0 * np.pi * whole_harmonics)
        # A step of more than pi per period is aliased by the unwrap, so those
        # tones are dropped rather than believed.
        usable = (np.isfinite(per_tone) & (np.abs(phase_slope) < np.pi)
                  & (whole_harmonics >= 1))
        if np.any(usable):
            timebase_error_folded = float(np.median(per_tone[usable]))

    # --- the verdict --------------------------------------------------------
    whole_record = (np.isfinite(periods_in_record)
                    and abs(periods_in_record - round(periods_in_record)) < 1e-6)
    # Only the tones the fit was allowed to use. A line found at the edge of
    # its search window may be a different line altogether, and letting one
    # of those set the headline number is how a single mistyped frequency
    # comes to look like a broken clock.
    trusted = np.isfinite(bin_offset) & ~at_search_edge
    moved = float(np.max(np.abs(bin_offset[trusted]))) if np.any(trusted) else np.nan
    if np.isfinite(timebase_error):
        implied_fs = sampling_frequency / (1.0 + timebase_error)
        implied_period = period_used / (1.0 + timebase_error)
        implied_periods = periods_in_record * (1.0 + timebase_error)
        implied_whole = (np.isfinite(implied_periods)
                         and abs(implied_periods - round(implied_periods)) < 2e-3)
        significant = np.isfinite(moved) and moved > ON_BIN_TOLERANCE
        common_scale = timebase_scatter <= max(0.05, 0.3 * moved)
        readings = (f"fs = {implied_fs:.10g} Hz, or the period = "
                    f"{implied_period:.10g} s, or the record holding "
                    f"{implied_periods:.6f} periods")

        if significant and not common_scale:
            add("fail", "off_grid_scatter",
                f"the tones are off their bins by up to {moved:.3g} bins, and "
                f"not by a common factor: the scatter around the best "
                f"proportional fit is {timebase_scatter:.3g} bins against a fit "
                f"of {abs(timebase_error) * np.max(bin_nominal):.3g}. No "
                f"sampling rate, period or record length does that. The tone "
                f"list itself is wrong, or the generator is not producing a "
                f"harmonic multisine. search_freq_indexes will find the lines; "
                f"nothing will make them leakage-free.")
        elif significant and implied_whole and not whole_record:
            add("fail", "diagnosis",
                f"the record length is right and the sampling rate or the "
                f"period is wrong. The tones sit {timebase_error:+.4g} of their "
                f"frequency away from where the metadata puts them, and at that "
                f"correction the record holds {round(implied_periods)} whole "
                f"periods exactly. So: {readings}. Which of the two it is, one "
                f"record cannot say -- only the product Tp*fs is observable -- "
                f"so take it from whichever of the generator and the card you "
                f"trust less.")
        elif significant:
            add("fail", "diagnosis",
                f"the tones are {timebase_error:+.4g} of their frequency away "
                f"from where the metadata puts them (up to {moved:.3g} bins at "
                f"the top tone), which reads as {readings}. That corrected count "
                f"is not a whole number either, so the record length is wrong as "
                f"well as the rate or the period.")
        elif not whole_record:
            add("fail", "diagnosis",
                f"the sampling rate and the period are right and the record "
                f"length is wrong. Every tone is within {moved:.2g} bins of "
                f"where fs and the period put it (timebase error "
                f"{timebase_error:+.2g}, far too small to explain "
                f"{periods_in_record:.6f} periods), so nothing is wrong with the "
                f"timebase: the record simply does not end on a period boundary. "
                f"Trim it.")
        elif grid_residual > ON_BIN_TOLERANCE:
            add("warn", "diagnosis",
                f"N, P, Tp and fs agree with each other and with the signals -- "
                f"{periods_in_record:.0f} whole periods, timebase error "
                f"{timebase_error:+.2g} -- and it is the frequency list that "
                f"does not fit the grid those four define, by up to "
                f"{grid_residual:.3g} bins. The tones named above are the ones "
                f"to correct; the metadata is not the problem.")
        else:
            add("ok", "diagnosis",
                f"N, P, Tp and fs agree with each other and with the signals: "
                f"{periods_in_record:.0f} whole periods, every tone on a bin "
                f"centre to {grid_residual:.1g} bins, and the measured timebase "
                f"error is {timebase_error:+.2g} ({abs(timebase_error) * np.max(bin_nominal):.2g} "
                f"bins at the top tone).")

        if (np.isfinite(timebase_error_folded)
                and abs(timebase_error_folded - timebase_error)
                > max(5e-6, 3 * abs(timebase_error))):
            add("info", "phase_drift",
                f"the two measurements of the timebase error disagree: "
                f"{timebase_error:+.3g} from the whole-record spectrum, "
                f"{timebase_error_folded:+.3g} from the phase walk period to "
                f"period. They are sensitive to different things, which is why "
                f"both are here: the first compares the lines against the "
                f"frequency list and is blind to the period, the second cuts "
                f"the record into periods and so feels the period as well. "
                f"Three things separate them -- a period that is wrong while "
                f"the frequency list is right, a cell that is drifting (whose "
                f"phase also walks, but not in proportion to the harmonic "
                f"number), or an actual clock error. The last panel of "
                f"visualize_consistency tells them apart: a slope proportional "
                f"to h is timing.")

    # --- is every listed tone actually there? -------------------------------
    for name, snr in (("voltage", snr_db_voltage), ("current", snr_db_current)):
        weak = np.isfinite(snr) & (snr < snr_warn_db)
        if np.any(weak):
            worst = int(np.argmin(np.where(weak, snr, np.inf)))
            add("warn", "tone_missing",
                f"{int(np.count_nonzero(weak))} of {frequencies.size} tones are "
                f"less than {snr_warn_db:g} dB above the local noise floor in "
                f"the {name} (worst {frequencies[worst]:.6g} Hz at "
                f"{snr[worst]:.1f} dB). Either the tone was not excited, or it "
                f"is not at the frequency it is listed at; the impedance "
                f"reported there is a ratio of two noise samples, and it will "
                f"look like a wildly time-varying cell.")
    if np.any(at_search_edge):
        add("info", "search_edge",
            f"{int(np.count_nonzero(at_search_edge))} tone(s) peaked at the edge "
            f"of their search window, so the line found may not be the right "
            f"one and those positions are excluded from the timebase fit.")

    if stray_lines:
        biggest = max(stray_lines, key=lambda row: row[2])
        add("warn" if biggest[2] > 0.2 else "info", "stray_line",
            f"{len(stray_lines)} line(s) that are not on the tone list reach "
            f"{100 * biggest[2]:.0f}% of the median tone ({biggest[0]:.6g} Hz, "
            f"{biggest[3]}). A harmonic of the fundamental is distortion or a "
            f"tone left off the list; anything else is pickup, and it biases "
            f"the bins it lands near.")

    # --- the two faults that make all of the above meaningless --------------
    for name, signal in (("voltage", voltage), ("current", current)):
        span = float(np.ptp(signal))
        if span <= 0:
            add("fail", "constant",
                f"the {name} does not change over the whole record.")
            continue
        at_rail = (np.count_nonzero(np.abs(signal - signal.max()) <= 1e-9 * span)
                   + np.count_nonzero(np.abs(signal - signal.min()) <= 1e-9 * span))
        if at_rail > max(8, 0.001 * n_samples):
            add("warn", "clipping",
                f"{at_rail} samples of the {name} ({100 * at_rail / n_samples:.2g}%) "
                f"sit exactly at the extreme value: the channel is clipping, or "
                f"the range was set too small. Clipping is a nonlinearity, so "
                f"the harmonics it generates appear as distortion and, on an "
                f"odd-harmonic grid, partly on the excited bins themselves.")

    if np.any(np.isfinite(amplitude_voltage)) and n_samples > 2:
        index = np.arange(n_samples, dtype=float)
        slope = np.polyfit(index, voltage, 1)[0]
        trend_span = abs(slope) * (n_samples - 1)
        # The median tone, not the sum: a trend this size leaks into the
        # lowest bins and inflates the tones there, so a sum would partly
        # measure the drift it is being compared against.
        typical = float(np.nanmedian(amplitude_voltage))
        periods = periods_in_record if np.isfinite(periods_in_record) else 1.0
        per_period = trend_span / max(periods, 1.0)
        if typical > 0 and per_period > typical:
            add("info", "drift",
                f"the voltage trends {trend_span:.4g} V across the record, "
                f"{per_period:.4g} V within one multisine period, against a "
                f"typical tone amplitude of {typical:.4g} V -- a factor "
                f"{per_period / typical:.3g}. That is the cell doing something "
                f"besides answering the excitation, which is what this package "
                f"is for; it is reported because a one-period window sees that "
                f"trend as leakage into the lowest tones. See "
                f"deistools.processing.detrending, and prefer run_bltva, which "
                f"models the trend, over run_stft_eis, which does not.")

    return ConsistencyReport(
        n_samples=n_samples,
        sampling_time=sampling_time,
        sampling_frequency=sampling_frequency,
        duration=duration,
        period_claimed=period_claimed,
        period=period_used,
        base_frequency=base_frequency,
        samples_per_period=samples_per_period,
        periods_in_record=periods_in_record,
        n_periods_claimed=None if n_periods is None else int(n_periods),
        frequencies=frequencies,
        harmonics=harmonics,
        subdivision=subdivision,
        bin_nominal=bin_nominal,
        bin_measured=bin_measured,
        bin_offset=bin_offset,
        measured_on=measured_on,
        at_search_edge=at_search_edge,
        timebase_error=timebase_error,
        timebase_scatter=timebase_scatter,
        timebase_error_folded=timebase_error_folded,
        phase_drift=phase_drift,
        phase_slope=phase_slope,
        amplitude_voltage=amplitude_voltage,
        amplitude_current=amplitude_current,
        snr_db_voltage=snr_db_voltage,
        snr_db_current=snr_db_current,
        stray_lines=stray_lines,
        bin_frequencies=bin_frequencies,
        magnitude_voltage=magnitude_voltage,
        magnitude_current=magnitude_current,
        voltage=voltage,
        current=current,
        findings=findings,
    )


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #

def visualize_consistency(report, max_points=4000, n_shown=3):
    """Six panels of a :class:`ConsistencyReport`, in the order you read them.

    ===========================  =========================================
    panel                        what a fault looks like
    ===========================  =========================================
    the record, with the period  the shaded tail is the part of the record
    boundaries drawn             that is not a whole period
    the two spectra, the listed   a marker beside a line instead of on it,
    tones marked                 or a line with no marker at all
    bin offsets against          *measured minus nominal* rising along the
    harmonic number              dashed fit: the rate or the period is
                                 wrong. *Nominal off its bin centre*
                                 rising instead: the record length is.
                                 Scatter with no slope: the tone list is.
    signal to noise per tone     a tone near or below the dotted line is
                                 not reliably in the record at all
    the periods, overlaid        a fan instead of one curve: the period is
                                 not the one the record was measured with
    phase, period by period      sloping lines mean the period is wrong,
                                 and the slope is then proportional to the
                                 harmonic number. Sloping but not in
                                 proportion: the cell is drifting, which
                                 is a result and not a fault.
    ===========================  =========================================

    Returns ``(fig, axs)``.
    """
    fig, axs = plt.subplots(2, 3, figsize=(16.5, 9))
    fig.suptitle(f"record against its metadata -- {report.worst.upper()}"
                 f"    (N = {report.n_samples}, fs = "
                 f"{report.sampling_frequency:.10g} Hz, Tp = "
                 f"{report.period:.10g} s, P = {report.periods_in_record:.6f})")

    frequencies = report.frequencies
    if frequencies.size:
        harmonics = np.round(report.harmonics).astype(int)
    else:
        harmonics = np.array([], dtype=int)

    # --- the record --------------------------------------------------------
    ax = axs[0, 0]
    step = max(1, report.n_samples // max_points)
    time = np.arange(0, report.n_samples, step) * report.sampling_time
    ax.plot(time, report.voltage[::step], lw=0.7, color="C0", label="voltage")
    twin = ax.twinx()
    twin.plot(time, report.current[::step], lw=0.7, color="C1", alpha=0.55)
    if np.isfinite(report.periods_in_record):
        complete = int(np.floor(report.periods_in_record + 1e-9))
    else:
        complete = 0
    if 1 <= complete <= 60:
        for p in range(complete + 1):
            ax.axvline(p * report.period, color="0.6", lw=0.6, ls=":")
    if (np.isfinite(report.periods_in_record)
            and report.periods_in_record > complete + 1e-9):
        ax.axvspan(complete * report.period, report.duration, color="C3",
                   alpha=0.18, label="incomplete period")
    ax.set(xlabel="time / s", ylabel="voltage / V",
           title=f"the record: {report.periods_in_record:.4f} periods of "
                 f"{report.period:.6g} s")
    twin.set_ylabel("current / A")
    ax.legend(fontsize=7, loc="upper left")

    # --- the spectra -------------------------------------------------------
    ax = axs[0, 1]
    keep = report.bin_frequencies > 0
    ax.loglog(report.bin_frequencies[keep], report.magnitude_voltage[keep],
              lw=0.6, color="C0", alpha=0.8, label="voltage")
    ax.loglog(report.bin_frequencies[keep], report.magnitude_current[keep],
              lw=0.6, color="C1", alpha=0.8, label="current")
    if frequencies.size:
        ax.loglog(frequencies, report.amplitude_voltage / 2, "v", ms=6,
                  color="C0", mfc="none", label="listed tones")
        ax.loglog(frequencies, report.amplitude_current / 2, "v", ms=6,
                  color="C1", mfc="none")
    for row in report.stray_lines:
        ax.axvline(row[0], color="C3", lw=0.8, ls="--")
    ax.set(xlabel="frequency / Hz", ylabel="DFT magnitude",
           title="spectra, with the listed tones marked")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.2)

    # --- the two discriminants ---------------------------------------------
    ax = axs[0, 2]
    if frequencies.size:
        nominal_off_bin = report.bin_nominal - np.round(report.bin_nominal)
        ax.plot(harmonics, nominal_off_bin, "s", ms=5, color="C2", mfc="none",
                label="nominal bin off its centre (blames N)")
        ax.plot(harmonics, report.bin_offset, "o", ms=5, color="C3",
                label="measured minus nominal (blames fs or Tp)")
        if np.isfinite(report.timebase_error) and harmonics.size:
            bins_per_harmonic = report.bin_nominal[-1] / max(harmonics[-1], 1)
            grid = np.linspace(0, harmonics.max() * 1.05, 32)
            ax.plot(grid, report.timebase_error * grid * bins_per_harmonic,
                    "--", color="C3", lw=1,
                    label=f"fit e = {report.timebase_error:+.3g}")
        ax.axhline(0, color="0.5", lw=0.6)
        for sign in (-1, 1):
            ax.axhline(sign * ON_BIN_TOLERANCE, color="0.7", lw=0.6, ls=":")
    ax.set(xlabel="harmonic number", ylabel="bins",
           title="where the tones are, against where they should be")
    ax.legend(fontsize=7)

    # --- signal to noise ---------------------------------------------------
    ax = axs[1, 0]
    if frequencies.size:
        finite_voltage = np.isfinite(report.snr_db_voltage)
        finite_current = np.isfinite(report.snr_db_current)
        ax.semilogx(frequencies[finite_voltage],
                    report.snr_db_voltage[finite_voltage], "o-", ms=4,
                    color="C0", label="voltage")
        ax.semilogx(frequencies[finite_current],
                    report.snr_db_current[finite_current], "o-", ms=4,
                    color="C1", label="current")
        if not np.any(finite_voltage) and not np.any(finite_current):
            ax.text(0.5, 0.5, "noiseless record", ha="center",
                    transform=ax.transAxes, fontsize=9, color="0.4")
        ax.axhline(SNR_WARN_DB, color="C3", lw=0.8, ls=":",
                   label=f"{SNR_WARN_DB:g} dB")
    ax.set(xlabel="frequency / Hz", ylabel="tone over local floor / dB",
           title="is every listed tone really there?")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.2)

    # --- the periods, overlaid ---------------------------------------------
    ax = axs[1, 1]
    if np.isfinite(report.samples_per_period):
        spp = int(round(report.samples_per_period))
    else:
        spp = 0
    if spp >= 8 and complete >= 2 and complete * spp <= report.current.size:
        segments = report.current[:complete * spp].reshape(complete, spp)
        inner = max(1, spp // 2000)
        position = np.arange(0, spp, inner) / spp
        colours = plt.cm.viridis(np.linspace(0, 1, complete))
        for p in range(complete):
            ax.plot(position, segments[p, ::inner], lw=0.6, color=colours[p],
                    alpha=0.8)
        ax.set(xlabel="position within the period", ylabel="current / A",
               title=f"{complete} periods overlaid, dark to light in time")
    else:
        ax.text(0.5, 0.5, "too few whole periods to overlay", ha="center",
                transform=ax.transAxes, fontsize=9, color="0.4")
        ax.set(title="the periods, overlaid")

    # --- the phase, period by period ---------------------------------------
    ax = axs[1, 2]
    if report.phase_drift.size:
        usable = np.flatnonzero(np.isfinite(report.phase_slope))
        if usable.size:
            pick = np.unique(np.linspace(0, usable.size - 1,
                                         min(n_shown, usable.size)).astype(int))
            index = np.arange(report.phase_drift.shape[0])
            for i, colour in zip(usable[pick], ("C0", "C1", "C2", "C4", "C5")):
                curve = report.phase_drift[:, i] - report.phase_drift[0, i]
                implied = report.phase_slope[i] / (2 * np.pi * max(harmonics[i], 1))
                ax.plot(index, curve, "o-", ms=4, color=colour,
                        label=f"h = {harmonics[i]}  ({implied:+.2g})")
            ax.legend(fontsize=7, title="tone (implied e)", title_fontsize=7)
    else:
        ax.text(0.5, 0.5, "fewer than three whole periods", ha="center",
                transform=ax.transAxes, fontsize=9, color="0.4")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set(xlabel="period index", ylabel="phase, relative to the first / rad",
           title="phase walk: a slope proportional to h means timing")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    return fig, axs
