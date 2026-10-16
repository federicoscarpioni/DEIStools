"""
Standard additive drifts, for testing trend removal.

A trend here is a slow voltage excursion **added to the measured response**:
the disturbance that :mod:`deistools.processing.detrending` exists to remove,
and that the lowest excited tones have to be recovered from underneath. It is
not part of the system. That distinction matters because the package also has
a system-level voltage curve --
:func:`~deistools.simulation.models.randles_diffusion.default_ocv` -- which
drives the diffusion resistance and is therefore physics, not a disturbance.

Three shapes are provided, and they are deliberately different in kind.
Residue left by a least-squares Legendre fit of order ``q``, relative to the
trend's own standard deviation, on a 160 s record:

=========================  ========  ========  ========  ========
trend                      q = 1     q = 3     q = 8     q = 15
=========================  ========  ========  ========  ========
:func:`polynomial_trend`   3.2e-01   9.8e-16   6.9e-16   6.9e-16
:func:`relaxation_trend`   2.1e-01   6.9e-04   8.7e-11   1.1e-15
:func:`logit_trend`        1.3e-01   3.1e-02   2.6e-03   3.3e-05
=========================  ========  ========  ========  ========

The cubic is gone exactly at its own order, as it must be. The other two are
not, and they resist in different ways.

.. warning::

   **The paper's trend is only hard if the record is long enough**, and that
   is easy to get wrong. :func:`relaxation_trend` rings with a period of about
   1000 s at its published coefficients, so what a record sees depends on how
   much of that fits inside it. Residue after a Legendre fit of order 8:

   ========  ==============  ========
   record    cycles seen     q = 8
   ========  ==============  ========
   160 s     0.16            8.7e-11
   400 s     0.40            2.8e-08
   1000 s    1.00            1.9e-04
   2000 s    2.00            3.2e-02
   5000 s    5.00            5.1e-01
   ========  ==============  ========

   Nine orders of magnitude, from the same trend. On a short record the
   published "cannot be captured by a finite number of polynomials" shape is
   *easy*, because only its smooth leading edge is in view. Scale it with
   ``time_scale`` rather than assuming the defaults are hard.

:func:`logit_trend` has no such dependence: its shape in normalised time is
fixed, so the residue above is the same on a 160 s record and on a 2000 s one.
That makes it the only one of the three whose difficulty cannot be changed by
accident, which is a useful property for a suite meant to be compared across
studies.

Conventions
-----------
Every function takes the time axis first, returns volts, and takes all of its
shape parameters by keyword so that defaults can be overridden one at a time.

* ``duration`` is the record length used to normalise time. Left as ``None``
  it is inferred as ``N * dt`` from the axis -- not ``t[-1]``, which is one
  sample short, and that one sample is the difference between reproducing a
  published figure and not.
* ``remove_mean`` defaults to ``True``. The estimators all exclude the dc bin,
  so only the *variation* is the disturbance, and the basis functions of
  Hallemans TIM 2021 (after eq. 26) have their sampled mean subtracted for the
  same reason.
* ``peak_to_peak`` rescales the result to a given excursion. Left as ``None``
  each trend keeps its own natural size, which for the two taken from papers
  is the published one. Set it to compare shapes at equal severity, which is
  what :data:`TRENDS` is for.

On provenance
-------------
:func:`relaxation_trend` is from Hallemans *et al.*, *IEEE TIM* **71** (2022)
6501711, eq. (31)-(32), with that paper's coefficients as defaults.

:func:`polynomial_trend` has no single published parameter set. The *form* is
the paper's -- BLTVA models the trend as a Legendre expansion, eq. (58) -- but
the coefficients below are this package's, and the cubic that
``examples/09_detrending.py`` used before this module existed was chosen by
hand to be hard, not taken from anywhere. It is reproducible here
(see :func:`polynomial_trend`) and labelled for what it is.

:func:`logit_trend` is built from scratch, with the Nernst relation as its
justification rather than a citation.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "polynomial_trend", "relaxation_trend", "logit_trend",
    "TRENDS", "trend_by_name",
]


#: Monomial coefficients of the cubic that ``examples/09_detrending.py``
#: carried inline before this module existed, lowest order first, to be used
#: with ``basis="monomial"`` and ``scale=48.0``. Chosen by hand to put a large
#: excursion on the record; **not** from any paper.
HARD_CUBIC = (0.0, 0.7, -1.1, 0.5)


def _duration(t, duration):
    """Record length ``N*dt``, inferred from the axis when not given.

    ``t[-1]`` is ``(N-1)*dt`` and so one sample short of the record. The
    difference is O(1/N) and irrelevant to the shape, but it shifts the
    normalised axis just enough to break an exact comparison against a figure
    computed as ``t/(N*dt)``, which is how the rest of this package
    normalises.
    """
    t = np.asarray(t, dtype=float)
    if duration is not None:
        return float(duration)
    if t.size < 2:
        raise ValueError("cannot infer duration from fewer than two samples")
    return float(t[-1] + (t[1] - t[0]))


def _finish(values, remove_mean, peak_to_peak):
    values = np.asarray(values, dtype=float)
    if remove_mean:
        values = values - values.mean()
    if peak_to_peak is not None:
        span = np.ptp(values)
        if span == 0.0:
            raise ValueError(
                "the trend is flat, so it cannot be rescaled to a given "
                "peak-to-peak; check the coefficients")
        values = values * (float(peak_to_peak) / span)
    return values


def polynomial_trend(t, *, coefficients=(0.0, 0.5, 0.2, 0.1), basis="legendre",
                     scale=1.0, duration=None, remove_mean=True,
                     peak_to_peak=None):
    """A trend that a polynomial model can remove exactly.

    This is the easy case, and it is in the suite because it is the one every
    strategy should pass. BLTVA models the trend as a Legendre expansion
    (Hallemans TIM 2021, the ``Nq`` block of eq. 58), so a Legendre trend of
    order ``Nq`` or lower is inside its model and should come out without
    residue.

    Parameters
    ----------
    t : (N,) array
        Time axis in seconds.
    coefficients : sequence of float
        Coefficients, **lowest order first**. The default is this package's
        standard test drift; it is not a published parameter set, and the
        docstring of the module says why there is none. Pass a shorter
        sequence for a lower order -- ``(0.0, 0.5, 0.2)`` is quadratic,
        ``(0.0, 0.5)`` linear.
    basis : {"legendre", "monomial"}
        ``"legendre"`` uses the estimator's own basis on the normalised axis
        ``xi = 2t/T - 1``, which is what makes "order Nq or lower" a
        meaningful statement. It agrees with
        :func:`~deistools.simulation.models.legendre.legendre_drift` to 7e-16
        once the mean is removed from both; the 4e-06 difference before that
        is the mean subtraction which that function applies and this one
        leaves to ``remove_mean``. ``"monomial"`` uses powers of ``x = t/T``,
        which is how a trend is usually written by hand.
    scale : float
        Multiplies the whole polynomial, so the coefficients can describe a
        shape and the scale its size.
    duration, remove_mean, peak_to_peak
        See the module docstring.

    Examples
    --------
    The cubic that example 09 used before this module existed::

        polynomial_trend(t, coefficients=HARD_CUBIC, basis="monomial",
                         scale=48.0)
    """
    t = np.asarray(t, dtype=float)
    T = _duration(t, duration)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("coefficients must be a non-empty 1-D sequence")

    if basis == "legendre":
        from numpy.polynomial.legendre import legvander
        xi = 2.0 * t / T - 1.0
        values = legvander(xi, coefficients.size - 1) @ coefficients
    elif basis == "monomial":
        x = t / T
        values = np.polynomial.polynomial.polyval(x, coefficients)
    else:
        raise ValueError(
            f"basis must be 'legendre' or 'monomial', got {basis!r}")

    return _finish(scale * values, remove_mean, peak_to_peak)


def relaxation_trend(t, *, gain=5.92e-4, damping=1.25e-3, stiffness=9.87e-6,
                     time_scale=2.0, duration=None, remove_mean=True,
                     peak_to_peak=None):
    r"""The trend of Hallemans 2022 eq. (31)-(32): a second-order step response.

    .. math::
        C(s) = \frac{K}{s^2 + c_1 s + c_0}

    driven by a step and evaluated at ``time_scale * t``. The paper picks this
    shape deliberately, because it "cannot be exactly captured by a finite
    number of polynomials" -- which is what makes parametric trend removal
    struggle with it and what motivates differencing. It is the hard case of
    this suite, and the defaults are the paper's own coefficients.

    Parameters
    ----------
    gain, damping, stiffness : float
        $K$, $c_1$ and $c_0$. The defaults are eq. (32).
    time_scale : float
        The paper evaluates the response at ``2t``; this is that factor. It
        is the knob that matters: at the published coefficients the response
        rings with a period near ``1000 / time_scale`` seconds, and the trend
        is hard only when a cycle or more of that fits inside the record. See
        the warning in the module docstring -- a Legendre fit of order 8
        leaves 8.7e-11 of it on a 160 s record and 3.2e-02 on a 2000 s one.
    duration, remove_mean, peak_to_peak
        See the module docstring. ``duration`` is unused here -- the shape is
        set by absolute time, not by the record length -- and is accepted only
        so that every trend in :data:`TRENDS` has the same signature.

    Notes
    -----
    The published coefficients are underdamped ($\zeta = 0.20$), and the
    closed form usually quoted for this response divides by
    $\sqrt{1-\zeta^2}$. Since the point of this module is that the parameters
    can be changed, all three regimes are implemented: the critically damped
    and overdamped branches are taken where $\zeta \ge 1$, rather than
    returning NaN as the quoted form would.
    """
    t = np.asarray(t, dtype=float)
    if stiffness <= 0.0:
        raise ValueError(f"stiffness (c0) must be positive, got {stiffness}")
    if damping < 0.0:
        raise ValueError(f"damping (c1) must be non-negative, got {damping}")

    natural = np.sqrt(stiffness)                       # omega_n
    zeta = damping / (2.0 * natural)
    tau = time_scale * t
    decay = np.exp(-zeta * natural * tau)

    if zeta < 1.0:
        ringing = natural * np.sqrt(1.0 - zeta ** 2)
        shaped = decay * (np.cos(ringing * tau)
                          + zeta / np.sqrt(1.0 - zeta ** 2)
                          * np.sin(ringing * tau))
    elif zeta == 1.0:
        shaped = decay * (1.0 + natural * tau)
    else:
        creeping = natural * np.sqrt(zeta ** 2 - 1.0)
        shaped = decay * (np.cosh(creeping * tau)
                          + zeta / np.sqrt(zeta ** 2 - 1.0)
                          * np.sinh(creeping * tau))

    values = (gain / stiffness) * (1.0 - shaped)
    return _finish(values, remove_mean, peak_to_peak)


def logit_trend(t, *, thermal_voltage=0.025693, soc_start=0.05, soc_end=0.95,
                duration=None, remove_mean=True, peak_to_peak=None):
    r"""A battery-like drift: the Nernst term of an ideal insertion electrode.

    .. math::
        U(t) = \frac{RT}{F}\,\ln\!\frac{s(t)}{1 - s(t)},
        \qquad s(t) = s_0 + (s_1 - s_0)\,\frac{t}{T}

    State of charge moves linearly through the record, as it does under
    constant current, and the open-circuit voltage follows the ideal-solution
    (Nernst) relation. The result is the shape every cell shows: nearly flat
    across the middle of its range and steepening sharply towards both ends.

    This is **not** from a paper and the module docstring says so. What it has
    instead is a reason to be the shape it is -- the Nernst term is the
    configurational entropy of a lattice gas, so it is the part of a real OCV
    curve that is there regardless of material. What it leaves out is
    everything material-specific: staging plateaus, phase transitions, and
    the enthalpic terms. For a system-level curve with those features see
    :func:`~deistools.simulation.models.randles_diffusion.default_ocv`, which
    is a different object -- part of the system rather than a disturbance on
    it.

    Why it belongs in this suite: it has **no finite derivative at the ends
    of its range**, so a polynomial fit cannot follow it near the limits
    however high its order -- the residue falls only from 3.1e-02 at order 3
    to 2.0e-07 at order 25, where the cubic is exact at order 3 and the
    relaxation reaches round-off by order 15.

    More useful than being hard, it is hard *reproducibly*: the shape is fixed
    in normalised time, so the residue is identical on a 160 s record and a
    2000 s one. :func:`relaxation_trend` swings by nine orders of magnitude
    over that range, which means a study that changes its record length
    silently changes how hard its trend was.

    Parameters
    ----------
    thermal_voltage : float
        $RT/F$, 25.693 mV at 25 degrees C. Divide by the number of electrons
        for a multi-electron reaction, or raise it to exaggerate the shape.
    soc_start, soc_end : float
        The state-of-charge window swept, both strictly inside ``(0, 1)``.
        Pushing either towards its limit makes the trend far more severe: the
        default 0.05-0.95 gives about 150 mV, while 0.01-0.99 gives 236 mV
        with almost all of the extra excursion in the last few per cent of the
        record. Reverse them for a discharge.
    duration, remove_mean, peak_to_peak
        See the module docstring.
    """
    t = np.asarray(t, dtype=float)
    T = _duration(t, duration)
    for name, value in (("soc_start", soc_start), ("soc_end", soc_end)):
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"{name} must lie strictly inside (0, 1), got {value}. The "
                "Nernst term diverges at both limits, so a state of charge of "
                "exactly 0 or 1 has no voltage.")

    soc = soc_start + (soc_end - soc_start) * (t / T)
    soc = np.clip(soc, 1e-12, 1.0 - 1e-12)
    return _finish(thermal_voltage * np.log(soc / (1.0 - soc)),
                   remove_mean, peak_to_peak)


#: The standard drifts, by name. Every entry takes ``(t, **kwargs)`` and
#: accepts ``duration``, ``remove_mean`` and ``peak_to_peak``, so a study can
#: sweep all three at equal severity::
#:
#:     for name, make in TRENDS.items():
#:         trend = make(time, peak_to_peak=6.6)
TRENDS = {
    "polynomial": polynomial_trend,
    "relaxation": relaxation_trend,
    "logit": logit_trend,
}


def trend_by_name(name, t, **kwargs):
    """Look one up in :data:`TRENDS`, with a message that lists the options."""
    try:
        make = TRENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown trend {name!r}; choose from "
            f"{', '.join(sorted(TRENDS))}") from None
    return make(t, **kwargs)
