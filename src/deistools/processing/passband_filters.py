import numpy as np
from dataclasses import dataclass, field

def create_fermi_dirac_fun(fr, fc, bw, n):
    """
    Create a digital filter shaped as a symmetric Fermi-Dirac function.

        Y = cosh(n) / (cosh(n*X) + cosh(n)),    X = (fr - fc) / bw

    Evaluated as ``1 / (1 + cosh(n*X)/cosh(n))`` with the ratio of hyperbolic
    cosines expanded in exponentials. Written literally, ``cosh(n*X)``
    overflows for the far tails of a wide axis -- the answer is still right,
    since an infinite denominator correctly gives zero, but it warns on every
    call. The identity

        cosh(a)/cosh(b) = exp(|a| - b) * (1 + exp(-2|a|)) / (1 + exp(-2b))

    is therefore taken in log space and finished with a numerically stable
    logistic, so no exponential of a large positive number is ever formed.
    """
    a = np.abs(n * np.asarray(fr - fc, dtype=float) / bw)
    b = abs(n)
    # log of cosh(n*X)/cosh(n); both exponentials have non-positive arguments
    log_ratio = (a - b) + np.log1p(np.exp(-2.0 * a)) - np.log1p(np.exp(-2.0 * b))
    # 1/(1 + exp(log_ratio)), evaluated without overflow for either sign
    small = np.exp(-np.abs(log_ratio))
    return np.where(log_ratio >= 0.0, small / (1.0 + small), 1.0 / (1.0 + small))

@dataclass
class FermiDiracFilter:
    frequencies : np.array
    centered_frequency : float
    cutoff: int
    order : int
    values : np.array = field(init=False) 

    def __post_init__(self):
        self.values = create_fermi_dirac_fun(
            self.frequencies,
            self.centered_frequency,
            self.cutoff,
            self.order
        )


def create_raised_cosine_fun(fr, fc, bw, alpha=1.0):
    """Raised-cosine (Tukey-in-frequency) passband, the usual DSP choice.

        Y = 1                                          |x| <= 1 - alpha
          = (1 + cos(pi*(|x| - 1 + alpha)/(2*alpha)))/2  1 - alpha < |x| < 1 + alpha
          = 0                                          |x| >= 1 + alpha

    with ``x = (fr - fc)/bw``. Like :func:`create_fermi_dirac_fun` it is 1/2 at
    ``|x| = 1``, so ``bw`` means the same thing in both: the half-amplitude
    half-width.

    ``alpha`` is the roll-off, from a rectangle at 0 to a Hann-shaped bell at
    1. It plays the same role as ``n`` in the Fermi-Dirac filter but inverted:
    small ``alpha`` is flat and rings, large ``alpha`` is smooth and does not.

    Offered beside the Fermi-Dirac because it is the shape most readers will
    recognise, and because it is **compactly supported** -- exactly zero beyond
    ``bw*(1 + alpha)`` -- so extracting a finite block of bins around a tone
    truncates nothing. The Fermi-Dirac has exponential tails and needs a block
    wide enough for them to have decayed.

    On the transistor system the Fermi-Dirac at ``n = 2`` still measures best
    (worst-case 5.0e-04 against 7.4e-04 for ``alpha = 1``), because on a
    band-limited extraction the dominant error is time-domain ringing and its
    time concentration is exceptional. See the method documentation.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    x = np.abs(np.asarray(fr, dtype=float) - fc) / bw
    out = np.zeros_like(x)
    out[x <= 1 - alpha] = 1.0
    edge = (x > 1 - alpha) & (x < 1 + alpha)
    out[edge] = 0.5 * (1 + np.cos(np.pi * (x[edge] - 1 + alpha) / (2 * alpha)))
    return out


#: Cosine-sum coefficients, with the usual names. Each is evaluated as
#: ``a0 + a1*cos(pi*u) + a2*cos(2*pi*u) + ...`` on ``u`` in [-1, 1].
_COSINE_SUM = {
    "hann": (0.5, 0.5),
    "blackman": (0.42, 0.5, 0.08),
    "blackman-harris": (0.35875, 0.48829, 0.14128, 0.01168),
    "nuttall": (0.3635819, 0.4891775, 0.1365995, 0.0106411),
}

_HALF_POINT_CACHE = {}


def _window_profile(u, window, beta):
    """A window shape on ``u`` in [-1, 1], peaking at 1 in the middle."""
    u = np.asarray(u, dtype=float)
    if window in _COSINE_SUM:
        a = _COSINE_SUM[window]
        out = np.full_like(u, a[0])
        for m, coeff in enumerate(a[1:], start=1):
            out = out + coeff * np.cos(m * np.pi * u)
        return out
    if window == "kaiser":
        return np.i0(beta * np.sqrt(np.maximum(0.0, 1 - u ** 2))) / np.i0(beta)
    raise ValueError(
        f"unknown window {window!r}; choose from "
        f"{sorted(_COSINE_SUM) + ['kaiser', 'gaussian']}")


def _half_point(window, beta):
    """Where the profile falls to 1/2, so that ``bw`` can mean the same thing."""
    key = (window, beta)
    if key not in _HALF_POINT_CACHE:
        u = np.linspace(0.0, 1.0, 20001)
        y = _window_profile(u, window, beta)
        below = np.argmax(y < 0.5)
        _HALF_POINT_CACHE[key] = float(
            np.interp(0.5, [y[below], y[below - 1]], [u[below], u[below - 1]]))
    return _HALF_POINT_CACHE[key]


def create_window_filter(fr, fc, bw, window="blackman", beta=8.0):
    """Passband built from a standard DSP window.

    All of these are scaled so that the response is **1/2 at**
    ``|fr - fc| = bw``, exactly as :func:`create_fermi_dirac_fun` and
    :func:`create_raised_cosine_fun` are, so ``bw`` means the same thing
    throughout and the families can be compared at equal bandwidth.

    Parameters
    ----------
    window : str
        ``'blackman'``, ``'blackman-harris'``, ``'nuttall'``, ``'hann'``,
        ``'kaiser'`` or ``'gaussian'``.
    beta : float
        Shape parameter, for ``'kaiser'`` only.

    Notes
    -----
    These were added after a comparison against the Fermi-Dirac filter the
    DMFA papers specify, which it loses. Scored by worst-case relative error
    against known ground truth, with the band extracted wide and the output
    decimated to the independent spacing:

        ================  ============  ==========  ===============
        filter            transistor    legendre    block-oriented
        ================  ============  ==========  ===============
        blackman          2.40e-04      5.21e-04    1.18e-01
        kaiser, beta=8    2.92e-04      5.94e-04    1.18e-01
        blackman-harris   2.96e-04      6.94e-04    1.18e-01
        hann              4.71e-04      9.67e-04    1.17e-01
        gaussian          3.05e-04      9.77e-04    1.18e-01
        Fermi-Dirac n=2   4.98e-04      6.47e-04    1.27e-01
        ================  ============  ==========  ===============

    ``'blackman'`` is the best of them on two of the three systems and never
    the worst; the block-oriented column is a tie because its error there is
    dominated by the branch filters' roll-off rather than by the extraction.

    A first Slepian sequence (DPSS, NW = 4) was also tried and lands within
    1% of ``kaiser`` with ``beta=8``, which is the expected result since
    Kaiser is a closed-form approximation to it. It is not offered here
    because computing it needs an eigensolve for no measurable gain.
    """
    x = np.abs(np.asarray(fr, dtype=float) - fc) / bw
    if window == "gaussian":
        return np.exp(-np.log(2.0) * x ** 2)

    scale = _half_point(window, beta)
    out = np.zeros_like(x)
    inside = x * scale <= 1.0
    out[inside] = _window_profile(x[inside] * scale, window, beta)
    return out


def rejection_db(band_filter, band_axis, offset):
    """How far the passband suppresses a disturbance at a given offset, in dB.

    ``band_axis`` is the band's local frequency axis, as
    :func:`deistools.processing.dmfa_functions.elaboration_frequency_axis`
    returns it, and ``offset`` is the distance in Hz from the band centre.
    Large positive numbers are good.

    The use this exists for is **trends**. A drift of the dc level lives at
    zero frequency, so for the tone at ``f_k`` it sits ``f_k`` away from the
    band centre; ``rejection_db(g, axis, f_k)`` is therefore how much of it
    survives into that tone's estimate. The lowest tone is the one at risk,
    because every other one is further from dc.

    This is DMFA's whole defence against a trend, and it is worth a great
    deal. Measured on a transistor record carrying a 5 V peak-to-peak
    order-3 trend against a 1.53 V signal, varying nothing but the filter:

        =================  ==============  =====================
        filter             rejection       error, lowest tone
        =================  ==============  =====================
        Fermi-Dirac n=2    18 dB           2.9e-01
        Fermi-Dirac n=4    35 dB           3.6e-02
        Fermi-Dirac n=8    69 dB           1.4e-03
        Fermi-Dirac n=16   139 dB          2.1e-02
        blackman           29 dB           6.8e-02
        nuttall            27 dB           8.8e-02
        =================  ==============  =====================

    Two things to take from that table. Rejection buys accuracy until the
    filter's own ringing takes over -- ``n = 16`` has 70 dB more rejection
    than ``n = 8`` and is 15x worse, because at that sharpness it rings. And
    **a trend reverses the ranking of the filter families**: the compactly
    supported windows win on a trend-free record but reach only about 30 dB
    here, where the Fermi-Dirac's exponential tail reaches 69 dB.

    Choose the filter against the trend you expect, not against a clean
    simulation.
    """
    band_axis = np.asarray(band_axis, dtype=float)
    band_filter = np.asarray(band_filter, dtype=float)
    order = np.argsort(band_axis)
    gain = np.interp(abs(float(offset)), band_axis[order], band_filter[order],
                     left=0.0, right=0.0)
    return -20.0 * np.log10(max(gain, 1e-300))
