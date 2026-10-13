"""
Digital lock-in amplifier for multi-frequency impedance.

The classical instrument, implemented offline. For each excited frequency the
record is multiplied ("mixed") with a cosine and a sine at that frequency and
the product low-pass filtered, which leaves the slowly varying complex
amplitude of that tone. The ratio of the voltage and current amplitudes is
Z(f, t).

Mathematics
-----------
Write the response near the ``k``-th tone as a carrier with a slowly varying
phasor::

    v(t) = A_v(t) * cos(w_k * t + phi_v(t)),      w_k = 2*pi*f_k

Mixing with the in-phase and quadrature references gives

    v(t) * cos(w_k t) = (A_v/2) * cos(phi_v) + (A_v/2) * cos(2 w_k t + phi_v)
    v(t) * sin(w_k t) = -(A_v/2) * sin(phi_v) + (A_v/2) * sin(2 w_k t + phi_v)

so a low-pass filter that rejects ``2*w_k`` and every other excited tone
leaves ``X = (A_v/2) cos(phi_v)`` and ``Y = -(A_v/2) sin(phi_v)``. Twice the
complex combination is the phasor::

    V(t) = 2 * (X - 1j*Y) = A_v(t) * exp(1j * phi_v(t))

The same is done for the current, and

    Z(f_k, t) = V(t) / I(t)

Choosing the integrator
-----------------------
The low-pass filter is the whole method, because it must reject

* the image at ``2*f_k`` produced by the mixing, and
* **every other tone of the multisine**, which after mixing appears at the
  difference frequency ``|f_j - f_k|``.

The second requirement is the binding one. For a multisine whose tones are
all harmonics of a base frequency ``f_b``, a **boxcar** average over exactly
an integer number of periods of ``f_b`` has spectral nulls at every multiple
of ``f_b`` -- that is, at every difference frequency that can occur. It
rejects all of them exactly.

A Hann (or any tapered) window has its nulls in the same places but a
mainlobe twice as wide, so with a one-period window the nearest harmonic sits
*inside* the mainlobe and is not rejected. A Hann integrator therefore needs
at least **two** base periods. :func:`integrator_length` enforces this, and
:func:`tone_rejection` measures it.

The integrator length is also the time resolution: a window of ``m`` periods
of ``f_b`` cannot follow anything faster than ``m / f_b`` seconds, which is
the same trade STFT makes with its window and DMFA with its bandwidth.

Edges
-----
The output within half an integrator length of either end of the record is
computed from a window that extends past the data. This module reflects the
signal at the boundaries, which is exact for an envelope with zero slope
there and good otherwise, and :func:`valid_mask` marks the affected samples
so they can be excluded from a fit.
"""
import numpy as np

__all__ = [
    "integrator_length", "integrator_kernel", "tone_rejection", "valid_mask",
    "lia_impedance_optim", "lia_impedance_ls", "lia_eis", "local_ls",
]


# --------------------------------------------------------------------------- #
# Integrator design
# --------------------------------------------------------------------------- #
def integrator_length(sampling_time, base_frequency, n_periods=1,
                      window="boxcar"):
    """Samples in the lock-in integrator, as a whole number of base periods.

    Parameters
    ----------
    sampling_time : float
        Seconds between samples.
    base_frequency : float
        The frequency every excited tone is a harmonic of, in Hz. For a
        multisine of period ``Tp`` this is ``1/Tp``, *not* the tone being
        demodulated.
    n_periods : int
        How many base periods the integrator spans. This sets the time
        resolution, ``n_periods / base_frequency`` seconds.
    window : {'boxcar', 'hann'}
        A tapered window has a mainlobe twice as wide as a boxcar, so it needs
        at least two base periods to put the nearest harmonic in a null.

    Returns
    -------
    int
        Exactly ``n_periods`` base periods in samples. The length is *not*
        rounded to an odd number: the nulls sit at multiples of
        ``1/(samples*sampling_time)`` and only land on the harmonics if the
        window spans a whole number of base periods exactly. Forcing the
        length odd to centre the kernel on a sample detunes them -- it costs
        about 40 dB of rejection on a 20 000-sample period. The resulting
        half-sample group delay is common to voltage and current and cancels
        in their ratio.
    """
    if n_periods < 1:
        raise ValueError("n_periods must be at least 1")
    if window == "hann" and n_periods < 2:
        raise ValueError(
            "a Hann integrator needs n_periods >= 2: its mainlobe is twice as "
            "wide as a boxcar's, so over one base period the neighbouring "
            "harmonic falls inside it and is not rejected. Use "
            "window='boxcar' for a single period.")
    samples = n_periods / (base_frequency * sampling_time)
    if abs(samples - round(samples)) > 1e-6 * samples:
        raise ValueError(
            f"{n_periods} period(s) of {base_frequency} Hz is {samples:.4f} "
            "samples, not a whole number. The integrator cannot null the other "
            "tones unless it spans whole base periods; adjust the sampling "
            "rate or the base frequency.")
    return int(round(samples))


def integrator_kernel(samples_window, window="boxcar"):
    """The normalised low-pass kernel, with unit DC gain."""
    if window == "hann":
        kernel = np.hanning(samples_window + 2)[1:-1]   # periodic, no zeros
    elif window == "boxcar":
        kernel = np.ones(samples_window)
    else:
        raise ValueError("window must be 'boxcar' or 'hann'")
    return kernel / kernel.sum()


def tone_rejection(kernel, sampling_time, frequencies, f_k):
    """How much the integrator suppresses each other tone, in dB.

    After mixing at ``f_k`` a tone at ``f_j`` lands at ``|f_j - f_k|``; this
    evaluates the kernel's frequency response there. Large positive numbers
    are good. A boxcar over whole base periods gives essentially infinite
    rejection (nulls), limited only by arithmetic.

    The returned array has one entry per frequency, **including ``f_k``
    itself**, whose offset is zero and whose "rejection" is therefore 0 dB by
    construction. Drop that entry before taking a worst case over the others,
    or the answer is always 0 dB.
    """
    offsets = np.abs(np.asarray(frequencies) - f_k)
    n = np.arange(kernel.size)
    response = np.array([np.abs(np.sum(kernel * np.exp(-2j * np.pi * df *
                                                       n * sampling_time)))
                         for df in offsets])
    with np.errstate(divide="ignore"):
        return -20 * np.log10(np.maximum(response, 1e-300))


def valid_mask(n_samples, samples_window):
    """False where the integrator window ran past the end of the record."""
    mask = np.ones(n_samples, dtype=bool)
    half = samples_window // 2
    mask[:half] = False
    mask[n_samples - half:] = False
    return mask


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def _fft_convolve_same(signal, kernel):
    """Zero-phase 'same'-length convolution, using numpy only.

    ``scipy.signal.fftconvolve`` would do this, but scipy is not a dependency
    of this package and the kernel here is long enough that ``np.convolve``
    (which is O(N*M)) is impractical on a million-sample record.
    """
    n = signal.size + kernel.size - 1
    nfft = 1 << int(n - 1).bit_length()
    full = np.fft.irfft(np.fft.rfft(signal, nfft) *
                        np.fft.rfft(kernel, nfft), nfft)[:n]
    start = (kernel.size - 1) // 2
    return full[start:start + signal.size]


def _lowpass(signal, kernel):
    """Apply the integrator, reflecting the signal at both boundaries."""
    half = kernel.size // 2
    if half == 0:
        return signal.copy()
    half = min(half, signal.size - 1)
    padded = np.pad(signal, half, mode="reflect")
    return _fft_convolve_same(padded, kernel)[half:half + signal.size]


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
def lia_impedance_optim(t, v, i, dt, f0, samples_window, window="boxcar"):
    """Lock-in impedance at one frequency, by mixing and low-pass filtering.

    Parameters
    ----------
    t, v, i : ndarray
        Time, voltage and current, all the same length.
    dt : float
        Sampling time in seconds.
    f0 : float
        The frequency to demodulate, in Hz.
    samples_window : int
        Length of the integrator. Use :func:`integrator_length` to get a value
        that rejects the other tones.
    window : {'boxcar', 'hann'}

    Returns
    -------
    ndarray
        Complex ``Z(f0, t)``, one value per input sample. Samples within half
        an integrator length of either end rest on reflected data; see
        :func:`valid_mask`.
    """
    omega = 2 * np.pi * f0
    kernel = integrator_kernel(samples_window, window)

    ref_cos, ref_sin = np.cos(omega * t), np.sin(omega * t)

    # Factor 2 undoes the halving that mixing introduces, so that the complex
    # result is the phasor itself rather than half of it. It cancels in the
    # ratio below, but makes V and I meaningful on their own.
    V = 2 * (_lowpass(v * ref_cos, kernel) - 1j * _lowpass(v * ref_sin, kernel))
    I = 2 * (_lowpass(i * ref_cos, kernel) - 1j * _lowpass(i * ref_sin, kernel))

    return V / I


# --------------------------------------------------------------------------- #
# Local least squares, an alternative demodulator
# --------------------------------------------------------------------------- #
def local_ls(t, x, omega, idx, Nw):
    """Fit cos, sin, an offset and a local slope in a window centred on ``idx``.

    Returns the complex amplitude **referenced to the centre of the window**,
    not to t = 0. The reference cancels in ``V/I`` because both are fitted on
    the same window, which is why :func:`lia_impedance_ls` is correct; a
    caller using this function alone must undo the ``exp(-1j*omega*t0)``
    itself.
    """
    half = Nw // 2
    sl = slice(max(0, idx - half), min(len(t), idx + half + 1))
    tt, xx = t[sl], x[sl]
    tau = tt - tt.mean()

    design = np.column_stack([np.cos(omega * tau), np.sin(omega * tau),
                              np.ones_like(tau), tau])
    weight = np.sqrt(np.hanning(tau.size + 2)[1:-1])
    theta, *_ = np.linalg.lstsq(design * weight[:, None], xx * weight,
                                rcond=None)
    return theta[0] - 1j * theta[1]


def lia_impedance_ls(t, v, i, dt, f0, samples_window, stride=None):
    """Lock-in impedance by local least squares instead of mixing.

    Fits a cosine, a sine, an offset and a linear slope in a sliding window.
    Carrying the offset and slope makes it insensitive to drift *within* the
    window, which plain mixing is not -- useful on a record with a trend.

    It is far more expensive: one least-squares solve per reported instant,
    against two FFT convolutions for the whole record. ``stride`` therefore
    evaluates every ``stride``-th sample; the default keeps about 1000
    instants, which is already finer than the integrator can resolve.

    Returns
    -------
    (Z, t_out) : ndarray, ndarray
        Unlike :func:`lia_impedance_optim` this returns its own time axis,
        because it reports on a decimated grid.
    """
    omega = 2 * np.pi * f0
    if stride is None:
        stride = max(1, len(t) // 1000)

    idx = np.arange(0, len(t), stride)
    V = np.array([local_ls(t, v, omega, k, samples_window) for k in idx])
    I = np.array([local_ls(t, i, omega, k, samples_window) for k in idx])
    return V / I, t[idx]


# --------------------------------------------------------------------------- #
# All frequencies
# --------------------------------------------------------------------------- #
def lia_eis(time, voltage, current, dt, frequencies, samples_window,
            impedance_calculation_fun=lia_impedance_optim, window="boxcar",
            decimate=None):
    """Lock-in impedance at every excited frequency.

    Parameters
    ----------
    samples_window : int
        Integrator length, from :func:`integrator_length`. The same window is
        used at every frequency, which is what makes the spectra comparable:
        every point of a spectrum is then an average over the same interval.
    impedance_calculation_fun : callable
        :func:`lia_impedance_optim` (default) or :func:`lia_impedance_ls`.
    decimate : int, optional
        Report every ``decimate``-th sample. The output is smooth on the scale
        of the integrator, so nothing is lost by decimating to a few times
        ``len(time) * base_frequency / n_periods`` points, and the full-rate
        array is large: 13 frequencies over a million samples is 200 MB.

    Returns
    -------
    (impedance, time_out) : ndarray, ndarray
        ``impedance`` has shape ``(n_frequencies, n_instants)``.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    if voltage.shape != current.shape or voltage.shape != time.shape:
        raise ValueError("time, voltage and current must have the same shape")

    columns = []
    for f in frequencies:
        if impedance_calculation_fun is lia_impedance_ls:
            Z_f, t_out = lia_impedance_ls(time, voltage, current, dt, f,
                                          samples_window)
        else:
            Z_f = impedance_calculation_fun(time, voltage, current, dt, f,
                                            samples_window, window=window)
            t_out = time
            if decimate:
                Z_f, t_out = Z_f[::decimate], time[::decimate]
        columns.append(Z_f)

    return np.vstack(columns), t_out
