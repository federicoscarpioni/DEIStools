"""
Goertzel filter for multi-frequency impedance.

The Goertzel algorithm evaluates a single DFT bin with a second-order
recurrence instead of a transform, costing ``O(N)`` per bin rather than
``O(N log N)`` for all of them. It is the standard way to detect a handful of
known tones -- DTMF decoding is the textbook case -- and it is attractive here
because a multisine measurement wants exactly a handful of known tones.

Mathematics
-----------
For a block of ``N`` samples and a bin ``k`` (the frequency in units of the
block's own resolution, ``k = f * N * dt``), define

    w     = 2*pi*k/N
    coeff = 2*cos(w)

and run, from ``s1 = s2 = 0``,

    s[n]  = x[n] + coeff*s[n-1] - s[n-2],     n = 0 .. N-1

After the last sample the DFT value is recovered from the final two states::

    X[k] = (exp(1j*w) * s[N-1] - s[N-2]) * exp(-1j*2*pi*k)

That combination is asymmetric, and the symmetric-looking rearrangement is
wrong::

    X[k] = (s[N-1] - exp(-1j*w) * s[N-2]) * exp(-1j*2*pi*k)    # WRONG

It has the right *magnitude* and is out by exactly ``exp(-1j*w)`` in phase.
The error cancels in any ratio -- including ``V/I`` -- so an impedance test
will not catch it; only a comparison against the DFT itself will, which is
what ``test_output_formula_is_not_the_tempting_rearrangement`` does. The
correct form is what one further iteration with a zero input would give,
``s[N] - exp(-1j*w)*s[N-1]``, written closed-form.

The trailing factor is unity whenever ``k`` is an integer, which is the usual
presentation. Keeping it is the *generalised* Goertzel of Sysel and Rajmic,
and it makes the result correct for a non-integer ``k`` as well -- useful when
a tone does not fall exactly on a bin of the block you chose.

References
----------
Goertzel, *Am. Math. Monthly* **65** (1958) 34 -- the original recurrence.

Sysel and Rajmic, *EURASIP J. Adv. Signal Process.* (2012) 56 -- the
generalisation to non-integer bins, which is the form implemented here.

Only two state variables are carried, so nothing but the recurrence has to be
stored. That is what makes the filter usable on a stream, one sample at a
time, with no buffer: see :class:`GoertzelFilter`.

What it is, and is not, good for
--------------------------------
Offline, on the records this package deals with, Goertzel is **slower** than
an FFT. The crossover is at roughly ``n_bins ~ log2(N)``: below that Goertzel
wins, above it the FFT does. A 13-tone multisine over a 20 000-sample block
has ``log2(N) = 14.3``, so the two are a wash, and the FFT is better
optimised.

It is included because:

* it is a standard estimator and the package exists to collect them;
* it computes exactly the same quantity as a rectangular-window STFT, which
  makes it a useful independent check on that path;
* its per-sample recurrence is the natural route to online processing, where
  no FFT is possible because the record has not finished.
"""
import numpy as np

__all__ = ["goertzel_dft", "goertzel_eis", "GoertzelFilter", "cost_ratio"]


def goertzel_dft(block, bins):
    """DFT values at the given bins, by recurrence rather than transform.

    Parameters
    ----------
    block : ndarray
        Either one block, shape ``(N,)``, or several, shape ``(n_blocks, N)``.
    bins : array_like
        Bin indices ``k = f * N * dt``. May be non-integer; the generalised
        correction is always applied.

    Returns
    -------
    ndarray
        Complex, shape ``(n_bins,)`` or ``(n_blocks, n_bins)``. Scaled like
        :func:`numpy.fft.fft` -- that is, **not** divided by ``N``.
    """
    block = np.asarray(block, dtype=float)
    single = block.ndim == 1
    if single:
        block = block[None, :]
    if block.ndim != 2:
        raise ValueError("block must be 1-D or 2-D")

    n_blocks, n = block.shape
    k = np.asarray(bins, dtype=float)
    omega = 2 * np.pi * k / n
    coeff = 2 * np.cos(omega)

    s1 = np.zeros((n_blocks, k.size))
    s2 = np.zeros((n_blocks, k.size))
    for sample in range(n):
        s = block[:, sample][:, None] + coeff[None, :] * s1 - s2
        s2, s1 = s1, s

    values = (np.exp(1j * omega)[None, :] * s1 - s2) * \
        np.exp(-1j * 2 * np.pi * k)[None, :]
    return values[0] if single else values


def goertzel_eis(voltage, current, frequencies, period, sampling_time):
    """Impedance per block, computed with the Goertzel recurrence.

    The signature mirrors
    :func:`deistools.processing.stft_functions.estimate_impedance`, and so
    does the result: the record is cut into consecutive blocks of ``period``
    seconds and one spectrum is produced per block, stamped at its midpoint.
    Only the excited bins are evaluated, which is the whole point of the
    algorithm.

    ``period`` must hold a whole number of periods of every tone, for the same
    reason a rectangular STFT window must: otherwise each bin picks up the
    others.

    Returns
    -------
    (impedance, time) : ndarray, ndarray
        ``impedance`` has shape ``(n_frequencies, n_blocks)``.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    if voltage.shape != current.shape or voltage.ndim != 1:
        raise ValueError("voltage and current must be 1-D arrays of equal length")

    samples_window = int(round(period / sampling_time))
    if samples_window < 2:
        raise ValueError("period is shorter than two samples")
    n_blocks = voltage.size // samples_window
    if n_blocks < 1:
        raise ValueError(
            f"the record holds {voltage.size} samples, less than one block of "
            f"{samples_window}")

    usable = n_blocks * samples_window
    v = voltage[:usable].reshape(n_blocks, samples_window)
    i = current[:usable].reshape(n_blocks, samples_window)

    bins = np.asarray(frequencies, dtype=float) * period
    V = goertzel_dft(v, bins)
    I = goertzel_dft(i, bins)

    time = (np.arange(n_blocks) + 0.5) * period
    return (V / I).T, time


class GoertzelFilter:
    """One bin of a running DFT, updated a sample at a time.

    The offline functions above exist because this package analyses recorded
    files, but the recurrence itself needs no record: it carries two numbers.
    This class is the streaming form, for an acquisition that has not
    finished.

    >>> filt = GoertzelFilter(frequency=0.05, sampling_time=1e-3, block=20000)
    >>> for sample in stream:                           # doctest: +SKIP
    ...     value = filt.push(sample)
    ...     if value is not None:
    ...         ...                                     # one block completed

    ``push`` returns ``None`` until a block is complete, then the complex DFT
    value for that block, and resets itself for the next one.
    """

    def __init__(self, frequency, sampling_time, block):
        self.block = int(block)
        self.k = float(frequency) * self.block * float(sampling_time)
        self.omega = 2 * np.pi * self.k / self.block
        self.coeff = 2 * np.cos(self.omega)
        self.reset()

    def reset(self):
        self._s1 = 0.0
        self._s2 = 0.0
        self._n = 0

    def push(self, sample):
        """Feed one sample; returns the block's DFT value when it completes."""
        s = float(sample) + self.coeff * self._s1 - self._s2
        self._s2, self._s1 = self._s1, s
        self._n += 1
        if self._n < self.block:
            return None
        value = (np.exp(1j * self.omega) * self._s1 - self._s2) * \
            np.exp(-1j * 2 * np.pi * self.k)
        self.reset()
        return value


def cost_ratio(n_bins, block):
    """Goertzel cost divided by FFT cost, in multiply-accumulates.

    Goertzel is about ``n_bins * N`` operations for ``n_bins`` bins of an
    ``N``-sample block; an FFT is about ``N * log2(N)`` for all of them. The
    ratio below one means Goertzel is the cheaper choice.

    This counts operations only. In practice a compiled FFT beats a recurrence
    at equal operation count by a wide margin, so the real crossover sits well
    below one.
    """
    return n_bins / np.log2(block)
