import numpy as np

def _zero_frequency_index(freq_axis):
    """Index of the 0 Hz bin in a shifted axis.

    ``fftshift`` always places it at ``n // 2``. Looked up rather than assumed
    so that a caller passing an axis built some other way fails here, with a
    clear message, instead of silently reading the wrong bins.
    """
    found = np.flatnonzero(np.asarray(freq_axis) == 0.0)
    if found.size != 1:
        raise ValueError(
            "freq_axis must be a shifted axis containing exactly one 0 Hz bin, "
            f"i.e. fftshift(fftfreq(n, sampling_time)); found {found.size}")
    return int(found[0])

def compute_freq_indexes(
          freq_axis,
          frequencies,
          sampling_time,
          ):
    """Bin index of each excited frequency in a shifted spectrum.

    ``freq_axis`` must be ``fftshift(fftfreq(n, sampling_time))`` built with
    the **same n as the spectrum you are going to index**, because the bin
    spacing ``1/(n*sampling_time)`` is taken from its length. Mixing a
    full-record axis with a windowed spectrum, or the reverse, silently reads
    the wrong bins; that is the single easiest way to misuse this function.

    Uses round rather than ceil: the aim is the *nearest* bin. With ceil a
    frequency landing a hair above a bin centre is pushed to the next one.

    """
    freq_axis = np.asarray(freq_axis)
    frequencies = np.asarray(frequencies, dtype=float)
    index_f0 = _zero_frequency_index(freq_axis)
    n = freq_axis.size

    offsets = frequencies * sampling_time * n
    rounded = np.round(offsets)

    return (rounded.astype(int) + index_f0).tolist(), index_f0

def search_peaks_position(
          spectrum,
          freq_axis,
          frequencies,
          sampling_time
          ):
        """Refine the computed bin indexes by finding the local spectral peak.

        ``spectrum`` must be a *shifted spectrum*, not a time series: the
        search indexes it with DFT bin numbers. Use this when the tones may
        not land exactly where asked -- a generator whose clock differs
        slightly from the acquisition card's, say. For a synthesised record
        :func:`compute_freq_indexes` is exact and this adds nothing.

        The window is bounded two ways: by half the distance to the nearer
        neighbouring excited frequency, and by the gap to the nearest
        neighbouring excited *bin*. Without the second bound the search can
        lock onto a neighbour, and when the two are equally strong ``argmax``
        silently prefers the lower index. Adjacent excited bins leave no room
        to search at all, and the computed index stands.
        """
        frequencies = np.asarray(frequencies, dtype=float)
        spectrum = np.asarray(spectrum)
        if spectrum.size != np.asarray(freq_axis).size:
            raise ValueError(
                f"spectrum has {spectrum.size} bins but freq_axis has "
                f"{np.asarray(freq_axis).size}. They must describe the same "
                "transform -- mixing a windowed spectrum with a full-record "
                "axis, or the reverse, reads the wrong bins.")
        # tol=None: the whole point here is that the tones may be off-grid.
        guesses = np.asarray(
            compute_freq_indexes(freq_axis, frequencies, sampling_time)[0], dtype=int)
        index_f0 = _zero_frequency_index(freq_axis)

        # Distance in Hz to the nearer neighbouring excited frequency. The
        # outermost tones have a neighbour on one side only; DC bounds the
        # first from below, and the last simply reuses its lower gap rather
        # than inventing a width for the side that has nothing on it.
        dist = np.empty(frequencies.size)
        if frequencies.size == 1:
            dist[0] = frequencies[0]
        else:
            dist[0] = min(frequencies[1] - frequencies[0], frequencies[0])
            dist[-1] = frequencies[-1] - frequencies[-2]
            for f in range(1, frequencies.size - 1):
                dist[f] = min(frequencies[f] - frequencies[f - 1],
                              frequencies[f + 1] - frequencies[f])
        half_hz = np.floor(dist * sampling_time * np.asarray(freq_axis).size / 4).astype(int)

        # The window must never reach a neighbouring excited bin, or the search
        # can lock onto the neighbour -- and when the two are equally strong,
        # argmax silently prefers the lower index. Adjacent excited bins
        # therefore leave no room to search and the computed index stands.
        if guesses.size == 1:
            half_bins = half_hz
        else:
            gaps = np.diff(guesses)
            nearest = np.empty(guesses.size, dtype=int)
            nearest[0], nearest[-1] = gaps[0], gaps[-1]
            nearest[1:-1] = np.minimum(gaps[:-1], gaps[1:])
            half_bins = np.minimum(half_hz, (nearest - 1) // 2)
        half_bins = np.maximum(half_bins, 0)

        index_multisine_freq = np.zeros(frequencies.size, dtype='int64')
        for f in range(frequencies.size):
            lo = max(0, guesses[f] - half_bins[f])
            hi = min(spectrum.size, guesses[f] + half_bins[f] + 1)
            index_multisine_freq[f] = lo + int(np.argmax(np.abs(spectrum[lo:hi])))
        return index_multisine_freq.tolist(), index_f0