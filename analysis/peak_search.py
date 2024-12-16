import numpy as np

def find_freq_indexes_bnd(input_signal, multisine_freq, SAMPLING_RATE, freq_axis, bnd):
    """
    Find the indexes in the frequency domain corresponding to the peak. Due to
    instrumental artifacts, the frequencies can shift by a few indexes respect
    to the values used in the multisine.
    This algorithm search an interval of 2*bnd in the neigborhood of the
    calculated ideal position.

    Parameters
    ----------
    input_signal : array
        Voltage (for quasi-CV) or current (for galvanostatic).
    multisine_freq : array
        Frequencies in the multisine.
    SAMPLING_RATE : float
        Sampling rate of the oscilloscope.
    bnd : int


    Returns
    -------
    index_multisine_freq : array
        Indexes in the frequency axis vector corrisponding to the frequencies in the multisine.

    """

    index_f0 = np.where(freq_axis == 0)[0][0]
    n_samples = input_signal.size

    guess_i_multisine_freq = np.ceil(multisine_freq * SAMPLING_RATE * n_samples) + index_f0
    index_multisine_freq = np.zeros(multisine_freq.size, dtype='int64')
    dist_between_freq = np.zeros(multisine_freq.size, dtype='float32')
    dist_between_freq[0] = np.min([multisine_freq[1] - multisine_freq[0], multisine_freq[0]])
    for f in range(1, multisine_freq.size - 1):
        dist_between_freq[f] = np.min(
            [multisine_freq[f] - multisine_freq[f - 1], multisine_freq[f + 1] - multisine_freq[f]])
    dist_between_freq[-1] = multisine_freq[-1] - multisine_freq[-2]  # Fixed
    # dist_peak = dist_between_freq/(dt*N_samples)
    dist_between_freq_P = 2 * np.floor(dist_between_freq * SAMPLING_RATE * n_samples / 4)

    for f in range(0, multisine_freq.size):
        if multisine_freq[f] < 1:
            min_searching = int(guess_i_multisine_freq[f]) - bnd
            max_searching = int(guess_i_multisine_freq[f]) + bnd
        else:
            min_searching = int(guess_i_multisine_freq[f]) - int(dist_between_freq_P[f] / 2)
            max_searching = int(guess_i_multisine_freq[f]) + int(dist_between_freq_P[f] / 2)
        searching_freq_rng = np.arange(min_searching, max_searching)
        index_multisine_freq[f] = min_searching + int(np.argmax((np.abs(input_signal[(searching_freq_rng)])), axis=0))

    return index_multisine_freq