import numpy as np
from dataclasses import dataclass
from numpy.fft import ifft
from deistools.processing.frequency_band import get_band_bins
from deistools.processing.passband_filters import PassbandFilter

@dataclass
class DmfaResult:
    frequency_response_function : np.ndarray
    # time : np.ndarray
    # excitation_drift : np.ndarray
    # response_drift : np.ndarray
    method_name : str = "DMFA"


def elaboration_frequency_axis(Npts_elab, measurement_time):
    return (np.arange(Npts_elab) - Npts_elab // 2) / float(measurement_time)


def elaboration_time_axis(Npts_elab, measurement_time):
    return np.arange(Npts_elab) * (float(measurement_time) / Npts_elab)


def get_band_size(n_samples, fs, multisine_period, periods_per_sample):
    return int(round(n_samples / (fs * multisine_period * periods_per_sample)))


def get_band_frequencies(n_bins, n_samples, fs):
    return np.fft.fftfreq(n_bins, d=1/n_bins) * fs / n_samples


def dmfa(
        U : np.ndarray,
        V : np.ndarray,
        k_exc : np.ndarray,
        passband_filter : PassbandFilter,
        fs : float,
        multisine_period : float,
        periods_per_sample : float,
):
    n_samples = U.size
    n_bins = get_band_size(n_samples, fs, multisine_period, periods_per_sample)
    filter = passband_filter.evaluate(get_band_frequencies(n_bins, n_samples, fs))
    G = np.zeros((k_exc.size, n_bins),dtype='complex128') # Initialize empty array
    for i in range(0, k_exc.size):
        k_band = get_band_bins(k_exc[i], n_bins, n_samples)
        U_filt = n_bins * ifft(U[k_band] * filter)
        V_filt = n_bins * ifft(V[k_band] * filter)
        G[i] = V_filt / U_filt
    time = np.arange(n_bins) * periods_per_sample * n_samples / (n_bins * fs)
    return G, time