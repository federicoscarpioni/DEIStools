import numpy as np
from numpy.fft import ifft
from deistools.processing.frequency_band import get_band_bins

# def extract_zero_frequency(
#         ft_voltage : np.ndarray,
#         ft_current : np.ndarray,
#         filter : np.ndarray,
#         n_bins : int,
# ):
#     index_f0 = np.where(freq_axis == 0)[0][0]
    
#     voltage = Npts_elab * ifft(ft_voltage[freq_range] * filter).real
#     current = Npts_elab * ifft(ft_current[freq_range] * filter).real
#     time = time_resolution * np.arange(Npts_elab)
#     return voltage, current, time

def elaboration_frequency_axis(Npts_elab, measurement_time):
    return (np.arange(Npts_elab) - Npts_elab // 2) / float(measurement_time)


def elaboration_time_axis(Npts_elab, measurement_time):
    return np.arange(Npts_elab) * (float(measurement_time) / Npts_elab)


# def max_elaboration_points(indexes_multisine_freq, n_samples):
#     lowest = min(indexes_multisine_freq)
#     highest = max(indexes_multisine_freq)
#     return 2 * min(lowest, n_samples - highest)
def get_band_size(n_samples, fs, multisine_period, periods_per_sample):
    return int(round(n_samples / (fs * multisine_period * periods_per_sample)))


# def _check_elaboration_range(indexes_multisine_freq, Npts_elab, n_samples):
#     limit = max_elaboration_points(indexes_multisine_freq, n_samples)
#     if Npts_elab > limit:
#         raise ValueError(
#             f"Npts_elab={Npts_elab} does not fit: the extraction window would "
#             f"run past the edge of the spectrum at bin "
#             f"{max(indexes_multisine_freq)} of {n_samples}. Use at most "
#             f"{limit} (see max_elaboration_points).")
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
# def extract_impedance_with_error(
#         ft_voltage,
#         ft_current,
#         variance_voltage,
#         variance_current,
#         indexes_multisine_freq: list,
#         filter,
#         Npts_elab,
# ):
#     _check_elaboration_range(indexes_multisine_freq, Npts_elab, ft_voltage.size)
#     n_freq = len(indexes_multisine_freq)
#     impedance = np.zeros((n_freq, Npts_elab), dtype="complex128")
#     variance = np.zeros((n_freq, Npts_elab), dtype="float64")

#     # N' uses the length of the record, which is what the noise was spread
#     # over, not the width of the extracted band.
#     N_prime = ft_voltage.size / np.sum(np.abs(filter) ** 2)

#     for f in range(n_freq):
#         if Npts_elab % 2 == 0:
#             elaboration_rng = np.arange(indexes_multisine_freq[f] - Npts_elab // 2,
#                                         indexes_multisine_freq[f] + Npts_elab // 2)
#         else:
#             elaboration_rng = np.arange(indexes_multisine_freq[f] - Npts_elab // 2,
#                                         indexes_multisine_freq[f] + Npts_elab // 2 + 1)
#         voltage_filtered = Npts_elab * ifft(ifftshift(ft_voltage[elaboration_rng] * filter))
#         current_filtered = Npts_elab * ifft(ifftshift(ft_current[elaboration_rng] * filter))
#         impedance[f] = voltage_filtered / current_filtered
#         numerator = variance_voltage + np.abs(impedance[f]) ** 2 * variance_current
#         variance[f] = numerator / (2 * N_prime * np.abs(current_filtered) ** 2)

#     return impedance, variance    n_samples = U.size
    n_bins = get_band_size(n_samples, fs, multisine_period, periods_per_sample)
    filter = passband_filter.evaluate(get_band_frequencies(n_bins, n_samples, fs))
    G = np.zeros((k_exc.size, n_bins),dtype='complex128') # Initialize empty array
    for i in range(0, k_exc.size):
        k_band = get_band_bins(k_exc[i], n_bins, n_samples)
        U_filt = n_bins * ifft(U[k_band] * filter)
        V_filt = n_bins * ifft(V[k_band] * filter)
        G[i] = V_filt / U_filt
    return G