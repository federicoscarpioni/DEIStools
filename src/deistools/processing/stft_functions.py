import numpy as np
from numpy.fft import rfft


def fft_eis_2d(voltage, current, i_exc)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    ft_voltage =  rfft(voltage) /voltage.shape[-1]
    ft_current =  rfft(current) / current.shape[-1]
    impedance = (ft_voltage[..., i_exc]/ft_current[..., i_exc])
    voltage_avg = ft_voltage[..., 0].real
    current_avg = ft_current[..., 0].real
    return impedance, voltage_avg, current_avg
def get_excited_bins(N, frequencies, fs)->np.ndarray:
    k_exc = (N * frequencies) / fs
    k_exc = np.rint(k_exc).astype(np.int32)
    return k_exc

def stft_eis(
        voltage,
        current,
        frequencies,
        fs,
        multisine_period,
        periods_per_window,
    )->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:

    L = int(round(periods_per_window * multisine_period * fs))          # samples per window, represents the frequency axis
    M = voltage.size // L                       # whole windows only, represents the time direction
    v = voltage[:M * L].reshape(M, L)           # one window per row
    i = current[:M * L].reshape(M, L)
    Z, v_mean, i_mean = fft_eis_2d(v, i, get_excited_bins(L, frequencies, fs))
    time = (np.arange(M) + 0.5) * L / fs        # window centres
    return Z, v_mean, i_mean, time