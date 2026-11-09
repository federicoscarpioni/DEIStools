import numpy as np
from numpy.fft import rfft

def excited_bins(n_samples, frequencies, fs)->np.ndarray:
    i_exc = (n_samples * frequencies) / fs
    i_exc = np.rint(i_exc).astype(np.int32)
    return i_exc

def fft_eis(voltage, current, i_exc):
    ft_voltage =  fft(voltage) /voltage.size
    ft_current =  fft(current) / current.size
    impedance = (ft_voltage[i_exc]/ft_current[i_exc])
    voltage_avg = abs(ft_voltage[0])
    current_avg = abs(ft_current[0])
    return impedance, voltage_avg, current_avg

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
    Z, v_mean, i_mean = fft_eis_2d(v, i, excited_bins(L, frequencies, fs))
    time = (np.arange(M) + 0.5) * L / fs        # window centres
    return Z, v_mean, i_mean, time