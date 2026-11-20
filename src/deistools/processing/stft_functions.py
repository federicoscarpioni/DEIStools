import numpy as np
from dataclasses import dataclass
from numpy.fft import rfft 
from deistools.processing.detrending import get_baseline_mean

@dataclass
class StftEisResult:
    frequency_response_function : np.ndarray
    time : np.ndarray
    excitation_drift : np.ndarray
    response_drift : np.ndarray
    method_name : str = "STFT-EIS"

def get_excited_bins(N, frequencies, fs)->np.ndarray:
    k_exc = (N * frequencies) / fs
    k_exc = np.rint(k_exc).astype(np.int32)
    return k_exc

def fft_eis_2d(u, v, k_exc)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''
    Uses rfft instead of fft to spare memory since the negative bins are not 
    necessary for this method.
    '''
    U =  rfft(u) /u.shape[-1]
    V =  rfft(v) / v.shape[-1]
    G = (V[..., k_exc]/U[..., k_exc])
    u_drift = U[..., 0].real
    v_drift = V[..., 0].real
    return G, u_drift, v_drift

def stft_eis(
        u,
        v,
        frequencies,
        fs,
        multisine_period,
        periods_per_window,
        detrending = False,
    )->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:

    L = int(round(periods_per_window * multisine_period * fs))          # samples per window, represents the frequency axis
    M = u.size // L                       # whole windows only, represents the time direction
    u = u[:M * L].reshape(M, L)           # one window per row
    v = v[:M * L].reshape(M, L)
    if detrending:
        samples_per_period = int(round(multisine_period * fs))
        window_time = np.arange(L) / fs
        u_detrended = np.empty((M, L))
        v_detrended = np.empty((M, L))
        for m in range(M):
            u_detrended[m] = u[m] - get_baseline_mean(u[m], samples_per_period, fs).evaluate(window_time)
            v_detrended[m] = v[m] - get_baseline_mean(v[m], samples_per_period, fs).evaluate(window_time)
        G, _, _ = fft_eis_2d(u_detrended, v_detrended, get_excited_bins(L, frequencies, fs))
        u_drift, v_drift = u.mean(axis=-1), v.mean(axis=-1)
    else:
        G, u_drift, v_drift = fft_eis_2d(u, v, get_excited_bins(L, frequencies, fs))
    G = G.T # output the matrix with size (n_freq, n_time)
    time = (np.arange(M) + 0.5) * L / fs        # window centres
    return G, u_drift, v_drift, time