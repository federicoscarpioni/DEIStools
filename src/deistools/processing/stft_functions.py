import numpy as np
from dataclasses import dataclass
from numpy.fft import rfft 

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
    )->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:

    L = int(round(periods_per_window * multisine_period * fs))          # samples per window, represents the frequency axis
    M = u.size // L                       # whole windows only, represents the time direction
    u = u[:M * L].reshape(M, L)           # one window per row
    v = v[:M * L].reshape(M, L)
    G, u_drift, v_drift = fft_eis_2d(u, v, get_excited_bins(L, frequencies, fs))
    G = G.T # output the matrix with size (n_freq, n_time)
    time = (np.arange(M) + 0.5) * L / fs        # window centres
    return G, u_drift, v_drift, time