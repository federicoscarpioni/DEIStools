import numpy as np
import math

def get_band_bins(k_e: int, n_bins: int, n_record: int) -> np.ndarray:
    '''
    ke : excited frequency
    N : number of bins in the band. For DMFA is it dictated by the window width
    '''
    # if N % 2==0:
    #     band_bins = np.arange(ke - int(N/2), ke + int(N/2), 1)
    # else:   
    #     band_bins = np.arange(ke - math.floor(N/2), ke + math.ceil(N/2), 1) # Note: is this the right way to manage even?
    off = np.fft.fftfreq(n_bins, d=1/n_bins).astype(int)   # 0, 1, …, -1
    band_bins = (k_e + off) % n_record

    return band_bins

def get_band_bins_halfwidth(
        k_e: int, # bin of the excited frequency 
        hw: int, # halfband width
        N: int, # number of samples in the signal
    ):
    low = k_e - hw
    high = k_e + hw + 1
    if low < 0: raise ValueError('Band too wide (extends below min)')
    if high >= N//2: raise ValueError('Band too wide (extends above max)')
    return np.arange(low, high)

# def get_three_ke_band_bins(e:int, k_exc:np.ndarray, halfwidth:int):
#     '''
#     DFT bins of the local band centred on ``k_exc[e]``.
    
#     ``halfwidth=None`` builds the band of Hallemans TIM 2021 eq. (55)::

#         K_w = {k_e^- - dk^-, ..., k_e^-, ..., k_e, ..., k_e^+, ..., k_e^+ + dk^+}

#     which spans the three consecutive excited frequencies ``k_e^-, k_e, k_e^+``
#     and extends half the distance towards the next excited bin beyond each of
#     the outer two. Only the centre frequency's coefficients are kept; the band
#     then slides to the next excited bin.
#     '''
#     k_e = int(k_exc[e])
#     if halfwidth is not None:
#         lo, hi = k_e - halfwidth, k_e + halfwidth
#     elif k_exc.size == 1:
#         raise ValueError("halfwidth must be given when only one frequency is excited")
#     else:
#         inner_lo, inner_hi = _excited_at(k_exc, e - 1), _excited_at(k_exc, e + 1)
#         outer_lo, outer_hi = _excited_at(k_exc, e - 2), _excited_at(k_exc, e + 2)
#         lo = inner_lo - (inner_lo - outer_lo) // 2
#         hi = inner_hi + (outer_hi - inner_hi) // 2
#     lo = max(1, lo)                 # bin 0 carries the DC offset of the response
#     hi = min(N // 2 - 1, hi)        # positive frequencies only
#     if hi <= lo:
#         raise ValueError(f"empty band at excited bin {k_e}")
#     return np.arange(lo, hi + 1)