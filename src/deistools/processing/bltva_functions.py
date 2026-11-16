"""
Notation (code -> Hallemans et al., TIM 2021)
---------------------------------------------
k_exc, k_e      excited bins; k_e is the one being fitted        k_e
k_band          bins of the local band around k_e                K_w (eq. 55)
k_neighbours    other excited bins inside k_band                 k' in eq. (58)
basis           sampled Legendre basis, (N, order+1)             b_p(t)
basis_spectra   DFT of basis / N                                 B_p(k)
regressors      regression matrix on k_band                      K (eq. 58)
V_band          response spectrum on k_band                      Y_w
theta           least-squares solution                           θ (eq. 64)

"Window" always refers to the time domain and "band" to the frequency domain.
"""
import numpy as np
from dataclasses import dataclass
from deistools.processing.frequency_band import get_band_bins_halfwidth
from numpy.polynomial.legendre import legvander


@dataclass
class BltvaResult:
    frequency_response_function : np.ndarray
    # time : np.ndarray
    # excitation_drift : np.ndarray
    # response_drift : np.ndarray
    method_name : str = "BLTVA"


def legendre_basis(N: int, order: int) -> np.ndarray:
    """
    Hallemans TIM 2021, after eq. (26): the mean values of the sampled and windowed b_p(t), p ≥ 1 are subtracted.
    
    Rescaled Legendre basis b_p(t) = L_p(2t/T - 1), sampled on N points.

    Returns a real array of shape ``(N, order+1)``.

    The sample mean is removed from every p >= 1. On a finite sampled window the
    continuous property (1/T) int b_p dt = 0 does not hold exactly, so without this
    every time-varying regressor leaks into the DC bin.
    """
    x = 2.0 * np.arange(N) / N - 1.0
    b = legvander(x, order)                       # (N, order+1)
    b[:, 1:] -= b[:, 1:].mean(axis=0, keepdims=True)
    return b


def legendre_spectra(b: np.ndarray) -> np.ndarray:
    """Scaled DFT of the basis functions: B_p(k) = FFT(b_p)/N.

    Kept at full length N. The skirts are two-sided about their own centre, so a
    column centred at harmonic k' is indexed as ``B[(K_w - k') % N, p]``.
    """
    return np.fft.fft(b, axis=0) / b.shape[0]


def build_regressors(k_e, k_band, centres, B, Np, Nq, Na):
    """
    TIM 2021 eq. (58)
    """
    N = B.shape[0]
    blocks, cmap, j = [], {}, 0

    def add(name, block):
        nonlocal j
        cmap[name] = slice(j, j + block.shape[1])
        blocks.append(block)
        j += block.shape[1]

    add("target", B[(k_band - k_e) % N, : Np + 1])
    for kp in centres:
        add(f"h{kp}", B[(k_band - kp) % N, : Np + 1])

    if Nq >= 1:
        add("drift", B[k_band, 1 : Nq + 1])

    xi = 2.0 * (k_band - k_band[0]) / (k_band[-1] - k_band[0]) - 1.0
    add("transient", legvander(xi, Na).astype(complex))
    return np.hstack(blocks), cmap


def select_na(P, Np, d=0):
    """Transient polynomial order, Hallemans TIM 2022 eq. (22).

        Na = floor(3P/4) - Np - d + 1

    ``d`` is the differencing order used for trend removal; it is 0 while no
    detrending is applied, which reduces the rule to ``floor(3P/4) - Np + 1``.
    Verified against the 2022 simulation example: P = 20, Np = 4, d = 3 gives
    Na = 9, as the paper states.

    The order is an upper bound. Algorithm 2 step 3 prunes transient columns
    that the skirts already span, so choosing generously costs little.
    """
    Na = int(np.floor(3 * P / 4)) - Np - d + 1
    if Na < 0:
        raise ValueError(
            f"eq. (22) gives Na={Na} for P={P}, Np={Np}, d={d}. Measure more "
            "periods or lower Np.")
    return Na


def solve_band(K, V_band):
    """
    eq. (64)/(67): theta = K^+ Y_w, the Moore-Penrose minimum-norm solution.
    """
    theta, _, rank, _ = np.linalg.lstsq(K, V_band)
    return theta, int(rank) 


def fit_band(
        V:np.ndarray, # DFT of the response signal
        k_e:int, # bin index of the excited frequency
        K : np.ndarray, # regression matrix
        k_band:np.ndarray, # bin band of the fitting
    ):
    n_theta = K.shape[1]
    if n_theta >= k_band.size:
        raise ValueError(
            f"band at bin {k_e}: {n_theta} unknowns for {k_band.size} equations."
        )
    V_band = V[k_band]
    theta, rank = solve_band(K, V_band)
    fit = K @ theta
    res_abs = np.linalg.norm(V_band - fit) # I am not sure this is what I want
    res_rel = np.linalg.norm(V_band - fit) / np.linalg.norm(V_band) # I am not sure this is what I want
    return theta, res_abs, res_rel, rank


def bltva(
        U : np.ndarray,
        V : np.ndarray, 
        k_exc : np.ndarray,
        P : int, 
        halfband : int, 
        Np : int, 
        Nq : int, 
        Na:int | None = None,
    ):
    N = U.size
    order = max(Np,Nq)
    basis = legendre_basis(N, order)
    dft_basis = legendre_spectra(basis)
    if Na is None: Na = select_na(P, Np)
    G_p = np.zeros((len(k_exc), Np + 1), dtype=complex)
    res_abs = np.zeros(len(k_exc))
    res_rel = np.zeros(len(k_exc))
    rank = np.zeros(len(k_exc), dtype=int)
    for e, k_e in enumerate(k_exc):
        k_band = get_band_bins_halfwidth(k_e, halfband, N)
        centres = [k for k in k_exc if k_band[0] <= k <= k_band[-1] and k != k_e] # Removes other exited bins from the band
        K, cmap = build_regressors(k_e, k_band,centres, dft_basis, Np, Nq, Na)
        theta, res_abs[e], res_rel[e], rank[e] = fit_band(V, k_e, K, k_band)
        G_p[e] = theta[cmap["target"]] / U[k_e]
    return G_p, basis, res_abs, res_rel, rank

def evaluate_transfer_function(
        G_p:np.ndarray,
        b:np.ndarray,
    ):
    Npp  = G_p.shape[1]
    return G_p @ b[:, :Npp].T