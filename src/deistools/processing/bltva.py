"""
BLTVA — Band-Limited Time-Varying Analysis estimator.

Estimates the time-varying transfer function H(w,t) between an excitation signal and a
response signal measured under (odd random phase) multisine excitation, following

    Hallemans et al., Electrochim. Acta 466 (2023) 142939, eq. (71)-(81)
    Hallemans et al., IEEE TIM 70 (2021) 6503814, eq. (55), (58), Algorithm 2
    Breugelmans et al., Electrochim. Acta 76 (2012) 375, eq. (1)-(14)   [LTV-only variant]

Model, per excited bin k_e, over a local band K_w:

    V(k) = sum_p     theta_p(k_e) B_p(k - k_e)          # target skirt
         + sum_{k'}  sum_p theta_p(k') B_p(k - k')      # other harmonics in band
         + sum_q     theta_q B_q(k)                     # drift skirt at DC
         + sum_r     alpha_r L_r(xi(k))                 # transients + out-of-band

    H_p(w_ke) = theta_p(k_e) / U(k_e)
    H(w_ke, t) = sum_p H_p(w_ke) b_p(t)

with the scaled DFT X(k) = (1/N) sum_n x(n) exp(-j2 pi k n / N) and Legendre basis
b_p(t) = L_p(2t/T - 1).

Naming is deliberately excitation/response rather than current/voltage: in galvanostatic
mode pass (excitation=i, response=v) and read the result as an impedance; in potentiostatic
mode pass (excitation=v, response=i) and read it as an admittance.

Not included (deliberately): nonlinear distortion variances, uncertainty bounds,
differencing-based detrending.

Dependencies: numpy, matplotlib.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.legendre import legvander
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

__all__ = [
    "BLTVAResult", "bltva", "legendre_basis", "legendre_spectra",
    "visualize_bltva_fit",
]


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class BLTVAResult:
    """Output of :func:`bltva`.

    Attributes
    ----------
    f_exc : (M,) float
        Excited frequencies in Hz.
    t : (N,) float
        Time axis in s.
    H_p : (M, Np+1) complex
        Basis coefficients of the transfer function at each excited frequency.
        ``H_p[:, 0]`` is the best stationary (time-averaged) transfer function.
    H_wt : (M, N) complex
        Time-varying transfer function, reconstructed on the full time axis.
    drift : (N,) float
        Reconstructed drift of the response signal, from the lowest band.
        All-zero if no band contained a drift block.
    theta_q : (Nq+1,) complex or None
        Raw drift coefficients. Should be near-real; the imaginary part is a
        diagnostic (the drift of a real signal is real).
    diagnostics : list of dict
        One entry per excited frequency: band width, number of unknowns,
        condition number, numerical rank, relative residual.
    """

    f_exc: np.ndarray
    t: np.ndarray
    H_p: np.ndarray
    H_wt: np.ndarray
    drift: np.ndarray
    theta_q: np.ndarray | None
    diagnostics: list = field(default_factory=list)

    def worst_condition(self) -> float:
        return max(d["cond"] for d in self.diagnostics)

    def worst_residual(self) -> float:
        return max(d["rel_residual"] for d in self.diagnostics)


# --------------------------------------------------------------------------- #
# Basis functions
# --------------------------------------------------------------------------- #

def legendre_basis(N: int, order: int) -> np.ndarray:
    """Rescaled Legendre basis b_p(t) = L_p(2t/T - 1), sampled on N points.

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


# --------------------------------------------------------------------------- #
# Band construction
# --------------------------------------------------------------------------- #

def _band_bins(k_exc: np.ndarray, e: int, N: int, halfwidth: int | None) -> np.ndarray:
    """DFT bins of the local band centred on ``k_exc[e]``.

    halfwidth=None reproduces Hallemans TIM 2021 eq. (55): edges at half the distance
    to the neighbouring excited bins (bands tile the spectrum). For sparse,
    log-spaced multisines prefer a fixed halfwidth of a few multiples of P, otherwise
    the high-frequency bands become enormous and pull in many nuisance harmonics.
    """
    k_e = int(k_exc[e])
    if halfwidth is not None:
        lo, hi = k_e - halfwidth, k_e + halfwidth
    elif k_exc.size == 1:
        raise ValueError("halfwidth must be given when only one frequency is excited")
    else:
        prev = int(k_exc[e - 1]) if e > 0 else 2 * k_e - int(k_exc[1])
        nxt = int(k_exc[e + 1]) if e < k_exc.size - 1 else 2 * k_e - int(k_exc[-2])
        lo, hi = k_e - (k_e - prev) // 2, k_e + (nxt - k_e) // 2
    lo = max(1, lo)                 # bin 0 carries the DC offset of the response
    hi = min(N // 2 - 1, hi)        # positive frequencies only
    if hi <= lo:
        raise ValueError(f"empty band at excited bin {k_e}")
    return np.arange(lo, hi + 1)


def build_regressors(K_w, k_e, centres, B, Np, Nq, Na, include_drift):
    N = B.shape[0]
    blocks, cmap, j = [], {}, 0

    def add(name, block):
        nonlocal j
        cmap[name] = slice(j, j + block.shape[1])
        blocks.append(block)
        j += block.shape[1]

    add("target", B[(K_w - k_e) % N, : Np + 1])
    for kp in centres:
        add(f"h{kp}", B[(K_w - kp) % N, : Np + 1])

    # q = 0 is identically zero off DC; its coefficient is the mean, V[0].
    # Only include the drift where its skirt still carries shape information.
    if include_drift and Nq >= 1:
        add("drift", B[K_w, 1 : Nq + 1])

    xi = 2.0 * (K_w - K_w[0]) / (K_w[-1] - K_w[0]) - 1.0
    add("transient", legvander(xi, Na).astype(complex))
    return np.hstack(blocks), cmap


# --------------------------------------------------------------------------- #
# Shared setup and per-band fit (used by both the estimator and its visualizer)
# --------------------------------------------------------------------------- #

def _prepare(response, excitation, fs, P, H_exc, Np, Nq, harmonics):
    """Validate inputs and precompute the arrays shared by every band fit.

    Returns N, T, t, k_exc, all_centres, b, B, V, U.
    """
    response = np.asarray(response, dtype=float)
    excitation = np.asarray(excitation, dtype=float)
    if response.shape != excitation.shape or response.ndim != 1:
        raise ValueError("response and excitation must be 1-D arrays of equal length")

    N = response.size
    if N % P:
        raise ValueError(
            f"N={N} is not an integer number of periods of P={P}. Trim the record "
            "to P*(fs*Tp) samples before calling; a non-integer record leaks and the "
            "leakage is indistinguishable from time variation."
        )
    if harmonics not in ("all_nl", "excited"):
        raise ValueError("harmonics must be 'all_nl' or 'excited'")

    T = N / fs
    t = np.arange(N) / fs

    H_exc = np.asarray(H_exc, dtype=int)
    k_exc = P * H_exc                                   # excited DFT bins, 0-based
    if k_exc.max() >= N // 2:
        raise ValueError("an excited harmonic lies at or above Nyquist")
    if Np >= P // 3:
        warnings.warn(f"Np={Np} violates Np < floor(P/3) = {P // 3}; "
                      "skirts cannot be resolved with this many periods")

    b = legendre_basis(N, max(Np, Nq))                  # (N, max+1) real
    B = legendre_spectra(b)                             # (N, max+1) complex
    V = np.fft.fft(response) / N
    U = np.fft.fft(excitation) / N

    all_centres = k_exc if harmonics == "excited" else np.arange(0, N // 2, P)

    return N, T, t, k_exc, all_centres, b, B, V, U


def _fit_band(e, k_exc, N, band_halfwidth, all_centres, B, Np, Nq, Na, drift_bands, V, rcond):
    """Solve the local least-squares problem for one excited frequency.

    Extracted from ``bltva``'s main loop so both the estimator and
    ``visualize_bltva_fit`` share the exact same fitting logic.
    """
    k_e = int(k_exc[e])
    K_w = _band_bins(k_exc, e, N, band_halfwidth)
    centres = [int(kp) for kp in all_centres
               if K_w[0] <= kp <= K_w[-1] and kp != k_e]

    K, cmap = build_regressors(K_w, k_e, centres, B, Np, Nq, Na,
                       include_drift=(e < drift_bands))
    n_theta = K.shape[1]
    if n_theta >= K_w.size:
        raise ValueError(
            f"band at bin {k_e}: {n_theta} unknowns for {K_w.size} equations. "
            "Widen the band, lower Np/Na, or use harmonics='excited'."
        )

    V_w = V[K_w]

    # Column equilibration: the skirt columns and the transient polynomial differ
    # by orders of magnitude in norm, which inflates the condition number for no
    # good reason. Scale to unit norm, solve, scale back.
    scale = np.linalg.norm(K, axis=0)
    scale[scale == 0.0] = 1.0
    Ks = K / scale

    theta, _, rank, sv = np.linalg.lstsq(Ks, V_w, rcond=rcond)
    cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
    theta = theta / scale

    fit = K @ theta
    residual = V_w - fit

    return {
        "k_e": k_e, "K_w": K_w, "centres": centres, "K": K, "cmap": cmap,
        "V_w": V_w, "theta": theta, "fit": fit, "residual": residual,
        "cond": float(cond), "rank": int(rank), "n_theta": int(n_theta),
    }


def _reconstruct_drift(b, Nq, theta_q, V0):
    """Time-domain drift from its Legendre coefficients: b_{1..Nq} @ theta_q + V0.

    Returns the complex reconstruction; the imaginary part is a diagnostic
    only (the drift of a real signal must be real).
    """
    return b[:, 1 : Nq + 1] @ theta_q + V0.real


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #

def bltva(response, excitation, fs, P, H_exc, *,
          Np=6, Nq=5, Na=6, drift_bands=2, harmonics="all_nl",
          band_halfwidth=None, rcond=None):
    """Estimate the time-varying transfer function H(w_k, t).

    Parameters
    ----------
    response, excitation : (N,) real
        Measured time series. Galvanostatic: response=voltage, excitation=current.
    fs : float
        Sampling frequency [Hz].
    P : int
        Number of measured multisine periods. Must divide N exactly.
    H_exc : sequence of int
        Excited harmonic numbers of the multisine (h_m, so f_m = h_m/T_p).
    Np : int
        Order of the Legendre expansion of the transfer function. Requires
        Np < floor(P/3). Controls time resolution.
    Nq : int
        Order of the Legendre expansion of the drift.
    Na : int
        Order of the transient / out-of-band polynomial.
    harmonics : {"all_nl", "excited"}
        "all_nl"  -> regressors at every harmonic {0, P, 2P, ...} in band. This is the
                     BLTVA of Hallemans TIM 2021; needed when odd nonlinear distortion
                     is present, since its skirts sit on non-excited harmonics.
        "excited" -> regressors only at excited bins. The LTV estimator of
                     Breugelmans 2012 eq. (11). Far fewer unknowns; correct when the
                     response is linear.
    band_halfwidth : int or None
        Band half-width in DFT bins. None uses the half-distance-to-neighbours rule.
        For sparse log-spaced multisines, a few multiples of P (e.g. 5*P) is better.
    rcond : float or None
        Singular-value truncation passed to ``np.linalg.lstsq``.

    Returns
    -------
    BLTVAResult
    """
    N, T, t, k_exc, all_centres, b, B, V, U = _prepare(
        response, excitation, fs, P, H_exc, Np, Nq, harmonics)

    # --- sweep the excited frequencies -------------------------------------
    M = k_exc.size
    H_p = np.zeros((M, Np + 1), dtype=complex)
    theta_q = None
    diagnostics = []

    for e in range(M):
        fit = _fit_band(e, k_exc, N, band_halfwidth, all_centres, B,
                         Np, Nq, Na, drift_bands, V, rcond)
        cmap, theta, K_w = fit["cmap"], fit["theta"], fit["K_w"]

        H_p[e] = theta[cmap["target"]] / U[fit["k_e"]]
        if e == 0 and "drift" in cmap:
            theta_q = theta[cmap["drift"]].copy()

        diagnostics.append({
            "k_e": fit["k_e"],
            "f_exc": fit["k_e"] / T,
            "N_w": int(K_w.size),
            "n_theta": fit["n_theta"],
            "n_centres": len(fit["centres"]),
            "cond": fit["cond"],
            "rank": fit["rank"],
            "rel_residual": float(np.linalg.norm(fit["residual"]) / np.linalg.norm(fit["V_w"])),
        })

    # --- reconstruct over time ---------------------------------------------
    H_wt = H_p @ b[:, : Np + 1].T                       # (M, N)

    if theta_q is None:
        drift = np.zeros(N)
    else:
        drift_c = _reconstruct_drift(b, Nq, theta_q, V[0])
        rel_imag = np.abs(drift_c.imag).max() / max(np.abs(drift_c.real).max(), 1e-300)
        if rel_imag > 1e-3:
            warnings.warn(f"drift has a large imaginary part ({rel_imag:.2e} of the "
                          "real part); the drift of a real signal must be real")
        drift = drift_c.real

    return BLTVAResult(
        f_exc=k_exc / T, t=t, H_p=H_p, H_wt=H_wt,
        drift=drift, theta_q=theta_q, diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# Visual check of the per-band least-squares fit
# --------------------------------------------------------------------------- #

def visualize_bltva_fit(response, excitation, fs, P, H_exc, *,
                       Np=6, Nq=5, Na=6, drift_bands=2, harmonics="all_nl",
                       band_halfwidth=None, rcond=None):
    """Slider-based visual check of the band-limited least-squares fit.

    For each excited frequency (selected with a slider, as in
    :func:`deistools.processing.dmfa_functions.visualize_dmfa_filtering`),
    shows:

    - top: the local DFT band data ``|V(k)|`` against the least-squares model
      ``|K @ theta|``, restricted to the band actually used for that
      frequency.
    - bottom: the fitted target-block Legendre coefficients ``|theta_p|``
      (the ones that become ``H_p``), which exposes overfitting or
      ill-conditioning as unreasonably large high-order coefficients.

    Parameters are identical to :func:`bltva`, so the same call used
    to produce a result can be reused here to inspect it.

    Returns
    -------
    fig, axs
    """
    N, T, _t, k_exc, all_centres, _b, B, V, U = _prepare(
        response, excitation, fs, P, H_exc, Np, Nq, harmonics)
    M = k_exc.size

    def plot_band(e):
        fit = _fit_band(e, k_exc, N, band_halfwidth, all_centres, B,
                         Np, Nq, Na, drift_bands, V, rcond)
        f_band = fit["K_w"] / T
        f_exc = fit["k_e"] / T
        theta_p = fit["theta"][fit["cmap"]["target"]]

        axs[0].plot(f_band, np.abs(fit["V_w"]), '-o', color='C0', label='Data')
        axs[0].plot(f_band, np.abs(fit["fit"]), 'x', color='C1', label='Fit')
        axs[0].axvline(f_exc, color='green', linestyle='--')
        axs[0].set_xlabel('Frequency / Hz')
        axs[0].set_ylabel('|V(k)|')
        axs[0].set_title(
            f'Excited frequency {f_exc:.4g} Hz   '
            f'cond={fit["cond"]:.2e}   rel. residual={np.linalg.norm(fit["residual"]) / np.linalg.norm(fit["V_w"]):.2e}'
        )
        axs[0].legend()

        axs[1].bar(np.arange(theta_p.size), np.abs(theta_p), color='C0')
        axs[1].set_yscale('log')
        axs[1].set_xlabel('Legendre order p')
        axs[1].set_ylabel(r'$|\theta_p|$ (target block)')

    def update(val):
        for ax in axs:
            ax.clear()
        plot_band(val)
        fig.canvas.draw_idle()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.subplots_adjust(bottom=0.25)

    sliders_freq_ax = fig.add_axes([0.20, 0.15, 0.60, 0.03])
    slider_frequency = Slider(
        ax=sliders_freq_ax,
        label="Excited freq. index",
        valmin=0,
        valmax=M - 1,
        valinit=0,
        valstep=1,
    )

    plot_band(slider_frequency.val)
    slider_frequency.on_changed(update)
    plt.show()

    return fig, axs


# --------------------------------------------------------------------------- #
# Minimal self-test: LTI ground truth, no drift.
# The estimator must return H_0 = V(k_e)/U(k_e) exactly and H_p = 0 for p >= 1.
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    fs, Tp, P = 200.0, 180.0, 30
    N = int(fs * Tp) * P
    T = N / fs
    t = np.arange(N) / fs

    H_exc = np.array([1, 3, 5, 9, 13, 21, 33, 51, 81, 129])
    A = 0.05 / H_exc ** 0.4
    phi = rng.uniform(0, 2 * np.pi, H_exc.size)

    def Z_true(f):                                   # simple R + R||C
        w = 2 * np.pi * f
        return 0.02 + 0.01 / (1 + 1j * w * 0.5)

    exc = np.zeros(N)
    resp = np.zeros(N)
    for h, a, p in zip(H_exc, A, phi):
        f = h / Tp
        z = Z_true(f)
        exc += a * np.cos(2 * np.pi * f * t + p)
        resp += a * np.abs(z) * np.cos(2 * np.pi * f * t + p + np.angle(z))

    r = bltva(resp, exc, fs, P, H_exc,
              Np=4, Nq=4, Na=6, harmonics="excited", band_halfwidth=5 * P)

    Z_ref = Z_true(r.f_exc)
    err0 = np.abs(r.H_p[:, 0] - Z_ref) / np.abs(Z_ref)
    leak = np.abs(r.H_p[:, 1:]).max(axis=1) / np.abs(r.H_p[:, 0])

    print(f"max relative error on H_0 : {err0.max():.3e}   (target < 1e-10)")
    print(f"max spurious |H_p|/|H_0|  : {leak.max():.3e}   (target < 1e-10)")
    print(f"worst cond(K)             : {r.worst_condition():.3e}")
    print(f"worst relative residual   : {r.worst_residual():.3e}")

    # Visual check of the per-band fit on the same signal, ground truth is LTI so
    # the fit should overlay the data exactly and theta_p>=1 should sit at the
    # numerical noise floor.
    visualize_bltva_fit(resp, exc, fs, P, H_exc,
                        Np=4, Nq=4, Na=6, harmonics="excited", band_halfwidth=5 * P)