import numpy as np
from numpy.polynomial.legendre import legvander

def legendre_basis(t, T, order):
    """
    Real Legendre polynomial basis b_0..b_order at t given the window duration T,
    using x = 2t/T - 1 (numpy.polynomial.legendre.legvander).

    b_0 = 1, left as-is. b_p, p >= 1, are mean-subtracted over the window: they
    integrate to zero only in continuous time, and on a sampled window B_p(0) != 0,
    so every time-varying regressor would otherwise leak into the DC bin (Sec. 5).

    Returns a (len(t), order+1) real array.
    """
    x = 2 * t / T - 1
    b = legvander(x, order)
    if order >= 1:
        b[:, 1:] -= b[:, 1:].mean(axis=0, keepdims=True)
    return b

#------------------------------------------------------------------------------#

def harmonic_grid_bins(P, N):
    """All harmonic-grid bins {0, P, 2P, ...} strictly below Nyquist (N//2)."""
    return np.arange(0, N // 2, P)

#------------------------------------------------------------------------------#

def _band_bounds(k_exc, e, P):
    """
    Unclipped (raw_lo, raw_hi) local band bounds for excited bin e: half the bin
    distance to each neighbouring excited tone, mirroring the available half-
    distance at the first/last excited tone since they have no neighbour on one
    side (Sec. 2). Falls back to +-P/2 when there is only one excited tone.
    """
    n = len(k_exc)
    if n == 1:
        half = max(1, P // 2)
        return k_exc[0] - half, k_exc[0] + half
    if e == 0:
        dk_plus = (k_exc[1] - k_exc[0]) // 2
        dk_minus = dk_plus
    elif e == n - 1:
        dk_minus = (k_exc[e] - k_exc[e - 1]) // 2
        dk_plus = dk_minus
    else:
        dk_minus = (k_exc[e] - k_exc[e - 1]) // 2
        dk_plus = (k_exc[e + 1] - k_exc[e]) // 2
    return k_exc[e] - dk_minus, k_exc[e] + dk_plus

#------------------------------------------------------------------------------#

def bltva_band_fit(V, I, B_full, k_exc, e, P, N, Np, Nq, Na):
    """
    Fit the local band around one excited bin k_exc[e] (eq. 58): the target tone's
    own Np-order Legendre expansion, an Np-order expansion for every other
    harmonic-grid bin also inside the band, a DC/drift skirt (Nq-order, or
    max(Np,Nq)-order if bin 0 falls within the band's unclipped extent -- see
    Sec. 2 -- to avoid double-counting bin 0's contribution), and an Na-order
    polynomial absorbing transients/out-of-band leakage on the band's own local
    axis (Sec. 3).

    V, I     : full-length positive-and-negative-frequency FFTs (V(k)/N, I(k)/N);
               only bins 1..N//2-1 are ever read from them (Sec. 1)
    B_full   : (N, order_max+1) complex, fft(legendre_basis(...), axis=0)/N,
               kept full-length so B_p(k-k') can be indexed as B_full[(k-k') % N]
               for any centre k' (Sec. 1)
    k_exc    : all excited bins (P*H_exc), sorted ascending
    e        : index into k_exc of the tone being fit
    P, N     : harmonic-grid spacing (bins) and total number of samples
    Np, Nq, Na: polynomial orders, see module docstring

    Returns (Z_p, theta_q, cond_K, residual_rel, k_band):
        Z_p          : (Np+1,) complex, theta_p(k_e) / I(k_e)
        theta_q      : (Nq+1,) complex, drift coefficients (real part is v_drift(t))
        cond_K       : condition number of the band's design matrix
        residual_rel : ||K@theta - V(k_band)|| / ||V(k_band)||
        k_band       : the band's bin indices
    """
    raw_lo, raw_hi = _band_bounds(k_exc, e, P)
    k_lo = max(1, raw_lo)
    k_hi = min(N // 2 - 1, raw_hi)
    k_band = np.arange(k_lo, k_hi + 1)

    k_e = int(k_exc[e])
    K_nl = harmonic_grid_bins(P, N)
    K_nl_w = sorted(set(K_nl.tolist()) & set(k_band.tolist()) - {k_e, 0})

    merge_dc = raw_lo <= 0          # bin 0 falls within the band's unclipped extent
    dc_order = max(Np, Nq) if merge_dc else Nq

    columns = []
    col_ranges = {}

    def add_block(name, center, order):
        start = sum(c.shape[1] for c in columns)
        cols = B_full[:, :order + 1][(k_band - center) % N]
        columns.append(cols)
        col_ranges[name] = slice(start, start + order + 1)

    add_block('target', k_e, Np)
    for k_other in K_nl_w:
        add_block(f'other_{k_other}', k_other, Np)
    add_block('dc', 0, dc_order)

    # Transient / out-of-band term: evaluated directly on the band's own
    # normalised local axis, not as a modulated skirt (Sec. 3).
    x_band = 2 * (k_band - k_band[0]) / (k_band[-1] - k_band[0]) - 1
    add_block_start = sum(c.shape[1] for c in columns)
    transient_cols = legvander(x_band, Na).astype(complex)
    columns.append(transient_cols)
    col_ranges['transient'] = slice(add_block_start, add_block_start + Na + 1)

    K = np.hstack(columns)
    n_theta = K.shape[1]
    if n_theta >= len(k_band):
        raise ValueError(
            f'Band for excited bin {k_e} is underdetermined: {n_theta} unknowns '
            f'>= {len(k_band)} band bins. Widen the band (larger P) or lower '
            'Np/Nq/Na.'
        )
    if n_theta >= 2 * P - 1:
        raise ValueError(
            f'Band for excited bin {k_e} violates n_theta < 2P-1 '
            f'({n_theta} >= {2 * P - 1}); lower Np/Nq/Na or increase P.'
        )

    y = V[k_band]
    theta, _, _, s = np.linalg.lstsq(K, y, rcond=None)
    cond_K = s.max() / s.min() if s.min() > 0 else np.inf
    residual_rel = np.linalg.norm(K @ theta - y) / np.linalg.norm(y)

    theta_p = theta[col_ranges['target']]
    theta_q = theta[col_ranges['dc']][:Nq + 1]
    Z_p = theta_p / I[k_e]

    return Z_p, theta_q, cond_K, residual_rel, k_band

#------------------------------------------------------------------------------#

def bltva_eis_band(V, I, dt, freqs, T_meas):
    H_exc = np.round(freqs / freqs[0]).astype(int)
    P = int(round(freqs[0] * T_meas))
    if V.size % P != 0:
        raise ValueError(
            f'N ({V.size}) is not an integer multiple of P ({P}); bltva_eis_band '
            'requires this. Adjust duration_s/sampling_time_s so that f_min*T_meas '
            'is an exact integer that also divides N.'
        )
    if not np.allclose(P * H_exc, freqs * T_meas, atol=1e-6):
        raise ValueError(
            'multisine_frequencies_Hz do not land exactly on P*H_exc bins; check '
            'the f_min/harmonics convention against T_meas.'
        )
    Np = 1
    Nq = 1
    Na = 1
    N = V.size
    if N % P != 0:
        raise ValueError(f'N ({N}) must be an integer multiple of P ({P})')
    if Np >= P // 3:
        raise ValueError(f'Np ({Np}) must be < floor(P/3) = {P // 3}')

    H_exc = np.asarray(H_exc, dtype=int)
    order = np.argsort(H_exc)
    k_exc = P * H_exc[order]

    T = N * dt
    t = np.arange(N) * dt
    b = legendre_basis(t, T, max(Np, Nq))
    B_full = np.fft.fft(b, axis=0) / N

    n_exc = len(k_exc)
    Z_t = np.zeros((n_exc, N), dtype=complex)
    v_drift_t = np.zeros((n_exc, N), dtype=complex)
    diagnostics = [None] * n_exc

    for e in range(n_exc):
        Z_p, theta_q, cond_K, residual_rel, k_band = bltva_band_fit(
            V, I, B_full, k_exc, e, P, N, Np, Nq, Na,
        )
        Z_t[e] = sum(Z_p[p] * b[:, p] for p in range(Np + 1))
        v_drift_t[e] = sum(theta_q[q] * b[:, q] for q in range(Nq + 1))
        diagnostics[order[e]] = dict(
            k_exc=int(k_exc[e]), Z_p=Z_p, theta_q=theta_q,
            cond_K=cond_K, residual_rel=residual_rel,
            k_band=(int(k_band[0]), int(k_band[-1])),
        )

    # Undo the ascending-order sort so outputs follow the caller's H_exc order.
    Z_t[order] = Z_t.copy()
    v_drift_t[order] = v_drift_t.copy()

    return Z_t, v_drift_t

#------------------------------------------------------------------------------#
