"""
Block-oriented nonlinear time-varying systems, Hallemans TIM 2021 Section II.

This is the system class the BLTVA is actually derived for: parallel branches,
each a *nonlinear time-invariant* block followed by a time-varying gain
(Fig. 2)::

    y(t) = sum_p  NL_p{u(t)} * b_p(t)

with each branch a Wiener-Hammerstein structure (Fig. 7)::

    y_p(t) = f2_p( L^-1{ H_p(s) L{ f1_p(u(t)) } } )

a static nonlinearity, an LTI filter, then another static nonlinearity.
Choosing which of the two nonlinearities is the identity gives a Hammerstein
branch, a Wiener branch, or a purely linear one.

Why this model and not the frozen-time one
------------------------------------------
``legendre.py`` and ``r_rc_transistor.py`` both prescribe an impedance and
synthesise a response from it. Neither can produce *nonlinear distortion*, so
neither can exercise Section V, and neither has a ground truth for the best
linear approximation beyond "the impedance we prescribed".

Here the ground truth has to be computed, because the BLA of a nonlinear
block is not its linear part. The static nonlinearities move it. The paper
obtains it by averaging ``Y(k)/U(k)`` over many random phase realisations of
the input, which is what :func:`best_linear_approximation` does. That gives a
genuine target for ``H_p`` and, from the residuals, for the distortion
variances.

Filtering without an ODE solver
-------------------------------
``f1_p(u)`` is a static function of a periodic multisine, so it is periodic
with the same period, and so is the filter's steady-state response. Filtering
in the DFT domain is therefore *exact* for the periodic steady state, needs
no integrator, and no scipy::

    Y(k) = H(j*2*pi*f_k) * X(k)

The one thing it omits is the startup transient, which the paper models
separately with the ``alpha_r`` polynomial rather than treating as signal.
Use ``r_rc_transistor`` when the transient itself is the object of study.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..engine import generate_multisine
from deistools.processing.bltva import legendre_basis


@dataclass
class SecondOrderLowpass:
    """``H(s) = dc_gain * wn**2 / (s**2 + 2*zeta*wn*s + wn**2)``.

    ``wn`` is in **rad/s**. The paper quotes its natural frequencies in Hz
    ("wn,0 = 0.7 Hz") while naming them ``wn``, which is ambiguous; use
    :meth:`from_hz` to be explicit about which you mean.
    """

    wn: float
    zeta: float
    dc_gain: float = 1.0

    @classmethod
    def from_hz(cls, f_n, zeta, dc_gain=1.0):
        return cls(2 * np.pi * f_n, zeta, dc_gain)

    def response(self, f):
        """``H(j*2*pi*f)``, conjugate-symmetric in ``f`` so the filtered
        signal comes back real."""
        s = 2j * np.pi * np.asarray(f, dtype=float)
        return self.dc_gain * self.wn ** 2 / (s ** 2 + 2 * self.zeta * self.wn * s + self.wn ** 2)


@dataclass
class Branch:
    """One parallel branch: ``f2( filter( f1(u) ) )``, scaled by ``b_p(t)``.

    ``f1`` and ``f2`` default to the identity, so a branch is linear unless
    you say otherwise. ``name`` is only for reporting.
    """

    filter: SecondOrderLowpass
    f1: Callable[[np.ndarray], np.ndarray] = field(default=lambda x: x)
    f2: Callable[[np.ndarray], np.ndarray] = field(default=lambda x: x)
    name: str = ""

    @property
    def is_linear(self):
        probe = np.linspace(-2.0, 2.0, 41)
        return (np.allclose(self.f1(probe), probe)
                and np.allclose(self.f2(probe), probe))

    def apply(self, u, fs):
        """Branch output before the time-varying gain."""
        x = np.asarray(self.f1(np.asarray(u, dtype=float)), dtype=float)
        f = np.fft.fftfreq(x.size, 1.0 / fs)
        filtered = np.fft.ifft(np.fft.fft(x) * self.filter.response(f)).real
        return np.asarray(self.f2(filtered), dtype=float)


def hallemans_2021_system():
    """The three branches of the simulation in Section VII, eq. (91d)-(91g).

    ``NL_0`` is Wiener-Hammerstein (nonlinear either side of the filter),
    ``NL_1`` Hammerstein (nonlinear before), ``NL_2`` Wiener (nonlinear
    after). The distortion of each is weak -- the paper reports the variances
    at least 20 dB below the linear part -- which is the regime where a linear
    approximation is worth making at all.

    Natural frequencies are read as Hz, as printed.
    """
    return [
        Branch(SecondOrderLowpass.from_hz(0.7, 0.15, 1.0),
               f1=lambda x: x + 0.02 * np.tanh(x - 2),
               f2=lambda x: x + 0.03 * np.tanh(x - 1),
               name="NL_0 Wiener-Hammerstein"),
        Branch(SecondOrderLowpass.from_hz(1.0, 0.10, 0.2),
               f1=lambda x: x + 0.08 * x ** 2,
               name="NL_1 Hammerstein"),
        Branch(SecondOrderLowpass.from_hz(1.5, 0.15, 0.2),
               f2=lambda x: x + 0.15 * x ** 3,
               name="NL_2 Wiener"),
    ]


def simulate_block_oriented(duration_s, sampling_time_s, frequencies,
                            amplitudes, phases, branches,
                            trend_coefficients=None, noise_std_V=0.0,
                            noise_std_A=0.0, seed=None):
    """Simulate ``y(t) = y_trend(t) + sum_p NL_p{u(t)} b_p(t) + v(t)``.

    Mirrors eq. (90)-(91). Branch ``p`` is multiplied by the ``p``-th Legendre
    basis function, so the branch list defines ``Np`` implicitly.

    Returns the usual ``time_s``/``voltage_V``/``current_A``/``frequencies_Hz``
    plus ``H_linear`` -- the branch filters evaluated at the excited
    frequencies, shape ``(n_freq, n_branches)``. That is the *linear part*,
    which equals the BLA only for a linear system; use
    :func:`best_linear_approximation` for the true target.
    """
    rng = np.random.default_rng(seed)
    N = int(round(duration_s / sampling_time_s))
    fs = 1.0 / sampling_time_s
    t = np.arange(N) * sampling_time_s
    frequencies = np.asarray(frequencies, dtype=float)

    current_A = generate_multisine(t, frequencies, np.asarray(amplitudes),
                                   np.asarray(phases))

    b = legendre_basis(N, max(len(branches) - 1, 0))
    voltage_V = np.zeros(N)
    for p, branch in enumerate(branches):
        voltage_V += branch.apply(current_A, fs) * b[:, p]

    drift_V = np.zeros(N)
    if trend_coefficients is not None:
        q = np.asarray(trend_coefficients, dtype=float)
        drift_V = legendre_basis(N, q.size - 1) @ q
        voltage_V = voltage_V + drift_V

    if noise_std_V > 0:
        voltage_V = voltage_V + rng.normal(0.0, noise_std_V, N)
    if noise_std_A > 0:
        current_A = current_A + rng.normal(0.0, noise_std_A, N)

    H_linear = np.stack([br.filter.response(frequencies) for br in branches],
                        axis=1)
    return {
        'time_s': t,
        'voltage_V': voltage_V,
        'current_A': current_A,
        'frequencies_Hz': frequencies,
        'H_linear': H_linear,
        'drift_V': drift_V,
        'branches': branches,
    }


def best_linear_approximation(branches, frequencies, amplitudes,
                              sampling_time_s, duration_s,
                              n_realisations=100, seed=None):
    """Monte Carlo BLA of each branch, as the paper computes its ground truth.

    For every random phase realisation of the multisine, each branch is driven
    on its own and ``Y_p(k)/U(k)`` taken at the excited bins; the mean over
    realisations is ``G_p``. Random phases are what make this converge: the
    distortion is uncorrelated with the input and averages away, while the
    linear part does not.

    Returns ``(G, scatter)``, each ``(n_freq, n_branches)`` -- the BLA and the
    standard error of its mean, so a caller can see whether
    ``n_realisations`` was enough.
    """
    rng = np.random.default_rng(seed)
    N = int(round(duration_s / sampling_time_s))
    fs = 1.0 / sampling_time_s
    t = np.arange(N) * sampling_time_s
    frequencies = np.asarray(frequencies, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    bins = np.round(frequencies * duration_s).astype(int)

    ratios = np.empty((n_realisations, frequencies.size, len(branches)),
                      dtype=complex)
    for r in range(n_realisations):
        phases = rng.uniform(0, 2 * np.pi, frequencies.size)
        u = generate_multisine(t, frequencies, amplitudes, phases)
        U = np.fft.fft(u)[bins]
        for p, branch in enumerate(branches):
            ratios[r, :, p] = np.fft.fft(branch.apply(u, fs))[bins] / U

    G = ratios.mean(axis=0)
    scatter = ratios.std(axis=0, ddof=1) / np.sqrt(n_realisations)
    return G, scatter


def frozen_impedance(G, N):
    """``Z(f, t) = sum_p G_p(f) b_p(t)``, shape ``(n_freq, N)``.

    The time-varying transfer function a BLTVA of this system should return.
    Identical in form to the Legendre model, which is the point: a *linear*
    block-oriented system is exactly that model with ``a_p(f) = H_p(j2*pi*f)``.
    """
    G = np.atleast_2d(G)
    return G @ legendre_basis(N, G.shape[1] - 1).T
