"""Shared simulation fixtures for estimator tests.

Two families live here.

**Stationary fixtures** (``stationary_sim``, ``z_true_stationary``) generate a
noiseless, time-invariant R+RC dataset. Every estimator must recover ``Z_true``
to numerical precision. Parameters are chosen so all excited frequencies land
exactly on DFT bins and the record is an exact number of fundamental periods.

**Oracle fixtures** (``make_oracle`` and the grid helpers) generate data from
the Legendre model, where the time-varying transfer function is prescribed as

    Z(f_k, t) = sum_p  a_p(f_k) * L_p(2t/T - 1)

For a purely linear time-varying system this is not an approximation of the
block-diagram model in Hallemans TIM 2021 Fig. 1 — it is algebraically
identical to it, because b_p(t) is real and the sums commute:

    sum_p b_p(t) * sum_k A_k Re{G_p(jw_k) e^{j0}}
      == sum_k A_k Re{(sum_p G_p(jw_k) b_p(t)) e^{j0}}

so no ODE solve is needed to produce the exact LTV response. BLTVA must
return ``H_p == a_p``, which makes these tests a true oracle: ground truth and
estimator output are the same mathematical objects.

A note on the basis
-------------------
Simulator and estimator share ``bltva.legendre_basis``, which subtracts the
sample mean from every order p >= 1. That is what the paper prescribes
(TIM 2021, after eq. 26: on a finite record ``(1/T) int b_p dt = 0`` does not
hold exactly, so the sampled means are subtracted), and an oracle has to be
built from the estimator's own model rather than a near neighbour of it.

The simulator previously used the raw ``legvander``. The two bases then
differed by O(1/N) per column, which put a floor of roughly ``1/N`` under
every recovery test -- about 2e-05 at N = 100000. With the bases shared, the
noiseless floor is ~1e-07 and the tolerances below are set by noise rather
than by the model.
"""
import numpy as np
import pytest

from deistools.simulation.models.r_rc import simulate_r_rc, impedance as r_rc_impedance
from deistools.simulation.models.legendre import simulate_legendre

# --------------------------------------------------------------------------- #
# Stationary reference dataset
# --------------------------------------------------------------------------- #

R_S = 10.0    # Ohm
R_P = 30.0    # Ohm
C = 1e-3      # F

F_MIN = 1.0                                               # Hz
H_EXC = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])    # harmonic indices
FREQUENCIES = H_EXC * F_MIN                               # Hz
AMPLITUDES = 0.01 * np.ones_like(FREQUENCIES)             # 10 mA
PHASES = np.zeros_like(FREQUENCIES)
FS = 10_000.0                                             # sampling freq, Hz
DT = 1.0 / FS
P_PERIODS = 10                                            # number of full periods
DURATION = P_PERIODS / F_MIN                              # 10 s


@pytest.fixture
def stationary_sim():
    """Stationary R+RC simulation: no noise, no time variation."""
    return simulate_r_rc(
        DURATION, DT, FREQUENCIES, AMPLITUDES, PHASES,
        R_S, R_P, C,
        noise_std_V=0.0, noise_std_A=0.0, seed=42,
    )


@pytest.fixture
def z_true_stationary():
    """Analytic stationary impedance at the excited frequencies, shape (n_freq,)."""
    return r_rc_impedance(FREQUENCIES, R_S, R_P, C)[:, 0]


# --------------------------------------------------------------------------- #
# Excitation grids
#
# These are test helpers, not the eventual deistools excitation module. They
# cover the two grid families that stress different code paths in BLTVA:
# a dense odd-linear grid (uniform bin spacing, the neighbour-distance band
# rule applies naturally) and a sparse quasi-log grid (spacing varies by
# orders of magnitude, which forces an explicit band_halfwidth).
# --------------------------------------------------------------------------- #

def odd_linear_grid(h_max=59, L=5, seed=0):
    """Odd harmonics up to ``h_max``, one randomly omitted per group of ``L``.

    This is the harmonic-grid half of the odd-random-phase construction
    (Hallemans TIM 2021, Definition 1). The random *phases* belong to the
    signal, not the grid, so they are applied separately.
    """
    rng = np.random.default_rng(seed)
    odd = np.arange(1, h_max + 1, 2)
    keep = []
    for i in range(0, odd.size, L):
        group = odd[i:i + L]
        if group.size < L:
            keep.extend(group.tolist())
        else:
            keep.extend(np.delete(group, rng.integers(group.size)).tolist())
    return np.array(sorted(keep), dtype=int)


def quasi_log_odd_grid(n=15, h_max=999):
    """``n`` strictly increasing odd harmonics, quasi-logarithmically spaced.

    Matches electrochemical practice (and the VUB group's own measurement
    examples, which are all quasi-log) rather than the dense linear grids
    used in their simulations. Strict increase guards against two tones
    landing on the same or adjacent bins, whose skirts would overlap inside
    a single band.
    """
    out = []
    for h in np.logspace(0, np.log10(h_max), n):
        k = 2 * int(round((h - 1) / 2)) + 1          # nearest odd
        if out and k <= out[-1]:
            k = out[-1] + 2
        out.append(k)
    return np.array(out, dtype=int)


GRIDS = {
    "odd_linear_L5": odd_linear_grid(h_max=59, L=5, seed=0),
    "quasi_log_15": quasi_log_odd_grid(n=15, h_max=999),
}


# --------------------------------------------------------------------------- #
# Legendre oracle
# --------------------------------------------------------------------------- #

ORACLE_FS = 50.0      # Hz   -> Nyquist 25 Hz, above the 9.99 Hz top tone
ORACLE_TP = 100.0     # s    multisine period
ORACLE_P = 20         # periods -> T = 2000 s, N = 100_000


def rrc_legendre_coefficients(freqs, R_s=R_S, R_p=R_P, C_dl=C, variation=(0.30, 0.10)):
    """Prescribed a_p(f) for an R+RC whose impedance varies smoothly in time.

    ``a_0`` is the stationary R+RC spectrum; each entry of ``variation`` scales
    a higher Legendre order relative to it, so the time variation is a fixed
    fraction of the impedance at every frequency.
    """
    freqs = np.asarray(freqs, dtype=float)
    w = 2 * np.pi * freqs
    a0 = R_s + R_p / (1 + 1j * w * R_p * C_dl)
    a = np.zeros((freqs.size, 1 + len(variation)), dtype=complex)
    a[:, 0] = a0
    for p, scale in enumerate(variation, start=1):
        a[:, p] = scale * a0
    return a


class Oracle:
    """A Legendre-oracle dataset together with its ground truth."""

    def __init__(self, H_exc, a, drift_coefficients, sim, fs, P, Tp):
        self.H_exc = H_exc
        self.a = a                              # (M, Np_true+1) prescribed
        self.drift_coefficients = drift_coefficients
        self.sim = sim
        self.fs = fs
        self.P = P
        self.Tp = Tp
        self.freqs = H_exc / Tp
        self.N = sim['time_s'].size
        self.T = self.N / fs

    @property
    def voltage(self):
        return self.sim['voltage_V']

    @property
    def current(self):
        return self.sim['current_A']

    @property
    def drift_true(self):
        return self.sim['drift_V']

    def relative_error(self, H_p):
        """|H_p - a_p| normalised by |a_0|, over the orders a prescribes."""
        n = self.a.shape[1]
        return np.abs(H_p[:, :n] - self.a) / np.abs(self.a[:, [0]])


@pytest.fixture
def make_oracle():
    """Factory for Legendre-oracle datasets.

    ``make_oracle(grid="quasi_log_15", variation=(0.3, 0.1), drift=None,
    noise_std_V=0.0, amplitude=0.01, seed=0)`` -> :class:`Oracle`.
    """
    def _make(grid="quasi_log_15", variation=(0.30, 0.10), drift=None,
              noise_std_V=0.0, noise_std_A=0.0, amplitude=0.01, seed=0,
              fs=ORACLE_FS, Tp=ORACLE_TP, P=ORACLE_P):
        H_exc = GRIDS[grid] if isinstance(grid, str) else np.asarray(grid, dtype=int)
        freqs = H_exc / Tp
        if freqs.max() >= fs / 2:
            raise ValueError("top excited tone is at or above Nyquist")

        N = int(fs * Tp) * P
        dt, T = 1.0 / fs, N / fs
        amps = np.full(freqs.size, amplitude, dtype=float)
        phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, freqs.size)

        a = rrc_legendre_coefficients(freqs, variation=variation)
        sim = simulate_legendre(T, dt, freqs, amps, phases, a,
                                drift_coefficients=drift,
                                noise_std_V=noise_std_V, noise_std_A=noise_std_A,
                                seed=seed + 1)
        return Oracle(H_exc, a, drift, sim, fs, P, Tp)

    return _make
