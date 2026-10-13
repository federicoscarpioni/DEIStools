"""Do the estimators agree, and are they being compared on the same terms?

This is deliberately *not* a benchmark. Which method performs best under which
conditions is a research question with no fixed answer, and it lives in
``examples/compare_estimators.py``. What is tested here is the much narrower
thing a benchmark depends on: that a comparison between the three is apples to
apples at all.

Two ways it could silently not be.

**Time alignment.** The three report on different axes. BLTVA gives
``H_wt`` at every sample; STFT gives one value per window, timestamped at the
window *centre*; DMFA gives ``Npts_elab`` values spanning the record, at a
step of ``T/Npts_elab``. Comparing them means mapping onto a common axis, and
a half-window offset looks exactly like a systematic method difference. The
convention is pinned below by showing the centre beats the window edge.

**Scaling.** All three must return the same impedance, not one of them an
admittance or a factor of N out. Pairwise agreement catches a normalisation
change in any one of them.
"""
import warnings

import numpy as np
import pytest
from numpy.fft import fft, fftshift, fftfreq
from numpy.polynomial.legendre import legvander

from deistools.processing.bltva import bltva
from deistools.processing.stft_functions import estimate_impedance
from deistools.processing.dmfa_functions import (
    extract_impedance, max_elaboration_points, elaboration_frequency_axis,
    elaboration_time_axis,
)
from deistools.processing.passband_filters import create_fermi_dirac_fun
from deistools.simulation.models.legendre import simulate_legendre
from deistools.simulation.models.r_rc import simulate_r_rc, impedance as r_rc_impedance
from tests.conftest import rrc_legendre_coefficients, R_S, R_P, C

# A grid that suits all three at once: odd harmonics on exact DFT bins of a
# one-period STFT window, low enough for DMFA's band to fit, and P periods
# for BLTVA.
FS, TP, P = 50.0, 20.0, 20
N = int(FS * TP) * P
T = N / FS
DT = 1.0 / FS
H = np.array([1, 3, 5, 9, 15, 25, 41, 59])
FREQS = H / TP
AMPS = np.full(H.size, 0.01)
PHASES = np.random.default_rng(0).uniform(0, 2 * np.pi, H.size)

# Tolerances are per method and reflect what each can actually do here, not a
# common standard: STFT averages over its window and so cannot report an
# instantaneous value, and DMFA's pointwise V/I division throws outliers near
# the current zero crossings.
TOL_BLTVA, TOL_STFT, TOL_DMFA_MEDIAN = 1e-4, 2e-2, 1e-3


def _simulate(variation, noise=1e-4):
    a = rrc_legendre_coefficients(FREQS, variation=variation)
    sim = simulate_legendre(T, DT, FREQS, AMPS, PHASES, a,
                            drift_coefficients=None, noise_std_V=noise,
                            noise_std_A=0.0, seed=1)
    return a, sim


def _z_true_at(a, t):
    """Ground truth Z(f, t) from the prescribed Legendre coefficients."""
    tau = 2.0 * np.asarray(t, dtype=float) / T - 1.0
    return a @ legvander(tau, a.shape[1] - 1).T


def _run_stft(sim):
    return estimate_impedance(sim['voltage_V'], sim['current_A'], FREQS, TP, DT)


def _run_bltva(sim, Np=2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bltva(sim['voltage_V'], sim['current_A'], FS, P, H,
                     Np=Np, Nq=5, harmonics="excited", band_halfwidth=5 * P)


def _run_dmfa(sim):
    ftv = fftshift(fft(sim['voltage_V'])) / N
    fti = fftshift(fft(sim['current_A'])) / N
    faxis = fftshift(fftfreq(N, DT))
    f0 = int(np.where(faxis == 0)[0][0])
    idx = (np.round(FREQS * N * DT) + f0).astype(int).tolist()
    npts = max_elaboration_points(idx, N)
    bw = float(np.diff(FREQS).min()) / 2.0
    filt = create_fermi_dirac_fun(elaboration_frequency_axis(npts, T), 0, bw, 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = extract_impedance(ftv, fti, idx, filt, npts)
    return z, elaboration_time_axis(npts, T)


def _sample_at(times, axis_step, n_max):
    """Nearest index on a uniform axis of the given step."""
    return np.round(np.asarray(times) / axis_step).astype(int).clip(0, n_max - 1)


# --------------------------------------------------------------------------- #
# Agreement with ground truth
# --------------------------------------------------------------------------- #

def test_all_methods_recover_a_stationary_impedance():
    """With nothing varying, all three must return the analytic R+RC."""
    sim = simulate_r_rc(T, DT, FREQS, AMPS, PHASES, R_S, R_P, C,
                        noise_std_V=0.0, noise_std_A=0.0, seed=42)
    z_true = r_rc_impedance(FREQS, R_S, R_P, C)[:, 0]

    z_stft, _, _, _ = _run_stft(sim)
    r = _run_bltva(sim, Np=1)
    z_dmfa, _ = _run_dmfa(sim)

    for label, est, tol in (
            ("STFT", z_stft, 1e-6),
            ("BLTVA", r.H_p[:, [0]], 1e-6),
            ("DMFA", z_dmfa, 1e-2)):
        err = np.abs(est - z_true[:, None]) / np.abs(z_true[:, None])
        stat = np.median(err) if label == "DMFA" else err.max()
        assert stat < tol, f"{label} deviates from Z_true by {stat:.3e}"


def test_all_methods_track_a_time_varying_impedance():
    """Each must follow Z(f, t), compared at matched instants."""
    a, sim = _simulate(variation=(0.30, 0.10))

    z_stft, _, _, t_stft = _run_stft(sim)
    r = _run_bltva(sim)
    z_dmfa, t_dmfa = _run_dmfa(sim)

    z_true = _z_true_at(a, t_stft)
    n_bl = _sample_at(t_stft, DT, N)
    n_dm = _sample_at(t_stft, t_dmfa[1] - t_dmfa[0], t_dmfa.size)

    e_bl = np.abs(r.H_wt[:, n_bl] - z_true) / np.abs(z_true)
    e_st = np.abs(z_stft - z_true) / np.abs(z_true)
    e_dm = np.abs(z_dmfa[:, n_dm] - z_true) / np.abs(z_true)

    assert e_bl.max() < TOL_BLTVA, f"BLTVA {e_bl.max():.3e}"
    assert e_st.max() < TOL_STFT, f"STFT {e_st.max():.3e}"
    # median, not max: DMFA's V/I division spikes where the current crosses zero
    assert np.median(e_dm) < TOL_DMFA_MEDIAN, f"DMFA {np.median(e_dm):.3e}"


def test_methods_agree_with_each_other():
    """Pairwise agreement, which catches a scaling change in any one of them
    even if the ground truth itself were wrong."""
    a, sim = _simulate(variation=(0.30, 0.10))

    z_stft, _, _, t_stft = _run_stft(sim)
    r = _run_bltva(sim)
    z_dmfa, t_dmfa = _run_dmfa(sim)
    scale = np.abs(_z_true_at(a, t_stft))

    bl = r.H_wt[:, _sample_at(t_stft, DT, N)]
    dm = z_dmfa[:, _sample_at(t_stft, t_dmfa[1] - t_dmfa[0], t_dmfa.size)]

    interior = slice(1, -1)          # the first and last window see edge effects
    for label, x, y in (("BLTVA/STFT", bl, z_stft),
                        ("BLTVA/DMFA", bl, dm),
                        ("STFT/DMFA", z_stft, dm)):
        d = np.median(np.abs(x - y)[:, interior] / scale[:, interior])
        assert d < TOL_STFT, f"{label} disagree by {d:.3e}"


# --------------------------------------------------------------------------- #
# The time-alignment convention
# --------------------------------------------------------------------------- #

def test_stft_timestamps_are_window_centres():
    """``estimate_impedance`` reports at the centre of each window."""
    _, sim = _simulate(variation=(0.30, 0.10))
    _, _, _, t_stft = _run_stft(sim)

    assert t_stft.size == P
    assert np.allclose(t_stft, (np.arange(P) + 0.5) * TP)


def test_window_centre_beats_the_window_edge():
    """The convention is load-bearing, not cosmetic.

    Comparing a window-averaged estimate against the truth at the window
    *start* instead of its centre introduces a half-window lag. On a
    monotonically drifting impedance that looks exactly like a systematic
    method bias, which is how a benchmark silently goes wrong. The centre must
    therefore be measurably better.
    """
    a, sim = _simulate(variation=(0.30, 0.10))
    z_stft, _, _, t_stft = _run_stft(sim)

    at_centre = np.abs(z_stft - _z_true_at(a, t_stft))
    at_edge = np.abs(z_stft - _z_true_at(a, t_stft - TP / 2))

    assert np.median(at_centre) < np.median(at_edge) / 5, (
        f"centre {np.median(at_centre):.3e} vs edge {np.median(at_edge):.3e}")


def test_bltva_sampled_at_window_centres_beats_a_lagged_sampling():
    """The same check for the BLTVA side of the mapping."""
    a, sim = _simulate(variation=(0.30, 0.10))
    r = _run_bltva(sim)
    _, _, _, t_stft = _run_stft(sim)

    z_true = _z_true_at(a, t_stft)
    aligned = r.H_wt[:, _sample_at(t_stft, DT, N)]
    lagged = r.H_wt[:, _sample_at(t_stft - TP / 2, DT, N)]

    assert (np.median(np.abs(aligned - z_true))
            < np.median(np.abs(lagged - z_true)) / 5)


def test_dmfa_axis_helpers_are_consistent():
    """DMFA reports on its own grid; the helpers must describe it exactly."""
    _, sim = _simulate(variation=(0.30, 0.10))
    z_dmfa, t_dmfa = _run_dmfa(sim)

    assert t_dmfa.size == z_dmfa.shape[1]
    assert t_dmfa[0] == 0.0
    assert np.isclose(t_dmfa[1] - t_dmfa[0], T / t_dmfa.size)
    assert t_dmfa[-1] < T

    axis = elaboration_frequency_axis(t_dmfa.size, T)
    assert axis.size == t_dmfa.size
    assert np.isclose(axis[1] - axis[0], 1.0 / T)
    assert np.isclose(np.ptp(axis), (t_dmfa.size - 1) / T)


def test_dmfa_filter_axis_is_not_the_sampling_rate():
    """Guards the bug this file's work uncovered: the extracted band spans
    ``Npts_elab/T`` Hz, and building the filter on a ``+/- fs/2`` axis scales
    its bandwidth wrongly by ``fs*T/Npts_elab``."""
    npts = 17640
    correct = elaboration_frequency_axis(npts, T)
    wrong = np.linspace(-FS / 2, FS / 2, npts)

    assert not np.isclose(np.ptp(correct), np.ptp(wrong))
    assert np.isclose(np.ptp(correct), (npts - 1) / T)
