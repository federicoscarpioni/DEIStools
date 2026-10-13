"""
Lock-in amplifier: the properties that make it correct.

The central claim is that a boxcar integrator spanning whole periods of the
base frequency nulls every other tone of the multisine exactly, and that the
resulting estimator coincides with a rectangular STFT. Both are pinned here.
"""
import numpy as np
import pytest

from deistools.processing.lock_in_amp import (
    integrator_length, integrator_kernel, tone_rejection, valid_mask,
    lia_impedance_optim, lia_impedance_ls, lia_eis)
from deistools.processing.stft_functions import estimate_impedance
from deistools.simulation.models.r_rc import simulate_r_rc
from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor


TP, FS, P = 20.0, 1000.0, 6
HARMONICS = np.array([1, 3, 5, 9, 13, 17, 27, 39, 59, 89, 133, 201, 301])
FREQS = HARMONICS / TP
AMPS = np.full(FREQS.size, 0.01)
PHASES = np.random.default_rng(3).uniform(0, 2 * np.pi, FREQS.size)


@pytest.fixture(scope="module")
def stationary():
    n = int(FS * TP) * P
    return simulate_r_rc(n / FS, 1.0 / FS, FREQS, AMPS, PHASES,
                         R_s=10.0, R_p=30.0, C=1e-3, noise_std_V=1e-4, seed=1)


@pytest.fixture(scope="module")
def drifting():
    n = int(FS * TP) * 20
    T = n / FS
    return simulate_r_rc_transistor(
        T, 1.0 / FS, FREQS, AMPS, PHASES, R_s=10.0, R_p=30.0, C=1e-3,
        V_gate_start=1.645, V_gate_rate=(2.455 - 1.645) / T, R_T_max=70.0,
        noise_std_V=1e-4, seed=1)


# --------------------------------------------------------------------------- #
# Integrator design
# --------------------------------------------------------------------------- #
def test_boxcar_over_whole_base_periods_nulls_every_other_tone():
    """The property the whole method rests on: exact nulls, not merely good.

    A boxcar of N_w samples has nulls at multiples of 1/(N_w*dt). When N_w
    spans whole periods of the base frequency those land on every harmonic,
    which is every difference frequency the multisine can produce.
    """
    dt = 1.0 / FS
    n_w = integrator_length(dt, 1.0 / TP, 1, "boxcar")
    kernel = integrator_kernel(n_w, "boxcar")
    for f_k in FREQS:
        rejection = tone_rejection(kernel, dt, FREQS, f_k)
        others = rejection[FREQS != f_k]
        assert others.min() > 200, f"only {others.min():.0f} dB at {f_k} Hz"


def test_odd_length_would_detune_the_nulls():
    """Why integrator_length does not round to an odd number.

    One extra sample on a 20 000-sample period moves the nulls off the
    harmonics and costs more than 200 dB of rejection. Keeping the kernel
    symmetric about a sample is not worth that; the half-sample group delay it
    avoids is common to voltage and current and cancels in their ratio.
    """
    dt = 1.0 / FS
    exact = integrator_length(dt, 1.0 / TP, 1, "boxcar")
    assert exact % 2 == 0, "this test assumes an even exact length"

    worst_exact = np.delete(
        tone_rejection(integrator_kernel(exact, "boxcar"), dt, FREQS, FREQS[0]), 0).min()
    worst_odd = np.delete(
        tone_rejection(integrator_kernel(exact + 1, "boxcar"), dt, FREQS, FREQS[0]), 0).min()

    assert worst_exact > 200
    assert worst_odd < 100
    assert worst_exact - worst_odd > 100


def test_hann_needs_two_base_periods():
    """Its mainlobe is twice as wide, so one period leaves the neighbour in it."""
    dt = 1.0 / FS
    with pytest.raises(ValueError, match="n_periods >= 2"):
        integrator_length(dt, 1.0 / TP, 1, "hann")

    n_w = integrator_length(dt, 1.0 / TP, 2, "hann")
    rejection = tone_rejection(integrator_kernel(n_w, "hann"), dt, FREQS, FREQS[0])
    assert np.delete(rejection, 0).min() > 60


def test_non_integer_number_of_base_periods_is_refused():
    """0.03 Hz at 1 kHz is 33333.3 samples per period, so there is no window
    length that nulls the harmonics. Better to refuse than to round."""
    with pytest.raises(ValueError, match="whole base periods"):
        integrator_length(1.0 / 1000.0, 0.03, 1, "boxcar")

    # and the ordinary case is accepted exactly
    assert integrator_length(1.0 / 1000.0, 1.0 / TP, 1, "boxcar") == 20000


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
def test_lock_in_equals_stft_at_the_window_centres(drifting):
    """A one-period boxcar lock-in IS a sliding rectangular STFT.

    They are the same estimator; STFT reports once per window, the lock-in at
    every sample. Where they coincide they must agree to arithmetic.
    """
    dt = 1.0 / FS
    t, v, i = drifting["time_s"], drifting["voltage_V"], drifting["current_A"]
    n_w = integrator_length(dt, 1.0 / TP, 1, "boxcar")

    Z_lia = np.vstack([lia_impedance_optim(t, v, i, dt, f, n_w) for f in FREQS])
    Z_stft, _, _, t_stft = estimate_impedance(v, i, FREQS, TP, dt)

    centres = np.round(t_stft / dt).astype(int)
    rel = np.abs(Z_lia[:, centres] - Z_stft) / np.abs(Z_stft)
    assert rel.max() < 1e-10


def test_recovers_a_stationary_impedance(stationary):
    dt = 1.0 / FS
    t = stationary["time_s"]
    n_w = integrator_length(dt, 1.0 / TP, 1, "boxcar")
    Z, _ = lia_eis(t, stationary["voltage_V"], stationary["current_A"], dt,
                   FREQS, n_w, decimate=500)
    truth = stationary["Z_true"]

    mask = valid_mask(t.size, n_w)[::500]
    err = np.abs(Z[:, mask] - truth) / np.abs(truth)
    assert np.median(err) < 1e-4


def test_local_least_squares_beats_mixing_on_a_drifting_system(drifting):
    """Carrying a local slope removes the within-window averaging bias.

    This is the one thing the least-squares demodulator buys, and on a
    smoothly drifting cell it is worth about an order of magnitude.
    """
    dt = 1.0 / FS
    t, v, i = drifting["time_s"], drifting["voltage_V"], drifting["current_A"]
    Z_true, n_w = drifting["Z_true"], integrator_length(dt, 1.0 / TP, 1, "boxcar")

    def median_error(Z, times):
        ref = np.array([np.interp(times, t, Z_true[k].real)
                        + 1j * np.interp(times, t, Z_true[k].imag)
                        for k in range(FREQS.size)])
        return float(np.median(np.abs(Z - ref) / np.abs(ref)))

    stride = 20000
    Z_ls = np.vstack([lia_impedance_ls(t, v, i, dt, f, n_w, stride=stride)[0]
                      for f in FREQS])
    t_ls = t[::stride]
    interior = (t_ls > 0.05 * t[-1]) & (t_ls < 0.95 * t[-1])

    Z_mix = np.vstack([lia_impedance_optim(t, v, i, dt, f, n_w)
                       for f in FREQS])[:, ::stride]

    err_ls = median_error(Z_ls[:, interior], t_ls[interior])
    err_mix = median_error(Z_mix[:, interior], t_ls[interior])
    assert err_ls < err_mix / 5


def test_valid_mask_marks_half_a_window_at_each_end():
    mask = valid_mask(1000, 101)
    assert not mask[:50].any()
    assert not mask[-50:].any()
    assert mask[50:-50].all()
