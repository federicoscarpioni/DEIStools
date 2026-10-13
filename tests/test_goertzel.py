"""
Goertzel filter: it must reproduce the DFT, and therefore the STFT.

The algorithm's only claim is that it evaluates one DFT bin by recurrence.
Everything else follows, so these tests check that claim directly -- against
numpy for the transform and against the STFT estimator for the impedance.
"""
import numpy as np
import pytest

from deistools.processing.goertzel import (
    goertzel_dft, goertzel_eis, GoertzelFilter, cost_ratio)
from deistools.processing.stft_functions import estimate_impedance
from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor


TP, FS, P = 20.0, 1000.0, 20
HARMONICS = np.array([1, 3, 5, 9, 13, 17, 27, 39, 59, 89, 133, 201, 301])
FREQS = HARMONICS / TP


@pytest.fixture(scope="module")
def drifting():
    n = int(FS * TP) * P
    T = n / FS
    phases = np.random.default_rng(3).uniform(0, 2 * np.pi, FREQS.size)
    return simulate_r_rc_transistor(
        T, 1.0 / FS, FREQS, np.full(FREQS.size, 0.01), phases,
        R_s=10.0, R_p=30.0, C=1e-3, V_gate_start=1.645,
        V_gate_rate=(2.455 - 1.645) / T, R_T_max=70.0,
        noise_std_V=1e-4, seed=1)


def test_reproduces_numpy_fft_at_integer_bins():
    x = np.random.default_rng(0).normal(size=512)
    bins = np.array([0, 1, 7, 64, 255])
    assert np.abs(goertzel_dft(x, bins) - np.fft.fft(x)[bins]).max() < 1e-8


def test_generalised_form_is_right_at_non_integer_bins():
    """The trailing exp(-2j*pi*k) is what makes a fractional bin correct.

    Without it the result is wrong by a phase for any k that is not an
    integer -- and a tone that does not land exactly on a bin of the chosen
    block is the ordinary case, not an exotic one.
    """
    x = np.random.default_rng(1).normal(size=256)
    n = np.arange(x.size)
    for k in (12.37, 5.5, 0.25):
        manual = np.sum(x * np.exp(-2j * np.pi * k * n / x.size))
        assert abs(goertzel_dft(x, [k])[0] - manual) < 1e-8


def test_output_formula_is_not_the_tempting_rearrangement():
    """Pins the phase factor that a V/I test would not catch.

    ``s1 - exp(-1j*w)*s2`` has the right magnitude and is wrong by exactly
    exp(-1j*w) in phase. It cancels in any ratio, so only a check against the
    DFT itself finds it.
    """
    x = np.random.default_rng(2).normal(size=128)
    k, n = 9, x.size
    omega = 2 * np.pi * k / n
    coeff = 2 * np.cos(omega)
    s1 = s2 = 0.0
    for sample in x:
        s1, s2 = sample + coeff * s1 - s2, s1

    correct = goertzel_dft(x, [k])[0]
    wrong = s1 - np.exp(-1j * omega) * s2
    assert abs(correct - np.fft.fft(x)[k]) < 1e-9
    assert abs(abs(wrong) - abs(correct)) < 1e-9          # same magnitude
    assert abs(wrong - correct) > 1e-3                    # different phase


def test_matches_stft_on_the_same_blocks(drifting):
    Z_g, t_g = goertzel_eis(drifting["voltage_V"], drifting["current_A"],
                            FREQS, TP, 1.0 / FS)
    Z_s, _, _, t_s = estimate_impedance(drifting["voltage_V"],
                                        drifting["current_A"], FREQS, TP,
                                        1.0 / FS)
    assert np.allclose(t_g, t_s)
    assert np.max(np.abs(Z_g - Z_s) / np.abs(Z_s)) < 1e-9


def test_recovers_the_known_impedance(drifting):
    Z, t = goertzel_eis(drifting["voltage_V"], drifting["current_A"],
                        FREQS, TP, 1.0 / FS)
    truth = np.array([
        np.interp(t, drifting["time_s"], drifting["Z_true"][k].real)
        + 1j * np.interp(t, drifting["time_s"], drifting["Z_true"][k].imag)
        for k in range(FREQS.size)])
    assert np.median(np.abs(Z - truth) / np.abs(truth)) < 1e-3


def test_streaming_filter_matches_the_block_call(drifting):
    block = int(TP * FS)
    v = drifting["voltage_V"][:3 * block]
    filt = GoertzelFilter(FREQS[0], 1.0 / FS, block)
    streamed = [value for s in v if (value := filt.push(s)) is not None]
    blocked = goertzel_dft(v.reshape(3, block), [FREQS[0] * TP])[:, 0]
    assert len(streamed) == 3
    assert np.abs(np.array(streamed) - blocked).max() < 1e-9


def test_cost_ratio_describes_the_crossover():
    assert cost_ratio(1, 1024) < 1        # one bin: Goertzel wins easily
    assert cost_ratio(64, 1024) > 1       # many bins: the FFT wins
