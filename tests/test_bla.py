"""Tests for the per-burst best linear approximation.

The module makes four claims that can be checked against known truth, and
each has a test here:

* a burst of one multisine period cannot separate noise from distortion,
  because every one of its bins is a harmonic;
* the level read off the non-harmonic bins is the standard deviation of the
  impedance, not merely proportional to it;
* the frequency-correlation ratio tells a varying cell from a noisy one;
* the distortion test does not fire on a linear cell and does fire on a
  nonlinear one.
"""
import numpy as np
import pytest

from deistools.processing.bla import (
    BurstBLA, bla_eis, burst_bla, classify_bins,
    ratio_noise_to_time_variation,
)
from deistools.simulation.models.r_rc import simulate_r_rc
from deistools.simulation.models.block_oriented import (
    Branch, SecondOrderLowpass, simulate_block_oriented,
)
from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor

TP, FS = 20.0, 1000.0
HARMONICS = np.array([1, 3, 5, 9, 13, 17, 27, 39, 59, 89, 133, 201, 301])
FREQS = HARMONICS / TP
AMPS = np.full(FREQS.size, 0.01)


def _record(periods, noise=1e-4, seed=1, phases=None):
    n = int(FS * TP) * periods
    if phases is None:
        phases = np.random.default_rng(3).uniform(0, 2 * np.pi, FREQS.size)
    return simulate_r_rc(n / FS, 1.0 / FS, FREQS, AMPS, phases,
                         R_s=10.0, R_p=30.0, C=1e-3,
                         noise_std_V=noise, noise_std_A=noise * 1e-3,
                         seed=seed)


# --------------------------------------------------------------------------- #
# Bin classification
# --------------------------------------------------------------------------- #

def test_one_period_per_burst_is_refused_with_the_reason():
    """At P = 1 the window fundamental is the multisine fundamental, so every
    bin is a harmonic and the noise class is empty. The error has to say so,
    because one period is exactly what the plain STFT prefers."""
    with pytest.raises(ValueError, match="cannot be separated"):
        classify_bins(20_000, 1, HARMONICS)


@pytest.mark.parametrize("periods", [2, 3, 4, 8])
def test_the_classes_partition_the_band(periods):
    n = int(FS * TP) * periods
    c = classify_bins(n, periods, HARMONICS)
    in_band = c["band"]
    union = c["excited"] | c["odd_nl"] | c["even_nl"] | c["noise"]
    assert np.array_equal(union, in_band), "classes must cover the band"
    for a, b in [("excited", "odd_nl"), ("excited", "even_nl"),
                 ("excited", "noise"), ("odd_nl", "even_nl"),
                 ("odd_nl", "noise"), ("even_nl", "noise")]:
        assert not (c[a] & c[b]).any(), f"{a} and {b} overlap"

    assert c["excited"].sum() == HARMONICS.size
    assert c["noise"].sum() > 0, "P >= 2 must leave noise-only bins"


def test_the_noise_class_grows_with_the_burst_and_the_detection_lines_do_not():
    """The measurement behind the P >= 2 rule, as the module docstring quotes
    it: 301 noise-only bins at two periods, 903 at four, while the odd and
    even detection lines stay at 138 and 150."""
    counts = {}
    for periods in (2, 4, 8):
        n = int(FS * TP) * periods
        counts[periods] = classify_bins(n, periods, HARMONICS)
    assert int(counts[2]["noise"].sum()) == 301
    assert int(counts[4]["noise"].sum()) == 903
    assert int(counts[8]["noise"].sum()) == 2107
    for periods in (2, 4, 8):
        assert int(counts[periods]["odd_nl"].sum()) == 138
        assert int(counts[periods]["even_nl"].sum()) == 150


def test_excited_bins_land_where_the_tones_are():
    periods = 4
    n = int(FS * TP) * periods
    c = classify_bins(n, periods, HARMONICS)
    assert np.array_equal(c["k_excited"], HARMONICS * periods)
    # and the spectrum really does have the signal there
    sim = _record(periods)
    V = np.abs(np.fft.fft(sim["voltage_V"])[1:n // 2]) / n
    at_tone = V[c["k_excited"] - 1]
    elsewhere = V[c["noise"]]
    assert at_tone.min() > 100 * elsewhere.max()


# --------------------------------------------------------------------------- #
# The noise level
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("noise", [1e-4, 1e-2])
def test_the_predicted_noise_matches_the_scatter_it_predicts(noise):
    """The headline claim: ``std_noise`` is the standard deviation of the
    impedance, checked against the scatter over independent realisations.

    Measured ratio 0.997 at three noise levels two decades apart, which also
    shows the level is not merely proportional to the truth.
    """
    periods = 2
    estimates, predicted = [], []
    for trial in range(40):
        rng = np.random.default_rng(1000 + trial)
        phases = rng.uniform(0, 2 * np.pi, FREQS.size)
        sim = _record(periods, noise=noise, seed=trial, phases=phases)
        r = burst_bla(sim["voltage_V"], sim["current_A"], HARMONICS, periods)
        estimates.append(r.impedance)
        predicted.append(r.std_noise)

    observed = np.abs(np.std(np.array(estimates), axis=0, ddof=1))
    ratio = np.median(np.array(predicted).mean(axis=0) / observed)
    assert 0.85 < ratio < 1.15, f"predicted/observed = {ratio:.3f}"


def test_the_noise_level_scales_with_the_noise():
    periods = 2
    levels = []
    for noise in (1e-4, 1e-3, 1e-2):
        sim = _record(periods, noise=noise)
        r = burst_bla(sim["voltage_V"], sim["current_A"], HARMONICS, periods)
        levels.append(np.median(r.std_noise))
    ratios = np.diff(np.log10(levels))
    assert np.allclose(ratios, 1.0, atol=0.1), levels


# --------------------------------------------------------------------------- #
# Noise against time variation
# --------------------------------------------------------------------------- #

def test_the_ratio_tells_a_varying_cell_from_a_noisy_one():
    """Lataire 2012 eq. (33). Stationary noise is uncorrelated between
    neighbouring bins; a time variation is not, because it reaches them by
    convolving the excited lines with the spectrum of the variation.

    Measured: ~12.7 for a stationary noisy cell, 0.73 for the transistor.
    """
    periods = 4
    stationary = _record(periods)
    r_st = burst_bla(stationary["voltage_V"], stationary["current_A"],
                     HARMONICS, periods)

    n = int(FS * TP) * 20
    total = n / FS
    phases = np.random.default_rng(3).uniform(0, 2 * np.pi, FREQS.size)
    tr = simulate_r_rc_transistor(
        total, 1.0 / FS, FREQS, AMPS, phases, R_s=10.0, R_p=30.0, C=1e-3,
        V_gate_start=1.645, V_gate_rate=(2.455 - 1.645) / total,
        R_T_max=70.0, noise_std_V=1e-4, seed=1)
    burst = int(FS * TP) * periods
    r_tv = burst_bla(tr["voltage_V"][:burst], tr["current_A"][:burst],
                     HARMONICS, periods)

    assert r_st.r_ntv > 3.0, r_st.r_ntv
    assert r_st.dominant_error() == "noise"
    assert r_tv.r_ntv < 2.0, r_tv.r_ntv
    assert r_tv.dominant_error() == "time variation"
    assert r_st.r_ntv > 4 * r_tv.r_ntv


def test_the_ratio_reports_infinity_rather_than_inventing_a_number():
    bins = np.arange(1, 11)
    nowhere = np.zeros(10, dtype=bool)
    assert ratio_noise_to_time_variation(np.ones(10, complex), nowhere, bins) \
        == float("inf")
    # non-adjacent bins give no lag-1 pair either
    scattered = np.zeros(10, dtype=bool)
    scattered[[0, 4, 8]] = True
    assert ratio_noise_to_time_variation(np.ones(10, complex), scattered,
                                         bins) == float("inf")


# --------------------------------------------------------------------------- #
# Nonlinear distortion
# --------------------------------------------------------------------------- #

def _stationary_nonlinear(strength, periods, noise=1e-4):
    """One branch, so b_0(t) = 1 and the system does not vary in time."""
    n = int(FS * TP) * periods
    phases = np.random.default_rng(3).uniform(0, 2 * np.pi, FREQS.size)
    branches = [Branch(SecondOrderLowpass.from_hz(0.7, 0.15, 1.0),
                       f1=lambda x, a=strength: x + a * np.tanh(x - 2),
                       f2=lambda x, a=strength: x + a * np.tanh(x - 1),
                       name="stationary NL")]
    return simulate_block_oriented(n / FS, 1.0 / FS, FREQS, AMPS, phases,
                                   branches, trend_coefficients=None,
                                   noise_std_V=noise, seed=1)


def test_no_distortion_is_reported_for_a_linear_cell():
    """The false-alarm test, and the reason the threshold exists.

    Merely requiring a positive difference between the detection lines and
    the noise fires on 10 of 13 tones here, where the two levels agree to
    2.5% and the sign is a coin toss.
    """
    periods = 4
    sim = _record(periods)
    r = burst_bla(sim["voltage_V"], sim["current_A"], HARMONICS, periods)
    assert np.all(np.isnan(r.distortion_odd()))
    assert np.all(np.isnan(r.distortion_even()))

    # the levels themselves are reported, and agree with the noise
    assert 0.8 < np.median(r.std_odd_lines / r.std_noise) < 1.25
    assert 0.8 < np.median(r.std_even_lines / r.std_noise) < 1.25


def test_distortion_is_reported_for_a_nonlinear_cell_and_grows_with_it():
    periods = 4
    detected, levels = [], []
    for strength in (0.02, 0.1, 0.3):
        sim = _stationary_nonlinear(strength, periods)
        r = burst_bla(sim["voltage_V"], sim["current_A"], HARMONICS, periods)
        detected.append(int(np.sum(~np.isnan(r.distortion_even()))))
        levels.append(float(np.median(r.std_even_lines / r.std_noise)))

    assert min(detected) >= 5, detected
    assert levels == sorted(levels), levels
    assert levels[-1] > 10 * levels[0]


def test_the_threshold_is_what_suppresses_the_false_alarm():
    """Guards the guard: with z = 0 the linear cell does fire."""
    periods = 4
    sim = _record(periods)
    r = burst_bla(sim["voltage_V"], sim["current_A"], HARMONICS, periods)
    lenient = r._excess(r.std_odd_lines, z=0.0)
    assert np.sum(~np.isnan(lenient)) >= 5, "expected false alarms at z = 0"
    assert np.all(np.isnan(r._excess(r.std_odd_lines, z=3.0)))


# --------------------------------------------------------------------------- #
# Sliding it over a record
# --------------------------------------------------------------------------- #

def test_bla_eis_walks_the_record_and_stamps_the_midpoints():
    periods = 2
    sim = _record(periods * 5)
    results, t = bla_eis(sim["voltage_V"], sim["current_A"], TP, 1.0 / FS,
                         HARMONICS, periods_per_window=periods)
    assert len(results) == 5
    assert all(isinstance(r, BurstBLA) for r in results)
    assert all(r.impedance.size == HARMONICS.size for r in results)
    span = periods * TP
    assert np.allclose(t, (np.arange(5) + 0.5) * span)


def test_bla_eis_refuses_a_record_too_short_for_one_burst():
    sim = _record(2)
    with pytest.raises(ValueError, match="short of"):
        bla_eis(sim["voltage_V"], sim["current_A"], TP, 1.0 / FS, HARMONICS,
                periods_per_window=8)
