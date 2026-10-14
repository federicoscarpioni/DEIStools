"""The record-against-its-metadata check.

The cases that matter are the ones where the arithmetic alone is ambiguous:
``N = P * Tp * fs`` can be made to balance by changing any of the four, and
what the check has to do is use the signal to say which of them the tones
agree with. Two faults produce the same complaint from the arithmetic and
must come apart here:

* the record was cut mid-period -- the lines stay put, the nominal bins slide
  off the grid, and the measured timebase error is zero.
* the sampling rate is not the one declared -- the lines move, in proportion
  to the harmonic number, and the timebase error reads as the true rate.

A test is included for each, with the true answer known by construction.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from deistools.processing.consistency import (
    check_consistency, visualize_consistency, _sub_bin_offset,
    _infer_fundamental,
)
from deistools.processing.mfa import MultiFrequencyAnalysis
from deistools.simulation.models.r_rc import simulate_r_rc

TP = 20.0
FS = 1000.0
P = 6
HARMONICS = np.array([1, 3, 5, 9, 13, 17, 27, 39, 59, 89, 133, 201, 301])
FREQUENCIES = HARMONICS / TP
N = int(FS * TP) * P


@pytest.fixture(scope="module")
def record():
    """A stationary cell, measured over exactly six multisine periods."""
    sim = simulate_r_rc(
        N / FS, 1.0 / FS, FREQUENCIES,
        np.full(FREQUENCIES.size, 0.01),
        np.random.default_rng(3).uniform(0, 2 * np.pi, FREQUENCIES.size),
        R_s=10.0, R_p=30.0, C=1e-3, noise_std_V=1e-4, seed=1)
    return sim["voltage_V"], sim["current_A"], sim["time_s"]


def run(record, **kwargs):
    voltage, current, _ = record
    settings = dict(frequencies=FREQUENCIES, sampling_time=1.0 / FS, period=TP)
    settings.update(kwargs)
    return check_consistency(kwargs.pop("voltage", voltage),
                             kwargs.pop("current", current),
                             settings["frequencies"], settings["sampling_time"],
                             period=settings["period"],
                             n_periods=settings.get("n_periods"))


# --------------------------------------------------------------------------- #
# Nothing wrong
# --------------------------------------------------------------------------- #

def test_correct_metadata_passes(record):
    report = run(record, n_periods=P)
    assert report.ok, [str(f) for f in report.findings]
    assert report.worst == "ok"
    assert report["diagnosis"][0].level == "ok"
    assert report.samples_per_period == int(FS * TP)
    assert report.periods_in_record == pytest.approx(P)
    assert abs(report.timebase_error) < 1e-9
    assert report.subdivision == 1


def test_the_tones_are_found_on_their_own_bins(record):
    report = run(record)
    assert np.allclose(report.bin_measured, HARMONICS * P, atol=1e-6)
    assert np.allclose(report.bin_nominal, HARMONICS * P)
    assert report.worst_offset() < 1e-6


def test_noiseless_record_reports_infinite_snr():
    """The simulated reference case has no noise floor to speak of."""
    sim = simulate_r_rc(N / FS, 1.0 / FS, FREQUENCIES,
                        np.full(FREQUENCIES.size, 0.01),
                        np.zeros(FREQUENCIES.size),
                        R_s=10.0, R_p=30.0, C=1e-3,
                        noise_std_V=0.0, noise_std_A=0.0, seed=1)
    report = check_consistency(sim["voltage_V"], sim["current_A"], FREQUENCIES,
                               1.0 / FS, period=TP)
    assert report.ok, [str(f) for f in report.findings]
    assert np.all(report.snr_db_current > 100)


# --------------------------------------------------------------------------- #
# The two faults that have to come apart
# --------------------------------------------------------------------------- #

def test_truncated_record_blames_the_length(record):
    voltage, current, _ = record
    report = check_consistency(voltage[:-137], current[:-137], FREQUENCIES,
                               1.0 / FS, period=TP)
    assert report["whole_periods"], "a partial period was not noticed"
    # The lines have not moved: fs and Tp are right.
    assert abs(report.timebase_error) < 1e-6
    assert report.worst_offset() < 1e-3
    assert "record length is wrong" in report["diagnosis"][0].message


def test_wrong_sampling_rate_blames_the_rate_and_recovers_it(record):
    voltage, current, _ = record
    declared = 1001.0                       # the record was taken at 1000 Hz
    report = check_consistency(voltage, current, FREQUENCIES, 1.0 / declared,
                               period=TP, n_periods=P)
    assert report["whole_periods"], "the triangle should not balance"
    assert report.timebase_error == pytest.approx(declared / FS - 1, rel=1e-3)
    implied_fs, implied_period, implied_periods = report.implied()
    assert implied_fs == pytest.approx(FS, rel=1e-6)
    assert implied_period == pytest.approx(TP * FS / declared, rel=1e-6)
    assert implied_periods == pytest.approx(P, rel=1e-6)
    message = report["diagnosis"][0].message
    assert "record length is right" in message
    assert "sampling rate or the period is wrong" in message


def test_both_routes_agree_on_a_clock_error(record):
    """The spectrum and the period-by-period phase measure the same thing."""
    voltage, current, _ = record
    report = check_consistency(voltage, current, FREQUENCIES, 1.0 / 1001.0,
                               period=TP)
    assert report.timebase_error_folded == pytest.approx(
        report.timebase_error, rel=5e-3)


def test_tones_that_do_not_share_a_scale_are_not_called_a_clock_error(record):
    """Individually wrong frequencies are a wrong list, not a wrong timebase."""
    voltage, current, _ = record
    scattered = FREQUENCIES + np.array(
        [0.0, 0.004, -0.003, 0.0, 0.0, 0.0035, 0.0, -0.004, 0.0, 0.0, 0.0,
         0.0036, 0.0])
    report = check_consistency(voltage, current, scattered, 1.0 / FS, period=TP)
    assert report["off_grid_scatter"], [str(f) for f in report.findings]
    assert not report["diagnosis"]


# --------------------------------------------------------------------------- #
# The rest of the triangle
# --------------------------------------------------------------------------- #

def test_claimed_period_count_is_checked(record):
    report = run(record, n_periods=5)
    assert report["n_periods"]
    assert report.n_periods_claimed == 5


def test_period_that_disagrees_with_the_tone_list(record):
    report = run(record, period=25.0)
    assert report["period_vs_tones"]
    assert report["whole_periods"]
    assert report["harmonics"], "non-integer harmonic numbers went unnoticed"


def test_non_integer_samples_per_period(record):
    voltage, current, _ = record
    # 20 s at 1000.07 Hz is 20001.4 samples, so no window can hold a whole
    # period and every estimator rounds.
    report = check_consistency(voltage, current, FREQUENCIES, 1.0 / 1000.07,
                               period=TP)
    assert report["samples_per_period"]
    assert report.samples_per_period == pytest.approx(20001.4)


def test_off_grid_tones_are_reported_the_way_compute_freq_indexes_refuses_them(record):
    voltage, current, _ = record
    off = FREQUENCIES.copy()
    off[4] += 0.5 / (N / FS)            # half a bin
    report = check_consistency(voltage, current, off, 1.0 / FS, period=TP)
    assert report["on_grid"]
    assert report["on_grid"][0].level == "fail"


def test_tones_sharing_a_bin(record):
    voltage, current, _ = record
    crowded = np.append(FREQUENCIES, FREQUENCIES[-1] + 1e-6)
    report = check_consistency(voltage, current, np.sort(crowded), 1.0 / FS,
                               period=TP)
    assert report["bin_collision"]
    assert report["bin_collision"][0].level == "fail"


def test_tone_above_nyquist(record):
    voltage, current, _ = record
    report = check_consistency(voltage, current, np.append(FREQUENCIES, 600.0),
                               1.0 / FS, period=TP)
    assert report["nyquist"]
    assert report["nyquist"][0].level == "fail"


# --------------------------------------------------------------------------- #
# The fundamental, which three estimators assume is the lowest tone
# --------------------------------------------------------------------------- #

def test_grid_that_does_not_start_at_harmonic_one(record):
    voltage, current, _ = record
    report = check_consistency(voltage, current, FREQUENCIES[1:], 1.0 / FS)
    assert report.subdivision == 3
    assert report["fundamental"]
    assert report.period == pytest.approx(TP)


def test_one_stray_frequency_does_not_subdivide_the_grid():
    """A single mistyped tone is a mistyped tone, not a finer fundamental."""
    listed = np.append(FREQUENCIES, 7.77)
    base, subdivision = _infer_fundamental(np.sort(listed))
    assert subdivision == 1
    assert base == pytest.approx(FREQUENCIES.min())


def test_subdivision_is_found_when_it_explains_the_whole_grid():
    base, subdivision = _infer_fundamental(FREQUENCIES[1:])
    assert subdivision == 3
    assert base == pytest.approx(1.0 / TP)


# --------------------------------------------------------------------------- #
# What is in the record that should not be, and what is missing
# --------------------------------------------------------------------------- #

def test_a_listed_tone_that_was_never_excited(record):
    voltage, current, _ = record
    listed = np.sort(np.append(FREQUENCIES, 7.75))     # 155/Tp, on a bin
    report = check_consistency(voltage, current, listed, 1.0 / FS, period=TP)
    assert report["tone_missing"]
    weak = int(np.argmin(report.snr_db_current))
    assert listed[weak] == pytest.approx(7.75)


def test_mains_pickup_is_reported(record):
    voltage, current, time = record
    report = check_consistency(voltage + 0.05 * np.sin(2 * np.pi * 50.0 * time),
                               current, FREQUENCIES, 1.0 / FS, period=TP)
    assert report["stray_line"]
    assert any(abs(row[0] - 50.0) < 0.6 for row in report.stray_lines)
    assert any("mains" in row[3] for row in report.stray_lines)


def test_a_small_stray_line_is_reported_once_you_ask_for_it(record):
    """The threshold is a fraction of the tones, and it is an argument."""
    voltage, current, time = record
    small = voltage + 0.002 * np.sin(2 * np.pi * 50.0 * time)
    assert not check_consistency(small, current, FREQUENCIES, 1.0 / FS,
                                 period=TP)["stray_line"]
    report = check_consistency(small, current, FREQUENCIES, 1.0 / FS,
                               period=TP, stray_fraction=0.005)
    assert any(abs(row[0] - 50.0) < 0.6 for row in report.stray_lines)


def test_clipping_is_reported(record):
    voltage, current, _ = record
    clipped = np.clip(voltage, np.quantile(voltage, 0.05), np.quantile(voltage, 0.95))
    report = check_consistency(clipped, current, FREQUENCIES, 1.0 / FS, period=TP)
    assert report["clipping"]


def test_drift_is_information_not_a_fault(record):
    voltage, current, time = record
    drifting = voltage + 5.0 * time / time[-1]
    report = check_consistency(drifting, current, FREQUENCIES, 1.0 / FS,
                               period=TP)
    assert report["drift"]
    assert report["drift"][0].level == "info"


# --------------------------------------------------------------------------- #
# Nothing here is allowed to raise on a bad record, or to touch it
# --------------------------------------------------------------------------- #

def test_mismatched_lengths_and_nans_are_reported_not_raised(record):
    voltage, current, _ = record
    damaged = voltage.copy()
    damaged[[10, 20, 30]] = np.nan
    report = check_consistency(damaged, current[:-1], FREQUENCIES, 1.0 / FS,
                               period=TP)
    assert report["arrays"] and report["not_finite"]
    assert np.count_nonzero(np.isnan(damaged)) == 3, "the caller's array was altered"


def test_a_constant_channel(record):
    voltage, _current, _ = record
    report = check_consistency(voltage, np.zeros_like(voltage), FREQUENCIES,
                               1.0 / FS, period=TP)
    assert report["constant"]


def test_sampling_time_must_be_a_positive_number(record):
    voltage, current, _ = record
    with pytest.raises(ValueError, match="positive"):
        check_consistency(voltage, current, FREQUENCIES, 0.0)


def test_summary_is_printable_in_every_case(record):
    voltage, current, _ = record
    for kwargs in ({}, {"period": 25.0}, {"period": TP, "n_periods": 5}):
        text = str(check_consistency(voltage, current, FREQUENCIES, 1.0 / FS,
                                     **kwargs))
        assert "Multi-frequency record check" in text
        assert "N = P x Tp x fs" in text


# --------------------------------------------------------------------------- #
# The sub-bin estimator the whole timebase measurement rests on
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("offset", [-0.4, -0.25, -0.05, 0.0, 0.05, 0.25, 0.4])
def test_sub_bin_offset_recovers_a_known_fractional_bin(offset):
    n = 4096
    k = 300
    index = np.arange(n)
    signal = np.cos(2 * np.pi * (k + offset) * index / n + 0.7)
    spectrum = np.fft.rfft(signal) / n
    peak = int(np.argmax(np.abs(spectrum)))
    estimate = peak + _sub_bin_offset(spectrum, peak)
    assert estimate == pytest.approx(k + offset, abs=0.02)


def test_sub_bin_offset_is_bounded():
    spectrum = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    assert abs(_sub_bin_offset(spectrum, 1)) <= 0.5
    assert _sub_bin_offset(spectrum, 0) == 0.0


# --------------------------------------------------------------------------- #
# The figure, and the method on the object
# --------------------------------------------------------------------------- #

def test_the_figure_draws_for_good_and_bad_records(record, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    voltage, current, _ = record
    for report in (check_consistency(voltage, current, FREQUENCIES, 1.0 / FS,
                                     period=TP),
                   check_consistency(voltage[:-137], current[:-137],
                                     FREQUENCIES, 1.0 / FS, period=TP),
                   check_consistency(voltage[:300], current[:300],
                                     FREQUENCIES[:1], 1.0 / FS, period=TP)):
        fig, axs = visualize_consistency(report)
        assert axs.shape == (2, 3)
        plt.close(fig)


def test_the_method_on_multifrequencyanalysis(record):
    voltage, current, _ = record
    mfa = MultiFrequencyAnalysis(FREQUENCIES, voltage, current, 1.0 / FS)
    report = mfa.check_consistency(period=TP, n_periods=P, plot=False)
    assert report.ok, [str(f) for f in report.findings]


def test_the_method_runs_before_anything_else_on_the_object(record):
    """It must work on a fresh instance: compute_freq_indexes is one of the
    things that raises when the metadata is wrong, so the check cannot
    depend on it having been called."""
    voltage, current, _ = record
    mfa = MultiFrequencyAnalysis(FREQUENCIES, voltage[:-137], current[:-137],
                                 1.0 / FS)
    report = mfa.check_consistency(period=TP, plot=False)
    assert not report.ok
    assert not hasattr(mfa, "freq_axis")
