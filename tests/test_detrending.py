import numpy as np
import pytest
from numpy.fft import rfft
from deistools.processing.detrending import get_baseline_mean, get_baseline_extremes, get_avarage_spline
from synthetic_records import stationary_r_rc

sampling_frequency_Hz = 20
multisine_period_s = 1
harmonics = np.array([1,3,5,7,9])

def test_get_baseline_mean():
    periods = 50
    _, multi_frequency_signal, _, _ = stationary_r_rc(
        fs=sampling_frequency_Hz,
        multisine_period=multisine_period_s,
        harmonics=harmonics,
        periods=periods,
    )
    record_samples = int(multi_frequency_signal.shape[0])
    time_s = np.arange(record_samples) / sampling_frequency_Hz
    trend = 3 + time_s * 0.01
    signal_with_trend = multi_frequency_signal + trend
    samples_per_period = sampling_frequency_Hz * multisine_period_s
    baseline = get_baseline_mean(
        signal_with_trend,
        samples_per_period,
        sampling_frequency_Hz
    )
    trend_est = baseline.evaluate(time_s)
    np.testing.assert_allclose(trend, trend_est, atol = 1e-12)
    detrended_signal = signal_with_trend - trend_est
    np.testing.assert_allclose(multi_frequency_signal, detrended_signal, atol = 1e-12)


def test_get_baseline_extremes():
    periods = 1
    _, multi_frequency_signal, _, _ = stationary_r_rc(
        fs=sampling_frequency_Hz,
        multisine_period=multisine_period_s,
        harmonics=harmonics,
        periods=periods,
    )
    record_samples = int(multi_frequency_signal.shape[0])
    time_s = np.arange(record_samples) / sampling_frequency_Hz
    trend = 3 + time_s * 0.01
    signal_with_trend = multi_frequency_signal + trend
    samples_per_period = sampling_frequency_Hz * multisine_period_s
    baseline = get_baseline_extremes(
        signal_with_trend,
        samples_per_period,
        sampling_frequency_Hz
    )
    trend_est = baseline.evaluate(time_s)
    # The tollarance for this tests is high because the method has a lot of sources of error, especially at low sampling frequencies
    np.testing.assert_allclose(trend, trend_est, atol = 1e-2)
    detrended_signal = signal_with_trend - trend_est
    np.testing.assert_allclose(multi_frequency_signal, detrended_signal, atol = 1e-2)
    # A slope with no multisine must instead be properly recovered
    expected_slope = (signal_with_trend[-1] - signal_with_trend[0]) * sampling_frequency_Hz / (signal_with_trend.size - 1)
    assert np.isclose(baseline.slope, expected_slope, atol=1e-12)
    assert np.isclose(baseline.intercept, signal_with_trend[0], atol=1e-12)


def _spline_record(periods=50):
    current, voltage, impedance, _ = stationary_r_rc(
        fs=sampling_frequency_Hz,
        multisine_period=multisine_period_s,
        harmonics=harmonics,
        periods=periods,
    )
    time_s = np.arange(voltage.size) / sampling_frequency_Hz
    samples_per_period = sampling_frequency_Hz * multisine_period_s
    return current, voltage, impedance, time_s, samples_per_period


def test_get_avarage_spline_linear():
    _, voltage, _, time_s, samples_per_period = _spline_record()
    trend = 3 + time_s * 0.01
    spline = get_avarage_spline(voltage + trend, samples_per_period, sampling_frequency_Hz)
    np.testing.assert_allclose(spline(time_s), trend, atol=1e-12)


def test_get_avarage_spline_quadratic():
    # The period mean of c*t^2 is its value at the centre plus c*var(t) over the
    # period, so the spline is exact up to that constant. A constant only
    # touches the DC bin, so the impedance is unaffected.
    _, voltage, _, time_s, samples_per_period = _spline_record()
    c = 1e-4
    trend = 3 + time_s * 0.01 + c * time_s**2
    spline = get_avarage_spline(voltage + trend, samples_per_period, sampling_frequency_Hz)
    difference = spline(time_s) - trend
    assert np.ptp(difference) < 1e-12
    expected_offset = c * (samples_per_period**2 - 1) / (12 * sampling_frequency_Hz**2)
    assert np.isclose(difference.mean(), expected_offset, atol=1e-12)


def test_get_avarage_spline_power():
    # Diffusion-like relaxation A*((1 + t/tau)^(-1/2) - 1). Not a polynomial, so
    # the spline is not exact; check the impedance at the tones instead.
    current, voltage, impedance, time_s, samples_per_period = _spline_record()
    tau = 0.1 * time_s.size / sampling_frequency_Hz
    trend = 0.05 * ((1 + time_s / tau) ** -0.5 - 1)
    spline = get_avarage_spline(voltage + trend, samples_per_period, sampling_frequency_Hz)
    k_exc = harmonics * (time_s.size // samples_per_period)

    def fft_eis(response):
        return rfft(response)[k_exc] / rfft(current)[k_exc]

    error_raw = np.abs(fft_eis(voltage + trend) - impedance) / np.abs(impedance)
    error_detrended = np.abs(fft_eis(voltage + trend - spline(time_s)) - impedance) / np.abs(impedance)
    assert error_raw.max() > 1e-2          # the trend really does corrupt the estimate
    assert error_detrended.max() < 1e-3    # measured 1.2e-4 at the lowest tone


def test_get_avarage_spline_ignores_multisine():
    # Period means hold no multisine content, so the spline must be the same
    # with and without the response on top of the trend.
    _, voltage, _, time_s, samples_per_period = _spline_record()
    tau = 0.1 * time_s.size / sampling_frequency_Hz
    trend = 0.05 * ((1 + time_s / tau) ** -0.5 - 1)
    with_signal = get_avarage_spline(voltage + trend, samples_per_period, sampling_frequency_Hz)
    trend_only = get_avarage_spline(trend, samples_per_period, sampling_frequency_Hz)
    np.testing.assert_allclose(with_signal(time_s), trend_only(time_s), atol=1e-12)


def test_get_avarage_spline_needs_two_periods():
    _, voltage, _, _, samples_per_period = _spline_record(periods=1)
    with pytest.raises(ValueError):
        get_avarage_spline(voltage, samples_per_period, sampling_frequency_Hz)
