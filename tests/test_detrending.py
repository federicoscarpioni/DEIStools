import numpy as np
from deistools.processing.detrending import get_baseline_mean, get_baseline_extremes
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