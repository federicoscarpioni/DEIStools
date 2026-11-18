from deistools.processing.mfa import MultiFrequencyAnalysis
import numpy as np
from synthetic_records import stationary_r_rc

multisine_period_s = 1
sampling_frequency_Hz = 20
harmonics = np.array([1,3,5,7,9])
periods = 50

def test_fft_eis():
    current_A, voltage_V, impedance_ohm, frequencies_Hz = stationary_r_rc(
        fs=sampling_frequency_Hz,
        multisine_period=multisine_period_s,
        harmonics=harmonics,
        periods=periods,
    )
    mfa = MultiFrequencyAnalysis(
        current_A,
        voltage_V,
        sampling_frequency_Hz,
        frequencies_Hz,
        multisine_period_s,
    )
    mfa.compute_fft()
    mfa.compute_freq_axis()
    impedance_est = mfa.run_fft_eis()
    np.testing.assert_allclose(impedance_est, impedance_ohm, rtol = 1e-12)

def test_stft_eis():
    current_A, voltage_V, impedance_ohm, frequencies_Hz = stationary_r_rc(
        fs=sampling_frequency_Hz,
        multisine_period=multisine_period_s,
        harmonics=harmonics,
        periods=periods,
    )
    mfa = MultiFrequencyAnalysis(
            current_A,
            voltage_V,
            sampling_frequency_Hz,
            frequencies_Hz,
            multisine_period_s,
        )
    mfa.compute_fft()
    mfa.compute_freq_axis()
    impedance_est, i_mean, v_mean, time = mfa.run_stft_eis(1)
    impedance_expected = np.tile(impedance_ohm[:, None], (1, periods))
    assert impedance_expected.shape == impedance_est.shape
    np.testing.assert_allclose(impedance_expected, impedance_est, rtol = 1e-12)
    assert v_mean.size == periods
    assert i_mean.size == periods
    assert time.size == periods
