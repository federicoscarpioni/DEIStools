from deistools.processing.stft_functions  import stft_eis
import numpy as np
from synthetic_records import stationary_r_rc, non_stationary_r_rc

sampling_frequency_Hz = 20
multisine_period_s = 1
harmonics = np.array([1,3,5,7,9])
periods = 50

def test_stft_eis_stationary():
    current_A, voltage_V, impedance_ohm, frequencies_Hz = stationary_r_rc(
        fs=sampling_frequency_Hz,
        multisine_period=multisine_period_s,
        harmonics=harmonics,
        periods=periods,
    )
    impedance_est, i_mean, v_mean, time = stft_eis(
        current_A,
        voltage_V,
        frequencies_Hz,
        sampling_frequency_Hz,
        multisine_period_s,
        1,
    )
    impedance_expected = np.tile(impedance_ohm[:, None], (1, periods))
    assert impedance_expected.shape == impedance_est.shape
    np.testing.assert_allclose(impedance_expected, impedance_est, rtol = 1e-12)
    assert v_mean.size == periods
    assert i_mean.size == periods

def test_stft_eis_slowly_varying():
    current_A, voltage_V, impedance_ohm, frequencies_Hz = non_stationary_r_rc(
        fs=sampling_frequency_Hz,
        multisine_period=multisine_period_s,
        harmonics=harmonics,
        periods=periods,
    )
    impedance_est, i_mean, v_mean, time = stft_eis(
        current_A,
        voltage_V,
        frequencies_Hz,
        sampling_frequency_Hz,
        multisine_period_s,
        1,
    )
    L = sampling_frequency_Hz * multisine_period_s
    impedance_expected = impedance_ohm.reshape(frequencies_Hz.size, periods, L).mean(axis=-1)
    assert impedance_expected.shape == impedance_est.shape
    np.testing.assert_allclose(impedance_expected, impedance_est, rtol = 1e-3)
    assert v_mean.size == periods
    assert i_mean.size == periods

# Other cases to implement
def tets_stft_mean_negative_sign():...
def test_stft_bins_scales_with_periods_per_windows():...
def test_stft_drops_incomplete_last_window():...
