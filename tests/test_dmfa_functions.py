import numpy as np, pytest
from numpy.fft import fft
from deistools.processing.dmfa_functions import dmfa
from deistools.processing.passband_filters import FermiDiracFilter
from deistools.processing.stft_functions import get_excited_bins
from synthetic_records import stationary_r_rc, non_stationary_r_rc

sampling_frequency_Hz = 20
multisine_period_s = 1
harmonics = np.array([1,3,5,7,9])
periods = 50

def test_dmfa():
        current_A, voltage_V, impedance_ohm, frequencies_Hz = stationary_r_rc(
                fs=sampling_frequency_Hz,
                multisine_period=multisine_period_s,
                harmonics=harmonics,
                periods=periods,
        )
        record_samples = int(voltage_V.shape[0])
        ft_voltage = fft(voltage_V)/record_samples
        ft_current = fft(current_A)/record_samples
        filter = FermiDiracFilter(halfwidth=0.25, order=4)
        excited_bins = get_excited_bins(
                record_samples,
                frequencies_Hz,
                sampling_frequency_Hz,
        )
        periods_per_sample =1
        impedance_est, time = dmfa(
                U=ft_current,
                V=ft_voltage,
                k_exc=excited_bins,
                passband_filter=filter,
                fs=sampling_frequency_Hz,
                multisine_period=multisine_period_s,
                periods_per_sample=1,
        )
        impedance_expected = np.tile(impedance_ohm[:, None], (1, periods))
        assert impedance_expected.shape == impedance_est.shape
        np.testing.assert_allclose(impedance_est,impedance_expected, rtol = 1e-12)

@pytest.mark.parametrize('periods_per_sample', [1,2,5])
def test_dmfa_time_array(periods_per_sample):
        current_A, voltage_V, impedance_ohm, frequencies_Hz = stationary_r_rc(
                fs=sampling_frequency_Hz,
                multisine_period=multisine_period_s,
                harmonics=harmonics,
                periods=periods,
        )
        record_samples = int(voltage_V.shape[0])
        ft_voltage = fft(voltage_V)/record_samples
        ft_current = fft(current_A)/record_samples
        filter = FermiDiracFilter(halfwidth=0.25, order=4)
        excited_bins = get_excited_bins(
                record_samples,
                frequencies_Hz,
                sampling_frequency_Hz,
        )
        impedance_est, time = dmfa(
                U=ft_current,
                V=ft_voltage,
                k_exc=excited_bins,
                passband_filter=filter,
                fs=sampling_frequency_Hz,
                multisine_period=multisine_period_s,
                periods_per_sample=periods_per_sample,
        )
        assert time.size == record_samples/(periods_per_sample * multisine_period_s *sampling_frequency_Hz)

def test_dmfa_slowly_varying():
        current_A, voltage_V, impedance_ohm, frequencies_Hz = non_stationary_r_rc(
                fs=sampling_frequency_Hz,
                multisine_period=multisine_period_s,
                harmonics=harmonics,
                periods=periods,
        )
        record_samples = voltage_V.size
        impedance_est, time = dmfa(
                U=fft(current_A)/record_samples,
                V=fft(voltage_V)/record_samples,
                k_exc=get_excited_bins(record_samples, frequencies_Hz, sampling_frequency_Hz),
                passband_filter=FermiDiracFilter(halfwidth=0.25, order=4),
                fs=sampling_frequency_Hz,
                multisine_period=multisine_period_s,
                periods_per_sample=1,
        )
        # One DMFA sample per period, taken at the start of each period
        L = sampling_frequency_Hz * multisine_period_s
        impedance_expected = impedance_ohm[:, ::L]
        assert impedance_expected.shape == impedance_est.shape
        # The ramp is not periodic: the DFT sees a jump from the end of the record
        # back to its start, and the filter spreads it over the first and last samples
        edge = 5
        rel_err = np.max(np.abs(impedance_est[:, edge:-edge] / impedance_expected[:, edge:-edge] - 1))
        assert 1e-6 < rel_err < 1e-4

def test_dmfa_fast_varying():
        current_A, voltage_V, impedance_ohm, frequencies_Hz = non_stationary_r_rc(
                fs=sampling_frequency_Hz,
                multisine_period=multisine_period_s,
                harmonics=harmonics,
                periods=periods,
                R1rate=0.1,
                R2rate=0.1,
        )
        record_samples = voltage_V.size
        impedance_est, time = dmfa(
                U=fft(current_A)/record_samples,
                V=fft(voltage_V)/record_samples,
                k_exc=get_excited_bins(record_samples, frequencies_Hz, sampling_frequency_Hz),
                passband_filter=FermiDiracFilter(halfwidth=0.25, order=4),
                fs=sampling_frequency_Hz,
                multisine_period=multisine_period_s,
                periods_per_sample=1,
        )
        L = sampling_frequency_Hz * multisine_period_s
        impedance_expected = impedance_ohm[:, ::L]
        assert impedance_expected.shape == impedance_est.shape
        edge = 5
        rel_err = np.max(np.abs(impedance_est[:, edge:-edge] / impedance_expected[:, edge:-edge] - 1))
        assert 1e-5 < rel_err < 1e-3 # about 10x the slowly varying case
