import numpy as np
from numpy.fft import fft
from deistools.processing.dmfa_functions import dmfa
from deistools.processing.passband_filters import FermiDiracFilter
from deistools.processing.stft_functions import get_excited_bins
from synthetic_records import stationary_r_rc

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
        impedance_est = dmfa(
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
