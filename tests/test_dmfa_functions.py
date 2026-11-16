import numpy as np
from numpy.fft import fft
from deistools.processing.dmfa_functions import dmfa
from deistools.processing.passband_filters import FermiDiracFilter
from deistools.processing.stft_functions import get_excited_bins

sampling_frequency_Hz = 20
harmonics = np.array([1,3,5,7,9])
base_frequency_Hz = 1
multisine_period_s = 1/base_frequency_Hz
frequencies_Hz = harmonics * base_frequency_Hz
n_samples = int(sampling_frequency_Hz * multisine_period_s)
time_s = np.linspace(0,1,n_samples, endpoint=False)
amplitudes_A = np.ones_like(frequencies_Hz) * 1e-4
R1 = 10
R2 =10
C = 5e-3
omega = 2 * np.pi * frequencies_Hz
impedance_ohm = R1 + R2 / (1+ 1j * omega * R2 * C)
current_A = np.zeros(len(time_s))
voltage_V = np.zeros(len(time_s))
for k, (f, A) in enumerate(zip(frequencies_Hz, amplitudes_A)):
        current_A += A * np.cos(2 * np.pi * f * time_s)
        voltage_V += A * np.abs(impedance_ohm[k]) * np.cos(2 * np.pi * f * time_s +  np.angle(impedance_ohm[k]))
periods = 50
voltage_V = np.tile(voltage_V, periods)
current_A = np.tile(current_A, periods)
record_length_s = multisine_period_s * periods

def test_dmfa():
        record_samples = int(voltage_V.shape[0])
        ft_voltage = fft(voltage_V)/record_samples
        ft_current = fft(current_A)/record_samples
        n_bins = periods
        DFT_bins  = np.fft.fftfreq(n_bins, d=1/n_bins).astype(int)
        frequency_axis = DFT_bins / record_length_s
        filter = FermiDiracFilter(frequency_axis,0,0.25,4)
        print(filter.values.shape)
        excited_bins = get_excited_bins(
                record_samples,
                frequencies_Hz,
                sampling_frequency_Hz,
        )
        impedance_est = dmfa(
                ft_voltage, 
                ft_current,
                excited_bins, 
                record_samples,
                filter.values,
        )
        impedance_expected = np.tile(impedance_ohm[:, None], (1, periods))
        assert impedance_expected.shape == impedance_est.shape
        np.testing.assert_allclose(impedance_est,impedance_expected, rtol = 1e-12)