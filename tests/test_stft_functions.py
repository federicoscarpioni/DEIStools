from deistools.processing.stft_functions  import stft_eis
import numpy as np

time_s = np.linspace(0,1,20, endpoint=False)
frequencies_Hz = np.array([1,3,5,7,9])
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

def test_stft_eis():
    impedance_est, v_mean, i_mean, time = stft_eis(voltage_V,
        current_A,
        frequencies_Hz,
        20,
        1,
        1,
    )
    impedance_expected = np.tile(impedance_ohm, (periods,1))
    assert impedance_expected.shape == impedance_est.shape
    np.testing.assert_allclose(impedance_expected, impedance_est, rtol = 1e-12)
    assert v_mean.size == periods
    assert i_mean.size == periods