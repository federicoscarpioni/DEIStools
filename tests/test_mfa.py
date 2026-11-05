from deistools.processing.mfa import MultiFrequencyAnalysis
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
periods = 5
voltage_V = np.tile(voltage_V, periods)
current_A = np.tile(current_A, periods)

def test_fft_eis():
    mfa = MultiFrequencyAnalysis(
        voltage_V,
        current_A,
        20,
        frequencies_Hz,
        periods,
    )
    mfa.compute_fft()
    mfa.compute_freq_axis()
    impedance_est = mfa.run_fft_eis()
    np.testing.assert_allclose(impedance_est, impedance_ohm, rtol = 1e-12)