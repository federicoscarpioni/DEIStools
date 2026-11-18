import numpy as np
from numpy.fft import fft
from deistools.processing.bltva_functions import bltva, evaluate_transfer_function
from deistools.processing.stft_functions import get_excited_bins
from synthetic_records import stationary_r_rc, legendre_oracle, non_stationary_r_rc

sampling_frequency_Hz = 20
multisine_period_s = 1
harmonics = np.array([1,3,5,7,9])
periods = 50

def test_bltva():
        current_A, voltage_V, impedance_ohm, frequencies_Hz = stationary_r_rc(
                fs=sampling_frequency_Hz,
                multisine_period=multisine_period_s,
                harmonics=harmonics,
                periods=periods,
        )
        record_samples = voltage_V.size
        ft_voltage = fft(voltage_V)/record_samples
        ft_current = fft(current_A)/record_samples
        excited_bins = get_excited_bins(
                        record_samples,
                        frequencies_Hz,
                        sampling_frequency_Hz,
                )
        halfband = 25
        Np = 2
        Nq = 1
        G_p, basis, res_abs, res_rel, rank = bltva(
                ft_current,
                ft_voltage,
                excited_bins,
                periods,
                halfband,
                Np,
                Nq,
        )
        impedance_est = evaluate_transfer_function(
                G_p,
                basis,
        )
        impedance_expected = np.tile(impedance_ohm[:, None], (1, record_samples))
        assert impedance_expected.shape == impedance_est.shape
        np.testing.assert_allclose(impedance_est,impedance_expected, rtol = 1e-12)


def test_bltva_time_varying():
        # Z(f,t) = sum_p G_p(f) b_p(t), built on the estimator's own basis, so
        # BLTVA must recover G_p exactly. A stationary record cannot check H_1, H_2.
        Np, Nq = 2, 2
        current, voltage, G, frequencies_Hz = legendre_oracle(
                fs=sampling_frequency_Hz,
                multisine_period=multisine_period_s,
                harmonics=harmonics,
                periods=periods,
                scales=(1, 0.05, -0.02),
        )
        record_samples = current.size
        G_p, basis, res_abs, res_rel, rank = bltva(
                U=fft(current) / record_samples,
                V=fft(voltage) / record_samples,
                k_exc=harmonics * periods,
                P=periods,
                halfband=25,
                Np=Np,
                Nq=Nq,
        )
        np.testing.assert_allclose(G_p, G, rtol=1e-11, atol=1e-11 * np.abs(G[:, 0]).max())

def test_bltva_slowly_varying():
        current_A, voltage_V, impedance_ohm, frequencies_Hz = non_stationary_r_rc(
                fs=sampling_frequency_Hz,
                multisine_period=multisine_period_s,
                harmonics=harmonics,
                periods=periods,
        )
        record_samples = voltage_V.size
        G_p, basis, res_abs, res_rel, rank = bltva(
                U=fft(current_A) / record_samples,
                V=fft(voltage_V) / record_samples,
                k_exc=harmonics * periods,
                P=periods,
                halfband=25,
                Np=2,
                Nq=1,
        )
        impedance_est = evaluate_transfer_function(G_p, basis)
        assert impedance_ohm.shape == impedance_est.shape
        # Z(t) is not a polynomial in t (R2 sits in the denominator), so a
        # second-order Legendre fit cannot reach machine precision
        rel_err = np.max(np.abs(impedance_est / impedance_ohm - 1))
        assert 1e-7 < rel_err < 1e-5

def test_bltva_fast_varying():
        current_A, voltage_V, impedance_ohm, frequencies_Hz = non_stationary_r_rc(
                fs=sampling_frequency_Hz,
                multisine_period=multisine_period_s,
                harmonics=harmonics,
                periods=periods,
                R1rate=0.1,
                R2rate=0.1,
        )
        record_samples = voltage_V.size
        G_p, basis, res_abs, res_rel, rank = bltva(
                U=fft(current_A) / record_samples,
                V=fft(voltage_V) / record_samples,
                k_exc=harmonics * periods,
                P=periods,
                halfband=25,
                Np=2,
                Nq=1,
        )
        impedance_est = evaluate_transfer_function(G_p, basis)
        assert impedance_ohm.shape == impedance_est.shape
        rel_err = np.max(np.abs(impedance_est / impedance_ohm - 1))
        assert 1e-4 < rel_err < 1e-2 # far more than 10x the slowly varying case: the curvature of Z(t) grows with the rate
