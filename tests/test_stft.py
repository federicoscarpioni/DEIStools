"""Regression test for the windowed-DFT (STFT) estimator on a stationary R+RC.

On a stationary signal with no noise, period-by-period FFT-EIS must return
the true impedance at every excited frequency in every time window.

Both tests now drive ``estimate_impedance`` directly. The accuracy test used
to bypass it, computing the bin indexes and calling ``fft_eis`` by hand,
because ``search_freq_indexes`` returned wrong bins -- see
``test_estimator_bugs`` for what that was and what it cost.
"""
import numpy as np

from deistools.processing.stft_functions import estimate_impedance
from tests.conftest import FREQUENCIES, F_MIN, DT, P_PERIODS


def test_stft_stationary_impedance(stationary_sim, z_true_stationary):
    """Each time window must recover Z_true to < 1e-6 relative error."""
    impedance, _, _, _ = estimate_impedance(
        stationary_sim['voltage_V'], stationary_sim['current_A'],
        FREQUENCIES, 1.0 / F_MIN, DT,
    )

    rel_err = np.abs(impedance - z_true_stationary[:, None]) / np.abs(z_true_stationary[:, None])
    assert rel_err.max() < 1e-6, (
        f"STFT deviates from Z_true: max relative error {rel_err.max():.3e}"
    )


def test_stft_returns_correct_shapes(stationary_sim):
    """Output shapes must be consistent with the number of windows."""
    impedance, voltage_avg, current_avg, time = estimate_impedance(
        stationary_sim['voltage_V'], stationary_sim['current_A'],
        FREQUENCIES, 1.0 / F_MIN, DT,
    )

    assert impedance.shape == (len(FREQUENCIES), P_PERIODS)
    assert voltage_avg.shape == current_avg.shape == (P_PERIODS,)
    assert time.shape == (P_PERIODS,)
