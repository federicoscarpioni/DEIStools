"""Stationary-limit tests for BLTVA.

On a time-invariant signal BLTVA must return:
  - H_p[:, 0] = Z_true  (the time-averaged transfer function is the known
    impedance at each excited frequency)
  - H_p[:, p>=1] ~ 0    (no time variation, so higher Legendre orders vanish)

These are the degenerate limit of the oracle tests in ``test_bltva_oracle``
and are deliberately weak: with no time variation almost any hyperparameter
choice passes. They guard the LTI case; the oracle tests carry the real load.

Hyperparameters satisfy the identifiability condition of Hallemans TIM 2021
eq. (49), ``Nq + 3*Np + Na + 1 < 2P - 1`` -> 4 + 6 + 6 + 1 = 17 < 19.
"""
import numpy as np
from deistools.processing.bltva import bltva
from tests.conftest import FS, P_PERIODS, H_EXC

FIT = dict(Np=2, Nq=4, Na=6, harmonics="excited",
           band_halfwidth=5 * P_PERIODS)


def test_bltva_stationary_impedance(stationary_sim, z_true_stationary):
    """H_p[:, 0] must match Z_true to < 1e-6 relative error."""
    r = bltva(stationary_sim['voltage_V'], stationary_sim['current_A'],
              FS, P_PERIODS, H_EXC, **FIT)

    rel_err = np.abs(r.H_p[:, 0] - z_true_stationary) / np.abs(z_true_stationary)
    assert rel_err.max() < 1e-6, (
        f"BLTVA H_p[:,0] deviates from Z_true: max relative error {rel_err.max():.3e}"
    )


def test_bltva_no_time_variation(stationary_sim):
    """Higher-order coefficients H_p[:, p>=1] must be negligible."""
    r = bltva(stationary_sim['voltage_V'], stationary_sim['current_A'],
              FS, P_PERIODS, H_EXC, **FIT)

    leak = np.abs(r.H_p[:, 1:]).max(axis=1) / np.abs(r.H_p[:, 0])
    assert leak.max() < 1e-6, (
        f"BLTVA leaks into higher Legendre orders: max |H_p>=1|/|H_0| = {leak.max():.3e}"
    )
