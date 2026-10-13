"""Regression test for the DMFA estimator on a stationary R+RC.

On a stationary signal with no noise, DMFA with a well-chosen Fermi-Dirac
filter must recover the true impedance at every excited frequency.  The
pointwise V(t)/I(t) division produces outliers near current zero crossings,
so we check the median relative error rather than the max.
"""
import numpy as np
from numpy.fft import fft, fftshift, fftfreq

from deistools.processing.dmfa_functions import (
    extract_impedance, max_elaboration_points, elaboration_frequency_axis,
)
from deistools.processing.passband_filters import create_fermi_dirac_fun
from tests.conftest import FREQUENCIES, FS, DT


def test_dmfa_stationary_impedance(stationary_sim, z_true_stationary):
    """Median relative error over time must be < 1e-2 at each frequency."""
    voltage = stationary_sim['voltage_V']
    current = stationary_sim['current_A']
    N = voltage.size

    # Shifted, amplitude-normalized DFT (DMFA convention)
    ft_voltage = fftshift(fft(voltage)) / N
    ft_current = fftshift(fft(current)) / N
    freq_axis = fftshift(fftfreq(N, DT))

    # Frequency bin indices in the shifted DFT
    index_f0 = np.where(freq_axis == 0)[0][0]
    freq_indexes = (np.round(FREQUENCIES * DT * N) + index_f0).astype(int).tolist()

    # The extraction window must fit inside the spectrum at every frequency;
    # extract_impedance now enforces this and max_elaboration_points reports
    # the limit, rather than the caller having to derive it.
    Npts_elab = max_elaboration_points(freq_indexes, N)

    bw = float(np.diff(FREQUENCIES).min()) / 2.0
    order = 10
    # The band spans Npts_elab/T Hz, not fs. Building the filter on a
    # linspace(-fs/2, fs/2) axis mis-scales its bandwidth by fs*T/Npts_elab.
    filt = create_fermi_dirac_fun(
        elaboration_frequency_axis(Npts_elab, N * DT), 0, bw, order)

    impedance = extract_impedance(ft_voltage, ft_current, freq_indexes, filt, Npts_elab)

    # impedance shape: (n_freq, Npts_elab)
    rel_err = np.abs(impedance - z_true_stationary[:, None]) / np.abs(z_true_stationary[:, None])

    # Median over time — robust to V/I division outliers at current zero crossings
    median_err = np.median(rel_err, axis=1)
    assert median_err.max() < 1e-2, (
        f"DMFA median relative error too large: {median_err.max():.3e} "
        f"(per-freq: {median_err})"
    )
