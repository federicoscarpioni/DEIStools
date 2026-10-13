"""Regression tests for two bugs in the STFT and DMFA estimators.

Both were found while building the oracle layer and were, until now, worked
around inside the tests that should have caught them -- the worst possible
state for a bug to be in, since the suite was green while the shipped
function was wrong.

**STFT.** ``estimate_impedance`` passed the *time series* of the first window
to ``search_freq_indexes``, which indexes its argument with *DFT bin numbers*
to locate spectral peaks. The peak search therefore ran over time samples at
frequency-bin offsets and returned plausible-looking but wrong bins -- 5004,
5009, 5018 instead of 5005, 5010, 5020 -- giving impedance errors of 23% to
118% at the affected frequencies while the high frequencies came out exact.
A separate fault crashed the search outright when the frequency spacing
approached the bin spacing, because the search window rounded to zero width.

**DMFA.** ``extract_impedance`` reads a window of ``Npts_elab`` bins centred
on each excited frequency without checking that the window fits inside the
spectrum. With the natural choice ``Npts_elab = N`` it indexed past the end
and raised ``IndexError`` from inside numpy.
"""
import numpy as np
import pytest
from numpy.fft import fft, fftshift, fftfreq

from deistools.processing.stft_functions import (
    estimate_impedance, search_freq_indexes, compute_freq_indexes,
)
from deistools.processing.dmfa_functions import (
    extract_impedance, max_elaboration_points,
)
from deistools.processing.passband_filters import create_fermi_dirac_fun
from deistools.simulation.models.r_rc import impedance as r_rc_impedance
from tests.conftest import (
    FREQUENCIES, F_MIN, FS, DT, R_S, R_P, C, P_PERIODS,
)


# --------------------------------------------------------------------------- #
# STFT
# --------------------------------------------------------------------------- #

def test_stft_recovers_impedance_end_to_end(stationary_sim, z_true_stationary):
    """The accuracy test now goes through estimate_impedance itself.

    It previously bypassed it, driving compute_freq_indexes and fft_eis by
    hand, because the function under test returned wrong bins.
    """
    impedance, _, _, time = estimate_impedance(
        stationary_sim['voltage_V'], stationary_sim['current_A'],
        FREQUENCIES, 1.0 / F_MIN, DT)

    rel_err = np.abs(impedance - z_true_stationary[:, None]) / np.abs(z_true_stationary[:, None])
    assert rel_err.max() < 1e-6, f"max relative error {rel_err.max():.3e}"
    assert time.shape == (P_PERIODS,)


def test_search_finds_the_same_bins_as_the_direct_computation(stationary_sim):
    """On a clean signal whose frequencies land exactly on bins, the peak
    search must agree with the arithmetic. It used to disagree by 1 to 7 bins
    at the lower frequencies."""
    samples_window = int(1.0 / F_MIN / DT)
    freq_axis = fftshift(fftfreq(samples_window, DT))
    spectrum = fftshift(fft(stationary_sim['current_A'][:samples_window])) / samples_window

    computed, _ = compute_freq_indexes(freq_axis, FREQUENCIES, DT)
    searched, _ = search_freq_indexes(spectrum, freq_axis, FREQUENCIES, DT)

    assert searched == computed


def test_search_survives_spacing_equal_to_bin_resolution(stationary_sim):
    """A 1 Hz spacing in a 1 s window gives a search half-width that rounds to
    zero. The window is now floored at one bin instead of being empty."""
    samples_window = int(1.0 / F_MIN / DT)
    freq_axis = fftshift(fftfreq(samples_window, DT))
    spectrum = fftshift(fft(stationary_sim['current_A'][:samples_window])) / samples_window

    tight = np.array([1.0, 2.0, 3.0, 10.0])      # 1 Hz apart, 1 Hz bins
    indexes, _ = search_freq_indexes(spectrum, freq_axis, tight, DT)

    assert len(indexes) == tight.size
    assert all(isinstance(i, int) for i in indexes)


def test_search_locates_an_off_bin_peak():
    """The point of the search: find the peak when it is not where the
    arithmetic predicts."""
    n, dt = 4096, 1e-3
    freq_axis = fftshift(fftfreq(n, dt))
    index_f0 = n // 2
    spectrum = np.zeros(n, dtype=complex)
    peak = index_f0 + 40
    spectrum[peak] = 1.0                          # true peak, 3 bins off

    nominal = np.array([37.0 / (n * dt) * 1.0])   # points at bin index_f0+37
    nominal = np.array([37.0 / (n * dt)])
    found, _ = search_freq_indexes(spectrum, freq_axis, nominal, dt)

    assert found[0] == peak


def test_compute_freq_indexes_rounds_to_the_nearest_bin():
    """Nearest bin, not the one above. ceil pushed an off-grid frequency a
    bin too high.

    ``tol=None`` is needed because these frequencies are deliberately off
    grid, which the function now rejects by default -- see
    ``test_compute_freq_indexes_rejects_off_grid_tones``.
    """
    n, dt = 1000, 1e-3
    freq_axis = fftshift(fftfreq(n, dt))
    index_f0 = n // 2

    # 10.4 Hz -> bin offset 10.4, nearest is 10
    idx, f0 = compute_freq_indexes(freq_axis, np.array([10.4]), dt, tol=None)
    assert f0 == index_f0
    assert idx == [index_f0 + 10]

    # 10.6 Hz -> nearest is 11
    idx, _ = compute_freq_indexes(freq_axis, np.array([10.6]), dt, tol=None)
    assert idx == [index_f0 + 11]


def test_compute_freq_indexes_rejects_off_grid_tones():
    """A multisine tone that does not land on a bin means the record length
    and the multisine period disagree, which is a design error rather than
    something to round away."""
    n, dt = 1000, 1e-3
    freq_axis = fftshift(fftfreq(n, dt))

    with pytest.raises(ValueError, match="from a bin centre"):
        compute_freq_indexes(freq_axis, np.array([10.0, 10.4]), dt)

    # exactly on grid is fine
    idx, _ = compute_freq_indexes(freq_axis, np.array([10.0, 25.0]), dt)
    assert idx == [n // 2 + 10, n // 2 + 25]


def test_compute_freq_indexes_rejects_frequencies_above_nyquist():
    n, dt = 1000, 1e-3                       # Nyquist 500 Hz
    freq_axis = fftshift(fftfreq(n, dt))

    with pytest.raises(ValueError, match="Nyquist"):
        compute_freq_indexes(freq_axis, np.array([600.0]), dt)


def test_compute_freq_indexes_rejects_an_unshifted_axis():
    """The contract is a shifted axis. An unshifted one has its zero bin at
    index 0, so every index would come out wrong by n//2."""
    n, dt = 1000, 1e-3
    with pytest.raises(ValueError, match="shifted axis"):
        compute_freq_indexes(np.arange(1.0, n + 1.0), np.array([10.0]), dt)


def test_search_accepts_off_grid_tones():
    """The search exists precisely for tones that are not where they were
    asked to be, so it must not inherit the on-grid check."""
    n, dt = 4096, 1e-3
    freq_axis = fftshift(fftfreq(n, dt))
    spectrum = np.zeros(n, dtype=complex)
    peak = n // 2 + 40
    spectrum[peak] = 1.0

    found, _ = search_freq_indexes(spectrum, freq_axis,
                                   np.array([37.4 / (n * dt)]), dt)
    assert found[0] == peak


# --------------------------------------------------------------------------- #
# DMFA
# --------------------------------------------------------------------------- #

def _dmfa_setup(stationary_sim):
    n = stationary_sim['voltage_V'].size
    ft_v = fftshift(fft(stationary_sim['voltage_V'])) / n
    ft_i = fftshift(fft(stationary_sim['current_A'])) / n
    freq_axis = fftshift(fftfreq(n, DT))
    index_f0 = np.where(freq_axis == 0)[0][0]
    indexes = (np.round(FREQUENCIES * DT * n) + index_f0).astype(int).tolist()
    return ft_v, ft_i, indexes, n


def test_max_elaboration_points_is_the_binding_constraint(stationary_sim):
    _, _, indexes, n = _dmfa_setup(stationary_sim)
    limit = max_elaboration_points(indexes, n)

    assert limit == 2 * min(min(indexes), n - max(indexes))
    assert max(indexes) + limit // 2 <= n
    assert min(indexes) - limit // 2 >= 0


def test_dmfa_rejects_an_oversized_window(stationary_sim):
    """Npts_elab = N is the natural first guess and used to raise IndexError
    from inside numpy. It now explains itself."""
    ft_v, ft_i, indexes, n = _dmfa_setup(stationary_sim)
    filt = create_fermi_dirac_fun(np.linspace(-FS / 2, FS / 2, n), 0, 1.0, 10)

    with pytest.raises(ValueError, match="does not fit"):
        extract_impedance(ft_v, ft_i, indexes, filt, n)


def test_dmfa_accepts_the_largest_valid_window(stationary_sim, z_true_stationary):
    ft_v, ft_i, indexes, n = _dmfa_setup(stationary_sim)
    limit = max_elaboration_points(indexes, n)
    filt = create_fermi_dirac_fun(
        np.linspace(-FS / 2, FS / 2, limit), 0,
        float(np.diff(FREQUENCIES).min()) / 2.0, 10)

    impedance = extract_impedance(ft_v, ft_i, indexes, filt, limit)

    assert impedance.shape == (len(indexes), limit)
    rel_err = np.abs(impedance - z_true_stationary[:, None]) / np.abs(z_true_stationary[:, None])
    assert np.median(rel_err, axis=1).max() < 1e-2
