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


# --------------------------------------------------------------------------- #
# Filter shapes
# --------------------------------------------------------------------------- #
def test_every_filter_family_means_the_same_by_bandwidth():
    """bw is the half-amplitude half-width in all three families.

    Without this they cannot be compared at equal bandwidth, and the sweep
    that found the published filter to be the weakest would be meaningless.
    """
    import numpy as np
    from deistools.processing.passband_filters import (
        create_fermi_dirac_fun, create_raised_cosine_fun, create_window_filter)

    bw = 0.25
    assert abs(create_fermi_dirac_fun(bw, 0.0, bw, 8) - 0.5) < 1e-3
    assert abs(create_raised_cosine_fun(np.array([bw]), 0.0, bw, 1.0)[0] - 0.5) < 1e-9
    for window in ("blackman", "blackman-harris", "nuttall", "hann",
                   "kaiser", "gaussian"):
        value = create_window_filter(np.array([bw]), 0.0, bw, window)[0]
        assert abs(value - 0.5) < 1e-3, window
        peak = create_window_filter(np.array([0.0]), 0.0, bw, window)[0]
        assert abs(peak - 1.0) < 1e-9, window


def test_compactly_supported_filters_really_vanish():
    """The cosine-sum windows truncate nothing, unlike the Fermi-Dirac.

    That is their one structural advantage: a finite block of bins around a
    tone loses no filter energy at all.
    """
    import numpy as np
    from deistools.processing.passband_filters import (
        create_fermi_dirac_fun, create_window_filter)

    far = np.array([6.0])
    assert create_window_filter(far, 0.0, 1.0, "blackman")[0] == 0.0
    assert create_window_filter(far, 0.0, 1.0, "nuttall")[0] == 0.0
    assert create_fermi_dirac_fun(far, 0.0, 1.0, 2)[0] > 0.0   # exponential tail


def test_unknown_window_is_refused():
    import pytest
    from deistools.processing.passband_filters import create_window_filter
    with pytest.raises(ValueError, match="unknown window"):
        create_window_filter(0.0, 0.0, 1.0, "bartlett")


def test_independent_time_step_is_half_over_bandwidth():
    from deistools.processing.dmfa_functions import independent_time_step
    assert independent_time_step(0.025) == 20.0


def test_zero_padding_fixes_the_edges_without_touching_the_interior():
    """The edge oscillation is circular wrap-around, and padding removes it.

    Multiplying a spectrum by a filter convolves circularly in time, so the
    kernel reaching past the end of the record picks up the beginning. On a
    time-varying system the two ends differ, and the estimate near either
    boundary is contaminated by the far one. Padding makes the convolution
    linear.

    The interior must not move: the kernel never reaches that far.
    """
    import numpy as np
    from numpy.fft import fft, fftshift, fftfreq, ifft, ifftshift
    from deistools.processing.dmfa_functions import (
        pad_record, elaboration_frequency_axis)
    from deistools.processing.passband_filters import create_fermi_dirac_fun
    from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor

    Tp, fs, P = 20.0, 1000.0, 10
    harmonics = np.array([1, 3, 5, 9, 13, 17, 27, 39, 59])
    freqs = harmonics / Tp
    n = int(fs * Tp) * P
    T, dt = n / fs, 1.0 / fs
    phases = np.random.default_rng(3).uniform(0, 2 * np.pi, freqs.size)
    sim = simulate_r_rc_transistor(
        T, dt, freqs, np.full(freqs.size, 0.01), phases,
        R_s=10.0, R_p=30.0, C=1e-3, V_gate_start=1.645,
        V_gate_rate=(2.455 - 1.645) / T, R_T_max=70.0,
        noise_std_V=1e-4, seed=1)
    bw = 0.5 * freqs.min()

    def run(pad):
        v, i = sim["voltage_V"], sim["current_A"]
        if pad:
            v, i = pad_record(v), pad_record(i)
        n_rec = v.size
        T_rec = n_rec * dt
        FV, FI = fftshift(fft(v)) / n_rec, fftshift(fft(i)) / n_rec
        axis = fftshift(fftfreq(n_rec, dt))
        idx = np.round(freqs * T_rec).astype(int) + int(np.argmin(np.abs(axis)))
        npts = int(round(24 * bw * T_rec)); npts -= npts % 2
        npts = min(npts, 2 * min(idx.min(), n_rec - idx.max()))
        g = create_fermi_dirac_fun(elaboration_frequency_axis(npts, T_rec), 0, bw, 2)
        Z = np.empty((freqs.size, npts), complex)
        for k in range(freqs.size):
            r = np.arange(idx[k] - npts // 2, idx[k] + npts // 2)
            Z[k] = ifft(ifftshift(FV[r] * g)) / ifft(ifftshift(FI[r] * g))
        times = np.arange(npts) * (T_rec / npts)
        inside = times <= T * 1.0000001
        return Z[:, inside], times[inside]

    def error(Z, times):
        ref = np.array([
            np.interp(times, sim["time_s"], sim["Z_true"][k].real)
            + 1j * np.interp(times, sim["time_s"], sim["Z_true"][k].imag)
            for k in range(freqs.size)])
        return np.abs(Z - ref) / np.abs(ref)

    Z_plain, t_plain = run(False)
    Z_pad, t_pad = run(True)
    e_plain, e_pad = error(Z_plain, t_plain), error(Z_pad, t_pad)

    assert np.max(e_pad) < np.max(e_plain) / 5      # measured ~25x
    assert 0.8 < np.median(e_pad) / np.median(e_plain) < 1.25   # interior fixed


def test_dmfa_is_least_squares_in_a_fourier_basis():
    """DMFA with a rectangular filter IS a linear least-squares fit.

    The model is 'the envelope is band-limited', which is the same as 'the
    envelope is a trigonometric polynomial of degree bw*T'. Fitting those
    coefficients by least squares gives exactly the band extraction -- the
    design matrix is orthogonal, so the normal equations are diagonal and the
    solve degenerates into reading the DFT bins.

    This is the sense in which DMFA and BLTVA are the same kind of estimator
    in different bases: Fourier for one, Legendre for the other.
    """
    import numpy as np
    from numpy.fft import fft, fftshift, fftfreq, ifft, ifftshift

    Tp, fs, P = 1.0, 200.0, 8
    freqs = np.array([1.0, 3.0, 7.0])
    n = int(fs * Tp) * P
    T, dt = n / fs, 1.0 / fs
    t = np.arange(n) * dt
    phases = np.random.default_rng(0).uniform(0, 2 * np.pi, freqs.size)
    Z = np.vstack([(20 + 10 * t / T) * np.exp(-0.3j),
                   (12 + 6 * t / T) * np.exp(-0.5j),
                   (6 + 3 * t / T) * np.exp(-0.8j)])
    v = sum(0.01 * np.abs(Z[k]) * np.cos(2 * np.pi * freqs[k] * t + phases[k]
                                         + np.angle(Z[k]))
            for k in range(freqs.size))

    tone, M = 0, 4
    npts = 2 * M + 1
    modes = np.arange(-M, M + 1)
    t_out = np.arange(npts) * (T / npts)

    FV = fftshift(fft(v)) / n
    axis = fftshift(fftfreq(n, dt))
    k_e = int(round(freqs[tone] * T)) + int(np.argmin(np.abs(axis)))
    band = np.arange(k_e - M, k_e + M + 1)
    # the one-sided band carries half the real phasor, hence the 2
    by_filtering = 2 * npts * ifft(ifftshift(FV[band]))

    columns = []
    for m in modes:
        arg = 2 * np.pi * (freqs[tone] + m / T) * t
        columns += [np.cos(arg), -np.sin(arg)]
    design = np.column_stack(columns)
    theta, *_ = np.linalg.lstsq(design, v, rcond=None)
    coeffs = theta[0::2] + 1j * theta[1::2]
    by_least_squares = np.array(
        [np.sum(coeffs * np.exp(2j * np.pi * modes * tt / T)) for tt in t_out])

    assert np.linalg.cond(design) < 1.001        # orthogonal columns
    rel = np.abs(by_filtering - by_least_squares).max() / np.abs(by_filtering).max()
    assert rel < 1e-10, rel


def test_the_variance_matches_the_scatter_it_predicts():
    """extract_impedance_with_error against 120 noise realisations.

    On a stationary cell the true impedance never moves, so the spread of the
    estimate over noise realisations IS its error. The prediction comes from
    treating DMFA as what it is -- a linear map applied to the measurement --
    and it lands within a few percent.
    """
    import numpy as np
    from numpy.fft import fft, fftshift, fftfreq
    from deistools.processing.dmfa_functions import (
        extract_impedance_with_error, elaboration_frequency_axis)
    from deistools.processing.passband_filters import create_fermi_dirac_fun
    from deistools.simulation.models.r_rc import simulate_r_rc

    Tp, fs, P = 20.0, 1000.0, 6
    harmonics = np.array([1, 3, 5, 9, 13, 17])
    freqs = harmonics / Tp
    n = int(fs * Tp) * P
    T, dt = n / fs, 1.0 / fs
    phases = np.random.default_rng(3).uniform(0, 2 * np.pi, freqs.size)
    amps = np.full(freqs.size, 0.01)
    bw = 0.5 * freqs.min()
    npts = int(round(16 * bw * T)); npts -= npts % 2
    g = create_fermi_dirac_fun(elaboration_frequency_axis(npts, T), 0, bw, 2)
    axis = fftshift(fftfreq(n, dt))
    idx = (np.round(freqs * T).astype(int)
           + int(np.argmin(np.abs(axis)))).tolist()

    sigma_v = 2e-3
    estimates, variance = [], None
    for seed in range(120):
        sim = simulate_r_rc(T, dt, freqs, amps, phases, R_s=10.0, R_p=30.0,
                            C=1e-3, noise_std_V=sigma_v, seed=seed)
        FV = fftshift(fft(sim["voltage_V"])) / n
        FI = fftshift(fft(sim["current_A"])) / n
        Z, variance = extract_impedance_with_error(
            FV, FI, sigma_v ** 2, 0.0, idx, g, npts)
        estimates.append(Z)

    observed = np.std(np.array(estimates).real, axis=0)
    ratio = np.sqrt(variance) / observed
    assert 0.85 < np.median(ratio) < 1.15, np.median(ratio)


def test_stopband_rejection_at_dc_is_what_controls_a_trend():
    """DMFA has no trend model; the filter's dc rejection is the whole defence.

    A trend lives at zero frequency, so for the tone at f_k it sits f_k from
    the band centre. Raising the Fermi-Dirac sharpness from 2 to 8 takes the
    rejection there from 18 dB to 69 dB, and that alone removes most of a
    trend two orders of magnitude larger than the signal.
    """
    import numpy as np
    from numpy.fft import fft, fftshift, fftfreq, ifft, ifftshift
    from deistools.processing.passband_filters import (
        create_fermi_dirac_fun, rejection_db)
    from deistools.processing.dmfa_functions import (
        elaboration_frequency_axis, elaboration_time_axis)
    from deistools.processing.bltva import legendre_basis
    from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor

    Tp, fs, P = 20.0, 1000.0, 10
    harmonics = np.array([1, 3, 5, 9, 13, 17, 27, 39, 59])
    freqs = harmonics / Tp
    n = int(fs * Tp) * P
    T, dt = n / fs, 1.0 / fs
    phases = np.random.default_rng(3).uniform(0, 2 * np.pi, freqs.size)
    sim = simulate_r_rc_transistor(
        T, dt, freqs, np.full(freqs.size, 0.01), phases,
        R_s=10.0, R_p=30.0, C=1e-3, V_gate_start=1.645,
        V_gate_rate=(2.455 - 1.645) / T, R_T_max=70.0,
        noise_std_V=1e-4, seed=1)
    trend = legendre_basis(n, 3) @ np.array([0.0, 2.0, 1.0, 0.5])

    bw = 0.5 * freqs.min()
    npts = int(round(24 * bw * T)); npts -= npts % 2
    axis = elaboration_frequency_axis(npts, T)
    times = elaboration_time_axis(npts, T)
    spectrum_axis = fftshift(fftfreq(n, dt))
    idx = np.round(freqs * T).astype(int) + int(np.argmin(np.abs(spectrum_axis)))
    truth = np.array([np.interp(times, sim["time_s"], sim["Z_true"][k].real)
                      + 1j * np.interp(times, sim["time_s"], sim["Z_true"][k].imag)
                      for k in range(freqs.size)])

    def lowest_tone_error(sharpness):
        g = create_fermi_dirac_fun(axis, 0, bw, sharpness)
        FV = fftshift(fft(sim["voltage_V"] + trend)) / n
        FI = fftshift(fft(sim["current_A"])) / n
        r = np.arange(idx[0] - npts // 2, idx[0] + npts // 2)
        Z = ifft(ifftshift(FV[r] * g)) / ifft(ifftshift(FI[r] * g))
        err = np.abs(Z - truth[0]) / np.abs(truth[0])
        cut = max(1, npts // 10)
        return np.median(err[cut:-cut])

    soft = create_fermi_dirac_fun(axis, 0, bw, 2)
    sharp = create_fermi_dirac_fun(axis, 0, bw, 8)
    assert rejection_db(soft, axis, freqs[0]) < 25
    assert rejection_db(sharp, axis, freqs[0]) > 60
    # 7.8x on this 10-period record; the effect grows with the record,
    # reaching 211x at the 20 periods the documentation measures.
    assert lowest_tone_error(8) < lowest_tone_error(2) / 5
