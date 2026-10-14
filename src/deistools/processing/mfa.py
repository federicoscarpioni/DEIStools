from numpy.fft import fft, fftshift, fftfreq, ifft, ifftshift
import numpy as np

from deistools.visualise import inspect_spectrum,inspect_spectrum_phase, visualise_peaks, visualise_peaks_phase
# from deistools.processing import fermi_dirac_filter
from deistools.processing.dmfa_functions import extract_zero_frequency, extract_impedance, visualize_dmfa_filtering, extract_impedance_with_error
from deistools.processing.stft_functions import (
    estimate_impedance as stft_eis, compute_freq_indexes, search_freq_indexes,
)
from deistools.processing.bltva import (
    bltva, visualize_bltva_fit, estimate_noise_variance,
    select_na, check_identifiability,
)
from deistools.processing.consistency import (
    check_consistency, visualize_consistency,
)
from deistools.processing.goertzel import goertzel_eis, cost_ratio
from deistools.processing.bla import bla_eis
from deistools.processing.lock_in_amp import (
    lia_eis, lia_impedance_optim, lia_impedance_ls,
    integrator_length, integrator_kernel, tone_rejection, valid_mask,
)

class MultiFrequencyAnalysis:
    
    def __init__(self, frequencies: np.array, voltage: np.array, current: np.array, sampling_time:float, freq_indexes = None, impedance = None):
        self.frequencies = frequencies
        self.voltage = voltage
        self.current = current
        self.sampling_time = sampling_time
        self.freq_indexes = freq_indexes
    
    def compute_freq_axis(self):
        self.freq_axis = fftshift(fftfreq(self.voltage.size, self.sampling_time))

    def compute_fft(self):
        self.ft_voltage = fftshift(fft(self.voltage)) / self.voltage.size
        self.ft_current = fftshift(fft(self.current)) / self.current.size
    
    def check_consistency(self, period=None, n_periods=None, plot=True,
                          **kwargs):
        """Check the record against the metadata, before estimating anything.

        The first thing to run on a new dataset, and the thing to run when an
        estimate comes back looking like a cell that cannot exist. It answers
        one question -- do ``N``, ``P``, ``T_p`` and ``f_s`` agree with each
        other and with the signals? -- and when they do not, it says which of
        them the signals disagree with. See
        :mod:`deistools.processing.consistency` for how.

        Parameters
        ----------
        period : float, optional
            The multisine period in seconds: the ``period`` of
            :meth:`run_stft_eis` and the ``P`` of :meth:`run_bltva`. Pass it.
            Without it the fundamental is inferred from ``self.frequencies``,
            which can only show that the tone list is self-consistent.
        n_periods : int, optional
            How many periods you believe the record holds.
        plot : bool
            Draw :func:`~deistools.processing.consistency.visualize_consistency`.

        Returns
        -------
        :class:`~deistools.processing.consistency.ConsistencyReport`
            ``print`` it for the numbers, ``.ok`` for one boolean,
            ``.findings`` for the verdicts.

        Notes
        -----
        Deliberately independent of the rest of the object: it builds its own
        spectra and takes the period as an argument, so it works on a freshly
        constructed instance and before :meth:`compute_freq_indexes`, which is
        itself one of the things that raises when the metadata is wrong.
        """
        report = check_consistency(
            self.voltage, self.current, self.frequencies, self.sampling_time,
            period=period, n_periods=n_periods, **kwargs)
        if plot:
            visualize_consistency(report)
        return report

    def run_fft_eis(self):
        return self.ft_voltage[self.freq_indexes] / self.ft_current[self.freq_indexes]
    
    def run_stft_eis(self, period):
        impedance, voltage, current, time = stft_eis( 
            self.voltage,
            self.current,
            self.frequencies,
            period,
            self.sampling_time
        )
        return impedance, voltage, current, time

    def run_dmfa(self, 
                 filter, 
                 time_resolution,
                 Npts_elab
        ):
        voltage, current, time = extract_zero_frequency(
            self.ft_voltage,
            self.ft_current,
            self.freq_axis,
            filter,
            Npts_elab,
            time_resolution,
        )
        impedance = extract_impedance(
            self.ft_voltage,
            self.ft_current,
            self.freq_indexes,
            filter,
            Npts_elab
        )
        return impedance, voltage, current, time

    def visualize_dmfa_filtering(
            self,
            filter,
        ):
        visualize_dmfa_filtering(
            filter,
            self.ft_voltage,
            self.ft_current,
            self.freq_axis,
            self.frequencies,
            self.freq_indexes,
        )

    def run_dmfa_with_error(
            self, 
            filter, 
            time_resolution,
            Npts_elab,
            voltage_variance,
            current_variance
        ):
        voltage, current, time = extract_zero_frequency(
            self.ft_voltage,
            self.ft_current,
            self.freq_axis,
            filter,
            Npts_elab,
            time_resolution,
        )
        impedance, variance = extract_impedance_with_error(
            self.ft_voltage,
            self.ft_current,
            voltage_variance,
            current_variance,
            self.freq_indexes,
            filter,
            Npts_elab
        )
        return impedance, variance, voltage, current, time

    def run_bla(self, period, *, periods_per_window=4, n_local=30):
        """Per-burst impedance with the error attributed to its source.

        The short-time Fourier estimate, plus what the bins it discards say
        about *why* it is wrong: measurement noise, nonlinear distortion, or
        the cell changing while the burst was open.

        Parameters
        ----------
        period : float
            One multisine period in seconds.
        periods_per_window : int
            Multisine periods per burst. **At least 2**: at one period every
            bin of the burst is a harmonic of the multisine, so noise and
            distortion land in the same places. The uncertainty is paid for
            in time resolution.
        n_local : int
            Bins averaged for each level at each tone.

        Returns
        -------
        (results, t) : list of :class:`~deistools.processing.bla.BurstBLA`, ndarray
            One entry per burst, stamped at the burst midpoint.

        Notes
        -----
        Check ``r_ntv`` before reading the levels. On a cell that varies, the
        non-harmonic bins carry the time-variation skirt as well as the
        noise, so ``std_noise`` is an upper bound and the distortion cannot
        be resolved at all. :meth:`run_bltva` is the estimator that models
        the variation and separates them.
        """
        harmonics = np.round(
            self.frequencies / self.frequencies.min()).astype(int)
        return bla_eis(self.voltage, self.current, period, self.sampling_time,
                       harmonics, periods_per_window=periods_per_window,
                       n_local=n_local)

    def run_goertzel(self, period):
        """Impedance per block, evaluated with the Goertzel recurrence.

        Computes the same quantity as :meth:`run_stft_eis` -- the record cut
        into blocks of ``period`` seconds, one spectrum each -- but evaluates
        only the excited bins, by a second-order recurrence instead of a
        transform. The two agree to about 1e-10.

        On a record of this size it is the slower of the two, because an FFT
        gives every bin for ``N*log2(N)`` operations while Goertzel costs
        ``N`` per bin: :func:`goertzel_cost_ratio` reports the crossover. It
        earns its place as an independent check on the STFT path, and because
        its recurrence is the one that also runs on a live stream.

        Returns
        -------
        (impedance, time) : ndarray, ndarray
        """
        return goertzel_eis(self.voltage, self.current, self.frequencies,
                            period, self.sampling_time)

    def goertzel_cost_ratio(self, period):
        """Goertzel operations divided by FFT operations; below 1 favours Goertzel."""
        return cost_ratio(np.size(self.frequencies),
                          int(round(period / self.sampling_time)))

    def run_lia(self, n_periods=1, *, base_frequency=None, window="boxcar",
                method="mix", decimate=None, stride=None):
        """Digital lock-in estimate of Z(f,t) at every excited frequency.

        Each tone is mixed down with a cosine and a sine at its own frequency
        and the product low-pass filtered, leaving that tone's slowly varying
        phasor. The ratio of the voltage and current phasors is Z(f,t).

        Parameters
        ----------
        n_periods : int
            Integrator length, in periods of ``base_frequency``. This is the
            time resolution: ``n_periods / base_frequency`` seconds.
        base_frequency : float, optional
            The frequency every tone is a harmonic of, in Hz. Defaults to the
            lowest excited frequency, which is correct for a multisine built
            on harmonics of ``1/Tp``. The integrator has to span whole periods
            of *this* frequency, not of the tone being demodulated, because
            that is what puts the other tones in its nulls.
        window : {'boxcar', 'hann'}
            ``'boxcar'`` nulls every harmonic exactly and is the default.
            ``'hann'`` rolls off faster but needs ``n_periods >= 2``.
        method : {'mix', 'ls'}
            ``'mix'`` demodulates and filters. ``'ls'`` fits a cosine, a sine,
            an offset and a local slope in a sliding window instead, which is
            insensitive to drift within the window but far slower.
        decimate : int, optional
            Report every n-th sample (``'mix'`` only). The result is smooth on
            the scale of the integrator, and the full-rate array is large.
        stride : int, optional
            Spacing of the reported instants for ``method='ls'``.

        Returns
        -------
        (impedance, time) : ndarray, ndarray
            ``impedance`` has shape ``(n_frequencies, n_instants)``.

        Notes
        -----
        A boxcar integrator spanning exactly one base period makes this
        identical to :meth:`run_stft_eis` with a one-period window -- verified
        to 1e-13 -- evaluated at every sample instead of once per window. Use
        :meth:`lia_valid_mask` to drop the instants whose window ran past the
        end of the record.
        """
        if base_frequency is None:
            base_frequency = float(np.min(self.frequencies))
        samples_window = integrator_length(
            self.sampling_time, base_frequency, n_periods, window)
        self.lia_samples_window = samples_window

        time = np.arange(self.voltage.size) * self.sampling_time
        fun = lia_impedance_ls if method == "ls" else lia_impedance_optim
        if method == "ls":
            columns, t_out = [], None
            for f in np.asarray(self.frequencies, dtype=float):
                Z_f, t_out = lia_impedance_ls(
                    time, self.voltage, self.current, self.sampling_time, f,
                    samples_window, stride=stride)
                columns.append(Z_f)
            return np.vstack(columns), t_out
        return lia_eis(time, self.voltage, self.current, self.sampling_time,
                       self.frequencies, samples_window,
                       impedance_calculation_fun=fun, window=window,
                       decimate=decimate)

    def lia_tone_rejection(self, n_periods=1, *, base_frequency=None,
                           window="boxcar"):
        """How far the lock-in integrator suppresses each other tone, in dB.

        After mixing at one frequency, every other tone of the multisine lands
        at the difference frequency. This reports the integrator's response
        there -- one row per demodulated tone. A boxcar over whole base
        periods nulls them all, so the numbers are limited only by arithmetic.
        """
        if base_frequency is None:
            base_frequency = float(np.min(self.frequencies))
        samples_window = integrator_length(
            self.sampling_time, base_frequency, n_periods, window)
        kernel = integrator_kernel(samples_window, window)
        return np.vstack([
            tone_rejection(kernel, self.sampling_time, self.frequencies, f)
            for f in self.frequencies])

    def lia_valid_mask(self, samples_window=None):
        """False where the lock-in integrator ran past the end of the record."""
        if samples_window is None:
            samples_window = getattr(self, "lia_samples_window", None)
            if samples_window is None:
                raise ValueError("run_lia has not been called yet; pass "
                                 "samples_window explicitly")
        return valid_mask(self.voltage.size, samples_window)

    def run_bltva(self, P, *, Np=6, Nq=5, Na=None, harmonics="all_nl",
                  band_halfwidth=None, rcond=None, noise=None, prune=True,
                  detrend=None):
        """Estimate the time-varying transfer function (TIM 2021/2022).

        ``detrend`` takes a strategy from :mod:`deistools.processing.detrending`,
        in practice ``Differencing(d)``: it is the only one whose attenuation
        the eq. (26) weighting can undo, and the estimator refuses the rest
        with an explanation.

        It is an argument here rather than state on the object on purpose. The
        other estimators cannot compensate for the shaping a detrending filter
        applies, so a detrended signal must never reach ``run_stft_eis`` or
        ``run_dmfa`` -- differencing in particular would annihilate the lowest
        tones there rather than cost nothing."""
        fs = 1.0 / self.sampling_time
        H_exc = np.round(self.frequencies / self.frequencies.min()).astype(int)
        return bltva(
            self.voltage, self.current, fs, P, H_exc,
            Np=Np, Nq=Nq, Na=Na, harmonics=harmonics,
            band_halfwidth=band_halfwidth, rcond=rcond,
            noise=noise, prune=prune, detrend=detrend,
        )

    def visualize_bltva_fit(self, P, *, Np=6, Nq=5, Na=None, harmonics="all_nl",
                            band_halfwidth=None, rcond=None, noise=None,
                            prune=True, detrend=None):
        fs = 1.0 / self.sampling_time
        H_exc = np.round(self.frequencies / self.frequencies.min()).astype(int)
        return visualize_bltva_fit(
            self.voltage, self.current, fs, P, H_exc,
            Np=Np, Nq=Nq, Na=Na, harmonics=harmonics,
            band_halfwidth=band_halfwidth, rcond=rcond,
            noise=noise, prune=prune, detrend=detrend,
        )

    def estimate_noise_variance(self, P, *, Np=6, Nq=5, Na=None, dof=5,
                                safety=10.0, k_max=None, step=None, detrend=None):
        """Noise variance spectrum of the response (TIM 2021, Algorithm 1).

        Returns a ``NoiseEstimate``; ``.at(k)`` interpolates it onto arbitrary
        DFT bins. The variance is that of the 1/N-scaled DFT, so white output
        noise of standard deviation ``s`` gives ``s**2 / N``.
        """
        fs = 1.0 / self.sampling_time
        H_exc = np.round(self.frequencies / self.frequencies.min()).astype(int)
        return estimate_noise_variance(
            self.voltage, self.current, fs, P, H_exc,
            Np=Np, Nq=Nq, Na=Na, dof=dof, safety=safety,
            k_max=k_max, step=step, detrend=detrend,
        )

    @staticmethod
    def select_na(P, Np, d=0):
        """Transient order from TIM 2022 eq. (22); see :func:`bltva.select_na`."""
        return select_na(P, Np, d)

    @staticmethod
    def check_identifiability(P, Np, Nq, Na):
        """TIM 2021 eq. (49); see :func:`bltva.check_identifiability`."""
        return check_identifiability(P, Np, Nq, Na)

    def compute_freq_indexes(self, input_signal=None):
        """Bin index of each excited frequency, delegating to stft_functions.

        These were duplicated here until the module versions were corrected
        (``ceil`` to ``round``, and a peak search that could stray onto a
        neighbouring excited bin), at which point the two copies diverged.
        One implementation only.

        ``input_signal`` is accepted for backwards compatibility and ignored:
        the frequency axis already determines the bins.
        """
        self.freq_indexes, self.index_f0 = compute_freq_indexes(
            self.freq_axis, self.frequencies, self.sampling_time)
        return self.freq_indexes

    def search_freq_indexes(self, spectrum):
        """Refine the bin indexes by locating the local spectral peak.

        ``spectrum`` must be a *shifted spectrum* such as ``self.ft_current``,
        not a time series -- the search indexes it with DFT bin numbers.
        """
        self.freq_indexes, self.index_f0 = search_freq_indexes(
            spectrum, self.freq_axis, self.frequencies, self.sampling_time)
        return self.freq_indexes

    # def lp_filter(self, cutoff, order):
    #     """
    #     Apply a low-pass filter to the signal
    #     """
    #     filter = fermi_dirac_filter(self.freq_axis,0,2*cutoff, order)
    #     self.voltage_filt = self.voltage.size * ifft(ifftshift(self.voltage * filter)).real
    #     self.current_filt = self.current.size * ifft(ifftshift(self.current * filter)).real

    def inspect_spectrum(self):
        positive_range = range(self.ft_voltage.size//2, self.ft_voltage.size)
        fig, axs = inspect_spectrum(
            self.ft_voltage[positive_range], 
            self.ft_current[positive_range], 
            self.freq_axis[positive_range],
        )
        return fig, axs

    def inspect_spectrum_phase(self):
        positive_range = range(self.ft_voltage.size//2, self.ft_voltage.size)
        fig, axs = inspect_spectrum_phase(
            self.ft_voltage[positive_range], 
            self.ft_current[positive_range], 
            self.freq_axis[positive_range],
        )
        return fig, axs
    
    def visualise_peaks(self):
        visualise_peaks(
            self.ft_voltage,
            self.ft_current,
            self.freq_axis,
            self.frequencies,
            self.freq_indexes,
        )

    def visualise_peaks_phase(self):
        visualise_peaks_phase(
            self.ft_voltage,
            self.ft_current,
            self.freq_axis,
            self.frequencies,
            self.freq_indexes,
        )