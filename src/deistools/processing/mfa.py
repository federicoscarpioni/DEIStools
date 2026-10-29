from numpy.fft import fft, fftshift, fftfreq, ifft, ifftshift
import numpy as np

from deistools.visualise import inspect_spectrum,inspect_spectrum_phase, visualise_peaks, visualise_peaks_phase
# from deistools.processing import fermi_dirac_filter
from deistools.processing.dmfa_functions import extract_zero_frequency, extract_impedance, visualize_dmfa_filtering, extract_impedance_with_error
from deistools.processing.stft_functions import (
    estimate_impedance as stft_eis, compute_freq_indexes, search_freq_indexes,
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