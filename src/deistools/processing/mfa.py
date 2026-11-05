from dataclasses import dataclass
from numpy.fft import fft, fftfreq
import numpy as np

from deistools.visualise import inspect_spectrum,inspect_spectrum_phase, visualise_peaks, visualise_peaks_phase
from deistools.processing.dmfa_functions import extract_zero_frequency, extract_impedance, visualize_dmfa_filtering, extract_impedance_with_error
from deistools.processing.stft_functions import (
    estimate_impedance as stft_eis, 
)
from deistools.processing.peak_search import search_peaks_position

@dataclass
class EstimationResult:
    impedance : np.ndarray
    time : np.ndarray

@dataclass
class MultiFrequencyAnalysis:
    voltage : np.ndarray 
    current : np.ndarray 
    sampling_frequency : float
    frequencies : np.ndarray
    period: float

    def __post_init__(self) -> None:
        self.num_samples = self.voltage.size
        self.harmonics = self.frequencies/self.frequencies[0]
        self.sampling_time = 1/self.sampling_frequency
        self.i_exc =  (self.num_samples * self.frequencies) / self.sampling_frequency # Indexes of the array
        self.i_exc = np.rint(self.i_exc).astype(np.int32)
    
    def compute_freq_axis(self) -> None:
        self.freq_axis = fftfreq(self.voltage.size, self.sampling_time)

    def compute_fft(self) -> None:
        self.ft_voltage = fft(self.voltage) / self.voltage.size
        self.ft_current = fft(self.current) / self.current.size

    def get_excited_freqs(self) -> np.ndarray:
        self.frequencies_exact = (self.i_exc * self.sampling_frequency) / self.num_samples
        return self.frequencies_exact

    def run_fft_eis(self):
        return self.ft_voltage[self.i_exc] / self.ft_current[self.i_exc]
    
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

    # def compute_freq_indexes(self, input_signal=None):
    #     """Bin index of each excited frequency, delegating to stft_functions.

    #     These were duplicated here until the module versions were corrected
    #     (``ceil`` to ``round``, and a peak search that could stray onto a
    #     neighbouring excited bin), at which point the two copies diverged.
    #     One implementation only.

    #     ``input_signal`` is accepted for backwards compatibility and ignored:
    #     the frequency axis already determines the bins.
    #     """
    #     self.freq_indexes, self.index_f0 = compute_freq_indexes(
    #         self.freq_axis, self.frequencies, self.sampling_time)
    #     return self.freq_indexes

    def search_freq_indexes(self, spectrum):
        """Refine the bin indexes by locating the local spectral peak.

        ``spectrum`` must be a *shifted spectrum* such as ``self.ft_current``,
        not a time series -- the search indexes it with DFT bin numbers.
        """
        self.freq_indexes, self.index_f0 = search_peaks_position(
            spectrum, self.freq_axis, self.frequencies, self.sampling_time)
        return self.freq_indexes


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