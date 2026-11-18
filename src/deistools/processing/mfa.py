from dataclasses import dataclass
from numpy.fft import fft, fftfreq
import numpy as np

from deistools.visualise import inspect_spectrum,inspect_spectrum_phase, visualise_peaks, visualise_peaks_phase
from deistools.processing.dmfa_functions import dmfa, get_band_size #, extract_zero_frequency
from deistools.visualise.dmfa import visualize_dmfa_filtering
from deistools.processing.stft_functions import get_excited_bins, stft_eis
from deistools.processing.peak_search import search_peaks_position

@dataclass
class MultiFrequencyAnalysis:
    excitation : np.ndarray 
    response : np.ndarray 
    sampling_frequency : float
    frequencies : np.ndarray
    period: float

    def __post_init__(self) -> None:
        self.num_samples = self.excitation.size
        self.harmonics = self.frequencies * self.period
        self.sampling_time = 1/self.sampling_frequency
        self.k_exc = get_excited_bins(self.num_samples, self.frequencies, self.sampling_frequency)
    
    def compute_freq_axis(self) -> None:
        self.freq_axis = fftfreq(self.excitation.size, self.sampling_time)

    def compute_fft(self) -> None:
        self.dft_excitation = fft(self.excitation) / self.num_samples
        self.dft_response = fft(self.response) / self.num_samples

    def get_excited_freqs(self) -> np.ndarray:
        self.frequencies_exact = (self.k_exc * self.sampling_frequency) / self.num_samples
        return self.frequencies_exact

    def run_fft_eis(self):
        return self.dft_response[self.k_exc] / self.dft_excitation[self.k_exc]
    
    def run_stft_eis(self, periods_per_window):
        frequency_response_function, excitation, response, time = stft_eis( 
            self.excitation,
            self.response,
            self.frequencies,
            self.sampling_frequency,
            self.period,
            periods_per_window,
        )
        return frequency_response_function, excitation, response, time

    def run_dmfa(self,
                 passband_filter,
                 periods_per_sample,
        ):
        frequency_response_function, time = dmfa(
            self.dft_excitation,
            self.dft_response,
            self.k_exc,
            passband_filter,
            self.sampling_frequency,
            self.period,
            periods_per_sample,
        )
        return frequency_response_function, time

    def visualize_dmfa_filtering(
            self,
            passband_filter,
            periods_per_sample,
        ):
        n_bins = get_band_size(self.num_samples, self.sampling_frequency, self.period, periods_per_sample)
        band_offsets = np.arange(-(n_bins // 2), n_bins - n_bins // 2)
        filter_values = passband_filter.evaluate(band_offsets * self.sampling_frequency / self.num_samples)
        visualize_dmfa_filtering(
            filter_values,
            band_offsets,
            self.dft_excitation,
            self.dft_response,
            self.freq_axis,
            self.frequencies,
            self.k_exc,
        )


    def search_freq_indexes(self, spectrum):
        self.freq_indexes, self.index_f0 = search_peaks_position(
            spectrum, self.freq_axis, self.frequencies, self.sampling_time)
        return self.freq_indexes


    def inspect_spectrum(self):
        positive_range = range(0, self.num_samples//2)
        fig, axs = inspect_spectrum(
            self.dft_excitation[positive_range], 
            self.dft_response[positive_range], 
            self.freq_axis[positive_range],
        )
        return fig, axs

    def inspect_spectrum_phase(self):
        positive_range = range(0, self.num_samples//2)
        fig, axs = inspect_spectrum_phase(
            self.dft_excitation[positive_range], 
            self.dft_response[positive_range], 
            self.freq_axis[positive_range],
        )
        return fig, axs
    
    def visualise_peaks(self):
        visualise_peaks(
            self.dft_excitation, 
            self.dft_response,
            self.freq_axis,
            self.frequencies,
            self.k_exc,
        )

    def visualise_peaks_phase(self):
        visualise_peaks_phase(
            self.dft_excitation, 
            self.dft_response,
            self.freq_axis,
            self.frequencies,
            self.k_exc,
        )