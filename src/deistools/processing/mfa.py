from numpy.fft import fft, fftshift, fftfreq, ifft, ifftshift
import numpy as np

from deistools.visualise import inspect_spectrum, visualise_peaks
# from deistools.processing import fermi_dirac_filter
from deistools.processing.dmfa_functions import extract_zero_frequency, extract_impedance

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
    
    def run_dmfa(self, 
                 filter, 
                 time_resolution,
                 Npts_eleab
        ):
        voltage, current, time = extract_zero_frequency(
            self.ft_voltage,
            self.ft_current,
            self.freq_axis,
            filter,
            Npts_eleab,
            time_resolution,
        )
        impedance = extract_impedance(
            self.ft_voltage,
            self.ft_current,
            self.freq_indexes,
            filter,
            Npts_eleab
        )
        return impedance, voltage, current, time

    def compute_freq_indexes(self, input_signal):
        self.index_f0 = np.where(self.freq_axis == 0)[0][0]
        indexes = np.ceil(self.frequencies * self.sampling_time * input_signal.size) + self.index_f0
        indexes = indexes.astype(int)
        self.freq_indexes = indexes.tolist()
        return self.freq_indexes

    def search_freq_indexes(self, input_signal):
        guess_i_multisine_freq = self.compute_freq_indexes(input_signal)
        index_multisine_freq = np.zeros(self.frequencies.size,dtype='int64')
        dist_between_freq = np.zeros(self.frequencies.size, dtype='float32')
        dist_between_freq[0] = np.min([self.frequencies[1] - self.frequencies[0], self.frequencies[0]])
        for f in range(1, self.frequencies.size-1):
            dist_between_freq[f] = np.min([self.frequencies[f] - self.frequencies[f-1], self.frequencies[f+1] - self.frequencies[f]])    
        dist_between_freq[-1] = self.frequencies[-1] *0.2# - multisine_freq[-2] # Fixed
        # dist_peak = dist_between_freq/(dt*N_samples)
        dist_between_freq_P = 2 * np.floor(dist_between_freq * self.sampling_time  * input_signal.size / 4)
        for f in range(0, self.frequencies.size):   
            min_searching = int(guess_i_multisine_freq[f]) - int(dist_between_freq_P[f]/2)
            max_searching = int(guess_i_multisine_freq[f]) + int(dist_between_freq_P[f]/2)
            searching_freq_rng =  np.arange(min_searching, max_searching)
            index_multisine_freq[f] = min_searching + int(np.argmax((np.abs(input_signal[(searching_freq_rng)])),axis = 0))  
        self.freq_indexes = index_multisine_freq.tolist()
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
    
    def visualise_peaks(self):
        visualise_peaks(
            self.ft_voltage,
            self.ft_current,
            self.freq_axis,
            self.frequencies,
            self.freq_indexes,
        )
    