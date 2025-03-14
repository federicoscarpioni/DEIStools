from numpy.fft import fft, fftshift, fftfreq, ifft, ifftshift
import numpy as np
import matplotlib.pyplot as plt

def fermi_dirac_filter(fr, fc, bw, n):
    """
    Create a digital filter shaped as a symmetric Fermi-Dirac function.
    """
    X = (fr - fc) / bw
    Y = np.cosh(n) / (np.cosh(n * X) + np.cosh(n))
    return Y

class MultiFrequencyAnalysis:
    
    def __init__(self, frequencies, voltage, current, sampling_time):
        self.frequencies = frequencies
        self.voltage = voltage
        self.current = current
        self.sampling_time = sampling_time
    
    def compute_freq_axis(self):
        self.freq_axis = fftshift(fftfreq(self.voltage.size, self.sampling_time))
    
    def fft_eis(self):
        self.ft_voltage = fftshift(fft(self.voltage)) / self.voltage.size
        self.ft_current = fftshift(fft(self.current)) / self.current.size
        indexes = self.find_freq_indexes(self.ft_current)
        self.impedance = self.ft_voltage[indexes] / self.ft_current[indexes]


    def find_freq_indexes(self, input_signal):
        index_f0 = np.where(self.freq_axis == 0)[0][0]
        n_samples = input_signal.size
    
        guess_i_multisine_freq = np.ceil(self.frequencies * self.sampling_time * n_samples) + index_f0
        index_multisine_freq = np.zeros(self.frequencies.size,dtype='int64')
        dist_between_freq = np.zeros(self.frequencies.size, dtype='float32')
        dist_between_freq[0] = np.min([self.frequencies[1] - self.frequencies[0], self.frequencies[0]])
        for f in range(1, self.frequencies.size-1):
            dist_between_freq[f] = np.min([self.frequencies[f] - self.frequencies[f-1], self.frequencies[f+1] - self.frequencies[f]])    
        dist_between_freq[-1] = self.frequencies[-1] *0.2# - multisine_freq[-2] # Fixed
        # dist_peak = dist_between_freq/(dt*N_samples)
        dist_between_freq_P = 2 * np.floor(dist_between_freq * self.sampling_time  * n_samples / 4)
        
        for f in range(0, self.frequencies.size):   
            min_searching = int(guess_i_multisine_freq[f]) - int(dist_between_freq_P[f]/2)
            max_searching = int(guess_i_multisine_freq[f]) + int(dist_between_freq_P[f]/2)
            searching_freq_rng =  np.arange(min_searching, max_searching)
            index_multisine_freq[f] = min_searching + int(np.argmax((np.abs(input_signal[(searching_freq_rng)])),axis = 0))
              
        return index_multisine_freq

    def lp_filter(self, cutoff, order):
        """
        Apply a low-pass filter to the signal
        """
        filter = fermi_dirac_filter(self.freq_axis,0,2*cutoff, order)
        self.voltage_filt = self.voltage.size * ifft(ifftshift(self.voltage * filter)).real
        self.current_filt = self.current.size * ifft(ifftshift(self.current * filter)).real