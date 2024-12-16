from numpy.fft import fft, fftshift, fftfreq

class MultiFrequencyAnalysis:
    def __init__(self, frequencies, voltage, current, sampling_time):
        self.frequencies = frequencies
        self.voltage = voltage
        self.current = current
        self.sampling_time = sampling_time

    def fft_eis(self):
        self.ft_voltage = fftshift(fft(self.voltage)) / self.voltage.size
        self.ft_current = fftshift(fft(self.current)) / self.current.size
        self.freq_axis = fftshift(fftfreq(self.voltage.size, self.sampling_time))
        indexes = self.find_peaks(self.ft_voltage)
        self.impedance = self.ft_voltage(indexes)/self.ft_current(indexes)


    def find_peaks(self):
        ...
        # return indexes