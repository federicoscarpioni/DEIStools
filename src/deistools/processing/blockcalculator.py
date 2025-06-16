# from attrs import define
from dataclasses import dataclass, field
import numpy as np
from numpy.fft import ifft, ifftshift
from npbuffer import NumpyCircularBuffer
from deistools.processing.mfa import MultiFrequencyAnalysis
from deistools.processing import detrending


@dataclass
class BlockCalculator:
    input_size : int
    sampling_time : float
    high_z_calculator : MultiFrequencyAnalysis 
    lp_filter : np.array
    ds_factor : int
    buffer_size : int
    impedance_index: int = field(default= 0)
    impedance : np.array = field(init=False)
    voltage_ds: NumpyCircularBuffer = field(init=False)
    current_ds: NumpyCircularBuffer = field(init=False)
    
    def __post_init__(self):
        self.high_z_calculator.compute_freq_axis()
        self.voltage_ds = NumpyCircularBuffer(self.buffer_size, np.float32)
        self.current_ds = NumpyCircularBuffer(self.buffer_size, np.float32)
        self.reset_impedance_memory()

    def calculate(self, data_voltage, data_current):
        self.high_z_calculator.voltage, coordinates_voltage = detrending.remove_baseline(data_voltage, self.sampling_time)
        self.high_z_calculator.current, coordinates_current = detrending.remove_baseline(data_current, self.sampling_time)
        # Compute impedance of the high frequency band
        self.high_z_calculator.compute_fft()
        if self.high_z_calculator.freq_indexes == None:
            self.high_z_calculator.search_freq_indexes(
                self.high_z_calculator.ft_current
            )
        high_z = self.high_z_calculator.run_fft_eis()
        self.impedance[:,self.impedance_index] = high_z
        self.impedance_index += 1
        # Decimate the signals
        self.voltage_filt = self.input_size * ifft(ifftshift(self.high_z_calculator.ft_voltage * self.lp_filter)).real
        self.current_filt = self.input_size * ifft(ifftshift(self.high_z_calculator.ft_current * self.lp_filter)).real
        self.voltage_filt = detrending.redo_baseline(self.voltage_filt, coordinates_voltage)
        self.current_filt = detrending.redo_baseline(self.current_filt, coordinates_current)
        self.voltage_ds.push(self.voltage_filt[::self.ds_factor])
        self.current_ds.push(self.current_filt[::self.ds_factor])

    def save_results(self, directory):
        np.save(directory+'/impedance.npy', self.impedance[:,0:self.impedance_index])
        self.reset_impedance_memory()
        np.save(directory+'/voltage.npy', self.voltage_ds.empty())
        np.save(directory+'/current.npy', self.current_ds.empty())
        print('Data saved for the last technique!')

    def reset_impedance_memory(self):
        self.impedance = np.zeros(
            (self.high_z_calculator.frequencies.size, self.buffer_size), # This is too much allocation! 
            dtype = np.complex64,
            ) 
        self.impedance_index  = 0

    
    