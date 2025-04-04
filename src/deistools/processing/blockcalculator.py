# from attrs import define
from dataclasses import dataclass, field
import numpy as np
from numpy.fft import ifft, ifftshift
from npbuffer import NumpyCircularBuffer
from mfa import MultiFrequencyAnalysis


@ dataclass
class BlockCalculator:
    input_size : int
    sampling_time : float
    high_z_calculator : MultiFrequencyAnalysis 
    lp_filter : np.array
    ds_factor : int
    buffer_size : int
    impedance : np.array = field(default = np.zeroes((high_z_calculator.frequencies, input_size//ds_factor), dype = np.complex64))
    impedance_index: int = field(default= 0)
    voltage_ds: NumpyCircularBuffer = field(default=NumpyCircularBuffer(buffer_size, np.float32))
    current_ds: NumpyCircularBuffer = field(default=NumpyCircularBuffer(buffer_size, np.float32))
    saving_dir : str
    finished_tech : bool = field(default=False)
    
    def __post_init__(self):
        self.high_z_calculator.compute_freq_axis()

    def calculate(self, data_voltage, data_current):
        self.high_z_calculator.voltage = data_voltage
        self.high_z_calculator.current = data_current
        # Compute impedance of the high frequency band
        high_z = self.high_z_calculator.run_fft_eis()
        self.impedance[:,self.impedance_index] = high_z
        self.impedance_index += 1
        # Decimate the signals
        self.voltage_filt = self.input_size * ifft(ifftshift(self.high_z_calculator.ft_voltage * self.fd_filter)).real
        self.current_filt = self.input_size * ifft(ifftshift(self.high_z_calculator.ft_current * self.fd_filter)).real
        self.voltage_ds.push(self.voltage_filt[::self.ds_factor])
        self.current_ds.push(self.current_filt[::self.ds_factor])
        if self.finished_tech:
            self._save_results()
            self._reset_impedance_memory()
    
    def finish_technique(self):
        self.finished_tech = True

    def _save_results(self):
        np.save(self.saving_dir+'/impedance.npy', self.impedance[:,0:self.impedance_index])
        np.save(self.saving_dir+'/voltage.npy', self.voltage_ds.empty())
        np.save(self.saving_dir+'/current.npy', self.current_ds.empty())

    def _reset_impedance_memory(self):
        self.impedance = np.zeroes((self.multisine_frequencies, self.voltage.size//self.ds_factor))
        self.impedance_index  = 0
    
    