# from attrs import define
from dataclasses import dataclass, field
import numpy as np
from numpy.fft import ifft, ifftshift

from npbuffer import NumpyCircularBuffer

from deistools.processing import MultiFrequencyAnalysis, FermiDiracFilter
from deistools.processing import detrending

from pyeclab import Channel


@dataclass
class ConditionAverageScope:
    """
    Quantity should be "current" or "voltage". Operator instead can only be ">" 
    or "<".
    """
    technique_index : int
    quantity : str
    operator : str
    threshold : float

@dataclass
class BlockCalculator:
    input_size : int
    sampling_time : float
    high_z_calculator : MultiFrequencyAnalysis 
    lp_filter : FermiDiracFilter
    ds_factor : int
    buffer_size : int
    potentiostat : Channel
    conditions: list[ConditionAverageScope] = field(default_factory=list)
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
        # self.high_z_calculator.voltage = data_voltage
        # self.high_z_calculator.current = data_current

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
        self.voltage_filt = self.input_size * ifft(ifftshift(self.high_z_calculator.ft_voltage * self.lp_filter.values)).real
        self.current_filt = self.input_size * ifft(ifftshift(self.high_z_calculator.ft_current * self.lp_filter.values)).real
        self.voltage_filt = detrending.redo_baseline(self.voltage_filt, coordinates_voltage)
        self.current_filt = detrending.redo_baseline(self.current_filt, coordinates_current)
        self.voltage_ds.push(self.voltage_filt[::self.ds_factor])
        self.current_ds.push(self.current_filt[::self.ds_factor])
        if self.check_software_limits_average_lin(coordinates_voltage, coordinates_current): # I have written below two possible methods to use here
            print("Software limit scope met!")
            self.potentiostat.end_technique()

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
    

    def check_software_limits_average_lin(self, conditions_voltage, conditions_current):
        """
        Check if a certain averege condition (< or > of a treshold value) is met for a
        value of the sampled data during the window of the online computation. It 
        uses the initial and final coordinate of the linearization. It adds the 
        value of the zero-frequency of the FFT to correct the linearized avarage.
        """
        mfa = self.high_z_calculator
        for condition in self.conditions:
            if self.potentiostat.data_info.TechniqueIndex == condition.technique_index:
                if condition.quantity == 'voltage':
                    value_avarage = (conditions_voltage['yfinish']+conditions_voltage['ystart'])/2
                    value_avarage = value_avarage + abs(mfa.ft_voltage[mfa.index_f0])
                    print(f'Avarage voltage is {value_avarage} V')
                if condition.quantity == 'current':
                    value_avarage = (conditions_current['yfinish']+conditions_current['ystart'])/2
                    value_avarage = value_avarage + abs(mfa.ft_current[mfa.index_f0])
                    print(f'Avarage current is {value_avarage} A')
                if condition.operator == ">" and value_avarage >= condition.threshold:
                    print(f'{condition.quantity} > {condition.threshold}')
                    return True
                elif condition.operator == "<" and value_avarage <= condition.threshold:
                    print(f'{condition.quantity} < {condition.threshold}')
                    return True
        return False


    def check_software_limits_average_fft(self):
        """
        Check if a certain averege condition (< or > of a treshold value) is met for a
        value of the sampled data during the window of the online computation. It 
        uses the zero-frequency value of the FFT as avarage.
        """
        mfa = self.high_z_calculator
        for condition in self.conditions:
            if self.potentiostat.data_info.TechniqueIndex == condition.technique_index:
                if condition.quantity == 'voltage':
                    value_avarage = abs(mfa.ft_voltage[mfa.index_f0])
                    print(f'Avarage voltage is {value_avarage} V')
                if condition.quantity == 'current':
                    value_avarage = abs(mfa.ft_current[mfa.index_f0])
                    print(f'Avarage current is {value_avarage} A')
                if condition.operator == ">" and value_avarage >= condition.threshold:
                    print(f'{condition.quantity} > {condition.threshold}')
                    return True
                elif condition.operator == "<" and value_avarage <= condition.threshold:
                    print(f'{condition.quantity} < {condition.threshold}')
                    return True
        return False