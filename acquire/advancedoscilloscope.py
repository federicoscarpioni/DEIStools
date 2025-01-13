import numpy as np
from pypicostreaming.pypicostreaming.series5000.series5000 import Picoscope5000a
from computations.mfa import MultiFrequencyAnalysis
from scipy.signal import bessel, lfilter, lfilter_zi
from np_rw_buffer import RingBuffer
from pathlib import Path
from threading import Thread

class ZPico5000a(Picoscope5000a):
    '''
    A class to perform on-line short-time Fourier Transform EIS on data streamed by a PicoScope. Performs also filtering
    and downsampling.
    '''
    def __init__(self,
                 sample_size,
                 frequencies,
                 irange,
                 filter_order,
                 cutoff,
                 ds_factor,
                 buffer_size,
                 resolution,
                 serial = None,
                 ):
        self.sample_size = sample_size
        self.frequencies = frequencies
        self.irange = irange
        super().__init__(resolution, serial = None)
        # Downsampling filter
        self.order = filter_order
        self.cutoff = cutoff
        self.filter = self.create_lp_filter()
        self.ds_factor = ds_factor
        # Storage of the downsampled signals
        self.buffer_size = buffer_size
        self.voltage = RingBuffer(self.buffer_size, dtype=np.float32)
        self.current = RingBuffer(self.buffer_size, dtype=np.float32)
        # Allocate array for impedance
        self.impedance = np.zeros((frequencies.size,int(self.buffer_size/ds_factor)), dtype = np.complex64)
        self.impedance_index = 0


    def create_lp_filter(self):
        self.filt_b, self.filt_a = bessel(self.order, self.cutoff, btype='lowpass', analog=False, norm='phase')
        self.initial_cond = lfilter_zi(self.filt_b, self.filt_a)
    

    

    def compute_high_freq_z(self):
        # Convert the signals
        self.voltage_original = self.convert_ADC_numbers(self.channels['A'].buffer_total.read(self.sample_size)[:,0], # Note: it is foundamental to add the [:,0] slicing to make the array 1D otherwise np.fft.fft will considere it 2D!!!
                                           self.channels['A'].vrange,
                                           self.channels['A'].irange)
        self.current_original = self.convert_ADC_numbers(self.channels['B'].buffer_total.read(self.sample_size)[:,0],
                                           self.channels['B'].vrange,
                                           self.channels['B'].irange)
        # Compute the impedance at high frequency
        self.high_freqs_analysis = MultiFrequencyAnalysis(self.frequencies,
                                            self.voltage_original,
                                            self.current_original,
                                            self.time_step)
        self.high_freqs_analysis.fft_eis()
        # Save the high impedance
        self.impedance[:,self.impedance_index] = self.high_freqs_analysis.impedance
        self.impedance_index += 1
        # # Filter the signal
        self.voltage_filt= lfilter(self.filt_b, self.filt_a, self.voltage_original, zi=self.initial_cond * self.voltage_original[0])
        self.current_filt = lfilter(self.filt_b, self.filt_a, self.current_original, zi=self.initial_cond * self.current_original[0])
        # self.voltage.write(self.voltage_filt[0][::self.ds_factor])
        # self.current.write(self.current_filt[0][::self.ds_factor])
        self.voltage.write(self.voltage_original)
        self.current.write(self.current_original)
        print('pico msg: completed z calculation and downsampling')

    def _online_computation(self):
        super()._online_computation()
        if len(self.channels['A'].buffer_total) > self.sample_size:
            self.computation_thread = Thread(target = self.compute_high_freq_z)
            self.computation_thread.start()
            print('pico msg: starting online z calculation...')


    def save_signals(self):
        np.save(self.saving_dir + '/voltage.npy', self.voltage)
        np.save(self.saving_dir + '/current.npy', self.current)

    def save_intermediate_signals(self, subfolder_name):
        self.computation_thread.join()
        saving_file_path = self.saving_dir + subfolder_name
        Path(saving_file_path).mkdir(parents=True, exist_ok=True)
        np.save(saving_file_path + '/voltage.npy', self.voltage.read())
        np.save(saving_file_path + '/current.npy', self.current.read())
        np.savetxt(saving_file_path + '/fftEISimpedance.txt', self.impedance[:,0:self.impedance_index])
        
        # Reset the variables
        self.impedance = np.zeros((self.frequencies.size, int(self.buffer_size / self.ds_factor)))
        self.impedance_index = 0
