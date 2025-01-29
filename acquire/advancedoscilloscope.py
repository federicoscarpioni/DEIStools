import numpy as np
from numpy.fft import ifft,ifftshift
from pypicostreaming.pypicostreaming.series5000.series5000 import Picoscope5000a
from computations.mfa import MultiFrequencyAnalysis, fermi_dirac_filter
from scipy.signal import bessel, lfilter, lfilter_zi
from np_rw_buffer import RingBuffer
from pathlib import Path
from threading import Thread


class LowPassFilter:
    def __init__(self,order, cutoff):
        self.order = order
        self.cutoff = cutoff
        self.b, self.a = bessel(self.order,
                               self.cutoff,
                               btype='lowpass',
                               analog=False,
                               norm='phase')
        self.initial_cond = lfilter_zi(self.b, self.a)


class ZPico5000a(Picoscope5000a):
    '''
    A class to perform on-line short-time Fourier Transform EIS on data streamed by a PicoScope. Performs also filtering
    and downsampling on the sampled signal.
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
        # Low-pass filter paramters
        self.buffer_size = buffer_size
        self.ds_factor = ds_factor
        self.cutoff = cutoff
        self.filter_order = filter_order
        super().__init__(resolution, serial = None)
        

    def allocate_memory(self):
        # Prepare high frequency analysis object
        self.high_freqs_analysis = MultiFrequencyAnalysis(self.frequencies,
                                            np.zeros(self.sample_size),
                                            np.zeros(self.sample_size),
                                            self.time_step)
        self.high_freqs_analysis.compute_freq_axis()
        self.fd_filter = fermi_dirac_filter(self.high_freqs_analysis.freq_axis,0,2*self.cutoff, self.filter_order)
        # Allocate ring buffer fot th storage of the downsampled signals
    
        self.voltage = RingBuffer(self.buffer_size, dtype=np.float32)
        self.current = RingBuffer(self.buffer_size, dtype=np.float32)
        # Allocate array for impedance
        self.impedance = np.zeros((self.frequencies.size,int(self.buffer_size/self.ds_factor)), dtype = np.complex64)
        self.impedance_index = 0
    
    def set_pico(self, capture_size, samples_total, sampling_time, time_unit, saving_path):
        super().set_pico(capture_size, samples_total, sampling_time, time_unit, saving_path)
        self.allocate_memory()

    def compute_high_freq_z(self):
        # Convert the signals
        self.voltage_original = self.convert_ADC_numbers(self.channels['A'].buffer_total.read(self.sample_size)[:,0], # Note: it is foundamental to add the [:,0] slicing to make the array 1D otherwise np.fft.fft will considere it 2D!!!
                                           self.channels['A'].vrange,
                                           self.channels['A'].irange)
        self.current_original = self.convert_ADC_numbers(self.channels['B'].buffer_total.read(self.sample_size)[:,0],
                                           self.channels['B'].vrange,
                                           self.channels['B'].irange)
        # Compute the impedance at high frequency
        # self.high_freqs_analysis = MultiFrequencyAnalysis(self.frequencies,
        #                                     self.voltage_original,
        #                                     self.current_original,
        #                                     self.time_step)
        # Overwrite the new data into the analysis object
        self.high_freqs_analysis.voltage = self.voltage_original
        self.high_freqs_analysis.current = self.current_original
        self.high_freqs_analysis.fft_eis()
        # Save the high impedance
        self.impedance[:,self.impedance_index] = self.high_freqs_analysis.impedance
        self.impedance_index += 1
        
        ## Filter the signal and decimate 
        self.voltage_filt = self.sample_size * ifft(ifftshift(self.high_freqs_analysis.ft_voltage * self.fd_filter)).real
        self.current_filt = self.sample_size * ifft(ifftshift(self.high_freqs_analysis.ft_current * self.fd_filter)).real
        self.voltage.write(self.voltage_filt[::self.ds_factor])
        self.current.write(self.current_filt[::self.ds_factor])

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
        np.save(saving_file_path + '/voltage.npy', self.voltage.read()[:,0])
        np.save(saving_file_path + '/current.npy', self.current.read()[:,0])
        np.savetxt(saving_file_path + '/fftEISimpedance.txt', self.impedance[:,0:self.impedance_index])
        
        # Reset the variables
        self.impedance = np.zeros((self.frequencies.size, int(self.buffer_size)), dtype = np.complex64) # da correggere, buffer size è molto più grande
        self.impedance_index = 0
        self.reset_buffers()
