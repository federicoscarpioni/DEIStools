import numpy as np
from pypicostreaming.series5000.series5000 import Picoscope5000a
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
        self.voltage = RingBuffer(self.buffer_size, dtype=np.float16)
        self.current = RingBuffer(self.buffer_size, dtype=np.float16)
        # Allocate array for impedance
        self.impedance = np.zeros((frequencies.size,int(self.buffer_size/self.ds_factor)))
        self.impedance_index = 0


    def create_lp_filter(self):
        self.filt_b, self.filt_a = bessel(self.order, self.cutoff_freq, btype='low', analog=False, norm='phase')
        self.initial_cond = lfilter_zi(self.b, self.a)


    def compute_high_freq_z(self, signalA, signalB):
        # Convert the signals
        voltage = self.convert_ADC_numbers(signalA,
                                           self.channels['A'].vrange,
                                           self.channels['A'].irange)
        current = self.convert_ADC_numbers(signalB,
                                           self.channels['B'].vrange,
                                           self.channels['B'].irange)
        # Compute the impedance at high frequency
        high_freqs = MultiFrequencyAnalysis(self.frequencies,
                                            voltage,
                                            current,
                                            self.sampling_time)
        high_freqs.fft_eis()
        # Save the high impedance
        self.impedance[:,self.impedance_index] = high_freqs.impedance
        self.impedance_index =+ 1
        # Filter the signal
        voltage_filt, _ = lfilter(self.b, self.a, voltage, zi=self.initial_cond * voltage[0])
        current_filt, _ = lfilter(self.b, self.a, current, zi=self.initial_cond * current[0])
        self.voltage.write(voltage_filt)
        self.current.write(current_filt)


    def streaming_callback(self):
        super().streaming_callback()
        if len(self.channels['A'].total_buffer) > self.sample_size:
            signalA = self.channels['A'].total_buffer.read(self.sample_size)
            signalB = self.channels['B'].total_buffer.read(self.sample_size)
            computation_thread = Thread(target = self.compute_high_freq_z, args=(signalA, signalB))
            computation_thread.start()


    def save_signals(self):
        np.save(self.saving_dir + f'/voltage.npy', self.voltage)
        np.save(self.saving_dir + f'/current.npy', self.current)

    def save_intermediate_signals(self, subfolder_name):
        saving_file_path = self.saving_dir + subfolder_name
        Path(saving_file_path).mkdir(parents=True, exist_ok=True)
        np.save(saving_file_path + f'/voltage.npy', self.voltage)
        np.save(saving_file_path + f'/current.npy', self.current)
        np.save(saving_file_path + f'/impedance.npy', self.impedance[:self.impedance_index, :])
        # Reset the variables
        self.voltage.read(self.buffer_size)
        self.current.read(self.buffer_size)
        self.impedance = np.zeros((self.frequencies.size, int(self.buffer_size / self.ds_factor)))
        self.impedance_index = 0

    # def save_intermediate_impedance(self, subfolder_name):
    #     saving_file_path = self.saving_dir + subfolder_name
    #     Path(saving_file_path).mkdir(parents=True, exist_ok=True)
    #     np.save(saving_file_path + f'/impedance.npy', self.impedance[:self.impedance_index,:])