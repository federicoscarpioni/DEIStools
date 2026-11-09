import numpy as np
from numpy.fft import rfft


def fft_eis(voltage, current, i_exc):
    ft_voltage =  fft(voltage) /voltage.size
    ft_current =  fft(current) / current.size
    impedance = (ft_voltage[i_exc]/ft_current[i_exc])
    voltage_avg = abs(ft_voltage[0])
    current_avg = abs(ft_current[0])
    return impedance, voltage_avg, current_avg

def estimate_impedance(
        voltage,
        current,
        frequencies,
        period,
        sampling_time,
        window_fun = 'rectangular',
    ):
    # Calculate window length in samples
    samples_window = int(period/sampling_time)
    # Initialize time-frequency analysis arrays
    num_spectra = int(voltage.size / samples_window)
    num_freq = frequencies.size
    impedance = np.zeros((num_freq, num_spectra),dtype='complex128')
    voltage_avg = np.zeros(num_spectra)
    current_avg = np.zeros(num_spectra)
    time = np.zeros(num_spectra)
    # Create frequency axis
    freq_axis = fftshift(fftfreq(samples_window, sampling_time))
    # Compute the indexes of the excited frequencies. search_freq_indexes
    # locates spectral peaks, so it needs the spectrum of the first window --
    # passing the time series made it take argmax over time samples at
    # frequency-bin offsets, which returned plausible-looking but wrong bins.
    first_window = fftshift(fft(current[:samples_window])) / samples_window
    freq_indexes, index_f0 = search_peaks_position(
        first_window,
        freq_axis,
        frequencies,
        sampling_time)
    # Compute STFT
    for i in range(0,num_spectra):
        data_range = range(samples_window * i, samples_window * (i + 1) )
        voltage_block = voltage[data_range]
        current_block = current[data_range]
        impedance[:,i], voltage_avg[i], current_avg[i] = fft_eis(
            voltage_block,
            current_block,
            freq_indexes,
            index_f0)
        # Compute time array as middle of the window
        window_length = sampling_time * samples_window
        time[i] = i * window_length + window_length/2 
    return impedance, voltage_avg, current_avg, time