import numpy as np
from numpy.fft import fft, fftshift, fftfreq

def compute_freq_indexes(
          freq_axis, 
          frequencies, 
          sampling_time,
          ):
    index_f0 = np.where(freq_axis == 0)[0][0]
    indexes = np.ceil(frequencies * sampling_time * freq_axis.size) + index_f0
    indexes = indexes.astype(int)
    freq_indexes = indexes.tolist()
    return freq_indexes, index_f0

def search_freq_indexes(
          signal, 
          freq_axis,
          frequencies, 
          sampling_time
          ):
        guess_i_multisine_freq, index_f0 = compute_freq_indexes(freq_axis, frequencies, sampling_time)
        index_multisine_freq = np.zeros(frequencies.size,dtype='int64')
        dist_between_freq = np.zeros(frequencies.size, dtype='float32')
        dist_between_freq[0] = np.min([frequencies[1] - frequencies[0], frequencies[0]])
        for f in range(1, frequencies.size-1):
            dist_between_freq[f] = np.min([frequencies[f] - frequencies[f-1], frequencies[f+1] - frequencies[f]])    
        dist_between_freq[-1] = frequencies[-1] *0.2# - multisine_freq[-2] # Fixed
        # dist_peak = dist_between_freq/(dt*N_samples)
        dist_between_freq_P = 2 * np.floor(dist_between_freq * sampling_time  * freq_axis.size / 4)
        for f in range(0, frequencies.size):
            min_searching = int(guess_i_multisine_freq[f]) - int(dist_between_freq_P[f]/2)
            max_searching = int(guess_i_multisine_freq[f]) + int(dist_between_freq_P[f]/2)
            searching_freq_rng =  np.arange(min_searching, max_searching)
            index_multisine_freq[f] = min_searching + int(np.argmax((np.abs(signal[(searching_freq_rng)])),axis = 0))
        freq_indexes = index_multisine_freq.tolist()
        return freq_indexes, index_f0

def fft_eis(voltage, current, freq_indexes, index_f0):
    ft_voltage =  fftshift(fft(voltage)) /voltage.size
    ft_current =  fftshift(fft(current)) / current.size
    impedance = (ft_voltage[freq_indexes]/ft_current[freq_indexes])
    voltage_avg = abs(ft_voltage[index_f0])
    current_avg = abs(ft_current[index_f0])
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
    # Compute the indexes of the exited frequencies
    freq_indexes, index_f0 = search_freq_indexes(
        current[:samples_window],
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