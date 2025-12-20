import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift, fftfreq

# from deistools.processing import fermi_dirac_filter

def extract_zero_frequency(
        ft_voltage, 
        ft_current, 
        freq_axis, 
        filter,
        Npts_elab,
        time_resolution,
    ):
    index_f0 = np.where(freq_axis == 0)[0][0]
    freq_range = np.linspace(index_f0 - int(Npts_elab/2),index_f0 + int(Npts_elab/2)-1, Npts_elab, dtype = 'int64') # !!! The -1 is important 
    voltage = Npts_elab * ifft(ifftshift(ft_voltage[freq_range] * filter)).real
    current = Npts_elab * ifft(ifftshift(ft_current[freq_range] * filter)).real
    time = time_resolution * np.arange(Npts_elab)
    return voltage, current, time

def extract_impedance(
        ft_voltage,
        ft_current,
        indexes_multisine_freq:list,
        filter,
        Npts_elab,
):
    impedance = np.zeros((len(indexes_multisine_freq), Npts_elab),dtype='complex128')
    # Compute impedance Z(t) for each frequency
    for f in range(0, len(indexes_multisine_freq)):
        if Npts_elab%2==0:
            elaboration_rng = np.arange(indexes_multisine_freq[f] - int(Npts_elab/2),indexes_multisine_freq[f] + int(Npts_elab/2), 1)
        else:
            import math
            elaboration_rng = np.arange(indexes_multisine_freq[f] - math.floor(Npts_elab/2),indexes_multisine_freq[f] + math.ceil(Npts_elab/2), 1)
        voltage_filtered = Npts_elab * ifft(ifftshift(ft_voltage[elaboration_rng] * filter))
        current_filtered = Npts_elab * ifft(ifftshift(ft_current[elaboration_rng] * filter))
        impedance[f] = voltage_filtered / current_filtered
    return impedance


def extract_impedance_with_error(
        ft_voltage,
        ft_current,
        variance_voltage,
        variance_current,
        indexes_multisine_freq:list,
        filter,
        Npts_elab,
):
    impedance = np.zeros((len(indexes_multisine_freq), Npts_elab),dtype='complex128')
    variance = np.zeros((len(indexes_multisine_freq), Npts_elab), dtype = 'float32')
    # Compute the impedance and the variance at each frequency
    N_prime = Npts_elab/(np.sum(filter**2))
    for f in range(0, len(indexes_multisine_freq)):
        if Npts_elab%2==0:
            elaboration_rng = np.arange(indexes_multisine_freq[f] - int(Npts_elab/2),indexes_multisine_freq[f] + int(Npts_elab/2), 1)
        else:
            import math
            elaboration_rng = np.arange(indexes_multisine_freq[f] - math.floor(Npts_elab/2),indexes_multisine_freq[f] + math.ceil(Npts_elab/2), 1)
        voltage_filtered = Npts_elab * ifft(ifftshift(ft_voltage[elaboration_rng] * filter))
        current_filtered = Npts_elab * ifft(ifftshift(ft_current[elaboration_rng] * filter))
        impedance[f] = voltage_filtered / current_filtered
        variance_numerator = np.abs(impedance[f])**2 * variance_current**2 +  variance_voltage**2
        variance_denominator = 2 * N_prime * np.abs(current_filtered)**2
        variance[f] = variance_denominator/variance_numerator#/variance_denominator
    
    return impedance, variance

# def extract_zero_frequency(ft_voltage, 
#                       ft_current, 
#                       ft_potential_we, 
#                       freq_axis, 
#                       Npts_elab, 
#                       SAMPLING_RATE,
#                       DT, 
#                       bw, 
#                       n):
#     # N_samples = ft_voltage.size
#     # N_impedances = round(DT/SAMPLING_RATE)
#     index_f0 = np.where(freq_axis == 0)[0][0] # Zero-frequency index
#     # Npts_elab = math.ceil(N_samples/N_impedances)
#     Npts_elab = int(Npts_elab)
#     fd_filter = fermi_dirac_filter(freq_axis[index_f0] + np.linspace(-1/(2*DT), 1/(2*DT), Npts_elab), 0, bw, n)
#     freq_range = np.linspace(index_f0 - int(Npts_elab/2),index_f0 + int(Npts_elab/2)-1, Npts_elab, dtype = 'int64') # !!! The -1 is important 
#     V0 = Npts_elab * ifft(ifftshift(ft_voltage[freq_range] * fd_filter)).real
#     I0 = Npts_elab * ifft(ifftshift(ft_current[freq_range] * fd_filter)).real
#     if ft_potential_we.any() == True:
#         V0_we = Npts_elab * ifft(ifftshift(ft_potential_we[freq_range]*fd_filter)).real
#         V0_ce = V0_we - V0
#     else:
#         V0_we = np.array(())
#         V0_ce = np.array(())
        
#     time_experiment = DT * np.arange(Npts_elab)

#     print('Zero-frequency extracted.')
        
#     return V0, I0, time_experiment

# def extract_impedance(ft_voltage, ft_current, multisine_freq, Npts_elab, index_multisine_freq, SAMPLING_RATE, DT, bw, n):
#     # Prepare the needed parameters
#     dist_between_freq = np.zeros(multisine_freq.size, dtype='float32')
#     dist_between_freq[0] = np.min([multisine_freq[1] - multisine_freq[0], multisine_freq[0]])
#     for f in range(1, multisine_freq.size-1):
#         dist_between_freq[f] = np.min([multisine_freq[f] - multisine_freq[f-1], multisine_freq[f+1] - multisine_freq[f]])    
#     dist_between_freq[-1] = multisine_freq[-1] - multisine_freq[-2] # Fixed
#     # dist_peak = dist_between_freq/(dt*N_samples)
#     # N_samples = ft_voltage.size
#     # N_impedances = round(DT/SAMPLING_RATE)
#     # Npts_elab = math.ceil(N_samples/N_impedances)
#     # Find the correct indexes of the peaks and calculate the impedances
#     Z_cell = np.zeros((multisine_freq.size, Npts_elab),dtype='complex128')

#     # Compute impedance Z(t) for each frequency
#     for f in range(0, multisine_freq.size):
#         if Npts_elab%2==0:
#             elaboration_rng = np.arange(index_multisine_freq[f] - int(Npts_elab/2),index_multisine_freq[f] + int(Npts_elab/2), 1)
#         else:
#             import math
#             elaboration_rng = np.arange(index_multisine_freq[f] - math.floor(Npts_elab/2),index_multisine_freq[f] + math.ceil(Npts_elab/2), 1)
#         FD_filter = fermi_dirac_filter(np.linspace(-1/(2*DT), 1/(2*DT), Npts_elab),0,bw,n)
        
#         voltage_portion_cell = Npts_elab * ifft(ifftshift(ft_voltage[elaboration_rng] * FD_filter))
#         current_portion = Npts_elab * ifft(ifftshift(ft_current[elaboration_rng] * FD_filter))
#         # v_test[f,:] = voltage_portion_cell
#         # Fill the impedances matrices and create phase and module matrices
#         Z_cell[f] = voltage_portion_cell / current_portion
            
#     print('Impedance extracted.')

#     return Z_cell