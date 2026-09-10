import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift, fftfreq
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


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

def visualize_dmfa_filtering(
        filter : np.array,
        ft_voltage : np.array, 
        ft_current : np.array, 
        freq_axis: np.array,
        frequenies: list[float],
        freq_indexes: list[int],
    ):

    # Define auxiliary functions  
    def plot_data(
            val_frequency,
        ):
        visible_range = get_visible_range(slider_frequency.val)
        lineV, = axs[0].plot(
            freq_axis[visible_range], 
            np.abs(ft_voltage)[visible_range] * filter, 
            '-o',
            color = 'C0',
            alpha = 0.1,
            label='Original data',
        )
        lineV_filt, = axs[0].plot(
                freq_axis[visible_range], 
                np.abs(ft_voltage)[visible_range] * filter, 
                '-o',
                color = 'C0',
                label='Filtered data',
            )
        axs[0].legend()
        lineI, = axs[1].plot(
            freq_axis[visible_range], 
            np.abs(ft_current)[visible_range], 
            '-o',
            color = 'C0',
            alpha = 0.1,
        )
        lineI_filt, = axs[1].plot(
                freq_axis[visible_range], 
                np.abs(ft_current)[visible_range] * filter, 
                '-o',
                color = 'C0',
            )
        axfiltv = axs[0].twinx()
        axfiltv.set_ylim((0,1))
        filter_V, = axfiltv.plot(
                    freq_axis[visible_range], 
                    filter, 
                    '-o',
                    color='orangered',
                    label = 'Filter'
                )
        axfilti = axs[1].twinx()
        axfilti.set_ylim((0,1))
        filter_I, = axfilti.plot(
                    freq_axis[visible_range], 
                    filter, 
                    '-o',
                    color='orangered',
                )
        # Set name of axis
        axs[0].set_ylabel('Voltage / V')
        axs[1].set_ylabel('Current / A')
        axs[1].set_xlabel('Frequency / Hz')
        axs[0].set_title(f'Fourier Transform frequency {frequenies[0]} Hz')
        line_freq_V = axs[0].axvline(frequenies[slider_frequency.val], color = 'green')
        line_freq_I = axs[1].axvline(frequenies[slider_frequency.val], color = 'green')
        return lineV, lineV_filt, lineI, lineI_filt, line_freq_V, line_freq_I, filter_V, filter_I, axfiltv, axfilti
        
    def get_visible_range(freq_val):
        N_pts = filter.size

        return range(
            freq_indexes[freq_val]-N_pts//2, 
            freq_indexes[freq_val]+N_pts//2,
            1,
        )

    def update_freq(val):
        visible_range = get_visible_range(val)

        lineV.set_xdata(freq_axis[visible_range])
        lineV.set_ydata(np.abs(ft_voltage)[visible_range])

        lineV_filt.set_xdata(freq_axis[visible_range])
        lineV_filt.set_ydata(np.abs(ft_voltage)[visible_range] * filter)

        lineI.set_xdata(freq_axis[visible_range])
        lineI.set_ydata(np.abs(ft_current)[visible_range])

        lineI_filt.set_xdata(freq_axis[visible_range])
        lineI_filt.set_ydata(np.abs(ft_current)[visible_range] * filter)

        filter_V.set_xdata(freq_axis[visible_range])
        filter_I.set_xdata(freq_axis[visible_range])

        axs[0].set_title(f'Fouri(er Transform frequency {frequenies[val]} Hz')
        line_freq_V.set_xdata([frequenies[val]])
        line_freq_I.set_xdata([frequenies[val]])
        axs[0].relim()
        axs[0].autoscale_view()
        axs[1].relim()
        axs[1].autoscale_view()
        axfiltv.relim()
        axfiltv.autoscale_view()
        axfilti.relim()
        axfilti.autoscale_view()

    # Initialize figure
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.subplots_adjust(bottom=0.25)

    # Create slider
    sliders_freq_ax = fig.add_axes([0.20, 0.15, 0.60, 0.03])
    slider_frequency = Slider(
        ax=sliders_freq_ax,
        label="Frequency index",
        valmin=0,
        valmax=len(frequenies)-1,
        valinit=0,
        valstep = 1,
        )
  
    # Initialize plot
    lineV, lineV_filt, lineI, lineI_filt, line_freq_V, line_freq_I, filter_V, filter_I, axfiltv, axfilti = plot_data(slider_frequency.val)
    slider_frequency.on_changed(update_freq)
    plt.show()

    return fig, axs

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