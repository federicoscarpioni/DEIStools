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

def elaboration_frequency_axis(Npts_elab, measurement_time):
    return (np.arange(Npts_elab) - Npts_elab // 2) / float(measurement_time)


def elaboration_time_axis(Npts_elab, measurement_time):
    return np.arange(Npts_elab) * (float(measurement_time) / Npts_elab)


def max_elaboration_points(indexes_multisine_freq, n_samples):
    lowest = min(indexes_multisine_freq)
    highest = max(indexes_multisine_freq)
    return 2 * min(lowest, n_samples - highest)


def _check_elaboration_range(indexes_multisine_freq, Npts_elab, n_samples):
    limit = max_elaboration_points(indexes_multisine_freq, n_samples)
    if Npts_elab > limit:
        raise ValueError(
            f"Npts_elab={Npts_elab} does not fit: the extraction window would "
            f"run past the edge of the spectrum at bin "
            f"{max(indexes_multisine_freq)} of {n_samples}. Use at most "
            f"{limit} (see max_elaboration_points).")


def extract_impedance(
        ft_voltage,
        ft_current,
        indexes_multisine_freq:list,
        filter,
        Npts_elab,
):
    _check_elaboration_range(indexes_multisine_freq, Npts_elab, ft_voltage.size)
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
        filter : np.ndarray,
        ft_voltage : np.ndarray, 
        ft_current : np.ndarray, 
        freq_axis: np.ndarray,
        frequencies: list[float],
        freq_indexes: list[int],
    ):

    # Define auxiliary functions  
    def plot_data(
            val_frequency,
        ):
        visible_range = get_visible_range(slider_frequency.val)
        lineV, = axs[0].plot(
            freq_axis[visible_range], 
            np.abs(ft_voltage)[visible_range], 
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
        axs[0].set_title(f'Fourier Transform frequency {frequencies[val_frequency]} Hz')
        line_freq_V = axs[0].axvline(frequencies[slider_frequency.val], color = 'green')
        line_freq_I = axs[1].axvline(frequencies[slider_frequency.val], color = 'green')
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

        axs[0].set_title(f'Fourier Transform frequency {frequencies[val]} Hz')
        line_freq_V.set_xdata([frequencies[val]])
        line_freq_I.set_xdata([frequencies[val]])
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
        valmax=len(frequencies)-1,
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
        indexes_multisine_freq: list,
        filter,
        Npts_elab,
):
    _check_elaboration_range(indexes_multisine_freq, Npts_elab, ft_voltage.size)
    n_freq = len(indexes_multisine_freq)
    impedance = np.zeros((n_freq, Npts_elab), dtype="complex128")
    variance = np.zeros((n_freq, Npts_elab), dtype="float64")

    # N' uses the length of the record, which is what the noise was spread
    # over, not the width of the extracted band.
    N_prime = ft_voltage.size / np.sum(np.abs(filter) ** 2)

    for f in range(n_freq):
        if Npts_elab % 2 == 0:
            elaboration_rng = np.arange(indexes_multisine_freq[f] - Npts_elab // 2,
                                        indexes_multisine_freq[f] + Npts_elab // 2)
        else:
            elaboration_rng = np.arange(indexes_multisine_freq[f] - Npts_elab // 2,
                                        indexes_multisine_freq[f] + Npts_elab // 2 + 1)
        voltage_filtered = Npts_elab * ifft(ifftshift(ft_voltage[elaboration_rng] * filter))
        current_filtered = Npts_elab * ifft(ifftshift(ft_current[elaboration_rng] * filter))
        impedance[f] = voltage_filtered / current_filtered
        numerator = variance_voltage + np.abs(impedance[f]) ** 2 * variance_current
        variance[f] = numerator / (2 * N_prime * np.abs(current_filtered) ** 2)

    return impedance, variance