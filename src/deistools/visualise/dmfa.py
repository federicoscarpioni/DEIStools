import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_dmfa_filtering(
        filter : np.ndarray,
        band_offsets : np.ndarray,
        U : np.ndarray,
        V : np.ndarray, 
        freq_axis: np.ndarray,
        frequencies: np.ndarray,
        freq_indexes: np.ndarray,
    ):

    # Define auxiliary functions  
    def plot_data(
            val_frequency,
        ):
        visible_range = get_visible_range(slider_frequency.val)
        lineV, = axs[0].plot(
            freq_axis[visible_range], 
            np.abs(V)[visible_range], 
            '-o',
            color = 'C0',
            alpha = 0.1,
            label='Original data',
        )
        lineV_filt, = axs[0].plot(
                freq_axis[visible_range],
                np.abs(V)[visible_range] * filter,
                '-o',
                color = 'C0',
                label='Filtered data',
            )
        axs[0].legend()
        lineI, = axs[1].plot(
            freq_axis[visible_range],
            np.abs(U)[visible_range],
            '-o',
            color = 'C0',
            alpha = 0.1,
        )
        lineI_filt, = axs[1].plot(
                freq_axis[visible_range], 
                np.abs(U)[visible_range] * filter, 
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
        return freq_indexes[freq_val] + band_offsets

    def update_freq(val):
        visible_range = get_visible_range(val)

        lineV.set_xdata(freq_axis[visible_range])
        lineV.set_ydata(np.abs(V)[visible_range])

        lineV_filt.set_xdata(freq_axis[visible_range])
        lineV_filt.set_ydata(np.abs(V)[visible_range] * filter)

        lineI.set_xdata(freq_axis[visible_range])
        lineI.set_ydata(np.abs(U)[visible_range])

        lineI_filt.set_xdata(freq_axis[visible_range])
        lineI_filt.set_ydata(np.abs(U)[visible_range] * filter)

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