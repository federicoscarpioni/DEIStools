from matplotlib.widgets import RangeSlider, Slider
import matplotlib.pyplot as plt
import numpy as np

def inspect_spectrum(
        ft_voltage : np.array, 
        ft_current : np.array, 
        freq_axis: np.array,
    ):

    # Define auxiliary functions  
    def plot_data(values:tuple):
        lineV, = axs[0].plot(
            freq_axis[int(values[0]):int(values[1])], 
            np.abs(ft_voltage)[int(values[0]):int(values[1])], 
            '-o',
        )
        lineI, = axs[1].plot(
            freq_axis[int(values[0]):int(values[1])], 
            np.abs(ft_current)[int(values[0]):int(values[1])], 
            '-o'
        )
        # Set log scale
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        # Set name of axis
        axs[0].set_ylabel('Voltage / V')
        axs[1].set_ylabel('Current / A')
        axs[1].set_xlabel('Frequency / Hz')
        axs[0].set_title('Fourier Transform')
        return lineV, lineI

    def update(val):
        lineV.set_xdata(freq_axis[int(val[0]):int(val[1])])
        lineV.set_ydata(np.abs(ft_voltage)[int(val[0]):int(val[1])])
        lineI.set_xdata(freq_axis[int(val[0]):int(val[1])])
        lineI.set_ydata(np.abs(ft_current)[int(val[0]):int(val[1])])
        axs[0].autoscale_view()
        axs[0].relim()
        axs[1].autoscale_view()
        axs[1].relim()

    # Initialize figure
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.subplots_adjust(bottom=0.25)

    # Create the RangeSlider
    slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
    slider = RangeSlider(
        slider_ax, 
        'indexes', 
        0,  
        freq_axis.size, 
        valstep = 1,
        valinit=[0,1000],
        dragging = True,
    )

    # Initialize plot
    lineV, lineI = plot_data(slider.val)
    slider.on_changed(update)
    plt.show()

    return fig, axs

#------------------------------------------------------------------------------#

def visualise_peaks(
        ft_voltage : np.array, 
        ft_current : np.array, 
        freq_axis: np.array,
        frequenies: list[float],
        freq_indexes: list[int],
    ):

    # Define auxiliary functions  
    def plot_data(
            val_range, 
            val_frequency,
        ):
        visible_range = get_visible_range(slider_frequency.val, slider_range.val)
        lineV, = axs[0].plot(
            freq_axis[visible_range], 
            np.abs(ft_voltage)[visible_range], 
            '-o',
        )
        lineI, = axs[1].plot(
            freq_axis[visible_range], 
            np.abs(ft_current)[visible_range], 
            '-o'
        )
        # Set log scale
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        # Set name of axis
        axs[0].set_ylabel('Voltage / V')
        axs[1].set_ylabel('Current / A')
        axs[1].set_xlabel('Frequency / Hz')
        axs[0].set_title(f'Fourier Transform frequency {frequenies[0]} Hz')
        line_freq_V = axs[0].axvline(frequenies[slider_frequency.val], color = 'green')
        line_freq_I = axs[1].axvline(frequenies[slider_frequency.val], color = 'green')
        return lineV, lineI, line_freq_V, line_freq_I
        
    def get_visible_range(freq_val, range_val):
        return range(
            freq_indexes[freq_val]-range_val, 
            freq_indexes[freq_val]+range_val,
            1,
        )

    def update_freq(val):
        visible_range = get_visible_range(val, slider_range.val)
        lineV.set_xdata(freq_axis[visible_range])
        lineV.set_ydata(np.abs(ft_voltage)[visible_range])
        lineI.set_xdata(freq_axis[visible_range])
        lineI.set_ydata(np.abs(ft_current)[visible_range])
        axs[0].set_title(f'Fouri(er Transform frequency {frequenies[val]} Hz')
        line_freq_V.set_xdata([frequenies[val]])
        line_freq_I.set_xdata([frequenies[val]])
        axs[0].relim()
        axs[0].autoscale_view()
        axs[1].relim()
        axs[1].autoscale_view()

    def update_range(val):
        visible_range = get_visible_range(slider_frequency.val, val)
        lineV.set_xdata(freq_axis[visible_range])
        lineV.set_ydata(np.abs(ft_voltage)[visible_range])
        lineI.set_xdata(freq_axis[visible_range])
        lineI.set_ydata(np.abs(ft_current)[visible_range])
        axs[0].relim()
        axs[0].autoscale_view()
        axs[1].relim()
        axs[1].autoscale_view()


    # Initialize figure
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.subplots_adjust(bottom=0.25)

    # Create sliders
    sliders_range_ax = fig.add_axes([0.20, 0.10, 0.60, 0.03])
    slider_range = Slider(
        ax=sliders_range_ax,
        label="Points to plot",
        valmin=20,
        valmax=3000,
        valinit=50,
        valstep = 1,
        )
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
    lineV, lineI, line_freq_V, line_freq_I = plot_data(slider_range.val, slider_frequency.val)
    slider_range.on_changed(update_range)
    slider_frequency.on_changed(update_freq)
    plt.show()

    return fig, axs