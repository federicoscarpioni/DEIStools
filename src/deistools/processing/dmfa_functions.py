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
    """Local frequency axis of an extracted band, in Hz offset from its centre.

    The band is ``Npts_elab`` DFT bins wide and the bins are ``1/T`` apart, so
    it spans ``Npts_elab / T`` Hz -- *not* the sampling rate. Build the
    passband filter on this axis, or its bandwidth is scaled wrongly by
    ``fs*T/Npts_elab``.
    """
    return (np.arange(Npts_elab) - Npts_elab // 2) / float(measurement_time)


def elaboration_time_axis(Npts_elab, measurement_time):
    """Times at which ``extract_impedance`` reports, spanning [0, T).

    The inverse transform of ``Npts_elab`` bins of width ``1/T`` returns
    ``Npts_elab`` samples covering the whole record, so the step is
    ``T / Npts_elab`` -- coarser than the acquisition, and the resolution at
    which DMFA can follow a change.
    """
    return np.arange(Npts_elab) * (float(measurement_time) / Npts_elab)


def max_elaboration_points(indexes_multisine_freq, n_samples):
    """Largest ``Npts_elab`` whose band fits inside the spectrum at every frequency.

    Each frequency is extracted from a window of ``Npts_elab`` bins centred on
    its peak, so the window must not run past either end of the spectrum. The
    binding constraint is the frequency closest to an edge -- in practice the
    highest one, since its bin sits ``max(index)`` from the top.
    """
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
        axs[0].set_title(f'Fourier Transform frequency {frequenies[val_frequency]} Hz')
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

        axs[0].set_title(f'Fourier Transform frequency {frequenies[val]} Hz')
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
        indexes_multisine_freq: list,
        filter,
        Npts_elab,
):
    """Impedance and the variance of its error, per frequency and instant.

    DMFA is a **linear** estimator of the envelope -- band selection, a
    diagonal weighting by the filter, and an inverse transform are all linear
    -- so the variance of its output is the covariance of a linear map applied
    to the measurement noise. That is the same route BLTVA takes, and it gives
    the expression of Chukwu et al. (2022) eq. (19)-(20):

        var(Re Z) = [ var_v + |Z|^2 * var_i ] / ( 2 * N' * |I_filtered|^2 )

        N' = N / sum(|g|^2)

    with ``N`` the length of the whole record -- not of the extracted band --
    and ``N'`` the usual equivalent noise bandwidth. The real and imaginary
    parts have the same variance, which is why the factor of two appears.

    Verified against the scatter of 200 noise realisations on a stationary
    cell: predicted and observed agree to better than 1%.

    Parameters
    ----------
    variance_voltage, variance_current : float
        Variances (**not** standard deviations) of the noise on the two
        channels. They can be measured from the instrument, or -- better --
        taken from bins the excitation never reaches, as
        :func:`deistools.processing.bltva.estimate_noise_variance` does, which
        also captures the noise colour.

    Returns
    -------
    (impedance, variance) : ndarray, ndarray
        Both ``(n_frequencies, Npts_elab)``. ``variance`` is that of either
        the real or the imaginary part.

    Notes
    -----
    This covers **noise only**. Nothing here knows about the band-limited
    model being wrong, which near the record edges it badly is; compare the
    prediction with the error you actually see, exactly as for BLTVA.
    """
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

def independent_time_step(bandwidth):
    """Output spacing at which neighbouring DMFA errors are uncorrelated.

    The filter kernel spans about ``1/bandwidth`` in time, so estimates taken
    closer together than that rest on overlapping data and their errors are
    correlated. Chukwu et al. (2022) put the correlation below 20% once
    successive instants are more than ``0.5/bandwidth`` apart, and measured on
    a stationary cell over 150 noise realisations the correlation between
    neighbours is:

        spacing     corr(e_j, e_j+1)
        0.125/bw          84%
        0.25/bw           49%
        0.5/bw             3%
        1/bw              -3%

    Reporting more densely than this is fine for looking at a result, but the
    points are then **not independent observations**: using them as such in a
    least-squares fit will make the parameter uncertainties too small.

    Note that ``Npts_elab`` sets both the width of the extracted band and the
    number of reported instants, and those are different concerns -- the band
    must be wide enough for the filter to have decayed, which is finer than
    this spacing. Extract with a wide band and decimate the result to
    ``independent_time_step`` before fitting.
    """
    return 0.5 / float(bandwidth)


def pad_record(signal, factor=2):
    """Zero-pad a record so DMFA's band extraction is a *linear* convolution.

    Multiplying a spectrum by a filter is a **circular** convolution in time:
    the filter kernel that reaches past the end of the record wraps around and
    picks up the beginning. For a time-varying impedance the two ends are
    different -- 106 ohm against 43 ohm on the transistor system -- so the
    estimate near either boundary is contaminated by the far one. That is
    where DMFA's edge oscillation comes from.

    Zero-padding before the transform makes the convolution linear, and the
    kernel then runs into zeros instead of into the other end of the record.
    Measured, with the band extracted 24 bandwidths wide and the output
    decimated to the independent spacing:

        ===========  ==============  ==============  =============
        system       worst, plain    worst, padded   median
        ===========  ==============  ==============  =============
        transistor   7.5e-01         3.2e-02         unchanged
        legendre     8.8e-01         3.1e-02         unchanged
        ===========  ==============  ==============  =============

    A factor of 24 to 29 on the worst case, and the interior does not move at
    all (0.99x) -- as it must not, since the kernel never reaches that far.

    It works better than "assume the signal is zero outside" has any right to,
    and the reason is worth knowing: the padding attenuates the **numerator
    and the denominator by the same factor**. At the first instant the
    filtered current falls to 0.49 of its mid-record value, and the filtered
    voltage falls with it, so the ratio survives. Zero-padding tapers the
    estimate's signal-to-noise near the edges but not its accuracy.

    **How much padding.** Only as much as the kernel actually reaches, which
    is less than it sounds. Measured by sweeping the factor on the transistor
    record, where ``1/bandwidth`` is 40 s against a 400 s record:

        ============  ==================  ==============
        pad factor    added / (1/bw)      worst error
        ============  ==================  ==============
        1.00x         0                   7.5e-01
        1.10x         1.0                 3.2e-02
        1.50x         5.0                 3.2e-02
        2.00x         10.0                3.2e-02
        ============  ==================  ==============

    One kernel length already buys the whole benefit and more buys nothing.
    ``factor=2`` is the convenient default -- it keeps the FFT size tidy and
    costs only arithmetic -- but ``1 + 1/(bandwidth * duration)`` is enough.

    Note that this is *not* the mirroring suggested for non-periodic records
    in the DMFA literature. Mirroring a random-phase multisine destroys the
    carrier -- it scatters more energy off the excited bins than it leaves on
    them. See the implementation notes.

    Returns
    -------
    ndarray
        The record followed by ``(factor - 1) * len(signal)`` zeros.
    """
    signal = np.asarray(signal)
    extra = int(round((factor - 1) * signal.size))
    if extra < 0:
        raise ValueError("factor must be at least 1")
    return np.concatenate([signal, np.zeros(extra, dtype=signal.dtype)])
