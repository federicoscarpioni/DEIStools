import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def plot_impedance_set(Z):

    
    # --- Initialize figure --- #
    
    fig = plt.figure()
    ax = fig.subplots()
    start_index = 0
    p, = ax.plot(Z[:,start_index].real,- Z[:,start_index].imag, '-o') 
    ax.axis('equal')
    ax.set_xlabel('$Z_{real} / Ohm$',fontsize=12)
    ax.set_ylabel('$-Z_{imag} / Ohm$',fontsize=12)
    '''comma is important because: As per the documentation for plot(), it returns
    a list of Line2D object. If you are only plotting one line at a time, you can 
    unpack the list using (notice the comma) '''
    plt.subplots_adjust(bottom=0.25)


    # --- Define updater functions ---#
    
    # The function to be called anytime a slider's value changes
    def update(val):
        val = val-1
        i = index.val-1
        p.set_xdata(Z[:,i].real)
        p.set_ydata(-Z[:,i].imag)
        fig.canvas.draw()
    
    def auto_lim(val):
        ax.relim(visible_only=True)
        ax.autoscale_view()
        fig.canvas.draw()


    # --- Main code ---#
        
    # Index slider config and call
    ax_slide = plt.axes([0.1, 0.05, 0.65, 0.03])
    index = Slider(
        ax=ax_slide,
        label='Index',
        valmin=1,
        valmax=Z[0].size,
        valinit= start_index,
        valfmt = '%d',
        valstep= 1
        )
    index.on_changed(update)
    
    # Reset button config and call
    axes = plt.axes([0.82, 0.03,  0.15, 0.085])
    breset = Button(axes, 'Auto lim', hovercolor='tab:blue')
    breset.on_clicked(auto_lim)

    plt.show()

    return index, breset

def plot_impedance_set_with_error(Z, Zerr, frequencies):
    
    # --- Initialize figure --- #
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    start_index = 0
    
    # Left subplot: Nyquist plot
    p1, = ax1.plot(Z[:,start_index].real, -Z[:,start_index].imag, '-o') 
    ax1.axis('equal')
    ax1.set_xlabel('$Z_{real} / Ohm$', fontsize=12)
    ax1.set_ylabel('$-Z_{imag} / Ohm$', fontsize=12)
    ax1.set_title('Nyquist Plot')
    
    # Right subplot: Error vs Frequency (log-log scale)
    p2, = ax2.loglog(frequencies, Zerr[:,start_index], 'o', markersize=4)
    ax2.set_xlabel('Frequency / Hz', fontsize=12)
    ax2.set_ylabel('Error / Ohm', fontsize=12)
    ax2.set_title('Impedance Error vs Frequency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    # --- Define updater functions ---#
    
    # The function to be called anytime a slider's value changes
    def update(val):
        i = int(index.val) - 1
        # Update Nyquist plot
        p1.set_xdata(Z[:,i].real)
        p1.set_ydata(-Z[:,i].imag)
        # Update error plot
        p2.set_ydata(Zerr[:,i])
        fig.canvas.draw()
    
    def auto_lim(val):
        ax1.relim(visible_only=True)
        ax1.autoscale_view()
        fig.canvas.draw()

    # --- Main code ---#
        
    # Index slider config and call
    ax_slide = plt.axes([0.1, 0.05, 0.65, 0.03])
    index = Slider(
        ax=ax_slide,
        label='Index',
        valmin=1,
        valmax=Z.shape[1],
        valinit=start_index + 1,
        valfmt='%d',
        valstep=1
    )
    index.on_changed(update)
    
    # Reset button config and call
    axes = plt.axes([0.82, 0.03, 0.15, 0.085])
    breset = Button(axes, 'Auto lim', hovercolor='tab:blue')
    breset.on_clicked(auto_lim)

    plt.show()

    return index, breset