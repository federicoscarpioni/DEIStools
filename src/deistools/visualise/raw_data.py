import matplotlib.pyplot as plt

def plot_technique(voltage, current, time):
    # Initialize figure with two rows
    fig, axs = plt.subplots(2, 1, figsize=(14, 6))
    # Plot voltage
    axs[0].plot(time, voltage)
    axs[0].set_xlabel('Time / s')
    axs[0].set_ylabel('Voltage / V')
    axs[0].grid('on')
    # Plot current
    axs[1].plot(time, current)
    axs[1].set_xlabel('Time / s')
    axs[1].set_ylabel('Current / A')
    axs[1].grid('on')
    plt.show()
    return fig, axs