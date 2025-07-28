import numpy as np
import os
from tkinter import Tk, filedialog
import pandas as pd

def load_impedance_set(text : str = 'Select npy impedance file'):
    # Open dialog window to select pkl file containing data
    root = Tk() # pointing root to Tk() to use it as Tk() in program.
    root.withdraw() # Hides small tkinter window.
    root.attributes('-topmost', True) # Opened windows will be active. above all windows despite of selection.
    # Returns file path as str
    filename = filedialog.askopenfilename(title = text)
    directory = os.path.dirname(filename)
    # Import impedance data
    Z = np.load(filename)
    return Z, directory

def load_voltage_current(text : str = "Select directory with voltage and current npy files"):
    # Open dialog window to select pkl file containing data
    root = Tk() # pointing root to Tk() to use it as Tk() in program.
    root.withdraw() # Hides small tkinter window.
    root.attributes('-topmost', True) # Opened windows will be active. above all windows despite of selection.
    # Returns file path as str
    directory = filedialog.askdirectory(title = text)
    # Import impedance data
    voltage = np.load(directory + '/voltage.npy')
    current = np.load(directory + '/current.npy')
    return voltage, current, directory

def load_pyeclab_data(text: str = "Select csv file"):
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(title = text)
    return pd.read_csv(file_path, sep = '\t')

if __name__ == '__main__':
    load_impedance_set()
    load_voltage_current()