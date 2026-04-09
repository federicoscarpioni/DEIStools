import numpy as np

def cut_signal_tail(signal:np.array, size: int) -> np.array:
    return signal[0:size]

def cut_signal_head(signal:np.array, size: int) -> np.array:
    point_removed = signal.size - size
    return signal[point_removed:]

def cut_signal_extemes(signal:np.array, size: int) -> np.array:
    ... 