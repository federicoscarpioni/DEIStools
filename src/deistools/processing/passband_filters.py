import numpy as np

def fermi_dirac_filter(fr, fc, bw, n):
    """
    Create a digital filter shaped as a symmetric Fermi-Dirac function.
    """
    X = (fr - fc) / bw
    Y = np.cosh(n) / (np.cosh(n * X) + np.cosh(n))
    return Y