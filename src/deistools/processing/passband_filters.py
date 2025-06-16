import numpy as np
from dataclasses import dataclass, field

def create_fermi_dirac_fun(fr, fc, bw, n):
    """
    Create a digital filter shaped as a symmetric Fermi-Dirac function.
    """
    X = (fr - fc) / bw
    Y = np.cosh(n) / (np.cosh(n * X) + np.cosh(n))
    return Y

@dataclass
class FermiDiracFilter:
    frequencies : np.array
    centered_frequency : float
    bandwidth: int
    n : int
    values : np.array = field(init=False) 

    def __post_init__(self):
        self.values = create_fermi_dirac_fun(
            self.frequencies,
            self.centered_frequency,
            self.bandwidth,
            self.n
        )
