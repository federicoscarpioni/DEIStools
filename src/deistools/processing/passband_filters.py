import numpy as np
from dataclasses import dataclass, field

def create_fermi_dirac_fun(fr, fc, bw, n):
    """
    Create a digital filter shaped as a symmetric Fermi-Dirac function.

        Y = cosh(n) / (cosh(n*X) + cosh(n)),    X = (fr - fc) / bw

    Evaluated as ``1 / (1 + cosh(n*X)/cosh(n))`` with the ratio of hyperbolic
    cosines expanded in exponentials. Written literally, ``cosh(n*X)``
    overflows for the far tails of a wide axis -- the answer is still right,
    since an infinite denominator correctly gives zero, but it warns on every
    call. The identity

        cosh(a)/cosh(b) = exp(|a| - b) * (1 + exp(-2|a|)) / (1 + exp(-2b))

    is therefore taken in log space and finished with a numerically stable
    logistic, so no exponential of a large positive number is ever formed.
    """
    a = np.abs(n * np.asarray(fr - fc, dtype=float) / bw)
    b = abs(n)
    # log of cosh(n*X)/cosh(n); both exponentials have non-positive arguments
    log_ratio = (a - b) + np.log1p(np.exp(-2.0 * a)) - np.log1p(np.exp(-2.0 * b))
    # 1/(1 + exp(log_ratio)), evaluated without overflow for either sign
    small = np.exp(-np.abs(log_ratio))
    return np.where(log_ratio >= 0.0, small / (1.0 + small), 1.0 / (1.0 + small))

@dataclass
class FermiDiracFilter:
    frequencies : np.array
    centered_frequency : float
    cutoff: int
    order : int
    values : np.array = field(init=False) 

    def __post_init__(self):
        self.values = create_fermi_dirac_fun(
            self.frequencies,
            self.centered_frequency,
            self.cutoff,
            self.order
        )
