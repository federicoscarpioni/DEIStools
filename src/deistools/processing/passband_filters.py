import numpy as np
from dataclasses import dataclass


@dataclass
class PassbandFilter:
    """
    Low-pass filter centred at 0 Hz, equal to 1/2 at |f| = halfwidth.
    """
    halfwidth : float

    def evaluate(self, frequencies : np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def rejection_db(self, offset : float) -> float:
        """
        Attenuation in dB of a component at ``offset`` Hz from the band centre.
        """
        gain = self.evaluate(np.abs(offset))
        return float(-20 * np.log10(np.maximum(gain, 1e-300)))


@dataclass
class FermiDiracFilter(PassbandFilter):
    """
    Y = cosh(n) / (cosh(n*x) + cosh(n)),    x = f / halfwidth
    """
    order : int

    def evaluate(self, frequencies):
        n = self.order
        x = np.abs(frequencies) / self.halfwidth
        with np.errstate(over='ignore'):
            ratio = np.exp(n * x - n) * (1 + np.exp(-2 * n * x)) / (1 + np.exp(-2 * n))
        return 1 / (1 + ratio)


@dataclass
class RaisedCosineFilter(PassbandFilter):
    """
    Y = 1                                              x <= 1 - alpha
      = (1 + cos(pi * (x - 1 + alpha) / (2 * alpha))) / 2    1 - alpha < x < 1 + alpha
      = 0                                              x >= 1 + alpha
    """
    alpha : float = 1.0

    def __post_init__(self):
        if not 0 < self.alpha <= 1:
            raise ValueError('alpha must be in (0, 1]')

    def evaluate(self, frequencies):
        x = np.abs(frequencies) / self.halfwidth
        a = self.alpha
        edge = 0.5 * (1 + np.cos(np.pi * (x - 1 + a) / (2 * a)))
        return np.where(x <= 1 - a, 1.0, np.where(x < 1 + a, edge, 0.0))


@dataclass
class HannFilter(PassbandFilter):
    """
    Y = 0.5 + 0.5 cos(pi * u),    u = x / 2,    zero for u > 1
    """

    def evaluate(self, frequencies):
        u = np.abs(frequencies) / self.halfwidth / 2
        return np.where(u <= 1, 0.5 + 0.5 * np.cos(np.pi * u), 0.0)


@dataclass
class BlackmanFilter(PassbandFilter):
    """
    Y = 0.42 + 0.5 cos(pi * u) + 0.08 cos(2 pi * u),    zero for u > 1
    """

    def evaluate(self, frequencies):
        grid = np.linspace(0, 1, 20001)
        profile = 0.42 + 0.5 * np.cos(np.pi * grid) + 0.08 * np.cos(2 * np.pi * grid)
        half_point = np.interp(0.5, profile[::-1], grid[::-1])
        u = np.abs(frequencies) / self.halfwidth * half_point
        window = 0.42 + 0.5 * np.cos(np.pi * u) + 0.08 * np.cos(2 * np.pi * u)
        return np.where(u <= 1, window, 0.0)


@dataclass
class BlackmanHarrisFilter(PassbandFilter):
    """
    Y = 0.35875 + 0.48829 cos(pi u) + 0.14128 cos(2 pi u) + 0.01168 cos(3 pi u),    zero for u > 1
    """

    def evaluate(self, frequencies):
        grid = np.linspace(0, 1, 20001)
        profile = (0.35875 + 0.48829 * np.cos(np.pi * grid)
                   + 0.14128 * np.cos(2 * np.pi * grid) + 0.01168 * np.cos(3 * np.pi * grid))
        half_point = np.interp(0.5, profile[::-1], grid[::-1])
        u = np.abs(frequencies) / self.halfwidth * half_point
        window = (0.35875 + 0.48829 * np.cos(np.pi * u)
                  + 0.14128 * np.cos(2 * np.pi * u) + 0.01168 * np.cos(3 * np.pi * u))
        return np.where(u <= 1, window, 0.0)


@dataclass
class NuttallFilter(PassbandFilter):
    """
    Y = 0.3635819 + 0.4891775 cos(pi u) + 0.1365995 cos(2 pi u) + 0.0106411 cos(3 pi u),    zero for u > 1
    """

    def evaluate(self, frequencies):
        grid = np.linspace(0, 1, 20001)
        profile = (0.3635819 + 0.4891775 * np.cos(np.pi * grid)
                   + 0.1365995 * np.cos(2 * np.pi * grid) + 0.0106411 * np.cos(3 * np.pi * grid))
        half_point = np.interp(0.5, profile[::-1], grid[::-1])
        u = np.abs(frequencies) / self.halfwidth * half_point
        window = (0.3635819 + 0.4891775 * np.cos(np.pi * u)
                  + 0.1365995 * np.cos(2 * np.pi * u) + 0.0106411 * np.cos(3 * np.pi * u))
        return np.where(u <= 1, window, 0.0)


@dataclass
class KaiserFilter(PassbandFilter):
    """
    Y = I0(beta * sqrt(1 - u^2)) / I0(beta),    zero for u > 1
    """
    beta : float = 8.0

    def evaluate(self, frequencies):
        grid = np.linspace(0, 1, 20001)
        profile = np.i0(self.beta * np.sqrt(1 - grid ** 2)) / np.i0(self.beta)
        half_point = np.interp(0.5, profile[::-1], grid[::-1])
        u = np.abs(frequencies) / self.halfwidth * half_point
        window = np.i0(self.beta * np.sqrt(np.maximum(0, 1 - u ** 2))) / np.i0(self.beta)
        return np.where(u <= 1, window, 0.0)


@dataclass
class GaussianFilter(PassbandFilter):
    """
    Y = exp(-ln(2) * x^2),    x = f / halfwidth
    """

    def evaluate(self, frequencies):
        x = np.abs(frequencies) / self.halfwidth
        return np.exp(-np.log(2) * x ** 2)
