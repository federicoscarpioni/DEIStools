import numpy as np
from dataclasses import dataclass
from scipy.interpolate import CubicSpline

def get_signal_mean(signal, samples_per_period, sampling_frequency):
    P = signal.size // samples_per_period
    means = signal[:P*samples_per_period].reshape(P, samples_per_period).mean(axis=1)
    # Mean time of the samples in each period: sample n sits at n / fs, so the
    # centre of period k is half a sample before (k + 0.5) * samples_per_period.
    centres = (np.arange(P) * samples_per_period + (samples_per_period - 1) / 2) / sampling_frequency
    return means, centres

@dataclass
class Baseline:
    intercept : float
    slope : float

    def evaluate(self,x):
        return self.intercept + self.slope * x

def get_baseline_extremes(
        signal: np.ndarray, 
        samples_per_period: int, 
        fs: float,
    ) -> Baseline:
    '''
    Note: with this method, the estimation of the intercept is the first point 
    of the multisine not a real intercept.
    '''
    # Line through the first and last point of the window.
    slope = (signal[-1] - signal[0]) * fs / (signal.size - 1)
    intercept = signal[0]
    return Baseline(intercept=intercept, slope=slope)

def get_baseline_mean(
        signal: np.ndarray, 
        samples_per_period: int, 
        fs: float,
    ) -> Baseline:
    mean, centres = get_signal_mean(signal, samples_per_period, fs)
    if mean.size < 2:
        raise ValueError(
            f"A baseline needs at least 2 whole periods, got {mean.size}."
        )
    # Line through the first and last period means. The means hold no
    # multisine content, so the line is independent of the response, and
    # passing through both ends removes the edge jump that leaks at 1/k.
    slope = (mean[-1] - mean[0]) / (centres[-1] - centres[0])
    intercept = mean[0] - slope * centres[0]
    return Baseline(intercept=intercept, slope=slope)


