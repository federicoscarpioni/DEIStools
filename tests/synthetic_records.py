import numpy as np
from deistools.processing.bltva_functions import legendre_basis

def stationary_r_rc(fs=20, multisine_period=1, harmonics=(1, 3, 5, 7, 9), periods=50,
                    R1=10, R2=10, C=5e-3, amplitude=1e-4):
    t = np.arange(int(fs * multisine_period * periods)) / fs
    f = np.asarray(harmonics) / multisine_period
    Z = R1 + R2 / (1 + 1j * 2 * np.pi * f * R2 * C)
    current = np.sum([amplitude * np.cos(2 * np.pi * fk * t) for fk in f], axis =0)
    voltage = np.sum([amplitude * np.abs(Zk) * np.cos(2 * np.pi * fk * t + np.angle(Zk)) for fk, Zk in zip(f, Z)], axis = 0)
    return current, voltage, Z, f

def legendre_oracle(fs=20, multisine_period=1, harmonics=(1, 3, 5, 7, 9), periods=50,
                    R1=10, R2=10, C=5e-3, amplitude=1e-4, scales=(1, 0.05, -0.02)):
    # Z(f,t) = sum_p G_p(f) b_p(t) on BLTVA's own Legendre basis, with G_p = scales[p] * Z_R-RC(f)
    t = np.arange(int(fs * multisine_period * periods)) / fs
    f = np.asarray(harmonics) / multisine_period
    Z = R1 + R2 / (1 + 1j * 2 * np.pi * f * R2 * C)
    G = Z[:, None] * np.asarray(scales)
    b = legendre_basis(t.size, len(scales) - 1)
    current = sum(amplitude * np.cos(2 * np.pi * fk * t) for fk in f)
    voltage = sum((amplitude * (Gk @ b.T) * np.exp(2j * np.pi * fk * t)).real for fk, Gk in zip(f, G))
    return current, voltage, G, f

def non_stationary_r_rc(fs=20, multisine_period=1, harmonics=(1, 3, 5, 7, 9), periods=50,
                    R1=10, R1rate=0.01, R2=10, R2rate=0.01, C=5e-3, amplitude=1e-4):
    t = np.arange(int(fs * multisine_period * periods)) / fs
    f = np.asarray(harmonics) / multisine_period
    R1 = R1 + R1rate * t
    R2 = R2 + R2rate * t
    Z = R1[None, :] + R2[None, :] / (1 + 1j * 2 * np.pi * f[:, None] * R2[None, :] * C)   # (n_freq, N)
    current = np.sum([amplitude * np.cos(2 * np.pi * fk * t) for fk in f], axis =0)
    voltage = np.sum([amplitude * np.abs(Zk) * np.cos(2 * np.pi * fk * t + np.angle(Zk)) for fk, Zk in zip(f, Z)], axis=0)
    return current, voltage, Z, f