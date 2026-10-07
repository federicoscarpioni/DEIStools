import numpy as np


def generate_multisine(t, frequencies, amplitudes, phases):
    signal = np.zeros(len(t))
    for f, A, phi in zip(frequencies, amplitudes, phases):
        signal += A * np.cos(2 * np.pi * f * t + phi)
    return signal


def synthesize_voltage(t, frequencies, amplitudes, phases, Z):
    voltage = np.zeros(len(t))
    for k, (f, A, phi) in enumerate(zip(frequencies, amplitudes, phases)):
        voltage += A * np.abs(Z[k]) * np.cos(2 * np.pi * f * t + phi + np.angle(Z[k]))
    return voltage


def simulate(duration_s, sampling_time_s, frequencies, amplitudes, phases,
             impedance_fn, noise_std_V=0.0, noise_std_A=0.0, seed=None):
    """
    Simulate a multisine-excited electrochemical cell.

    impedance_fn(f, t) must return a (n_freq, N) complex array giving the
    frozen-time impedance at each excitation frequency and each sample time.

    Returns a dict with time_s, voltage_V, current_A, Z_true, frequencies_Hz.
    """
    rng = np.random.default_rng(seed)

    N = int(round(duration_s / sampling_time_s))
    t = np.arange(N) * sampling_time_s
    frequencies = np.asarray(frequencies)
    amplitudes = np.asarray(amplitudes)
    phases = np.asarray(phases)

    Z = impedance_fn(frequencies, t)

    current_A = generate_multisine(t, frequencies, amplitudes, phases)
    voltage_V = synthesize_voltage(t, frequencies, amplitudes, phases, Z)

    if noise_std_V > 0:
        voltage_V += rng.normal(0.0, noise_std_V, N)
    if noise_std_A > 0:
        current_A += rng.normal(0.0, noise_std_A, N)

    return {
        'time_s': t,
        'voltage_V': voltage_V,
        'current_A': current_A,
        'Z_true': Z,
        'frequencies_Hz': frequencies,
    }
