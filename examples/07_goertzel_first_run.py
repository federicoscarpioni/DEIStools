"""
07 -- A first Goertzel estimation.

The Goertzel algorithm evaluates one DFT bin with a second-order recurrence
instead of a transform. A multisine measurement wants only a handful of known
bins, which is exactly the case the algorithm was made for.

This runs it on the same transistor measurement as examples 04-06, and checks
it against both the ground truth and the STFT -- which it should reproduce
exactly, since the two compute the same thing by different routes.

Run:  python examples/07_goertzel_first_run.py [--save]
"""
import time as clock

import numpy as np
import matplotlib.pyplot as plt

from deistools.processing.goertzel import (
    goertzel_dft, goertzel_eis, GoertzelFilter, cost_ratio)
from deistools.processing.stft_functions import estimate_impedance
from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor

from _plotting import parse_args, finish, show_all, overview

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# The measurement to analyse (see example 01)
# --------------------------------------------------------------------------
Tp = 20.0
fs = 1000.0
P = 20

harmonics = np.array([1, 3, 5, 9, 13, 17, 27, 39, 59, 89, 133, 201, 301])
frequencies = harmonics / Tp
amplitudes = np.full(frequencies.size, 0.01)
phases = np.random.default_rng(3).uniform(0, 2 * np.pi, frequencies.size)

N = int(fs * Tp) * P
T = N / fs

sim = simulate_r_rc_transistor(
    T, 1.0 / fs, frequencies, amplitudes, phases,
    R_s=10.0, R_p=30.0, C=1e-3,
    V_gate_start=1.645, V_gate_rate=(2.455 - 1.645) / T, R_T_max=70.0,
    noise_std_V=1e-4, seed=1)

time = sim["time_s"]
voltage = sim["voltage_V"]
current = sim["current_A"]
Z_true = sim["Z_true"]


# --------------------------------------------------------------------------
# The estimation
# --------------------------------------------------------------------------
# The record is cut into blocks, exactly as STFT does, and within each block
# only the 13 excited bins are evaluated. 'period' is the block length in
# seconds and must hold a whole number of periods of every tone.

block_period = 1 * Tp

Z, t = goertzel_eis(voltage, current, frequencies, block_period, 1.0 / fs)

print(f"  goertzel_eis -> Z {Z.shape}, one spectrum per {block_period:.0f} s block")
print(f"  reported at t = {t[0]:.0f}, {t[1]:.0f} ... {t[-1]:.0f} s")


# --------------------------------------------------------------------------
# What the recurrence actually computes
# --------------------------------------------------------------------------
# One bin, directly, with no transform anywhere.

first_block = voltage[:int(block_period * fs)]
bins = frequencies * block_period               # = the harmonic numbers here
by_recurrence = goertzel_dft(first_block, bins)
by_fft = np.fft.fft(first_block)[np.round(bins).astype(int)]

print(f"\n  goertzel_dft on one block vs np.fft.fft at the same bins:")
print(f"    max relative difference {np.max(np.abs(by_recurrence - by_fft) / np.abs(by_fft)):.1e}")
print(f"    (the bins here are the harmonic numbers {bins[:4].astype(int)} ... {int(bins[-1])})")


# --------------------------------------------------------------------------
# How good is it?
# --------------------------------------------------------------------------
truth = np.array([np.interp(t, time, Z_true[k].real)
                  + 1j * np.interp(t, time, Z_true[k].imag)
                  for k in range(frequencies.size)])
error = np.abs(Z - truth) / np.abs(truth)

print(f"\n  relative error against the known truth:")
print(f"    median {np.median(error):.2e}, worst {np.max(error):.2e}")


# --------------------------------------------------------------------------
# It is the STFT, computed differently
# --------------------------------------------------------------------------
start = clock.time()
Z_g, _ = goertzel_eis(voltage, current, frequencies, block_period, 1.0 / fs)
t_goertzel = clock.time() - start

start = clock.time()
Z_s, _, _, t_s = estimate_impedance(voltage, current, frequencies,
                                    block_period, 1.0 / fs)
t_stft = clock.time() - start

print(f"\n  against STFT on the same blocks:")
print(f"    max relative difference {np.max(np.abs(Z_g - Z_s) / np.abs(Z_s)):.1e}")
print(f"    the two are the same estimator; only the arithmetic differs")

samples_per_block = int(block_period * fs)
print(f"\n  cost, for {frequencies.size} bins out of {samples_per_block}:")
print(f"    goertzel {t_goertzel:.2f} s, FFT-based STFT {t_stft:.2f} s")
print(f"    operation-count ratio n_bins / log2(N) = "
      f"{cost_ratio(frequencies.size, samples_per_block):.2f}")
print("    Goertzel wins below 1, and only on operation count -- a compiled")
print("    FFT beats a Python recurrence long before that. Its real advantage")
print("    is that it needs no record at all; see the streaming section.")


# --------------------------------------------------------------------------
# The same filter, on a stream
# --------------------------------------------------------------------------
# Two state variables per frequency, updated one sample at a time. Nothing is
# buffered, so this form runs during an acquisition rather than after it.

filt = GoertzelFilter(frequencies[0], 1.0 / fs, samples_per_block)
streamed = [value for s in voltage[:3 * samples_per_block]
            if (value := filt.push(s)) is not None]
blocked = goertzel_dft(
    voltage[:3 * samples_per_block].reshape(3, samples_per_block),
    [frequencies[0] * block_period])[:, 0]

print(f"\n  GoertzelFilter fed sample by sample over 3 blocks:")
print(f"    completed {len(streamed)} blocks, max difference from the batch "
      f"call {np.abs(np.array(streamed) - blocked).max():.1e}")


# --------------------------------------------------------------------------
# Figure 1: the standard view -- signals, spectra, estimate
# --------------------------------------------------------------------------
fig1 = overview(time, voltage, current, fs, frequencies, Z=Z, t_Z=t,
                Z_true=truth)
finish(fig1, "07_goertzel_overview", args.save,
       f"Goertzel with a {block_period:.0f} s block")


# --------------------------------------------------------------------------
# Figure 2: the result, and that it is the STFT
# --------------------------------------------------------------------------
fig2, axs = plt.subplots(1, 3, figsize=(15, 4.6))

for i, colour in zip((0, 6, 12), ("C0", "C1", "C2")):
    axs[0].plot(time, np.abs(Z_true[i]), color=colour, lw=2, alpha=0.35)
    axs[0].plot(t, np.abs(Z[i]), "o-", ms=4, color=colour,
                label=f"{frequencies[i]:.2f} Hz")
axs[0].set(xlabel="time / s", ylabel=r"$|Z|$ / $\Omega$",
           title="magnitude (pale line: truth)")
axs[0].legend(fontsize=8)

axs[1].loglog(frequencies, np.median(error, axis=1), "-o", ms=4, color="C3")
axs[1].set(xlabel="frequency / Hz", ylabel="median relative error",
           title="error against frequency")
axs[1].grid(True, which="both", alpha=0.25)

difference = np.abs(Z_g - Z_s) / np.abs(Z_s)
axs[2].semilogy(frequencies, np.median(difference, axis=1), "-o", ms=4,
                color="C2")
axs[2].set(xlabel="frequency / Hz", ylabel="relative difference",
           title="Goertzel minus STFT (arithmetic only)")
axs[2].grid(True, which="both", alpha=0.25)

finish(fig2, "07_goertzel_result", args.save,
       "one Goertzel estimation, and its agreement with STFT")

print("\n  Goertzel evaluates only the bins you ask for, which is the right")
print("  shape for a multisine, but offline an FFT already gives all of them")
print("  cheaply. Use it when the record has not finished -- or as an")
print("  independent check on the STFT path, which is what this file is.")

show_all(args.save)
