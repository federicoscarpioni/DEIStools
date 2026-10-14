# Roadmap

DEIStools exists to collect the published methods for estimating impedance
from multi-frequency signals of slowly time-varying systems, so that they can
be compared and each one's limitations studied. That makes the list of methods
*not yet* implemented part of the specification rather than a backlog.

## Done

| | item | outcome |
|---|---|---|
| 2 | BLA-style short-time Fourier method | [per-burst BLA](methods/bla.md): noise level calibrated to 0.3% against the observed scatter, odd/even distortion with no false alarm on a linear cell, and a frequency-correlation test that separates noise from time variation (9–19 against 0.73). *The varying-cell case is item 2b.* |
| 1 | Lock-in amplifier and Goertzel filter | both implemented, with [examples](https://github.com/federicoscarpioni/DEIStools/tree/main/examples) 07 and 08 and 15 tests. Both turned out to be the [STFT](methods/stft.md) by another route — Goertzel to 1e-10, the lock-in to 9e-14 — which makes them independent checks on it rather than fourth and fifth methods |
| 4 | Variance estimation for DMFA | `extract_impedance_with_error`, following Chukwu 2022 eq. (19)–(20). Predicted/observed = 0.997 across three noise configurations. *The comparison against traditional band filtering is still open.* |
| 6 | Other detrending strategies | moving average, Butterworth low-pass removal and polynomial fit, measured against differencing, with [example 09](https://github.com/federicoscarpioni/DEIStools/blob/main/examples/09_detrending.py). Conclusion: only differencing can be a BLTVA weighting factor, and only the polynomial fit helps DMFA — and differencing buys columns rather than accuracy. See [detrending](methods/detrending.md) |

## Planned

| | item | why |
|---|---|---|
| 2b | **The three-step estimator of Pintelon 2013 §V-B** | [the per-burst BLA](methods/bla.md) reports noise and distortion on a cell that is steady inside a burst, but on a varying one the time-variation skirt swamps both. Modelling the variation first — fitting the $H_p$ branches at the non-harmonic bins — would lift that. Blocked on one thing: its step 3 reconstructs $G_\text{BLTI}$ by central differences over *neighbouring excited harmonics*, which its Assumption 3 needs equispaced, and this package's grid is quasi-logarithmic |
| 3 | **End-to-end check of DMFA against an independent implementation** | the definition, filter and variance are now verified against Koster 2017, Battistel 2016/2019 and Chukwu 2022, and three bugs were fixed; what is left is a comparison against code that is not this code — see the [notes](implementation/dmfa.md) |
| 4b | **Slepian-basis DMFA** | the Fourier basis forces a periodicity that band-limitedness does not require; a DPSS basis is 5× better at the leading edge for the same cost |
| 5 | **BLTVA with the Gaussian method** for automatic $N_q$ and $N_p$ selection | would remove the two parameters that currently need judgement — and [the $N_q$ result](methods/detrending.md#what-it-buys-is-columns-not-accuracy) makes this more valuable than it looked, since $N_q$ matters more than the detrending choice |
| 6b | **Joint trend + envelope solve for DMFA** | a filter cannot separate a trend from the signal when both are in the same bins; putting the trend in the design matrix returns every affected tone to its trend-free accuracy. The normal equations are block-arrow and can be formed by FFT |
| 6c | **Trend-aware demodulation for the lock-in** | `method='ls'` already carries a local slope and is 11× better than mixing on a drifting cell. Extending it to a local quadratic, and to STFT's windows, is cheap and untested |

Also open: uncertainty estimation for STFT and DMFA, deferred until the
literature on it has been worked through. BLTVA has it because Hallemans *et
al.* derive it; offering error bars for the other two without that grounding
would be worse than offering none.

## Known gaps

- **Loading measured data.** `deistools.processing.data_loader` opens a
  Tkinter file dialog and takes no path argument, so it cannot be called from
  a script, a test or a headless session. `cut_signal_extemes` in
  `data_preparation` is an unimplemented stub, and its name is misspelled.
- **Record trimming is left to the user.** The examples all generate whole
  numbers of periods. `bltva` refuses a record that is not one; STFT and DMFA
  do not check.
- **No diffusion model.** All four simulated systems are RC-like. There is no
  Warburg or finite-length diffusion element, which is what most
  electrochemists would want to validate against.

## How a new estimator should arrive

To keep comparisons honest, a new method should come with:

1. a **first-run example** — one estimation at good settings, showing the call
   and what comes back;
2. a **parameter example** — each parameter swept against a known ground
   truth;
3. an **all-systems example** — the method on all four simulated systems;
4. entries in the four **comparison** scripts;
5. a row in the status table on the [home page](index.md), stating honestly
   what it has been validated against.

The existing examples 04–06, 10–15 and 20–23 follow exactly this pattern.
