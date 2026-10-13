# Roadmap

DEIStools exists to collect the published methods for estimating impedance
from multi-frequency signals of slowly time-varying systems, so that they can
be compared and each one's limitations studied. That makes the list of methods
*not yet* implemented part of the specification rather than a backlog.

## Planned

| | item | why |
|---|---|---|
| 1 | ~~Lock-in amplifier and Goertzel filter~~ **done** | both implemented, with examples and tests |
| 2 | **BLA-style short-time Fourier method**, with noise and nonlinearities estimated | would give STFT the uncertainty and distortion diagnostics BLTVA has |
| 3 | **Verify DMFA against the literature** | the current implementation is unverified — see the [notes](implementation/dmfa.md) |
| 4 | ~~Variance estimation for DMFA~~ **done** (`extract_impedance_with_error`); the comparison against traditional band filtering is still open | |
| 4b | **Slepian-basis DMFA** | the Fourier basis forces a periodicity that band-limitedness does not require; a DPSS basis is 5x better at the leading edge for the same cost |
| 5 | **BLTVA with the Gaussian method** for automatic $N_q$ and $N_p$ selection | would remove the two parameters that currently need judgement |
| 6 | **Other detrending strategies**, implemented and compared | only differencing exists today |
| 6b | **Joint trend + envelope solve for DMFA** | a filter cannot separate a trend from the signal when both are in the same bins; putting the trend in the design matrix returns every affected tone to its trend-free accuracy. The normal equations are block-arrow and can be formed by FFT |

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
