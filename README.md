# DEIStools

Impedance estimation for electrochemical systems that change while you measure
them.

Classical EIS assumes the cell is the same at the end of the sweep as at the
start. A cell that is charging, ageing or relaxing is not, and what comes back
is a plausible-looking spectrum that belongs to no single instant. DEIStools
collects the published methods for recovering `Z(f, t)` from a single
multi-frequency record, so that they can be compared rather than chosen on
faith.

It is hardware-agnostic: it works on any suitably sampled voltage/current pair,
not only on signals from a particular lab setup.

**📖 [Documentation](https://federicoscarpioni.github.io/deistools/)**

## Status of each method

| method | status | validated against |
|---|---|---|
| **STFT** | usable | analytic ground truth on four simulated systems |
| **DMFA** | ⚠️ **provisional** | behaves as expected, but **not** checked against the literature |
| **BLTVA** | usable | the source papers' algorithms, plus exact-recovery tests |
| **Detrending** (differencing) | usable | the 2022 paper's own worked example |
| Lock-in, Goertzel | not implemented | — |

Uncertainty estimation exists for BLTVA only. See the
[roadmap](https://federicoscarpioni.github.io/deistools/roadmap/).

## Install

```bash
git clone https://github.com/federicoscarpioni/deistools
cd deistools
pip install -e .
```

Requires only `numpy` and `matplotlib`.

## Quick start

```python
import numpy as np
from deistools.processing.bltva import bltva, legendre_basis

# a record of P whole periods of a multisine on these odd harmonics
harmonics = np.array([1, 3, 5, 9, 13, 17, 27, 39, 59, 89, 133, 201, 301])

result = bltva(
    voltage, current, fs, P, harmonics,
    Np=8,                   # order of the Legendre expansion of Z(f,t)
    Nq=2,                   # order of the trend polynomial
    Na=None,                # transient order: chosen for you
    harmonics="excited",
    band_halfwidth=50,
)

Z = result.H_p @ legendre_basis(len(voltage), 8).T   # (n_freq, n_samples)
sigma = result.std_H_p                               # and an uncertainty
```

The record must hold a **whole number of multisine periods** — a fractional
period leaks, and the leakage is indistinguishable from time variation.

## Subpackages

### `deistools.processing`

Signal processing and impedance estimation.

- **`bltva`** — best linear time-varying approximation. Fits `Z(f,t)` as a
  Legendre expansion in the frequency domain and returns the coefficients with
  an uncertainty. Also `estimate_noise_variance`, `select_na`,
  `check_identifiability`, and the slider diagnostic `visualize_bltva_fit`.
- **`stft_functions`** — short-time Fourier estimation (`estimate_impedance`).
- **`dmfa_functions`** — demodulated multi-frequency analysis
  (`extract_impedance` and its helpers), plus `visualize_dmfa_filtering`.
- **`passband_filters`** — the Fermi-Dirac passband DMFA filters with.
- **`detrending`** — trend removal by differencing, for use with BLTVA.
- **`mfa`** — `MultiFrequencyAnalysis`, a class wrapping the above for
  interactive use.

### `deistools.simulation`

Synthetic measurements with known ground truth, for testing estimators.

- **`simulate_r_rc`** — stationary R + (R‖C). Ground truth is analytic.
- **`simulate_r_rc_transistor`** — a transistor sweeps `R_ct(t)`; solved as a
  true time-varying ODE, so the frozen-time ground truth is approximate and
  the script measures by how much.
- **`simulate_legendre`** — a prescribed Legendre expansion. The only exactly
  known *time-varying* ground truth here.
- **`simulate_block_oriented`** — nonlinear branches; the target is a best
  linear approximation, obtained by Monte Carlo.

### `deistools.visualise`

Interactive plotting helpers: `plot_technique`, `inspect_spectrum`,
`visualise_peaks`, `plot_impedance_set` and variants.

## Examples

[`examples/`](examples/) holds seventeen self-contained scripts — each builds
its own simulation, calls the estimators directly and prints what it found.

| | |
|---|---|
| **00–03** | one simulated system per script |
| **04–06** | one estimation with each method: how to call it, what comes back |
| **10–15** | each estimator's parameters, swept against known truth |
| **20–23** | the three methods side by side, one system per script |

If you are new, read `00`, `01`, then `04`–`06`. See
[`examples/README.md`](examples/README.md).

```bash
python examples/04_stft_first_run.py --save
```

## Documentation

Built with MkDocs:

```bash
pip install -e ".[docs]"
mkdocs serve          # live preview at http://127.0.0.1:8000
mkdocs gh-deploy      # publish to GitHub Pages
```

## Sources

- Hallemans *et al.*, *IEEE TIM* **70** (2021) 6503814 — the BLTVA model,
  Algorithms 1 and 2, Sections V and VI.
- Hallemans *et al.*, *IEEE TIM* **71** (2022) 6501711 — trend removal.
- Breugelmans *et al.*, *Electrochim. Acta* **76** (2012) 375 — the short-time
  Fourier approach and the transistor circuit.

Where a paper is ambiguous, or this implementation makes a choice the paper
does not dictate, it is recorded in the
[implementation notes](https://federicoscarpioni.github.io/deistools/implementation/).

## History note

Before `v0.1.0`, this repo also contained lab-hardware acquisition code. That
has moved to a separate package,
[`elma`](https://github.com/federicoscarpioni/elma) (Electrochemistry Lab
Multi-frequency Acquisition), which depends on `deistools` for impedance
estimation. The last combined snapshot remains available at this repo's
`v0.1.0` tag.
