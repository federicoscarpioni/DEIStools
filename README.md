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

**📖 [Documentation](https://federicoscarpioni.github.io/DEIStools/)**

## Status of each method

| method | status | validated against |
|---|---|---|
| **STFT** | usable | analytic ground truth on four simulated systems |
| **DMFA** | ⚠️ **provisional** | definition, filter and variance now checked against the source papers; an end-to-end check against an independent implementation is still open |
| **BLTVA** | usable | the source papers' algorithms, plus exact-recovery tests |
| **Lock-in** | usable | agrees with STFT to 9e-14 at the window centres |
| **Goertzel** | usable | reproduces numpy's DFT and the STFT estimator to 1e-10 |
| **Detrending** | usable | four strategies, each measured against the others; only differencing is accepted by BLTVA |

Uncertainty estimation exists for BLTVA only. See the
[roadmap](https://federicoscarpioni.github.io/DEIStools/roadmap/).

## Install

```bash
git clone https://github.com/federicoscarpioni/DEIStools
cd DEIStools
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
- **`passband_filters`** — passband filters for DMFA: the Fermi-Dirac family
  and the standard DSP windows, plus `rejection_db`.
- **`lock_in_amp`** — digital lock-in (`lia_eis`), with two demodulators and
  the integrator design helpers that make the tone rejection exact.
- **`goertzel`** — the generalised Goertzel recurrence (`goertzel_eis`), batch
  and streaming.
- **`detrending`** — `Differencing`, `MovingAverage`, `LowPassRemoval` and
  `PolynomialFit`, with `check_response` to verify a strategy against the
  protocol. Only differencing may be used as a BLTVA weighting factor; the
  module documents why.
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

[`examples/`](examples/) holds twenty self-contained scripts — each builds
its own simulation, calls the estimators directly and prints what it found.

| | |
|---|---|
| **00–03** | one simulated system per script |
| **04–08** | one estimation with each method: how to call it, what comes back |
| **09** | trend removal: the four strategies, and what each costs |
| **10–15** | each estimator's parameters, swept against known truth |
| **20–23** | the three main methods side by side, one system per script |

If you are new, read `00`, `01`, then `04`–`08`. See
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

For DMFA:

- Battistel, Du and La Mantia, *Electroanalysis* **28** (2016) 2346 — the first
  statement of DMFA, with the first-order expansion of eq. (11) that the later
  papers replace.
- Koster, Du, Battistel and La Mantia, *Electrochim. Acta* **246** (2017) 553 —
  the quadrature-filter definition of eq. (9), which is what this package
  implements, and the bandwidth ceiling set by the multisine spacing.
- Battistel and La Mantia, *Electrochim. Acta* **304** (2019) 513 — the
  physical definition via a Volterra series, and the Fermi-Dirac filter of
  eq. (12).
- Chukwu, Mugisa, Brogioli and La Mantia, *ChemElectroChem* **9** (2022)
  e202200109 — the variance of the error on the admittance, eq. (19)–(20), and
  the correlation between neighbouring instants.

For Goertzel:

- Goertzel, *Am. Math. Monthly* **65** (1958) 34 — the original recurrence.
- Sysel and Rajmic, *EURASIP J. Adv. Signal Process.* (2012) 56 — the
  generalisation to non-integer bins, which is the form implemented here.

The lock-in has no source listed because it is the textbook demodulator; what
is specific to a multisine is the integrator design, and that is derived in
the [documentation](https://federicoscarpioni.github.io/DEIStools/methods/lock-in/).

Where a paper is ambiguous, or this implementation makes a choice the paper
does not dictate, it is recorded in the
[implementation notes](https://federicoscarpioni.github.io/DEIStools/implementation/).

## History note

Before `v0.1.0`, this repo also contained lab-hardware acquisition code. That
has moved to a separate package,
[`elma`](https://github.com/federicoscarpioni/elma) (Electrochemistry Lab
Multi-frequency Acquisition), which depends on `deistools` for impedance
estimation. The last combined snapshot remains available at this repo's
`v0.1.0` tag.
