# DEIStools

Impedance estimation for electrochemical systems that change while you measure
them.

Classical EIS assumes the cell is the same at the end of the sweep as at the
start. A cell that is charging, ageing or relaxing is not, and what comes back
is a plausible-looking spectrum that belongs to no single instant. DEIStools
collects the published methods for recovering $Z(f, t)$ from a single
multi-frequency record, so that they can be compared rather than chosen on
faith.

The package is deliberately *several* methods and not one. Each rests on a
different assumption, and which is best depends on the system in front of you —
a question this package is built to let you answer for yourself.

## Status of each method

Research software is only as useful as it is honest about its own maturity.

| method | status | validated against |
|---|---|---|
| **STFT** | usable | analytic ground truth on four simulated systems |
| **DMFA** | :warning: **provisional** | definition and filter now verified against the source papers; an end-to-end check against an independent implementation is still open |
| **BLTVA** | usable | the source papers' algorithms, plus exact-recovery tests on a prescribed Legendre system |
| **Lock-in** | usable | agrees with STFT to 1e-13 where the two coincide |
| **Detrending** (differencing) | usable | the 2022 paper's own worked example |
| **Goertzel** | usable | reproduces numpy's DFT and the STFT estimator to 1e-10 |

Uncertainty estimation exists for BLTVA only. STFT and DMFA return an estimate
with no error bars, because the corresponding theory has not been settled here
yet. See the [roadmap](roadmap.md).

!!! warning "DMFA"
    Treat DMFA results as provisional. Verifying the implementation against the
    original description is an open item on the roadmap.

## Install

```bash
pip install -e .
```

## A first estimation

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

# bltva returns model coefficients, not an impedance per instant
Z = result.H_p @ legendre_basis(len(voltage), 8).T   # (n_freq, n_samples)
sigma = result.std_H_p                               # and an uncertainty
```

## Where to go next

- **[Getting started](getting-started.md)** — the shortest path to a result on
  your own data, and which example scripts to read.
- **[Concepts](concepts/signal-model.md)** — the signal model, the notation,
  and when "the impedance at time $t$" is a meaningful thing to ask for.
- **[Methods](methods/stft.md)** — one page per estimator: what it assumes,
  what its parameters do, how it fails.
- **[Implementation notes](implementation/index.md)** — where the source papers
  are ambiguous and what this code chose, with the evidence.

## Sources

The methods implemented here come from:

- Hallemans *et al.*, *IEEE TIM* **70** (2021) 6503814 — the BLTVA model,
  Algorithms 1 and 2, the nonlinear-distortion and uncertainty expressions of
  Sections V and VI, eq. (49) and (55).
- Hallemans *et al.*, *IEEE TIM* **71** (2022) 6501711 — trend removal by
  differencing, the weighting of eq. (26), the rule for $N_a$ in eq. (22).
- Breugelmans *et al.*, *Electrochim. Acta* **76** (2012) 375 - the short-time
  Fourier approach, and the transistor circuit of Fig. 7.

For DMFA:

- Battistel, Du and La Mantia, *Electroanalysis* **28** (2016) 2346 - the first
  statement of DMFA, with the first-order expansion of eq. (11) that the later
  papers replace.
- Koster, Du, Battistel and La Mantia, *Electrochim. Acta* **246** (2017) 553 -
  the quadrature-filter definition, eq. (9), which is what this package
  implements, and the bandwidth ceiling set by the multisine spacing.
- Battistel and La Mantia, *Electrochim. Acta* **304** (2019) 513 - the
  physical definition via a Volterra series, the FFT-EIS/DMFA comparison, and
  the Fermi-Dirac filter of eq. (12).
- Chukwu, Mugisa, Brogioli and La Mantia, *ChemElectroChem* **9** (2022)
  e202200109 - the variance of the error on the admittance, eq. (19), and the
  correlation between neighbouring instants.