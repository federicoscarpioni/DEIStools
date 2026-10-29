# DEIStools

Impedance estimation for electrochemical systems that change while you measure
them.

Classical EIS assumes the cell is the same at the end of the sweep as at the
start. A cell that is charging, ageing or relaxing is not, and what comes back
is a plausible-looking spectrum that belongs to no single instant. DEIStools
collects published methods for recovering `Z(f, t)` from a single
multi-frequency record, so that they can be compared rather than chosen on
faith.

It is hardware-agnostic: it works on any suitably sampled voltage/current pair,
not only on signals from a particular lab setup.

> **The package is being rebuilt.** It has been cut back to code that has been
> reviewed and can be defended. Tests, examples and documentation are being
> rewritten by hand, and BLTVA is being reimplemented. See
> [Next steps](#next-steps).

## Current state

| subpackage | module | what it does |
|---|---|---|
| `processing` | `stft_functions` | short-time Fourier estimate of `Z(f, t)`, plus the bin-index helpers |
| | `dmfa_functions` | dynamic multi-frequency analysis, with and without an error estimate |
| | `passband_filters` | the DMFA passband filters: Fermi–Dirac, raised cosine and standard windows |
| | `mfa` | `MultiFrequencyAnalysis`, a class that wraps the above for interactive use |
| | `data_loader` | file-dialog loaders for impedance sets and recorded signals |
| | `data_preparation` | trimming the head or tail of a signal |
| `simulation` | `engine`, `models`, `trends` | synthetic records with a known ground truth — see [its README](src/deistools/simulation/README.md) |
| `visualise` | `fourier_space`, `impedance`, `raw_data` | interactive plots of spectra, impedance sets and raw signals |

There are no tests and no examples at the moment.

**Inputs are the caller's responsibility.** STFT and DMFA assume that the
record holds a whole number of multisine periods and that every excited tone
falls on a DFT bin. They do not check either. A record that breaks these
assumptions leaks, and the leakage looks like time variation.

## Install

```bash
git clone https://github.com/federicoscarpioni/DEIStools
cd DEIStools
pip install -e .
```

Requires `numpy` and `matplotlib`. `data_loader` also needs `pandas` and
`tkinter`, which are not declared yet (see [Next steps](#next-steps)).

## Methods: theory and sources

### STFT — short-time Fourier transform

The record is cut into consecutive windows, each a whole number of multisine
periods long, and each window is transformed on its own. In each window, the
impedance at an excited frequency is the ratio of the voltage and current DFT
coefficients at that tone's bin:

```
Z(f_k, t_i) = V_i(k) / I_i(k),      t_i = centre of window i
```

Because each window holds whole periods, every tone falls exactly on a bin and
nothing leaks between tones. The price is time resolution: one value per
window, and the cell is assumed stationary *within* each window. A longer
window resolves lower frequencies and averages more noise, but it follows the
time variation more coarsely.

*Source:* Breugelmans *et al.* (2012) compute an instantaneous impedance by
resolving the impedance over fractions of an odd-random-phase multisine record.

### DMFA — dynamic multi-frequency analysis

DMFA transforms the *whole* record once, then treats each excited tone
separately:

1. Take a band of `Npts_elab` bins centred on the tone.
2. Multiply it by a passband filter centred on the tone (a quadrature filter).
3. Inverse-transform the band. This gives the complex envelope of the voltage
   and of the current at that tone, as functions of time.
4. Take their ratio, `Z(f_k, t) = V_k(t) / I_k(t)`.

The filter bandwidth sets the time resolution: a wider band follows faster
changes but admits more noise. The bandwidth is also limited by the spacing of
the excited tones, because neighbouring bands must not overlap.
`extract_zero_frequency` applies the same filtering around 0 Hz and returns
the slow, DC part of the signals. `extract_impedance_with_error` also
propagates the noise variance of the two channels into a variance on `Z`.

The default filter is the Fermi–Dirac shape

```
Y = cosh(n) / (cosh(n X) + cosh(n)),      X = (f - f_c) / bw
```

which is flat in the pass band and falls off steeply at the edges, with the
sharpness set by `n`.

*Sources:*
- Battistel, Du and La Mantia (2016): DMFA introduced, for the analysis of
  non-stationary impedance spectra.
- Koster, Du, Battistel and La Mantia (2017): dynamic impedance redefined
  with quadrature filters, and the design of multisines for it.
- Battistel and La Mantia (2019): the physical definition of dynamic
  impedance through Volterra series, and the strategy for extracting it,
  including the filter.
- Chukwu, Mugisa, Brogioli and La Mantia (2022): the statistics of the
  measurement noise on dynamic impedance spectra, which is what the error
  estimate follows.

### BLTVA — best linear time-varying approximation *(being reimplemented)*

BLTVA writes the time-varying transfer function as a sum of fixed spectra
times Legendre polynomials in time:

```
Z(f, t) = sum_p  G_p(f) * b_p(t)
```

A polynomial trend is added, and the coefficients are fitted in the frequency
domain from the bins around each excited tone. Because the time variation
spreads every tone into its neighbouring bins, those bins carry the
information about how `Z` changes.

The simulation's Legendre oracle is built on exactly this model.
Any implementation must use the *same* basis, including the subtraction of
the sample mean from every order p ≥ 1, or it will disagree with the oracle at
the O(1/N) level. That basis is `deistools.simulation.models.legendre.legendre_basis`.

*Sources:* Hallemans *et al.* (2021) for the model and the estimator;
Hallemans *et al.* (2022) for trend removal.

### Simulation

The engines (frozen-time synthesis, ODE integration and DFT-domain
filtering), the conditions under which each one is exact, every model's
mathematics and its sources are in
[`src/deistools/simulation/README.md`](src/deistools/simulation/README.md).

## Next steps

- [ ] **Reimplement BLTVA** from Hallemans *et al.* (2021, 2022), on
  `legendre_basis` as described above.
- [ ] **Write the test suite.** Targets for the simulation are listed in §7 of
  [its README](src/deistools/simulation/README.md#7-what-to-test).
- [ ] **Write the examples**, one at a time.
- [ ] **Validate DMFA end to end** against an implementation that is not this
  one.
- [ ] **Back up or drop an unsupported claim in `dmfa_functions`.** The
  docstring of `extract_impedance_with_error` says the predicted variance
  agrees with the scatter of 200 noise realisations to better than 1%. The
  code that checked this went with the old test suite.
- [ ] **`data_loader`: declare or remove its dependencies.** It imports
  `pandas`, which is not in `pyproject.toml`, so the module fails to import on
  a clean install. It also opens a Tkinter file dialog and takes no path
  argument, so it cannot be used from a script or a test. Because it loads
  *experimental* data, consider moving it to
  [`elma`](https://github.com/federicoscarpioni/elma).
- [ ] **`data_preparation`: finish `cut_signal_extemes`.** It is an
  unimplemented stub, and its name is misspelled.
- [ ] **Confirm the paper attributions** listed in §6 of the
  [simulation README](src/deistools/simulation/README.md#6-attributions-to-check-against-the-papers)
  against the PDFs: equation numbers, figure numbers and coefficients.
- [ ] **Documentation site.** The MkDocs site on the `gh-pages` branch is
  still online and describes the previous version of the package. Rebuild it
  or take it down.

## References

- Battistel, A., Du, G., & La Mantia, F. (2016). On the analysis of
  non-stationary impedance spectra. *Electroanalysis* **28**, 2346.
  doi:[10.1002/elan.201600260](https://doi.org/10.1002/elan.201600260)
- Koster, D., Du, G., Battistel, A., & La Mantia, F. (2017). Dynamic impedance
  spectroscopy using dynamic multi-frequency analysis: a theoretical and
  experimental investigation. *Electrochim. Acta* **246**, 553–563.
- Battistel, A., & La Mantia, F. (2019). On the physical definition of dynamic
  impedance: how to design an optimal strategy for data extraction.
  *Electrochim. Acta* **304**, 513–520.
  doi:[10.1016/j.electacta.2019.03.033](https://doi.org/10.1016/j.electacta.2019.03.033)
- Chukwu, R., Mugisa, J., Brogioli, D., & La Mantia, F. (2022). Statistical
  analysis of the measurement noise in dynamic impedance spectra.
  *ChemElectroChem* **9**, e202200109.
  doi:[10.1002/celc.202200109](https://doi.org/10.1002/celc.202200109)
- Breugelmans, T., Lataire, J., Muselle, T., Tourwé, E., Pintelon, R., &
  Hubin, A. (2012). Odd random phase multisine electrochemical impedance
  spectroscopy to quantify a non-stationary behaviour: theory and validation
  by calculating an instantaneous impedance value. *Electrochim. Acta* **76**,
  375–382. doi:[10.1016/j.electacta.2012.05.051](https://doi.org/10.1016/j.electacta.2012.05.051)
- Hallemans, N., Pintelon, R., *et al.*, & Lataire, J. (2021). Best linear
  time-varying approximation of a general class of nonlinear time-varying
  systems. *IEEE Trans. Instrum. Meas.* **70**.
  doi:[10.1109/TIM.2021.3086891](https://doi.org/10.1109/TIM.2021.3086891)
- Hallemans, N., Pintelon, R., Zhu, X., Collet, T., Dabiri Havigh, M.,
  Wouters, B., Revilla, R. I., Claessens, R., Ramharter, K., Hubin, A., &
  Lataire, J. (2022). Trend removal in measurements of best linear
  time-varying approximations — with application to operando electrochemical
  impedance spectroscopy. *IEEE Trans. Instrum. Meas.* **71**, 1–11.

The simulation's own references are listed in
[its README](src/deistools/simulation/README.md#9-references).

## History note

Before `v0.1.0`, this repo also contained lab-hardware acquisition code. That
has moved to a separate package,
[`elma`](https://github.com/federicoscarpioni/elma) (Electrochemistry Lab
Multi-frequency Acquisition), which depends on `deistools` for impedance
estimation. The last combined snapshot remains available at this repo's
`v0.1.0` tag.
