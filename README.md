# DEIStools

Tools for the estimation of instantaneous impedance from multi-frequency voltage and current signals. This experimental techniques is commonly called dynamic electrochemical impedance spectroscopy (DEIS).
This package is agnostic to the signal acquisition stage and can be used for any single input - single output (SISO) system identification. 
The high-level API uses the object MultiFrequencyAnalysis that can call any of the three most used estimators: short-time Fourier transform EIS (STFT-EIS), dynamic multi-frequency analysis (DMFA), and best linear time-varying approximation (BLTVA). See examples below.


**Note of version 0.2.0:** the package has been rewritten in almost all parts so it brakes old scripts. The core methods are tested with pytest, put it is still in development.


## Install

```bash
git clone https://github.com/federicoscarpioni/DEIStools
cd DEIStools
pip install -e .
```

Requires `numpy`, `matplotlib` and `pandas`.

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

1. Take a band of `n_bins = N / (fs * period * periods_per_sample)` bins
   centred on the tone, where `N` is the record length.
2. Multiply it by a passband filter centred on the tone (a quadrature filter).
3. Inverse-transform the band. This gives the complex envelope of the voltage
   and of the current at that tone, sampled once every `periods_per_sample`
   periods.
4. Take their ratio, `Z(f_k, t) = V_k(t) / I_k(t)`.

Each output sample is a weighted average of the record, centred on the
*start* of its block of periods, `t_m = m * periods_per_sample * period`.

The filter bandwidth sets the time resolution: a wider band follows faster
changes but admits more noise. The bandwidth is also limited by the spacing of
the excited tones, because neighbouring bands must not overlap. The DFT is
circular, so a record that does not return to its starting state (a drift, a
ramp) disturbs the first and last few output samples.

The filter is passed as an object from `deistools.processing`:
`FermiDiracFilter`, `RaisedCosineFilter`, `HannFilter`, `BlackmanFilter`,
`BlackmanHarrisFilter`, `NuttallFilter`, `KaiserFilter` or `GaussianFilter`.
DMFA evaluates it on the band's frequency offsets from the tone, in Hz. Every
filter is defined by its `halfwidth`, the offset at which it equals 1/2, and
`rejection_db(offset)` gives its attenuation at any offset, such as the
distance to the next tone. The Fermi–Dirac shape is

```
Y = cosh(n) / (cosh(n X) + cosh(n)),      X = Δf / halfwidth
```

which is flat in the pass band and falls off steeply at the edges, with the
sharpness set by the `order` n.

*Sources:*
- Battistel, Du and La Mantia (2016): DMFA introduced, for the analysis of
  non-stationary impedance spectra.
- Koster, Du, Battistel and La Mantia (2017): dynamic impedance redefined
  with quadrature filters, and the design of multisines for it.
- Battistel and La Mantia (2019): the physical definition of dynamic
  impedance through Volterra series, and the strategy for extracting it,
  including the filter.
- Chukwu, Mugisa, Brogioli and La Mantia (2022): the statistics of the
  measurement noise on dynamic impedance spectra, the basis for a noise and
  variance estimate that is not implemented yet.

### BLTVA — best linear time-varying approximation

BLTVA writes the time-varying transfer function as a sum of fixed spectra
times Legendre polynomials in time:

```
Z(f, t) = sum_{p=0..Np}  G_p(f) * b_p(t)
```

The `b_p` are Legendre polynomials rescaled to the record, with the sample
mean removed from every order p ≥ 1 (`legendre_basis` in
`deistools.processing.bltva_functions`). Because the time variation spreads
every tone into its neighbouring bins, those bins carry the information about
how `Z` changes.

For each excited tone, `bltva` takes the bins within `halfband` of it and
fits them by least squares with:

- the `Np + 1` spectra of that tone, whose coefficients are the `G_p`;
- the spectra of any other excited tone inside the band;
- `Nq` Legendre orders for the slow drift of the response;
- a polynomial of order `Na` across the band for the transient. By default
  `Na` follows Hallemans *et al.* (2022), eq. (22), from `P` and `Np`.

It returns `G_p` with shape `(n_frequencies, Np + 1)` and the basis;
`evaluate_transfer_function(G_p, basis)` turns them into `Z(f, t)` at every
sample. The residuals `res_abs` and `res_rel` and the `rank` of each fit are
returned per tone as diagnostics. The band must hold more bins than there are
unknowns, otherwise `bltva` raises a `ValueError`.

*Sources:* Hallemans *et al.* (2021) for the model and the estimator;
Hallemans *et al.* (2022) for trend removal.


### Examples

`current` and `voltage` are 1-D arrays sampled at `fs`. The record must hold a
whole number of multisine periods, and every excited frequency must fall on a
DFT bin.

```python
import numpy as np
from deistools.processing import MultiFrequencyAnalysis, FermiDiracFilter
from deistools.processing.bltva_functions import bltva, evaluate_transfer_function

fs = 20                                         # sampling frequency, Hz
period = 1                                      # multisine period, s
frequencies = np.array([1, 3, 5, 7, 9]) / period  # excited frequencies, Hz

mfa = MultiFrequencyAnalysis(
    excitation=current,
    response=voltage,
    sampling_frequency=fs,
    frequencies=frequencies,
    period=period,
)

# DFT of both signals, and interactive plots of the spectrum
mfa.compute_fft()
mfa.compute_freq_axis()
mfa.inspect_spectrum()
mfa.inspect_spectrum_phase()

# Check that the excited bins sit on the spectral peaks
mfa.visualise_peaks()

# Every estimate of Z has shape (n_frequencies, n_times)

# STFT-EIS: one estimate per window of whole periods, time at window centres
Z_stft, current_mean, voltage_mean, time_stft = mfa.run_stft_eis(periods_per_window=1)

# DMFA: one estimate per periods_per_sample periods
Z_dmfa, time_dmfa = mfa.run_dmfa(
    FermiDiracFilter(halfwidth=0.25 / period, order=4),
    periods_per_sample=1,
)

# BLTVA: one estimate per sample (not yet wrapped by MultiFrequencyAnalysis)
n_periods = mfa.num_samples // int(fs * period)
G_p, basis, res_abs, res_rel, rank = bltva(
    mfa.dft_excitation,
    mfa.dft_response,
    mfa.k_exc,
    P=n_periods,
    halfband=n_periods // 2,
    Np=2,
    Nq=1,
)
Z_bltva = evaluate_transfer_function(G_p, basis)
```


### Simulation

DEIStools has an early-development simulation subpacage to get synthetized voltage and current signals from time-varying equivalent circuit models.
The engines (frozen-time synthesis, ODE integration and DFT-domain
filtering), the conditions under which each one is exact, every model's
mathematics and its sources are in
[`src/deistools/simulation/README.md`](src/deistools/simulation/README.md).


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


## History note

Before `v0.1.0`, this repo also contained lab-hardware acquisition code. That
has moved to a separate package,
[`elma`](https://github.com/federicoscarpioni/elma) (Electrochemistry Lab
Multi-frequency Acquisition), which depends on `deistools` for impedance
estimation. The last combined snapshot remains available at this repo's
`v0.1.0` tag.
