# DEIStools examples

Eighteen self-contained scripts. Each one builds its own simulation, calls the
estimators directly, and prints what it found. There is no shared science
module to look through: everything a script does is in that script, in the
order it happens, so it can be read top to bottom and copied from.

```bash
pip install -e .                                    # from the repository root
python examples/00_synth_stationary.py              # figures in windows
python examples/00_synth_stationary.py --save       # figures to examples/figures/
python examples/12_dmfa_parameters.py --no-widgets  # skip the interactive sliders
```

Nothing needs an instrument, a data file or a network connection. Every seed is
fixed, so the numbers you see are the numbers in this README. The whole set
takes about four minutes and produces 52 figures.

The only shared file is [`_plotting.py`](_plotting.py), which holds matplotlib
boilerplate — axis labels, colours, the two standard figure layouts — so that
sixty lines of formatting do not bury the four lines that matter.

## Layout

| | | |
|---|---|---|
| **00–03** | one system per script | how each simulation is built and what kind of ground truth it gives |
| **04–07** | one estimator per script | a single estimation at good settings: how to call it, and what comes back |
| **10–15** | one estimator per pair | its parameters swept against known truth, then applied to all four systems |
| **20–23** | one system per script | all three estimators side by side |

If you are new to the package, read 00–01 and then 04–07. That is enough to
run all three estimators on your own data. The rest is about choosing their
parameters well.

### 00–03 · The four systems

| | system | ground truth |
|---|---|---|
| [00](00_synth_stationary.py) | stationary R+(R‖C) | analytic, **exact** |
| [01](01_synth_transistor.py) | transistor-gated R_ct(t) | frozen-time, accurate to 0.014 % |
| [02](02_synth_legendre.py) | prescribed Legendre expansion | **exact**, and time-varying |
| [03](03_synth_block_oriented.py) | block-oriented nonlinear | best linear approximation |

All four use the same excitation: 13 odd harmonics on a quasi-log grid,
0.05–15.05 Hz, 10 mA per tone, sampled at 1 kHz. Each script defines it again
at the top, so you can change one without touching the others.

### 04–07 · One estimation with each method

All three analyse the same transistor-gated measurement, at settings the
parameter scripts arrive at later, so the outputs can be compared directly.

| | script | what it shows |
|---|---|---|
| [04](04_stft_first_run.py) | STFT | the four arrays `estimate_impedance` returns |
| [05](05_dmfa_first_run.py) | DMFA | the four steps that make up the method, and `visualize_dmfa_filtering` |
| [06](06_bltva_first_run.py) | BLTVA | the `BLTVAResult` fields, reconstructing Z(f,t) from the coefficients, the uncertainty, and `visualize_bltva_fit` |
| [07](07_goertzel_first_run.py) | Goertzel | the recurrence against `np.fft`, agreement with STFT, and the streaming `GoertzelFilter` |

### 10–15 · The three estimators

| | script | parameters |
|---|---|---|
| [10](10_stft_window_length.py) | STFT, window length | `period` |
| [11](11_stft_all_systems.py) | STFT on all four systems | |
| [12](12_dmfa_parameters.py) | DMFA, bandwidth and sharpness | `bw`, `n` |
| [13](13_dmfa_all_systems.py) | DMFA on all four systems | |
| [14](14_bltva_parameters.py) | BLTVA, Np / Nq / band width | `Np`, `Nq`, `band_halfwidth` |
| [15](15_bltva_all_systems.py) | BLTVA on all four systems | |

Each estimator is called in full at the top of its parameter script — the FFT,
the filter and the band for DMFA; the fit and the reconstruction of Z(f,t) from
the coefficients for BLTVA — before any sweeping starts.

12 and 14 also open the package's interactive diagnostics,
`visualize_dmfa_filtering` and `visualize_bltva_fit`, on their best and worst
settings. Run them without `--save`.

### 20–23 · Side by side

| | system |
|---|---|
| [20](20_compare_stationary.py) | stationary — does any method invent variation? |
| [21](21_compare_transistor.py) | transistor — the realistic case |
| [22](22_compare_legendre.py) | Legendre — what a correct assumption is worth |
| [23](23_compare_block_oriented.py) | block-oriented — what survives nonlinearity |

## A few results

- On the stationary cell BLTVA's time-varying coefficient comes back as
  **exactly zero** at all 13 tones — removed from the fit, not merely small —
  while on the transistor the same coefficient is 0.37 of `H_0`.
- `Np` and the band width **interact**: a wider band helps at low `Np` and hurts
  at high `Np`, because `Na` is spent from the same budget and it is `Na` that
  has to span the band. Tuning them separately gives a setting 29× worse.
- **DMFA's record edges are unusable** — several hundred times worse than the
  middle. The stationary system shows none of it, which is the proof that the
  rings come from representing a *changing* quantity with a finite band.
- **BLTVA's uncertainty is a model-error detector**: its ratio to the actual
  error is ≈1 where the model is right, and 190 on the transistor, whose
  impedance is a rational function of time rather than a polynomial.
- The **bottom decade is always the least reliable part** of a time-varying
  spectrum, whichever method produced it.
- **DMFA output must be decimated to 0.5/bandwidth before fitting.** Reported
  more densely than that, neighbouring errors are correlated — 84% at the
  spacing these examples used to report.

## Conventions

- black is ground truth; time runs blue to red
- Nyquist plots show −Im(Z) against Re(Z) on equal axes
- `--save` writes PNGs to `examples/figures/`, which is gitignored — they
  regenerate deterministically

## Not covered

Uncertainty estimation for STFT and DMFA; lock-in and Goertzel estimators,
which are not implemented yet. Record trimming is left to you: the examples
generate whole numbers of periods, and `bltva` refuses a record that is not one.

Why the parameters are what they are, and where each method comes from in the
literature, belongs in the documentation rather than in these scripts.
