# DEIStools examples

Twenty self-contained scripts. Each one builds its own simulation, calls the
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
takes about five minutes and produces 57 figures.

The only shared file is [`_plotting.py`](_plotting.py), which holds matplotlib
boilerplate — axis labels, colours, the two standard figure layouts — so that
sixty lines of formatting do not bury the four lines that matter.

## Layout

| | | |
|---|---|---|
| **00–03** | one system per script | how each simulation is built and what kind of ground truth it gives |
| **04–08** | one estimator per script | a single estimation at good settings: how to call it, and what comes back |
| **09** | trend removal | the four strategies: what each does to the signal, its spectrum and the estimate |
| **10–15** | one estimator per pair | its parameters swept against known truth, then applied to all four systems |
| **20–23** | one system per script | all three estimators side by side |

If you are new to the package, read 00–01 and then 04–08. That is enough to
run every estimator on your own data. The rest is about choosing their
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

### 04–08 · One estimation with each method

All five analyse the same transistor-gated measurement, at settings the
parameter scripts arrive at later, so the outputs can be compared directly.

| | script | what it shows |
|---|---|---|
| [04](04_stft_first_run.py) | STFT | the four arrays `estimate_impedance` returns |
| [05](05_dmfa_first_run.py) | DMFA | the four steps that make up the method, and `visualize_dmfa_filtering` |
| [06](06_bltva_first_run.py) | BLTVA | the `BLTVAResult` fields, reconstructing Z(f,t) from the coefficients, the uncertainty, and `visualize_bltva_fit` |
| [07](07_goertzel_first_run.py) | Goertzel | the recurrence against `np.fft`, agreement with STFT, and the streaming `GoertzelFilter` |
| [08](08_lock_in_first_run.py) | lock-in | designing the integrator, tone rejection, the record edges, agreement with STFT, and mixing against local least squares |

Two of those five are the same estimator as a third. Goertzel reproduces STFT
to 1e-10 and a one-period boxcar lock-in to 9e-14, which is why 07 and 08
spend as much space checking against STFT as estimating: agreement that close,
by completely different arithmetic, is a real check on all three.

### 09 · Trend removal

| | script | what it shows |
|---|---|---|
| [09](09_detrending.py) | detrending | the drift in the record and in the spectrum, the four strategies side by side, what each removes from the excited bins of a *stationary* cell, and which one BLTVA accepts |

Worth reading before 10–15 if your own cells drift, because the right answer
differs per estimator and one of the four wrong answers destroys the lowest
tones outright.

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
- **The lock-in is the STFT evaluated everywhere**, agreeing to 9e-14 at the
  window centres. Its median error over *all* instants is higher (1.2e-03
  against 7.1e-04) precisely because it also reports between them, where more
  of the variation has happened inside the window. Reporting densely is not
  resolving finely — the same trap as DMFA's decimation above.
- **A local-least-squares demodulator is 11× better than mixing** on the same
  data and the same window (1.05e-04 against 1.16e-03), because it carries a
  local slope and so follows drift *within* the window instead of averaging
  over it.
- **A boxcar integrator over whole base periods rejects the other tones by
  307 dB** — round-off, not a real limit — because its nulls land exactly on
  the harmonics. One extra sample detunes them and costs about 230 dB.
- **A trend overlaps the lowest tones, and no filter can separate them
  there.** A filter scales a bin's trend and signal by the same number, so
  their ratio is untouched. Only the polynomial fit — the one strategy that
  is not a filter — improves a drifting DMFA estimate, 8.7e-02 to 7.9e-03.
- **Subtracting a polynomial is not the gentle option.** On a stationary
  cell, where every change is signal destroyed, a cubic fit takes 2.1e-02 of
  the lowest tone against 1.9e-03 for a Butterworth and 5.0e-05 for a boxcar.
  The damage goes as 1/h, so it lands hardest exactly where the trend was.

## Conventions

- black is ground truth; time runs blue to red
- Nyquist plots show −Im(Z) against Re(Z) on equal axes
- `--save` writes PNGs to `examples/figures/`, which is gitignored — they
  regenerate deterministically

## Not covered

Uncertainty estimation for STFT and DMFA. The 20–23 comparisons show the
three main estimators rather than all five, since Goertzel and the lock-in
reproduce STFT anyway. Record trimming is left to you: the examples generate
whole numbers of periods, and `bltva` refuses a record that is not one.

Why the parameters are what they are, and where each method comes from in the
literature, belongs in the documentation rather than in these scripts.
