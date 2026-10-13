# Getting started

## Install

```bash
git clone https://github.com/federicoscarpioni/deistools
cd deistools
pip install -e .
```

The package needs only `numpy` and `matplotlib`.

## What you need from a measurement

Three things, and one constraint.

| | |
|---|---|
| `voltage`, `current` | the two channels, as 1-D arrays of equal length |
| `fs` | the sampling rate in Hz |
| the excitation | which frequencies were applied, and with what amplitudes |

The constraint: **the record must contain a whole number of multisine
periods**. A rectangular window over a fractional period leaks, and the
leakage is indistinguishable from time variation — the very thing you are
trying to measure. `bltva` refuses a record that is not an integer number of
periods rather than returning something plausible and wrong; STFT and DMFA
will not complain, so trim the record yourself before using them.

## The shortest path to a result

Each estimator is called differently, because each works in a different
domain. The three "first run" examples show one estimation each, at settings
that work, on the same simulated measurement:

| | | |
|---|---|---|
| [`04_stft_first_run.py`](https://github.com/federicoscarpioni/deistools/blob/main/examples/04_stft_first_run.py) | STFT | one function call, four arrays back |
| [`05_dmfa_first_run.py`](https://github.com/federicoscarpioni/deistools/blob/main/examples/05_dmfa_first_run.py) | DMFA | four steps: transform, locate, filter, extract |
| [`06_bltva_first_run.py`](https://github.com/federicoscarpioni/deistools/blob/main/examples/06_bltva_first_run.py) | BLTVA | one call, returning model coefficients and an uncertainty |

They are self-contained: each builds its own simulated measurement at the top,
so you can replace that block with your own data and the rest still runs.

## A reading path

If you are new to the package:

1. **[`00_synth_stationary.py`](https://github.com/federicoscarpioni/deistools/blob/main/examples/00_synth_stationary.py)** and
   **[`01_synth_transistor.py`](https://github.com/federicoscarpioni/deistools/blob/main/examples/01_synth_transistor.py)** —
   how a measurement is simulated, and what a time-varying one looks like.
2. **04–06** — one estimation with each method.
3. **[Concepts](concepts/signal-model.md)** — what the methods are actually
   doing, and the notation they are written in.
4. **10–15** — what each parameter does, measured against known truth.
5. **20–23** — the three methods side by side on one system each.

That is roughly an hour of reading and four minutes of running.

## Running the examples

```bash
python examples/04_stft_first_run.py            # figures in windows
python examples/04_stft_first_run.py --save     # figures to examples/figures/
python examples/05_dmfa_first_run.py --no-widgets   # skip interactive sliders
```

Every seed is fixed, so the numbers you get are the numbers in the commentary.
The full set of seventeen scripts takes about four minutes and produces 50
figures.

## Which method should I use?

Short answer: try all three — that is what the package is for.

Longer answer, from the comparisons in examples 20–23:

- **Your system is roughly stationary.** All three agree to within a factor of
  a few. Use whichever is convenient. BLTVA will tell you the system is
  stationary, by pruning its time-varying coefficient to exactly zero.
- **Your system varies smoothly** — charging, slow ageing, a temperature ramp.
  BLTVA is the most accurate and the only one that reports an uncertainty.
  DMFA gives finer time resolution than STFT and is a good cross-check.
- **Your system changes abruptly**, or in a way no low-order polynomial
  describes. BLTVA's assumption fails and no setting rescues it. Use STFT,
  which assumes nothing about the form of the variation.
- **You do not know.** Run BLTVA and compare its reported uncertainty with the
  error you actually see. A large gap means the model, not the measurement, is
  your limit — see [BLTVA](methods/bltva.md#the-uncertainty-is-a-model-error-detector).
