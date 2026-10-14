# Getting started

## Install

```bash
git clone https://github.com/federicoscarpioni/DEIStools
cd DEIStools
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

## Check the record before you estimate anything

`N`, the number of periods `P`, the period `Tp` and the rate `fs` are tied
together by `N = P * Tp * fs`. When one of them is wrong the estimators do not
stop; they return an impedance that moves in time because the arithmetic is
wrong, not because the cell is. One call says whether the four agree with each
other and with the signals, and when they do not, which one the tones disagree
with:

```python
from deistools.processing.consistency import check_consistency, visualize_consistency

report = check_consistency(voltage, current, frequencies, 1 / fs,
                           period=Tp, n_periods=P)
print(report)
if not report.ok:
    visualize_consistency(report)
```

It separates the two faults that look the same in the arithmetic: a record cut
mid-period leaves the lines where they are, while a wrong sampling rate or
period moves every line in proportion to its harmonic number — and the size of
that movement gives the rate the record was really taken at. It also says
whether every listed tone is present, whether anything is present that is not
listed, and whether the channels are clipping.

## The shortest path to a result

Each estimator is called differently, because each works in a different
domain. The five "first run" examples show one estimation each, at settings
that work, on the same simulated measurement:

| | | |
|---|---|---|
| [`04_stft_first_run.py`](https://github.com/federicoscarpioni/DEIStools/blob/main/examples/04_stft_first_run.py) | STFT | one function call, four arrays back |
| [`05_dmfa_first_run.py`](https://github.com/federicoscarpioni/DEIStools/blob/main/examples/05_dmfa_first_run.py) | DMFA | four steps: transform, locate, filter, extract |
| [`06_bltva_first_run.py`](https://github.com/federicoscarpioni/DEIStools/blob/main/examples/06_bltva_first_run.py) | BLTVA | one call, returning model coefficients and an uncertainty |
| [`07_goertzel_first_run.py`](https://github.com/federicoscarpioni/DEIStools/blob/main/examples/07_goertzel_first_run.py) | Goertzel | a recurrence per bin, and a streaming filter |
| [`08_lock_in_first_run.py`](https://github.com/federicoscarpioni/DEIStools/blob/main/examples/08_lock_in_first_run.py) | lock-in | mix, low-pass, divide — the familiar instrument |

They are self-contained: each builds its own simulated measurement at the top,
so you can replace that block with your own data and the rest still runs.

## A reading path

If you are new to the package:

1. **[`00_synth_stationary.py`](https://github.com/federicoscarpioni/DEIStools/blob/main/examples/00_synth_stationary.py)** and
   **[`01_synth_transistor.py`](https://github.com/federicoscarpioni/DEIStools/blob/main/examples/01_synth_transistor.py)** —
   how a measurement is simulated, and what a time-varying one looks like.
2. **04–08** — one estimation with each method.
3. **[`09_detrending.py`](https://github.com/federicoscarpioni/DEIStools/blob/main/examples/09_detrending.py)**
   — if your cells drift, which they do.
4. **[Concepts](concepts/signal-model.md)** — what the methods are actually
   doing, and the notation they are written in.
5. **10–15** — what each parameter does, measured against known truth.
6. **20–23** — the three main methods side by side on one system each.

That is roughly an hour of reading and five minutes of running.

## Running the examples

```bash
python examples/04_stft_first_run.py            # figures in windows
python examples/04_stft_first_run.py --save     # figures to examples/figures/
python examples/05_dmfa_first_run.py --no-widgets   # skip interactive sliders
```

Every seed is fixed, so the numbers you get are the numbers in the commentary.
The full set of twenty scripts takes about five minutes and produces 57
figures.

## Which method should I use?

Short answer: try them — that is what the package is for.

First, a simplification. There are five estimators but **three ideas**:
[Goertzel](methods/goertzel.md) and the [lock-in](methods/lock-in.md) both
compute the same thing as [STFT](methods/stft.md), verified to 1e-10 and
9e-14 respectively. Choose between those three on what they *offer* — Goertzel
runs on a stream, the lock-in gives a continuous output and a drift-corrected
demodulator — not on accuracy, where they coincide. The real choice is between
STFT, [DMFA](methods/dmfa.md) and [BLTVA](methods/bltva.md).

Longer answer, from the comparisons in examples 20–23:

- **Your system is roughly stationary.** All three agree to within a factor of
  a few. Use whichever is convenient. BLTVA will tell you the system is
  stationary, by pruning its time-varying coefficient to exactly zero.
- **Your system varies smoothly** — charging, slow ageing, a temperature ramp.
  BLTVA is the most accurate and the only one that reports an uncertainty.
  DMFA decouples the time resolution from the multisine period and is a good
  cross-check.
- **Your system changes abruptly**, or in a way no low-order polynomial
  describes. BLTVA's assumption fails and no setting rescues it. Use STFT,
  which assumes nothing about the form of the variation.
- **You do not know.** Run BLTVA and compare its reported uncertainty with the
  error you actually see. A large gap means the model, not the measurement, is
  your limit — see [BLTVA](methods/bltva.md#the-uncertainty-is-a-model-error-detector).

## My cell drifts

Almost all of them do, and the handling is not symmetric between the methods,
so it is worth reading [Detrending](methods/detrending.md) before reaching for
a filter. The short version:

- **With BLTVA**, raise `Nq` and do not detrend at all. It models the trend
  jointly with the impedance, which neither distorts the signal nor needs the
  trend to be guessed in advance.
- **With DMFA or STFT**, use `PolynomialFit` at the lowest degree that works,
  and know that it removes some signal along with the trend.
- **Never pass `Differencing` to DMFA or STFT.** It is built for BLTVA, which
  divides its attenuation back out; without that step it annihilates the
  lowest tones outright.

[`09_detrending.py`](https://github.com/federicoscarpioni/DEIStools/blob/main/examples/09_detrending.py)
shows all four strategies on one drifting record — the signals, the spectra,
and what each costs.
