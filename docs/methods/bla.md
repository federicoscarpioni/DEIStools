# Best linear approximation per burst

[STFT](stft.md) reads the excited bins of each window and divides. That gives
an impedance and nothing else — a single window offers no way to say whether
the number it produced is limited by measurement noise, by nonlinearity, or by
the cell changing while the window was open.

This page is about recovering that. It keeps STFT's assumption — the system is
whatever it is inside one window, with no model of how it varies — and uses the
bins STFT throws away.

## The one structural fact

Cut the record into windows of $P$ whole multisine periods. The DFT of a window
has fundamental $1/(PT_p)$, so a tone at harmonic $h$ lands on bin $Ph$ and
**only every $P$-th bin is a harmonic of the multisine at all**. A nonlinear
system of the PISPO class (period in, same period out) answers a periodic input
with a periodic output, so its distortion is harmonically related to the input
and can land *only* on those bins.

That splits one burst into four classes:

| bins | what is there |
|---|---|
| $k = Ph$, $h$ excited | signal, plus everything below |
| $k = Pm$, $m$ odd, unexcited | odd-order distortion + noise + drift |
| $k = Pm$, $m$ even, unexcited | even-order distortion + noise + drift |
| $k$ not a multiple of $P$ | noise + drift |

!!! warning "This needs $P \ge 2$, and that is the whole price"
    At $P = 1$ the window fundamental *is* the multisine fundamental, every bin
    is a harmonic, and the fourth class is empty. On the package's 13-tone
    grid, in band:

    | $P$ | noise | odd | even | excited |
    |---|---|---|---|---|
    | 1 | **0** | 168 | 180 | 13 |
    | 2 | 301 | 138 | 150 | 13 |
    | 4 | 903 | 138 | 150 | 13 |
    | 8 | 2107 | 138 | 150 | 13 |

    One period is exactly the window [plain STFT prefers](stft.md#the-one-parameter)
    on a strongly excited cell. So the uncertainty is bought with time
    resolution, and there is no way around it. `classify_bins` raises rather
    than returning an empty class.

## Calling it

```python
from deistools.processing import MultiFrequencyAnalysis

mfa = MultiFrequencyAnalysis(frequencies, voltage, current, sampling_time)
results, t = mfa.run_bla(period=Tp, periods_per_window=4)

for burst in results:
    burst.impedance          # what STFT would have returned
    burst.std_noise          # level at the non-harmonic bins
    burst.std_odd_lines      # level at the unexcited odd harmonics
    burst.std_even_lines     # level at the unexcited even harmonics
    burst.r_ntv              # noise, or the cell moving?
    burst.distortion_odd()   # the subtraction, where it is significant
```

Or directly, with `burst_bla` for a single burst and `bla_eis` to slide.

## The noise level is real

`std_noise` is the standard deviation of the impedance, not something
proportional to it. Checked against the scatter of $Z$ over 60 independent
noise realisations of a stationary cell:

| injected noise | predicted / observed |
|---|---|
| $10^{-4}$ V | 0.997 |
| $10^{-3}$ V | 0.997 |
| $10^{-2}$ V | 0.997 |

Two decades of noise, the same ratio — so it is calibrated, not merely
correlated.

## Noise, or the cell moving?

The two are told apart by their **correlation over frequency**, not their size.
Stationary noise is uncorrelated between neighbouring DFT bins. A time
variation is not: it reaches each bin by convolving the excited lines with the
spectrum of the variation, so neighbours share contributions.

For a linear-in-time variation the ratio of the autocorrelation at lags 0 and 1
works out to exactly $\pi^2/6$ (Lataire 2012, eq. 31), so

$$
R_{\mathrm{NTV}} = \frac{6}{\pi^2}\,
\frac{\overline{|V(k)|^2}}{\left|\overline{V(k)\,V^*(k{+}1)}\right|}
$$

puts "pure time variation" at 1. Measured on 80 s bursts:

| record | $R_{\mathrm{NTV}}$ | verdict |
|---|---|---|
| stationary R+RC, noise $10^{-4}$ V | 9.2 – 19.4 | noise |
| block-oriented (nonlinear *and* varying) | 0.97 | time variation |
| transistor (linear, varying) | 0.73 | time variation |

Five consecutive bursts of the transistor record all return 0.73; five of the
stationary record return 9.2 to 19.4. The separation is clean and it needs no
model.

!!! note "It needs a residual to work on"
    On a noiseless simulation the non-harmonic bins hold nothing but round-off
    and the statistic is meaningless — measured at 0.86 for a stationary
    noiseless cell, which reads as "time variation" for a system that has
    none. Real records always have noise; simulated ones should be given some.

## What `std_noise` really contains

On a varying cell the non-harmonic bins are **not** noise alone. Each tone
grows a skirt — for a linear-in-time variation $B_1(k) \sim 1/k$ — and with a
dozen tones the skirts cover the band. Measured on a 40 s burst cut from the
usual 400 s transistor record, against the pure-noise floor at the same bins:

| distance to the nearest tone | ratio to the noise floor |
|---|---|
| 1–2 bins | 7656 |
| 6–10 bins | 463 |
| 21–50 bins | 71 |
| 51–100 bins | 44 |

It decays, and **it never arrives**. So `std_noise` is an upper bound, and on a
strongly varying cell a loose one. That is not an implementation limit:
separating the two needs a model of the variation, which is what
[BLTVA](bltva.md) is for and what this page deliberately does not assume.
`r_ntv` tells you which case you are in.

## Nonlinear distortion

The levels are reported **as measured**, with nothing subtracted:
`std_odd_lines` and `std_even_lines` each contain the noise as well as any
distortion. The subtraction is available from `distortion_odd()` and
`distortion_even()`, and returns NaN unless the excess is significant.

That threshold is not decoration. Each level is a mean of `n_local` bin powers,
so it carries about $1/\sqrt{n_\text{local}}$ relative error and their
difference about $\sqrt{2/n_\text{local}}$. Merely requiring a positive
difference fires on **10 of 13 tones of a linear stationary cell**, where the
two levels agree to 2.5% and the sign is a coin toss. Requiring $3\sigma$
leaves none.

Measured on a stationary nonlinear cell — one branch, so no time variation —
against a linear one:

| cell | odd / noise | even / noise | even detected |
|---|---|---|---|
| linear | 1.02 | 0.92 | **0 / 13** |
| nonlinear, $a = 0.02$ | 1.02 | 2.65 | 9 / 13 |
| nonlinear, $a = 0.1$ | 1.02 | 12.1 | 9 / 13 |
| nonlinear, $a = 0.3$ | 1.06 | 30.3 | 9 / 13 |

No false alarms on the linear cell, and the level tracks the nonlinearity
monotonically over a factor of 15.

Odd-order distortion is the one that matters for an odd multisine: it lands
*on* the excited lines as well as the empty odd ones, so it biases the
impedance, where even-order distortion only sits beside it.

!!! danger "On a varying cell the distortion cannot be resolved this way"
    The non-harmonic bins sit *above* the harmonic ones, so the subtraction is
    negative everywhere and every tone returns NaN — measured on the
    block-oriented system, which is nonlinear *and* varying, although the
    distortion is certainly there. Check `r_ntv` first. If it is near 1, use
    [BLTVA](bltva.md), which models the variation and reports
    `var_distortion_odd` with it removed.

## Where this sits

| | STFT | **BLA per burst** | BLTVA |
|---|---|---|---|
| model of the variation | none | none | Legendre, global |
| time resolution | 1 period | $\ge$ 2 periods | $T/N_p$ |
| noise level | — | yes, on a steady cell | yes |
| distortion level | — | yes, on a steady cell | yes, always |
| works while the cell moves | yes | the estimate does; the error bars do not | yes |

The honest summary: this buys STFT an uncertainty on a cell that is *nearly*
steady within a burst, at the cost of halving its time resolution at best. When
the cell moves enough that `r_ntv` falls to 1, the impedance is still STFT's,
but the error attribution needs BLTVA.

## Sources

- Lataire, Louarroudi and Pintelon, *IEEE TIM* **61** (2012) 2132 — the best
  linear time-invariant approximation of a time-varying system, and the
  frequency-correlation test used here.
- Pintelon, Louarroudi and Lataire, *IEEE TIM* **61** (2012) 3387 — the
  multiple-input time-invariant equivalent, and separating leakage from time
  variation.
- Pintelon, Louarroudi and Lataire, *IEEE TIM* **62** (2013) 3361 — the
  periodic-excitation procedure this module follows: which bins carry what, and
  why two periods are the minimum.
- Pintelon and Schoukens, *System Identification: A Frequency Domain Approach*,
  2nd ed., Wiley-IEEE Press (2012) — the underlying framework.

## Not done yet

The three-step estimator of Pintelon 2013 §V-B goes further than this page: it
*models* the time variation (fitting the $H_p$ branches of the
multiple-input equivalent at the non-harmonic bins), subtracts it, and then
separates noise from distortion at the excited bins. That would lift the
restriction above — distortion on a varying cell — and is roadmap item 2b.

One thing blocks a direct transcription: step 3 reconstructs $G_\text{BLTI}$
from $H_0$ by central differences over *neighbouring excited harmonics*, which
Assumption 3 of that paper requires to be equispaced. This package's grid is
quasi-logarithmic (1, 3, 5, 9, 13, …, 301), so those differences are not
available and the $O(T^{-1})$ correction needs another route.
