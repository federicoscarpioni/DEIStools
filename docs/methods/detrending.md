# Detrending

A measured cell drifts. Its open-circuit voltage wanders, its temperature
moves, a reference electrode ages — and the resulting slow excursion can be
tens of volts peak-to-peak on a signal of one volt. That drift is not what the
measurement is for, so the instinct is to filter it out first and estimate
afterwards.

The instinct is wrong, and this page is mostly about why. Four strategies are
implemented. **Each one is useless or harmful with three of the four
estimators**, and the pattern is not arbitrary: it falls out of one fact about
what a filter can do.

## The one fact

A filter applied by convolution is **diagonal in frequency**. It multiplies
bin $k$ by $F(k)$ and does nothing else. So if a trend and the signal both
have content at bin $k$, they are scaled by the same number:

$$
\frac{|T(k)|}{|S(k)|} \quad\text{is unchanged, for every } k
\text{ and every diagonal filter}
$$

Measured here as *bit-identical* before and after, for both a boxcar and a
Butterworth. A filter cannot separate a trend from the signal where the two
overlap, because at a single bin there is nothing to separate them with. It
can only reject a disturbance that lives somewhere **else** in frequency.

Whether a trend overlaps is not a matter of taste. A trend that takes the
whole record to unfold has its energy in the lowest few bins — exactly where
the slowest excited tones sit. For a typical multisine, the lowest three to
five tones are inside the drift's skirt. For those tones, no filter helps.

Two things *can* work:

- **Annihilate the trend in the time domain.** $\Delta^d$ applied to a
  polynomial of degree $d-1$ gives exactly zero, pointwise. No multiplication
  in frequency can do that, and differencing achieves it by not being
  diagonal — the zero-padded history is what buys the extra power.
- **Model the trend and fit it jointly.** Put trend regressors in the design
  matrix alongside the signal model and let least squares apportion the shared
  bins. This is what BLTVA's $N_q$ block does.

Everything below is a consequence.

## Which strategy, which estimator

| strategy | BLTVA | DMFA / STFT |
|---|---|---|
| [`Differencing`](#differencing) | **the only one that works** | **destroys the signal** (error 1.00) |
| [`MovingAverage`](#movingaverage) | refused | at best 1.7×, and not consistently |
| [`LowPassRemoval`](#lowpassremoval) | refused | no help on an overlapping trend |
| [`PolynomialFit`](#polynomialfit) | refused — use $N_q$ | **the only one that reliably helps** |

The two columns are exact complements, and one question decides both: *can the
estimator undo an attenuation?*

BLTVA can. Its cost function is weighted by $W = (FF^H)^{-1}$, so whatever the
filter took out is divided back in. That is what lets differencing trade away
the entire low-frequency band and still get the answer. DMFA and STFT have no
such step, so for them whatever a filter removes is simply gone — and
differencing, which removes almost everything near DC, is catastrophic rather
than free.

!!! danger "Never pass `Differencing` to DMFA or STFT"
    `Differencing(1)` attenuates the lowest excited bin of a 160 000-sample
    record to $|F| = 3\times10^{-4}$. The measured DMFA error at that tone is
    **1.00** — the tone is gone. In BLTVA the same attenuation costs nothing.
    This is the most dangerous thing in the module, and the reason
    `frequency_response` is a mandatory part of the protocol: it is the only
    way to see the $3\times10^{-4}$ coming.

## Detrending for BLTVA is not a preprocessing step

Removing a trend before a BLTVA fit changes **four** things at once (2022
paper, eq. 18–27):

1. every skirt regressor is multiplied by $F(k)$ — eq. (21)
2. the cost function is weighted by $W = (FF^H)^{-1}$ — eq. (26)
3. the filter's boundary effects add their own regressors — eq. (21)
4. the transient order $N_a$ shrinks by the filter order — eq. (22)

So a strategy is not "a function returning a cleaned signal". It is the bundle
of those four things, which is what the `Detrending` protocol encodes. A
strategy that cannot report its `frequency_response` cannot be used, because
step 2 would be impossible.

### Why the weighting rules out three of the four

Step 2 is also what disqualifies them. For a real, zero-phase response
$F/|F| = 1$, so the weighted system is **identical** to the one built from the
raw record — verified to $5.6\times10^{-17}$ on the skirt regressors. Nothing
about the trend was removed. What *did* change is that the trend block of
eq. (58) is gone, replaced by nothing.

Measured on the 2022 paper's own trend, with $F(k) = 1$ exactly — a filter
that does nothing at all, which the protocol happily accepts:

| configuration | error at the lowest tone |
|---|---|
| no trend, $N_q = 5$ | 2.2e-04 |
| trend, $N_q = 5$, no detrending | 1.6e-02 |
| trend, $F(k) = 1$ "detrending" | **2.6e+00** |

A moving average scores well in the same test (1.8e-04), which looks like a
refutation until the averaging length is varied:

| length | error at the lowest tone |
|---|---|
| 101 | 1.83e-04 |
| 401 | 1.82e-04 |
| 1601 | 1.68e-04 |

An answer that does not depend on the averaging length is not coming from the
averaging. It comes from the $1/|F(k)|$ weighting applied to the *transient*
block, which eq. (21) does not multiply by $F$: since $F \sim k^2$ near DC for
any boxcar, the weighting hands the transient polynomial a $k^{-2}$ envelope,
and that reshaped basis happens to absorb the trend skirt. The trend removal is
an accident of the reparametrisation — and a Butterworth low-pass, same
protocol but a different $1/|F|$ shape, scores 2.6e-02, *worse* than no
detrending at all.

So BLTVA refuses them, with an error naming the alternative.

## The strategies

### `Differencing`

$$
y_d(n) = \Delta^d \tilde y(n), \qquad \tilde y(n) = 0 \text{ for } n < 0
$$

High-pass, eq. (12b): $F_d(k) = (1 - z_k^{-1})^d$ with
$z_k = e^{j2\pi k/N}$.

The published strategy, and the only one BLTVA accepts. It suppresses a smooth
trend *without modelling it*, which is the point of the 2022 paper: a trend
that no finite polynomial captures cannot be removed parametrically, and
trying inflates the variance of the low-frequency estimates.

Two properties make it work where the others fail. It annihilates a
degree-$(d-1)$ polynomial identically in the time domain, which no diagonal
filter can. And its $F$ is complex, so $F/|F|$ is a phase ramp the magnitude
weighting does not cancel.

The price is the boundary term — the zero-padded history leaves a polynomial
of degree $d-1$ in $z_k^{-1}$ (Theorem 2), which becomes the extra regressors
of step 3. `select_differencing_order` picks $d$ by eq. (11), with a documented
caveat: the criterion assumes "trend plus random series", and a strongly
low-pass response breaks that, so it can recommend differencing a record with
no trend at all. Treat it as a starting point.

### `MovingAverage`

Subtract a centred boxcar average of $L$ samples. The response is one minus
the Dirichlet kernel:

$$
F(k) = 1 - \frac{\sin(\pi k L/N)}{L \sin(\pi k /N)}
$$

$L$ must be odd, so the window has a true centre and $F$ carries no phase.

**A boxcar is a poor filter and this is where it shows.** $F$ has sidelobes,
overshoots above 1, and is not monotone — so checking $|F|$ *at* a tone is
worthless. `MovingAverage(20001)` on a 160 000-sample record has
$|F| = 1.00005$ at bin 8 and swings from **0.363 to 1.214** over the $\pm 4$
bins either side, a 3.3× ripple. That band is exactly what DMFA reconstructs
the time variation from, so the ripple is an uncorrected envelope distortion.
Use `band_response` and judge by `ripple`, never by the centre value.

Measured on DMFA at the lowest tone, against a clean baseline of 4.1e-03:

| record | no filter | with it |
|---|---|---|
| no trend | 4.1e-03 | **1.9e-02** |
| trended, $L = 20001$ | 8.7e-02 | 5.0e-02 |
| trended, $L = 40001$ | 8.7e-02 | 9.7e-02 |

Those three rows have to be read together. $L = 20001$ *does* improve the
trended record, by 1.7× — per-bin invariance does not forbid that, because a
filter can still reweight which bins dominate a band fit even though it cannot
change the ratio within any one of them. But $L = 40001$ makes it worse, so
the direction is not consistent across settings, and the win is paid for by
raising the trend-free floor 4.5×. Nothing is gained here that `PolynomialFit`
does not gain better, and more predictably, on the same record.

Present for completeness; `LowPassRemoval` is strictly the better shape.

### `LowPassRemoval`

Subtract the low-pass content, with a Butterworth magnitude:

$$
F(k) = \frac{r^n}{\sqrt{1 + r^{2n}}}, \qquad r = |f_k| / f_c
$$

Defined *by* this magnitude and applied as such, so it is zero-phase and has
no startup transient — unlike the same Butterworth run as a time-domain
recursion, which would have both and would then need boundary regressors it
cannot afford.

The cutoff is in **cycles per record**, which is the DFT bin index: bin $k$ is
$k/T$ Hz. That keeps the strategy independent of the sampling rate, and it is
the only unit `frequency_response(k, N)` can resolve. A cutoff of 3 means
"treat anything slower than three cycles over the whole record as trend".

Unlike the boxcar it is monotone and never exceeds 1, so a cutoff a few times
below the slowest tone gives a genuinely flat band: at `cutoff = 2` over
$\pm 4$ bins around bin 8, ripple 1.03× and a DMFA error of 5.8e-03 on a
trend-free record against the 4.1e-03 baseline.

It still does not remove a trend that *overlaps* the tones — 8.7e-02 before,
8.8e-02 after at `cutoff = 2`, and 1.2e-01 at `cutoff = 4, order = 6`. What it
does well is remove a trend genuinely slower than everything excited, which is
the case the cutoff describes. Whether your trend is that case is an empirical
question; `band_response` answers it.

### `PolynomialFit`

$$
y_d = \left(I - b (b^Tb)^{-1} b^T\right) y
$$

with $b$ the Legendre basis on the record — the same basis BLTVA uses for its
trend block. Legendre rather than monomial because the monomial Vandermonde is
hopeless beyond degree ~15.

This is the one strategy with **no frequency response at all**, and not for
want of deriving it. Subtracting a least-squares fit is an orthogonal
projection: a rank-$(q+1)$ operator whose frequency-domain kernel is
$\sum_q B_q(k) B_q(k')^*$, dense in $(k,k')$. It mixes bins, so no $F(k)$
exists, step 2 cannot be carried out, and `frequency_response` raises.

Being non-diagonal is also exactly what makes it the **only** strategy that
helps DMFA or STFT. On a trended record it takes the lowest tone from 8.7e-02
to **7.9e-03**, where both diagonal filters leave it at 8.8e-02 or worse.

The price is paid silently, and it is the thing to understand before using it.
Applying the same subtraction to a record with **no trend** — so that
everything removed is signal — gives, at the lowest tone against a clean error
of 4.1e-03:

| degree | DMFA error at the lowest tone |
|---|---|
| 1 | 4.9e-03 |
| 3 | 7.9e-03 |
| 8 | 4.0e-02 |
| 12 | 6.2e-02 |

The distortion grows with the degree and **there is nothing to correct it
with**. That is both why the 2022 paper reaches for differencing and why BLTVA
fits this basis instead of subtracting it. Keep the degree as low as the trend
allows, and read the result as a trade rather than a cleanup.

!!! tip "Inside BLTVA, use `Nq` instead"
    The $N_q$ drift block of eq. (58) is this same Legendre basis, *fitted
    jointly with the impedance rather than subtracted first*. The trend is
    then separated from the signal using the whole band of data, and the
    distortion above does not arise at all. `PolynomialFit` exists for the
    estimators that have no such block, and for measuring what its absence
    costs.

## Checking a strategy

`check_response` verifies step 1 of the protocol — that `apply` is what
`frequency_response` claims:

$$
\mathrm{DFT}(\texttt{apply}(y)) = F(k)\,\mathrm{DFT}(y) + \big(\text{something
in the span of } \texttt{extra\_regressors}\big)
$$

It returns the part of the residual the boundary regressors cannot absorb. All
four implemented strategies return $\sim 10^{-15}$:

| strategy | residual |
|---|---|
| `Differencing(1)` | 2.9e-15 |
| `Differencing(4)` | 9.6e-16 |
| `MovingAverage(401)` | 6.8e-13 |
| `LowPassRemoval(3, 2)` | 4.2e-16 |

Anything larger means the strategy is reporting a response that does not
describe what it did, and an estimator built on it is biased with no warning.
A constant column joins the projection, because the estimator has one:
eq. (21) omits the $m=0$ boundary term on the grounds that the transient
polynomial already spans a constant.

This is the test that decides which category a new strategy falls into, and it
is worth running on anything added later.

## In practice

- **A polynomial trend, with BLTVA.** Use `Nq`. Nothing else, no detrending.
- **A trend no polynomial captures, with BLTVA.** `Differencing(d)`, with $d$
  from `select_differencing_order` as a starting point, checked against the
  spectrum.
- **Any trend, with DMFA or STFT.** `PolynomialFit` at the lowest degree that
  works, accepting the distortion in the table above — or, for the tones the
  drift actually overlaps, use BLTVA instead. The
  [DMFA page](dmfa.md#trends-why-bltvas-detrending-does-not-transfer) has the
  measurements and the joint-solve alternative.
- **To study what naive detrending costs**, apply the strategy yourself and
  call `bltva` with `detrend=None`. That is the honest way to reproduce what a
  user who filters first would get, and it needs no escape hatch.
