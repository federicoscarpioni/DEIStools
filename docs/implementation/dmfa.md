# DMFA implementation notes

!!! info "Verification status"
    The **definition** and the **filter** have now been checked line by line
    against Battistel *et al.* (2016), Koster *et al.* (2017) and Battistel and
    La Mantia (2019), and both match — see below. What is still open is an
    end-to-end comparison against an independent implementation on the same
    data.

    One finding from that check is a caveat the papers do not state: the
    mirroring they recommend for non-periodic records **must not be applied to
    a random-phase multisine**. See
    [below](#mirroring-must-not-be-used-with-a-random-phase-multisine).

## Which definition is implemented

Three definitions appear in this literature, and they are not equivalent.

**Battistel 2016, eq. (11)** writes the dynamic impedance as the stationary
ratio times a correction,

\[
Z_k(t) = \frac{V_k}{I_k}\left\{1 - \sum_{l=1}^{bw}
\left[\frac{I_{k,l}}{I_k}e^{\,2\pi \mathrm{j} l \Delta f t}
    + \frac{I_{k,l}}{I_k}e^{-2\pi \mathrm{j} l \Delta f t}\right]\right\}
\]

This is a **first-order expansion**, valid only while the skirt is small
against its carrier. The odd look of it — a series that has to be truncated,
with the truncation doubling as the bandwidth — is real, and it is why the
later papers replaced it.

**Koster 2017, eq. (9)** is the clean statement: the ratio of two filtered
inverse transforms, with no expansion and no smallness assumption. **This is
what DEIStools implements**, and it is the current definition.

**Battistel 2019, eq. (4)** goes further and gives the *physical* definition
via a Volterra series, against which both heuristics can be judged. It is a
reference point rather than an algorithm; the paper finds DMFA within about
1% of it at the lowest frequency and below 0.1% at the highest.

So: the calculation that looked odd was the 2016 one, and it is not the one
in this package.

## The filter matches the published one up to a constant

`create_fermi_dirac_fun` evaluates

\[
\tilde g = \frac{\cosh n}{\cosh n + \cosh (n x)},
\qquad x = \frac{f - f_c}{bw}
\]

while Battistel 2019 eq. (12) and Chukwu 2022 eq. (3) specify the product of
two Fermi-Dirac functions, which simplifies exactly to

\[
\tilde g_{\text{paper}} = \frac{\cosh n + 1}{\cosh n + \cosh (n x)}
\]

Checked numerically: the two agree in **shape** to 1e-14, and their ratio is
the frequency-independent constant

\[
\frac{\tilde g_{\text{DEIStools}}}{\tilde g_{\text{paper}}}
= \frac{\cosh n}{\cosh n + 1} = 0.999329524658 \quad (n = 8)
\]

Since DMFA divides voltage by current through the *same* filter, the constant
cancels exactly and the impedance is unaffected. The only consequence is that
the DEIStools filter has gain 0.99933 at the centre rather than 1, which
matters if you use the filter for anything that is not a ratio.

Both conventions put $\tilde g = 1/2$ at $x = \pm 1$, so `bw` means the same
thing in both: the half-amplitude half-width.

## The one-sided extraction is correct

The papers stress that $\tilde g$ is a *quadrature* filter, defined only for
positive frequencies, because that is what makes the inverse transform the
analytic signal — the complex envelope — rather than a real band-limited
signal.

`extract_impedance` takes a block of bins around $+f_k$ only and never the
mirror at $-f_k$, so the output is inherently one-sided. That is correct and
deliberate. The resulting factor of $1/2$ (a real cosine carries half its
energy on each side) cancels in the ratio, as does the `Npts_elab` scaling.

## Block extraction is a controlled approximation

Here DEIStools departs from the papers' procedure, for cost. The papers
filter the **whole** spectrum and inverse-transform at full length;
`extract_impedance` takes only `Npts_elab` bins around each tone and
inverse-transforms those. That is equivalent to filtering the full spectrum
with a filter forced to zero outside the block, so the two agree only to the
extent that the filter has already decayed at the block edge.

Measured on the transistor record at `bw = 0.5 f_min`, `n = 2`:

| block width | filter value at the edge | difference from full-spectrum filtering (median / worst) |
|---|---|---|
| 4 bandwidths | 1.2e-01 | 1.1e-03 / 2.3e-02 |
| 8 bandwidths | 2.5e-03 | 1.5e-05 / 2.6e-03 |
| 16 bandwidths | 8.5e-07 | 4.5e-09 / 9.2e-07 |

The error tracks the edge value almost exactly, as it should.

!!! tip "Use at least 16 bandwidths"
    The examples size the block at 8 bandwidths, which with a soft filter
    (`n = 2`) leaves a worst-case 2.6e-03 — the same order as the method's own
    error. Sixteen costs nothing but arithmetic and puts the truncation at
    1e-09. A sharper filter needs a narrower block; the rule is to extend the
    block until the filter has fallen below about 1e-06.

## Mirroring must not be used with a random-phase multisine

Battistel 2019 §3.1 recommends mirroring a non-periodic record about its last
point, doubling its length, to suppress the Gibbs skirt. Applied to a
random-phase multisine it is **destructive**, and badly so.

Mirroring reflects the record in time. For a tone $\cos(\omega t + \varphi)$
occupying a whole number of periods, the reflection is
$\cos(\omega t - \varphi)$: the cosine part is even and survives, but the sine
part is folded into $\sin(\omega|t|)$, which has a derivative kink at the seam
and therefore a broadband spectrum. The scattered energy lands in every
extraction band.

Measured on the standard 13-tone grid:

| record | off-bin / on-bin energy |
|---|---|
| original | 1.1e-24 |
| mirrored, phases all zero | 6.2e-04 |
| **mirrored, random phases** | **1.5e+00** |

With random phases the multisine scatters more energy off its bins than it
leaves on them. The end-to-end effect on the impedance is what you would
expect:

| | worst error, edges included |
|---|---|
| plain | 4.1e-01 |
| mirrored | 7.5e+00 |

The paper's own case is different: their multisine was periodic and it was the
*cyclic voltammetry* that was not, so mirroring fixed the dc skirt without
touching the tones. The technique is sound there and wrong here.

**Conclusion: discard the edges.** It is the robust answer, it costs a fixed
fraction of the record, and it is what the examples do. If you need the edges,
the [lock-in](../methods/lock-in.md) filters in the time domain, where the
boundary can be reflected on the smooth baseband envelope instead of being
wrapped around by a circular DFT.

## Is the Fermi-Dirac filter the right choice? No.

The Fermi-Dirac distribution comes from statistical physics, not from signal
processing, and the natural question is whether a standard DSP window does
better. It does. Scored against known ground truth, **the published filter
comes last of seven**, and the best standard window is about twice as good on
the worst case.

!!! warning "This corrects an earlier conclusion on this page"
    A first comparison here tested only the raised-cosine (Tukey) family and
    reported that the Fermi-Dirac won. That comparison was incomplete: the
    cosine-sum windows — Blackman, Blackman-Harris, Nuttall — and Kaiser were
    not in it, and they are the ones that beat it. The table below supersedes
    it.

### What the filter actually does to the result

In a galvanostatic measurement the current is the excitation: exactly
periodic, so its envelope is constant and the filter merely scales it. All the
time variation sits in the voltage. The consequence is that

\[
Z_{\text{measured}}(f_k, t) \;\approx\;
\frac{\big(\tilde g * Z\big)(f_k, t)}{\tilde g(0)}
\]

— the reported impedance is the true $Z(f_k,t)$ **convolved with the filter's
time-domain kernel**. So the filter's frequency response is, directly, the
transfer function applied to the time variation, and its time-domain
sidelobes are the ringing: at the record edges, and wherever the variation
turns.

That is precisely the quantity the classical window designs optimise, which is
why they win.

### The measurement

Every candidate scaled to 1/2 at $|f - f_c| = bw$ so that the bandwidth means
the same thing in each, every bandwidth swept, the band extracted 24
bandwidths wide and the output decimated to the independent spacing. Scored by
worst-case relative error over the usable part of the record:

| filter | transistor | legendre | block-oriented | mean rank |
|---|---|---|---|---|
| **blackman** | **2.40e-04** | **5.21e-04** | 1.18e-01 | **2.7** |
| kaiser, β = 8 | 2.92e-04 | 5.94e-04 | 1.18e-01 | 3.0 |
| blackman-harris | 2.96e-04 | 6.94e-04 | 1.18e-01 | 3.7 |
| DPSS, NW = 4 | 2.97e-04 | 7.09e-04 | 1.18e-01 | 4.0 |
| hann (= raised cosine α=1) | 4.71e-04 | 9.67e-04 | 1.17e-01 | 4.3 |
| gaussian | 3.05e-04 | 9.77e-04 | 1.18e-01 | 4.7 |
| **Fermi-Dirac, n = 2** | 4.98e-04 | 6.47e-04 | 1.27e-01 | **5.7** |

Three systems, and the ordering is stable on the two that discriminate. The
block-oriented column is effectively a tie because its error there is
dominated by the branch filters' roll-off rather than by the extraction — see
[example 23](../methods/dmfa.md).

Two details worth noting:

* **DPSS lands on top of Kaiser**, within 1%, which is the expected result:
  Kaiser is a closed-form approximation to the first Slepian sequence. Since
  time concentration is what decides this problem, the provably optimal
  concentration was the obvious thing to try, and it confirms that Kaiser is
  already at that limit. It is not offered in the package because it needs an
  eigensolve for no measurable gain.
* **A rectangular filter — eq. (13) of the paper — is catastrophic**, at
  3.8e-02. The smooth filter of eq. (12) was a large improvement over it; it
  simply was not taken as far as the window literature already had.

### What `n` actually controls

A second correction. At `n = 8` the Fermi-Dirac is a flat top with a steep
roll-off. At `n = 2` — the value the bandwidth/sharpness sweep selects — it is
**not flat at all**, but a smooth bell. Calling `n` a sharpness parameter is
misleading: it trades passband flatness against time concentration, and on
this problem concentration is what pays. That is also why the two *least*
flat-topped candidates came out ahead of the flatter ones.

!!! note "A metric that did not survive scrutiny"
    An earlier version of this page quoted a "time concentration" figure — the
    fraction of kernel energy beyond $5/bw$ — and drew conclusions from it.
    That number turned out to depend on the frequency grid it was evaluated
    on, moving by orders of magnitude between two reasonable choices, and the
    ranking it implied did not match the end-to-end result. It has been
    removed. The error against ground truth is the measure that decides.

### What this package does about it

The Fermi-Dirac remains the **default**, because it is what the DMFA papers
specify and this package's job is to implement the published methods
faithfully. But `create_window_filter` makes every alternative a one-line
change:

```python
from deistools.processing.passband_filters import create_window_filter

band_filter = create_window_filter(band_axis, 0, bandwidth, "blackman")
```

and `create_raised_cosine_fun` covers the Tukey family. Example 12 runs the
comparison, so the ranking can be re-checked on your own system rather than
taken on trust — which is the point, since the block-oriented column already
shows a case where the filter choice does not matter at all.

## DMFA as linear algebra: it is BLTVA in a different basis

DMFA looks like a filtering operation and BLTVA looks like a least-squares
fit, which makes them seem like different kinds of thing. They are not. DMFA
is already a linear-algebra estimator; it just happens to use a basis in
which the solve is trivial.

### The implicit model

Filtering a band of width $bw$ around a tone and transforming back **assumes
the envelope is band-limited** to $|f| < bw$. That is a model, and on a record
of length $T$ it is this one:

\[
V_k(t) \;=\; \sum_{m=-M}^{M} c_m \, e^{\,\mathrm{j}2\pi m t / T},
\qquad M = bw\,T
\]

— the envelope is a trigonometric polynomial of degree $bw\,T$, with
$2M+1$ complex coefficients. Compare BLTVA, whose model is

\[
Z(f,t) \;=\; \sum_{p=0}^{N_p} H_p(f)\, b_p(t)
\]

with $b_p$ the Legendre polynomials and $N_p+1$ coefficients. **Same
structure, different basis.**

### Fitting it gives exactly DMFA

Build the design matrix in the time domain — one column pair per mode,
$\cos$ and $-\sin$ of $(\omega_k + 2\pi m/T)\,t$ — and solve the ordinary
least-squares problem. The result is the band extraction, to machine
precision:

```
max |DMFA − least squares| / max   =  1.3e-15
design-matrix condition number     =  1.000000
```

That condition number is the whole explanation. The columns are distinct
Fourier harmonics, so they are **orthogonal on the record**; the normal
equations $K^{\mathrm{H}}K\,\hat\theta = K^{\mathrm{H}}y$ have a diagonal
matrix on the left, and the solve degenerates into reading off the DFT bins.
There is nothing to invert. BLTVA's Legendre columns are not orthogonal in
the band, which is why it has a solve, a condition number worth reporting,
and a pruning step.

### A smooth filter is ridge regularisation

Multiplying coefficient $m$ by the filter gain $g_m$ is exactly diagonal
Tikhonov shrinkage. Minimising
$\|y - Kc\|^2 + \sum_m \lambda_m |c_m|^2$ gives
$\hat c_m = y_m/(1+\lambda_m)$, so

\[
g_m \;=\; \frac{1}{1+\lambda_m}
\qquad\Longleftrightarrow\qquad
\lambda_m \;=\; \frac{1}{g_m} - 1
\]

Verified to 1.2e-15. So the choice between a rectangular and a Fermi-Dirac
filter is not a choice between "no filter" and "a filter" — it is a choice
between **hard model-order truncation and a ridge prior**, with the penalty
rising smoothly towards the band edge. The filter *is* the regulariser.

### Three things this explains

**It gives DMFA an uncertainty.** A linear estimator $\hat c = Ay$ has
covariance $A\Sigma A^{\mathrm{H}}$, which is the route BLTVA already takes.
Working it through reproduces Chukwu eq. (19)–(20) exactly — their error
propagation *is* this covariance — and
[`extract_impedance_with_error`](../api/processing.md) now computes it,
agreeing with the scatter over 200 noise realisations to better than 1%.

**It explains the independent spacing.** The model has $2\,bw\,T$ free
coefficients over a record of length $T$, so there is one degree of freedom
per $T/(2\,bw\,T) = 1/(2\,bw)$ seconds. That is precisely the
$0.5/bw$ at which neighbouring errors stop being correlated — the rule is
just "do not report more points than the model has parameters".

**It explains the edges, and points at a better basis.** The Fourier basis is
**periodic by construction**. Band-limitedness does not require periodicity;
the basis imposes it, and a non-periodic envelope then produces Gibbs
oscillation. BLTVA's polynomial basis carries no such constraint, which is why
it has no edge problem at all.

The natural basis for *band-limited on a finite interval* is the Slepian
(DPSS) basis, not the Fourier one. Fitting the same band with the same number
of coefficients:

| basis | at $t=0$ | interior | at $t=T$ |
|---|---|---|---|
| Fourier (periodic) | 26.8% | 1.5% | 3.2% |
| **Slepian (not periodic)** | **5.0%** | 1.6% | 2.4% |

Five times better at the leading edge, for nothing — same bandwidth, same
degrees of freedom, same data. The interior is unchanged, as it must be.

!!! note "Not implemented"
    A Slepian-basis DMFA is a genuine improvement over zero-padding rather
    than an alternative to it: padding removes the wrap-around, a non-periodic
    basis removes the reason the wrap hurt. It needs an eigensolve per
    bandwidth and a different code path from `extract_impedance`, so it is on
    the [roadmap](../roadmap.md) rather than in the package. The same DPSS
    appeared in the filter comparison above, for the same underlying reason.

## Variance: propagation, or measured in-band?

Chukwu *et al.* (2022) derive the expected variance of the error on the
admittance by propagating the measurement noise through the DMFA chain:

\[
\sigma^2_{Y_{l,k_0}} \;=\;
\frac{\sigma_i^2 + \big|Y_{l,k_0}\big|^2\,\sigma_v^2}
     {2\,N'\,\big|\tilde v_{l,k_0}\big|^2},
\qquad
N' = \frac{N}{\sum_{k=-N/2}^{N/2}\big|\tilde g(k-k_0)\big|^2}
\tag{19, 20}
\]

The derivation is sound — $N'$ is the usual equivalent-noise-bandwidth factor,
and the assumptions (noise of constant variance, uncorrelated between samples,
Parseval for the filter norm) are stated rather than hidden.

The weakness is not the algebra; it is **where $\sigma_i$ and $\sigma_v$ come
from**. The paper obtains them from a separate calibration: zero volts across
resistors, histogrammed. That assumes the noise is white and that it has not
changed between the calibration and the measurement.

### The BLTVA route is available and is better

BLTVA already estimates the noise variance *from the measurement itself*,
using bins the excitation never reaches (Algorithm 1 of Hallemans TIM 2021,
exposed here as `estimate_noise_variance`). On an odd-harmonic grid with gaps
those bins are plentiful, and the estimate:

- needs no separate instrument calibration;
- captures the noise **colour**, not just its level;
- reflects the noise as it was during that measurement;
- can be separated from nonlinear distortion by the parity of the bins.

The two are complementary rather than rival. Keep eq. (19) — it is the correct
propagation through the filter — and feed it a $\sigma^2(f)$ measured from the
unexcited bins instead of from a resistor. That removes the assumption the
formula rests on while keeping its structure. The section above shows that eq. (19) is nothing more than the covariance of
a linear estimator, which is also why BLTVA can report one.

`extract_impedance_with_error` now computes it correctly. Three bugs had
survived in that function, unnoticed because it had no test coverage: it
returned the **reciprocal** of the variance (a half-finished edit was still
commented out on its last line), it squared arguments already named
`variance_*`, and its N' used the width of the extracted band where eq. (20)
calls for the length of the record. Corrected, it agrees with the scatter over
200 noise realisations to 0.3%. Passing it a data-driven noise variance from
the unexcited bins, rather than a resistor calibration, is left to the caller
and is the remaining improvement.

!!! note "A testable hypothesis about the paper's own residual"
    Chukwu *et al.* report that below 100 Hz their measured standard
    deviations run up to four times the predicted value, and state the reason
    could not be ascertained. A noise floor rising towards low frequency —
    $1/f$ noise, which is ubiquitous in electrochemical cells — would produce
    exactly that signature, and a white-noise calibration cannot see it. A
    data-driven estimate from the unexcited bins would. This has not been
    tested here and is offered as a hypothesis, not a result.

### Do not treat successive instants as independent

Neighbouring estimates rest on overlapping filter kernels, so their errors are
correlated. Chukwu *et al.* put the correlation below 20% once successive
instants are more than $0.5/bw$ apart. Measured here on a stationary cell —
where the impedance never moves, so the whole spread of the estimate is noise
— over 150 noise realisations:

| output spacing | instants over the record | corr($e_j$, $e_{j+1}$) | corr($e_j$, $e_{j+2}$) |
|---|---|---|---|
| 0.125/bw | 80 | **84%** | 50% |
| 0.25/bw | 40 | 49% | 4% |
| **0.5/bw** | 20 | **3%** | 0% |
| 1/bw | 10 | −3% | −3% |
| 2/bw | 4 | −9% | −1% |

The rule is confirmed, and the earlier examples sat at 0.125/bw — four times
too dense, with 84% correlated neighbours. Reporting densely is fine for
looking at a result; using every point as an independent observation in a
least-squares fit is not, and will make the parameter uncertainties too small.

`independent_time_step(bandwidth)` returns $0.5/bw$. Note that `Npts_elab`
sets the band width and the reporting rate together, and they are different
concerns — the band must be wide enough for the filter to decay, which is
*finer* than this spacing. **Extract wide, then decimate**, which is what the
examples now do.

!!! warning "The time resolution is not free after all"
    At the bandwidth that measures best here, $bw = 0.5 f_{\min}$, the
    independent spacing works out to $0.5/bw$ = one multisine period —
    exactly STFT's window. DMFA's resolution is *adjustable* where STFT's is
    fixed, which is a real advantage, but the claim that it is simply finer
    was an artefact of reporting more densely than the data supports. Buying
    genuinely finer resolution means widening the band, and example 12 shows
    what that costs.

## The record edges are a property of the method

The band-limited inverse transform rings at the record boundaries. Measured on
the four simulated systems:

| system | worst, edges removed | worst, edges kept |
|---|---|---|
| stationary | 1.35e-05 | 1.35e-05 |
| transistor | 1.39e-03 | 4.13e-01 |
| legendre | 1.33e-03 | 8.85e-01 |
| block-oriented | 1.71e-01 | 4.26e-01 |

The stationary row is the diagnostic one: identical with and without the
edges. A constant impedance occupies a single bin, so truncating the band
removes nothing and there is nothing to ring.

That establishes the cause. The artefact is not "the record ended" — it is the
cost of representing a *changing* quantity with a finite band, under a DFT
that treats the record as periodic when the envelope is not. No parameter
setting removes it. This package discards the first and last tenth throughout.

## The filter belongs on the band's local axis

`elaboration_frequency_axis(n_points, T)` returns the frequency axis of an
extracted band: `n_points` bins of width $1/T$, centred on zero — about 0.2 Hz
in the worked example. The full spectrum axis spans $f_s$, 1000 Hz there.

Build the passband on the wrong axis and its bandwidth is scaled wrongly by
$f_s T / n_{\text{points}}$, a factor of 2500. The result is not an error,
just a filter that does something entirely different from what was asked. This
is the easiest mistake to make with the method and the reason
`elaboration_frequency_axis` exists as a named function rather than being
inlined.

## Display fixes in `visualize_dmfa_filtering`

Three defects were corrected in the slider view. They were display-only — no
estimation path was affected — but they had survived because the function had
no test coverage:

- the voltage panel drew its "Original data" trace with the filter *applied*,
  making it identical to the filtered trace until the slider was first moved;
- the initial title hard-coded the first frequency rather than the selected
  one;
- a stray parenthesis in the updated title (`Fouri(er`).

The parameter name `frequenies` is still misspelled in the public signature,
so pass that argument positionally.
