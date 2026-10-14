# BLA implementation notes

## Why step 3 of Pintelon 2013 is not transcribed

The per-burst estimator on the [BLA page](../methods/bla.md) stops short of
the full three-step procedure of Pintelon, Louarroudi and Lataire (*IEEE TIM*
**62** (2013) 3361, §V-B). Steps 1 and 2 transcribe directly. Step 3 does not,
and the obstruction is this package's frequency grid rather than anything
about the method.

This page records why, with the numbers, because it decides how
[roadmap item 2b](../roadmap.md) has to be built.

### Where the derivatives come from

The time variation is carried by polynomial basis functions,
$G(s,t) = \sum_p G_p(s) f_p(t)$ with $f_p$ of degree $p$. Theorem 2 of the
2012 companion paper rewrites that single-input time-*varying* system as a
multiple-input time-*invariant* one whose inputs are $u(t) f_p(t)$.

The bridge is the Laplace duality

$$
\mathcal{L}\{t\,x(t)\} = -X'(s)
$$

Multiplying by $t$ in the time domain is differentiating with respect to $s$ in
the Laplace domain. So trading the $f_p(t)$ factors for extra inputs turns
powers of $t$ into $s$-derivatives, and they survive into the relation between
the BLTI approximation and the branch transfer functions (eq. 15):

$$
G_\text{BLTI}(s) = H_0(s)
+ \frac{2}{T}\sum_{p} H^{(1)}_{2p+1}(s)
+ \frac{4}{T^2}\sum_{p} \beta_{2p} H^{(2)}_{2p}(s)
+ O(T^{-3})
$$

Those are derivatives **along the frequency axis**, of quantities known only
as samples, so they have to be computed numerically. Eq. (27) does it with
central differences between **neighbouring excited harmonics** — and that is
the step that carries Assumption 3's requirement that the excited harmonics be
equispaced.

### Two separate things go wrong without it

This package's grid is 1, 3, 5, 9, 13, 17, 27, 39, 59, 89, 133, 201, 301, so
the gaps between neighbouring excited harmonics are

```
2  2  4  4  4  10  12  20  30  44  68  100
```

a **50× spread**. Both failure modes below are measured on an R+RC impedance
standing in for $H_p$, whose derivatives are analytic.

#### The step size

A central difference carries $O(h^2)$ truncation error. Symmetric stencil,
widened:

| half-span | relative error |
|---|---|
| ±2 harmonics | 1.18e-04 |
| ±4 | 4.74e-04 |
| ±10 | 2.97e-03 |
| ±30 | 2.69e-02 |
| ±100 | 3.14e-01 |

2661× over a 50× span, which is $h^2$ to within the fit. The top of a
quasi-log grid sits in the last row.

#### The asymmetry

The $O(h^2)$ accuracy exists *because* the odd-order Taylor terms cancel, and
that cancellation needs equal distance either side. Unequal spacing puts a
term in $f'$ back. For the first derivative this costs an order of accuracy.
For the **second** derivative — which the $O(T^{-2})$ term of eq. (15) needs —
the three-point formula written for a symmetric stencil is not merely less
accurate, it is **inconsistent**. Fixed 1:2 asymmetry, stencil halved each
row:

| $a$ | $b$ | formula as written | proper unequal-spacing weights |
|---|---|---|---|
| 40 | 80 | 1.07 | 1.31e-01 |
| 20 | 40 | 2.01 | 8.37e-02 |
| 10 | 20 | 4.02 | 4.77e-02 |
| 5 | 10 | 8.09 | 2.55e-02 |
| 2.5 | 5 | 1.62e+01 | 1.32e-02 |
| 1.25 | 2.5 | **3.26e+01** | 6.69e-03 |

The error **doubles every time the stencil is refined**. The leading term is
$(b-a)f'/h^2 \sim f'/h$, so it diverges as $h \to 0$: refining the grid makes
it worse. The proper weights converge at $O(h)$, as they should.

At the gap ratios this package's grid actually has:

| left | right | formula as written | proper weights |
|---|---|---|---|
| 30 | 30 | 2.69e-02 | 2.69e-02 |
| 30 | 44 | **9.59e-01** | 4.74e-02 |
| 44 | 68 | 7.53e-01 | 7.20e-02 |
| 68 | 100 | 5.18e-01 | 1.08e-01 |

Equal spacing is the only row where the two agree, which is the point. A 30/44
ratio — the gap pattern around harmonic 89 — already gives a 96% error.

### What it costs here

First derivative, over the whole grid:

| grid | median error | worst |
|---|---|---|
| quasi-log (this package) | **5.5e-02** | 8.1e-02 at $h = 133$ |
| equispaced, same tone count | 3.5e-04 | 3.6e-04 |
| equispaced, denser | 3.1e-04 | 3.6e-04 |

**156× on the median.** Per harmonic:

| harmonic | stencil span | relative error |
|---|---|---|
| 3 | 4 | 3.6e-04 |
| 5 | 6 | 1.9e-02 |
| 9 | 8 | 1.4e-03 |
| 17 | 14 | 5.5e-02 |
| 39 | 32 | 6.4e-02 |
| 89 | 74 | 7.2e-02 |
| 133 | 112 | 8.1e-02 |
| 201 | 168 | 6.3e-02 |

!!! note "It is not monotone, and that is worse than if it were"
    Harmonic 5 (span 6) is 13× worse than harmonic 9 (span 8), despite the
    narrower stencil. The step-size error and the asymmetry error have
    independent signs and partially cancel at some tones. So the damage is
    unpredictable per tone rather than simply growing with the gap, which
    rules out correcting it with a rule of thumb.

Since the correction is itself the $O(T^{-1})$ term, computing it to 5–8%
still beats using $H_0$ alone — but it does not deliver the $O(T^{-2})$ the
construction promises, and the $O(T^{-2})$ term is unusable outright.

### The route that does work

Do not difference at all. The local polynomial method **already fits** a
polynomial in the frequency index over a band of consecutive DFT bins,

$$
H(k+r) = \sum_{s=0}^{R} g_s\, r^s
$$

and those $g_s$ are Taylor coefficients. With $\Delta\omega$ the spacing of one
DFT bin,

$$
H' = \frac{g_1}{\Delta\omega}, \qquad H'' = \frac{2 g_2}{\Delta\omega^2}
$$

The fit runs over *consecutive bins*, which are equispaced by construction
whatever the tones do, so the grid stops mattering. The $H_p$ for $p \ge 1$ are
estimated at the non-excited bins in step 1 anyway, so the dense samples are
already there.

Recovered from a noiseless fit:

| band half-width | order $R$ | $dH/ds$ | $d^2H/ds^2$ |
|---|---|---|---|
| 5 | 2 | 8.6e-05 | 1.2e-04 |
| 5 | 3 | **4.5e-09** | 1.2e-04 |
| 5 | 4 | 4.5e-09 | **8.1e-09** |
| 10 | 3 | 6.6e-08 | 4.5e-04 |
| 10 | 4 | 6.6e-08 | 1.2e-07 |
| 20 | 4 | 9.8e-07 | 1.9e-06 |

Note the parity: $R = 3$ fixes the first derivative and leaves the second
where $R = 2$ had it; the second needs $R = 4$. Adding an odd-order term
removes the next *even* remainder and vice versa. Pintelon 2012 already
requires $R \ge 2$ for this correction, so raising it to 4 is a small change.

!!! warning "That table is noiseless"
    It isolates the approximation error. With noise, a narrow band and a high
    order inflate the variance — the usual LPM bias/variance trade, which the
    LPM's own residuals quantify. The orders above are therefore a starting
    point, not a recommendation.

    This is also why *"use adjacent DFT bins in the difference instead"* is not
    the fix it looks like. Adjacent LPM estimates share most of their data, so
    the difference between them is dominated by correlated estimation error
    rather than by the derivative. Reading the fitted coefficients avoids that;
    differencing the fitted outputs does not.

### Status

Nothing on this page is implemented. It is the design note for roadmap item
2b, recorded now because the measurements were made while deciding whether
step 3 could be transcribed as published. It cannot, on this grid, and the
replacement is specific enough to build from.

The numbers here are properties of the numerical scheme and the grid, not of
package code, so unlike the rest of these notes they are not pinned by tests —
there is nothing yet to pin. They are reproducible from the published
formulas and the grid in [the general convention](index.md#a-general-convention).
