# BLTVA implementation notes

## The rank threshold in eq. (53) is ambiguous

Algorithm 1 truncates the residual used for the noise-variance estimate at a
rank determined by a threshold. The paper prints the threshold in a form that
reads as $a \cdot \mathrm{snr}$, which always truncates at $m = 1$ — the
threshold is met immediately and nothing is retained.

Both readings were implemented and tested against a known noise level:

| reading | estimated / true noise variance |
|---|---|
| $a \cdot \mathrm{snr}$ | 1.86 – 5.81 |
| $1 / (a \cdot \mathrm{snr})$ | 0.97 – 0.98 |

The reciprocal reading is unbiased, so that is what the code uses. The choice
is pinned by `test_rank_threshold_is_unbiased`, so a future change that
reverts it will fail the test suite rather than silently shift every noise
estimate.

## The band solver deliberately does not equilibrate

Column equilibration is the obvious conditioning improvement for a
least-squares problem whose regressors have wildly different scales, and an
earlier version of this code did it.

It is wrong here, and subtly so. Algorithm 2 step 6 prunes coefficients whose
magnitude falls below the noise floor. With equilibration, collinear columns
that cancel against one another produce coefficients of order $10^{10}$ —
which are never below any threshold, so **pruning never fires**. The fit then
keeps every regressor it was given, including the ones the data does not
support.

Using the plain minimum-norm solution instead gives $|\theta| \approx
10^{-10}$ for the same columns, and all five spurious drift columns are
pruned, with an identical residual.

So `_solve_band` uses `np.linalg.lstsq` without equilibration, on purpose. The
pruning behaviour described in
[BLTVA § pruning](../methods/bltva.md#pruning-it-can-say-stationary) depends on
it.

## The Legendre basis convention

`legendre_basis` mean-removes every column of order $p \ge 1$, as the 2021
paper prescribes after eq. (26). That is what makes $H_0$ the time-averaged
impedance.

The simulators import the *same* function from
`deistools.processing.bltva` rather than building their own basis. This is not
tidiness: when `simulate_legendre` used a raw `legvander` while the estimator
mean-removed, the apparent error floor of an exact-recovery test sat at
$2\times10^{-5}$ — entirely a convention mismatch. Sharing the function moved
it to $10^{-7}$.

If you write a new simulator, import that basis.

## eq. (49) is necessary, not sufficient

The identifiability bound

\[
N_q + 3N_p + N_a + 1 \;<\; 2P - 1
\]

counts the parameters a band must support. `check_identifiability` enforces
it.

But Algorithm 1 estimates the noise variance from a *sliding band*, and that
band needs strictly more equations than regressors. On a short record it can
fail first. On a 6-period record, $N_p = 2$ with $N_q = 2$ satisfies eq. (49)
and is then refused:

```
ValueError: noise band at bin 6: 10 equations for 10 regressors.
```

The practical ceiling on 6 periods is $N_p = 1$ with $N_q \le 1$.

**If BLTVA refuses a configuration the inequality admits, this is usually
why.** Lower $N_q$ before lowering $N_p$.

## $N_p$ and the band width compete for $N_a$

The interaction documented on the
[BLTVA page](../methods/bltva.md#band_halfwidth-and-why-it-is-not-independent-of-n_p)
has a specific cause, and it is not the one that first suggests itself.

It is **not** ill-conditioning. Measured across the band width at $N_p = 8$,
the per-band condition number *improves* as the band widens — 148 at
halfwidth 50 down to 25 at halfwidth 400 — while the error gets worse. The
diagnostics say what is actually happening:

| band | bins | skirt centres | parameters | pruned | residual |
|---|---|---|---|---|---|
| 50 | 70 | 1 | 29 | 7 | 2.20e-06 |
| 100 | 120 | 2 | 38 | 6 | 3.03e-06 |
| 200 | 220 | 3 | 47 | 3 | 1.73e-05 |
| 400 | 420 | 5 | 65 | 0 | 5.71e-05 |

Two things happen together. More skirt centres fall inside a wider band, each
bringing its own block of $N_p + 1$ regressors. And the residual *rises* by
26× — the model fits the band **worse** despite having more than twice the
parameters, which means it is mis-specified rather than over-fitted.

The culprit is $N_a$. eq. (22) sets

\[
N_a = \left\lfloor 3P/4 \right\rfloor - N_p - d + 1
\]

so the budget spent on $N_p$ comes straight out of the transient polynomial:
$N_a = 13$ at $N_p = 3$, but only $8$ at $N_p = 8$. That polynomial has to
describe the smooth out-of-band background across the *whole* band. Over 70
bins an order-8 polynomial is ample; over 420 bins it is stretched far too
thin.

Hence: raise $N_p$ and you must narrow the band.

## Why `band_halfwidth` exists at all

The eq. (55) rule builds a band spanning three consecutive excited
frequencies, which is right and well-motivated: it parameterises both
immediate neighbours explicitly, so the nearest *unmodelled* skirt sits two
spacings away and is flat enough for the transient polynomial to absorb
honestly.

On a linearly spaced odd grid, as the papers use, those bands are modest. On a
quasi-logarithmic grid with 13 tones over 2.5 decades, the spacing at the top
is enormous and the rule produces a band of about 5360 bins — roughly 268
harmonics. The package warns when that happens.

`band_halfwidth` overrides the rule with a fixed number of bins. It is also
more robust where the grid spacing changes abruptly.

!!! note "A correction to an earlier claim"
    An earlier draft of this analysis stated that the neighbour rule degrades
    by 100× on logarithmic grids. That was wrong — it was an artefact of a
    buggy test helper that emitted adjacent harmonics. The real failure mode
    is an abrupt change in spacing, which is what
    `test_neighbour_rule_degrades_on_abrupt_spacing_change` pins.
