# `deistools.simulation` — models, engines and their mathematics

This subpackage makes synthetic measurements whose impedance is known, so that
an estimator's output can be compared against a ground truth. This document
explains what each piece computes, why it is correct, where it is only
approximately correct and by how much, and which paper each piece comes from.

It is meant to be read before the code. If you can follow the derivations
here, you can defend the code.

**Status of the claims in this document.** Every derivation here can be
checked by hand. Each numerical claim is labelled as one of:

- **verified** — re-run on branch `refactor/owned-code`, with code you can
  reproduce from [§8](#8-reproducing-the-two-engine-checks);
- **development result** — measured while the code was being written, with
  code that has since been deleted along with the test suite, and not re-run.

Attributions to specific equations, figures or coefficients in a paper
were copied from the code's docstrings and could not be checked against the
PDFs. They are collected in [§6](#6-attributions-to-check-against-the-papers)
for you to confirm.

---

## Contents

1. [The common setting](#1-the-common-setting)
2. [The three engines](#2-the-three-engines)
3. [When frozen time is exact, and when it is not](#3-when-frozen-time-is-exact-and-when-it-is-not)
4. [The models](#4-the-models)
5. [Trends: disturbances, not systems](#5-trends-disturbances-not-systems)
6. [Attributions to check against the papers](#6-attributions-to-check-against-the-papers)
7. [What to test](#7-what-to-test)
8. [Reproducing the two engine checks](#8-reproducing-the-two-engine-checks)
9. [References](#9-references)

---

## 1. The common setting

### Excitation

Every model is driven **galvanostatically** by a multisine current:

```math
i(t) = \sum_{k=1}^{K} A_k \cos(\omega_k t + \phi_k), \qquad \omega_k = 2\pi f_k .
```

`engine.generate_multisine` builds it directly from the lists of frequencies,
amplitudes and phases. The simulation does not care how the tones are chosen.
The estimators do, though. They need every $f_k$ to sit exactly on a DFT bin of
the record, which means the record length $T = N\,\Delta t$ must be a whole
number of multisine periods. Build every record that way.

### Time axis

`N = round(duration_s / sampling_time_s)` samples at $t_n = n\,\Delta t$,
$n = 0,\dots,N-1$. The record length is $T = N\,\Delta t$, **not** $t_{N-1}$,
which is one sample shorter. This matters wherever time is normalised to
$[-1, 1)$, as the Legendre basis does.

### Noise

Every model adds independent, white Gaussian noise to the **recorded**
voltage and current *after* computing the response (`noise_std_V`,
`noise_std_A`, `seed`). Current noise therefore models a noisy ammeter: it
corrupts the measurement but never drives the system. The voltage is always
computed from the noise-free current.

### What "frozen-time impedance" means

For a system whose parameters $\theta(t)$ change with time, the **frozen-time
impedance** at instant $t$ is the ordinary LTI impedance of the system with its
parameters frozen at their values at that instant:

```math
Z(f, t) \;=\; Z_{\text{LTI}}\big(f;\ \theta(t)\big).
```

That is the quantity every estimator in this package tries to recover, so
every model returns it as `Z_true`, of shape `(n_freq, N)`. Whether the
system's *actual* response matches the frozen-time impedance is a separate
question, and it is the subject of [§3](#3-when-frozen-time-is-exact-and-when-it-is-not).

---

## 2. The three engines

Three different calculations produce a voltage from the current. Knowing which
one a model uses tells you what its ground truth means.

### 2.1 Frozen-time synthesis — `engine.simulate`

Given a model's `impedance_fn(f, t)`, which returns $Z(f_k, t_n)$ as an array
of shape `(n_freq, N)`, the engine writes the voltage tone by tone:

```math
v(t) \;=\; \sum_{k} A_k\, \big|Z(f_k,t)\big| \cos\!\Big(\omega_k t + \phi_k + \arg Z(f_k,t)\Big)
\;=\; \sum_k A_k\, \operatorname{Re}\!\Big\{ Z(f_k,t)\, e^{j(\omega_k t+\phi_k)} \Big\}.
```

That is `engine.synthesize_voltage`. It is the steady-state response of an LTI
system with impedance $Z$, applied instant by instant. There is no ODE and no
time stepping, so the cost is $O(KN)$.

**Used by:** `r_rc`, `legendre`, and the
`voltage_quasistatic_V` reference inside `r_rc_transistor`.

### 2.2 Time-domain ODE integration — `r_rc_transistor`

When the frozen-time synthesis is not the true response (see
[§3.2](#32-a-time-varying-parameter-inside-a-loop-is-only-approximately-handled)),
the circuit equation is integrated directly. For the transistor model the state
is the voltage $v_C$ across the parallel branch:

```math
C\,\frac{dv_C}{dt} + \frac{v_C}{R_{ct}(t)} = i(t), \qquad v(t) = R_s\, i(t) + v_C(t).
```

This is integrated with the classical fixed-step **fourth-order Runge–Kutta**
method on the output grid. Two details keep it honest:

- RK4 needs $i$ and $R_{ct}$ at the half steps $t_n + \Delta t/2$. They are
  **evaluated analytically** there, not interpolated, because interpolating
  would quietly drop the method's order.
- The **initial state** is the frozen-time steady state at $t = 0$,
  $v_C(0) = \sum_k A_k \operatorname{Re}\{Z_p(f_k,0)\,e^{j\phi_k}\}$ with
  $Z_p = R_{ct}/(1+j\omega R_{ct} C)$. Starting from $v_C = 0$ would add a
  startup transient lasting about $5R_{ct}C$ that has nothing to do with the
  time variation. The frozen state is still not quite the true time-varying
  state at $t=0$, since that would depend on the history before the record.
  The leftover transient has the size of the first-order term in §3.2 and
  decays with $R_{ct}C$.

The step must stay small against the fastest time constant, and the code
warns when $\Delta t > 0.1\,\min_t R_{ct}(t)\,C$.

### 2.3 Periodic steady-state filtering in the DFT domain — `block_oriented`

The block-oriented model needs an LTI filter applied to a *nonlinear*
function of the input. A signal that is a static function of a periodic
multisine is itself periodic with the same period. So the filter's
steady-state output is periodic too, and its Fourier coefficients are
$H(j\omega_k)X_k$. The code therefore filters in the DFT domain:

```math
y = \operatorname{IDFT}\!\Big( H(j2\pi f_{\text{bin}}) \cdot \operatorname{DFT}(x) \Big),
```

with $H$ evaluated at the signed DFT bin frequencies (`np.fft.fftfreq`), so that
$H(-f)=\overline{H(f)}$ and the output is real.

This is **exact for the periodic steady state** under two conditions, and both
are worth stating whenever you defend it:

1. **The record must hold a whole number of periods.** Otherwise the DFT's
   circular convolution is not the steady state, and the error appears as
   wrap-around at the record edges.
2. **The signal entering the filter must be band-limited below $f_s/2$.** The
   input $x = f_1(u)$ has passed a static nonlinearity, which creates
   harmonics. A polynomial of degree $d$ applied to a multisine with top tone
   $f_{\max}$ has content up to $d\,f_{\max}$. Above $f_s/2$ those harmonics alias
   and the simulation no longer represents the continuous-time system. A tanh
   nonlinearity has infinitely many harmonics, but at these coefficients they
   decay fast. A nonlinearity *after* the filter ($f_2$) is applied sample by
   sample, so its samples are exact; its harmonics alias only in the *DFT* of
   the output, just as they would at a real ADC with no anti-aliasing filter.

No startup transient is produced, by construction. The papers model the
transient separately, as a polynomial in the estimator, rather than as part of
the signal.

---

## 3. When frozen time is exact, and when it is not

This is the most important section. The frozen-time impedance $Z(f,t)$ is
always well defined. Whether the system's *actual* response is the
frozen-time synthesis of $Z(f,t)$ depends on **where in the system the time
variation sits**, not on how fast or how slow the system is.

### 3.1 A time-varying gain on the output of LTI branches is handled exactly

Suppose the system is a sum of LTI branches, each followed by a real,
time-varying gain. This is the structure of Hallemans *et al.* (2021):

```math
y(t) = \sum_{p} b_p(t)\, \big(h_p * u\big)(t), \qquad b_p(t) \in \mathbb{R}.
```

Drive it with the multisine, in steady state. Each LTI branch gives

```math
(h_p * u)(t) = \sum_k A_k \operatorname{Re}\!\Big\{ H_p(j\omega_k)\, e^{j(\omega_k t + \phi_k)} \Big\}.
```

Multiply by the real $b_p(t)$ and sum over $p$. Because $b_p$ is real it can be
moved inside $\operatorname{Re}\{\cdot\}$, and the two sums commute:

```math
y(t) = \sum_k A_k \operatorname{Re}\!\Big\{ \underbrace{\Big(\sum_p H_p(j\omega_k)\, b_p(t)\Big)}_{Z(f_k,t)} e^{j(\omega_k t + \phi_k)} \Big\}.
```

That is exactly the frozen-time synthesis of §2.1 with
$Z(f,t) = \sum_p H_p(j\omega)\, b_p(t)$. **No approximation was made**, and in
particular *no assumption that $b_p$ varies slowly*. The derivation needs only
two things: $b_p$ real, and the branches in periodic steady state.

**Verified:** a linear block-oriented system computed by DFT filtering (§2.3)
and the same system computed by frozen-time synthesis of
$Z = \sum_p H_p b_p$ (§2.1) agree to **4.5e-14** relative, which is round-off.
These are two independent engines that agree on one system ([§8](#8-reproducing-the-two-engine-checks)).

Note what this means for "memory". The branches $h_p$ can have arbitrarily
long memory and the result still holds, because the variation is applied
*after* the memory, to the output.

### 3.2 A time-varying parameter inside a loop is only approximately handled

Now put the varying parameter inside the dynamics. The transistor circuit is
the simplest case: $R_{ct}(t)$ sits in parallel with $C$, so the state $v_C$
feeds back through the varying resistance. With $g(t) = 1/R_{ct}(t)$,

```math
C\,\dot v_C + g(t)\, v_C = i(t).
```

Take one tone, $i = \operatorname{Re}\{I e^{j\omega t}\}$, and look for
$v_C = \operatorname{Re}\{V(t)\, e^{j\omega t}\}$ with a slowly varying phasor
$V(t)$. Substituting gives $C(\dot V + j\omega V) + gV = I$, that is,

```math
V = Z_p(t)\,\big(I - C\,\dot V\big), \qquad Z_p(t) = \frac{1}{g(t) + j\omega C}.
```

Iterate this equation:

- **Zeroth order** ignores $\dot V$: $V^{(0)} = Z_p(t)\, I$. That *is* the
  frozen-time response.
- **First order** substitutes $V^{(0)}$ back in. Using
  $\dot Z_p = -\dot g\, Z_p^2$:

```math
V^{(1)} = -C\,Z_p\,\dot V^{(0)} = C\,\dot g\, Z_p^{3}\, I .
```

So the relative error of the frozen-time response at frequency $\omega$ is

```math
\frac{V^{(1)}}{V^{(0)}} = C\,\dot g\,Z_p^2
\quad\Longrightarrow\quad
\left|\frac{V^{(1)}}{V^{(0)}}\right| = \frac{\tau\,\big|\tfrac{d}{dt}\ln R_{ct}\big|}{1+(\omega\tau)^2},
\qquad \tau = R_{ct}C .
```

Three things follow, and they are what you would say to defend the transistor
model:

- The error is **first order in the drift rate**. Halve the ramp and the
  error halves.
- It is **largest at low frequency**, where it approaches
  $\tau\,|d\ln R_{ct}/dt|$, and it falls off as $1/(\omega\tau)^2$ above the arc.
- The expansion is valid while $\tau\,|d\ln R_{ct}/dt| \ll 1$, meaning the
  parameter changes little within one time constant.

**Verified:** adding the first-order term $V^{(1)}$ to the frozen-time
synthesis reproduces the RK4 solution of §2.2.

| record | ODE − frozen (rms) | left after adding $V^{(1)}$ |
|---|---|---|
| $R_s=10\ \Omega$, $R_p=30\ \Omega$, $R_{T,\max}=70\ \Omega$, $C=1$ mF; gate ramp across 90 % of the window in 400 s | 0.012 % | ≈ 0 % of it |
| same circuit, ramp 10× faster (over 40 s) | 0.119 % | 0.2 % of it |

Both records are noise-free; the exact settings are in §8.

This is why `r_rc_transistor` integrates its ODE rather than using the engine,
and why it returns both responses. `voltage_V` is the ODE solution and
`voltage_quasistatic_V` is the frozen-time synthesis, so their difference is
exactly this error.

### 3.3 The rule of thumb

> Ask **where** a varying parameter sits, not how much memory the system has.
> A parameter that scales the output of a fixed subsystem costs nothing,
> however slow that subsystem is. A parameter inside a loop costs roughly the
> loop's time constant times the rate of drift, however fast the loop is.

---

## 4. The models

### Summary

| model | engine | ground truth `Z_true` | is the response exactly the synthesis of `Z_true`? |
|---|---|---|---|
| `r_rc` | frozen-time | analytic, constant | yes (LTI) |
| `r_rc_transistor` | RK4 ODE | frozen-time | no — first-order error, §3.2 |
| `legendre` | frozen-time | prescribed $\sum_p a_p(f)b_p(t)$ | yes — branch form, §3.1 |
| `block_oriented` | DFT filtering | best linear approximation, by Monte Carlo | linear branches: yes. Nonlinear: the response contains distortion by design |

### 4.1 `r_rc` — stationary reference

```math
Z(f) = R_s + \frac{R_p}{1 + j\omega R_p C}.
```

All parameters are constant, so the system is LTI and the frozen-time
synthesis is its exact steady-state response. Use it to check that an
estimator recovers a constant spectrum and **does not invent time
variation**. Ground truth is analytic. No paper is needed: this is the
textbook parallel RC with a series resistance.

### 4.2 `r_rc_transistor` — transistor-gated charge-transfer resistance

**Circuit.** A transistor $T$ sits in series with $R_p$ inside the parallel
branch:

```
R_s ---+--- R_p --- T ---+
       |                 |
       +------- C -------+
```

The transistor is modelled as a voltage-controlled resistance with a
**piecewise-linear** characteristic:

```math
R_T(V_g) = R_{T,\max}\,\big(1-\alpha\big), \quad
\alpha = \operatorname{clip}\!\left(\frac{V_g - V_\text{off}}{V_\text{on}-V_\text{off}},\,0,\,1\right),
\qquad R_{ct} = R_p + R_T .
```

The gate is a linear ramp, $V_g(t) = V_{g,0} + \dot V_g\, t$, so $R_{ct}(t)$
changes continuously. The defaults are $V_\text{off} = 1.6$ V and
$V_\text{on} = 2.5$ V.

**Engine and ground truth.** The ODE of §2.2 gives `voltage_V`. `Z_true` is the
frozen-time impedance $R_s + R_{ct}(t)/(1+j\omega R_{ct}(t)C)$, and
`voltage_quasistatic_V` is its synthesis. The error of the frozen-time
approximation is derived in §3.2.

**Things to know when defending it:**

- The characteristic has **kinks** at $V_\text{off}$ and $V_\text{on}$, where
  $dR_{ct}/dt$ jumps. Keep the ramp strictly inside the window -- from 5 % to
  95 % of it, say -- so that $R_{ct}(t)$ is smooth. A ramp that crosses a threshold makes
  the variation non-smooth, which no smooth-variation estimator should be
  asked to follow.
- The linear characteristic and the threshold values are modelling choices.
  They are *not* known to come from a paper ([§6](#6-attributions-to-check-against-the-papers)).
- Its value is that it is a circuit you can **build on the bench**. That makes
  it the bridge between simulation and measurement.

**Source.** The code attributes the circuit to Breugelmans *et al.* (2012),
Fig. 7. The paper is confirmed to exist and to concern time-varying ORP-EIS
with an instantaneous impedance. That **its Fig. 7 is this circuit has not
been checked.**

### 4.3 `legendre` — prescribed time-varying impedance (the oracle)

**Model.** The impedance at each excited frequency is a Legendre expansion in
normalised time:

```math
Z(f_k, t) = \sum_{p=0}^{P} a_p(f_k)\, b_p(t), \qquad
b_p(t_n) = L_p(x_n) - \overline{L_p(x)}\ \ (p\ge1),\quad b_0 = 1,\quad x_n = \frac{2n}{N}-1 .
```

There is an optional additive drift with its own expansion,
$v_\text{drift}(t) = \sum_q c_q\, b_q(t)$. You supply the complex coefficients
`impedance_coefficients[k, p]` $= a_p(f_k)$ and the real
`drift_coefficients[q]` $= c_q$.

**Why it is exact.** It is a sum of LTI "branches" $a_p(f)$ each scaled by a
real $b_p(t)$, which is exactly the form of §3.1. So the frozen-time synthesis
is the exact response, for any coefficients and any order. That is what makes
it an **oracle**. An estimator that fits this model, as BLTVA does, should get
back $a_p$ itself, limited only by noise.

**The mean subtraction, and why it matters for your BLTVA.** On a finite
sampled record, $\frac{1}{N}\sum_n L_p(x_n)$ is not exactly zero for $p\ge1$.
Without correcting for it, every time-varying column leaks a little into the
DC bin. `legendre_basis` subtracts the sample mean from every $p\ge1$. The
code attributes this step to Hallemans *et al.* (2021), after eq. (26).

The oracle and the estimator **must use the identical basis, mean subtraction
included**. Otherwise they disagree by $O(1/N)$, which looks like an estimator
error but is not one. During development this showed up as a floor of about
2e-5 under every recovery test at $N=10^5$. `legendre_basis` now lives in
`models/legendre.py`. When you write your BLTVA, import it from there or
reproduce it exactly.

**Source.** Hallemans *et al.* (2021): the BLTVA model, with the time-varying
transfer function expanded in Legendre polynomials.

### 4.4 `block_oriented` — nonlinear branches, and the best linear approximation

**Model.** Parallel branches, each a Wiener–Hammerstein block followed by a
Legendre gain:

```math
y(t) = \sum_p b_p(t)\; f_{2,p}\!\Big( h_p * f_{1,p}\big(u(t)\big) \Big),
\qquad H_p(s) = \frac{G_p\,\omega_{n,p}^2}{s^2 + 2\zeta_p\omega_{n,p} s + \omega_{n,p}^2}.
```

Making $f_1$ or $f_2$ the identity gives a Wiener branch, a Hammerstein branch,
or a purely linear one. `hallemans_2021_system()` returns three branches, one
of each kind. The code attributes them to the paper's Section VII, with
natural frequencies 0.7, 1.0 and 1.5 Hz.

**Engine.** DFT filtering, §2.3. Both of its conditions apply: a whole number
of periods, and the harmonics of $f_1$ kept below $f_s/2$.

**Ground truth: the BLA, not the linear part.** With nonlinear branches the
response contains distortion, and no linear model reproduces it. The quantity
an estimator should recover is the **best linear approximation** (BLA) of
each branch. Because the static nonlinearities shift it, the BLA is **not**
$H_p$ (which is returned as `H_linear` and is only the BLA when the branch is
linear). `best_linear_approximation` computes it the standard way, by Monte
Carlo over random-phase realisations of the multisine:

```math
G_p(f_k) = \mathbb{E}_{\phi}\!\left[ \frac{Y_p(k)}{U(k)} \right]
\approx \frac{1}{R}\sum_{r=1}^{R} \frac{Y_p^{[r]}(k)}{U^{[r]}(k)} .
```

Each branch is driven separately, before its gain. With random phases, the
nonlinear distortion is uncorrelated with the input and averages away, while
the linear response does not. This is the BLA framework of Pintelon and
Schoukens (2012). The function also returns the standard error of the mean,
`scatter`, so you can check that $R$ was large enough.

The time-varying target is then $Z(f,t) = \sum_p G_p(f)\,b_p(t)$
(`frozen_impedance`). For linear branches $G_p = H_p$ and this reduces exactly
to the Legendre model. That is the identity checked in §3.1.

**Source.** Hallemans *et al.* (2021), Sections II and VII. See §6 for the
equation numbers and coefficients to confirm.

---

## 5. Trends: disturbances, not systems

`trends.py` provides slow voltage excursions to **add** to a response. Unlike
the models, they are not part of the system: they are the disturbance a
trend-removal step must take out from under the lowest tones. Each function
takes the time axis first and returns volts. By default the mean is removed
(the estimators all ignore the DC bin), and `peak_to_peak` rescales the trend
so that shapes can be compared at equal severity.

| trend | form | source | removable by a polynomial? |
|---|---|---|---|
| `polynomial_trend` | Legendre (or monomial) polynomial | this package's coefficients | exactly, at its own order |
| `relaxation_trend` | step response of $K/(s^2+c_1s+c_0)$, at time $2t$ | Hallemans *et al.* (2022) | depends on the record length |
| `logit_trend` | Nernst term $\tfrac{RT}{F}\ln\frac{x}{1-x}$ with $x$ linear in $t$ | built from first principles | no — infinite slope at the ends |

Two points to make when defending these:

- **The relaxation trend's difficulty depends on the record length.** At its
  published coefficients it rings with a period near 1000 s. A short record
  sees only the smooth leading edge, which a polynomial removes easily; a long
  one sees whole cycles. The code's docstring tabulates the residue left by a
  degree-8 fit: 8.7e-11 at 160 s, 3.2e-2 at 2000 s (development result).
- **The logit trend is hard reproducibly.** Its shape is fixed in normalised
  time, so its difficulty does not change with the record length. That makes
  it the fair one to compare across studies.

The underdamped, critically damped and overdamped branches of the step
response are all implemented, so changing the coefficients never returns NaN.

---

## 6. Attributions to check against the papers

These were written into the code's docstrings during the generated
development and **could not be checked against the PDFs**. You studied the
BLTVA papers yourself, so you are better placed to confirm them. Correct the
code where any of them is wrong.

| # | claim in the code | where | check |
|---|---|---|---|
| 1 | the transistor circuit is Breugelmans *et al.* (2012) **Fig. 7** | `r_rc_transistor.py` | figure number, and whether the paper's circuit matches |
| 2 | transistor thresholds $V_\text{off}=1.6$ V, $V_\text{on}=2.5$ V and a linear characteristic | `r_rc_transistor.py` | almost certainly a modelling choice; say so |
| 3 | mean subtraction of the Legendre basis, quoted from Hallemans (2021) **after eq. (26)** | `legendre.py` | the quotation and the equation number |
| 4 | block-oriented class = Hallemans (2021) **Section II, Fig. 2**; Wiener–Hammerstein = **Fig. 7** | `block_oriented.py` | figure numbers |
| 5 | the three branches and their coefficients = **Section VII, eq. (90)–(91g)**, natural frequencies read as Hz | `block_oriented.py` | every coefficient, and the Hz/rad·s⁻¹ reading |
| 6 | relaxation trend = Hallemans (2022) **eq. (31)–(32)**, $K=5.92\times10^{-4}$, $c_1=1.25\times10^{-3}$, $c_0=9.87\times10^{-6}$, evaluated at $2t$ | `trends.py` | the coefficients and the factor 2 |
| 7 | BLTVA models the trend as a Legendre expansion, **eq. (58)** | `trends.py` | equation number |
| 8 | article numbers **6503814** (2021) and **6501711** (2022) | old top-level README | neither was found. The 2021 DOI is confirmed (see §9) |

---

## 7. What to test

These are the properties that pin each piece down. They are listed here so
that you can write the tests yourself. Each comes with a closed form or an
independent check to test against, so none of them needs a magic tolerance.

**Engine**
- One tone through a constant $Z$: the output amplitude is $A|Z|$ and the phase
  shift is $\arg Z$.
- The same seed gives the same noise, and a different seed gives different noise.
- `noise_std_A` leaves the voltage unchanged.

**`r_rc`**
- FFT-EIS on a whole-period record returns the analytic $Z(f)$ to round-off.

**`r_rc_transistor`**
- A static gate makes the system LTI: FFT-EIS must return the analytic
  impedance, which also confirms the initial state.
- Under step refinement, RK4 converges at fourth order. Compare at a fixed
  *instant*, not at the last sample, because the grids differ.
- Halving the ramp rate halves `voltage_V - voltage_quasistatic_V`, with the
  ramp kept inside $(V_\text{off}, V_\text{on})$.
- Adding the first-order term of §3.2 removes the residual (§8 shows how).

**`legendre`**
- `legendre_basis`: column 0 is all ones, and columns $p\ge1$ have zero mean.
- With $P=0$ the model reduces to a stationary one.
- Two engines agree: a linear `block_oriented` system and `simulate_legendre`
  with $a_p = H_p$ match to round-off (§8).

**`block_oriented`**
- For linear branches, `best_linear_approximation` returns `H_linear`.
- `scatter` falls as $1/\sqrt{R}$.
- Aliasing: doubling $f_s$ does not change the excited bins, provided
  $d\,f_{\max} < f_s/2$.

**`trends`**
- A polynomial trend is removed exactly by a fit of its own order.
- The logit trend's residue after a fixed-order fit is independent of the
  record length.

---

## 8. Reproducing the two engine checks

The two **verified** numbers in §3 come from this script. Run it from the
repository root.

```python
import numpy as np
from deistools.simulation import (simulate_legendre, simulate_block_oriented,
                                  simulate_r_rc_transistor, Branch, SecondOrderLowpass)

# 1. Branch form is exact: DFT filtering + gain  vs  frozen-time synthesis
Tp, P, fs = 20.0, 4, 200.0
f = np.array([1, 3, 5, 9, 13, 17, 27]) / Tp
A = np.full(f.size, 0.01)
ph = np.random.default_rng(3).uniform(0, 2 * np.pi, f.size)
branches = [Branch(SecondOrderLowpass.from_hz(0.7, 0.15, 1.0)),
            Branch(SecondOrderLowpass.from_hz(1.0, 0.10, 0.2)),
            Branch(SecondOrderLowpass.from_hz(1.5, 0.15, 0.2))]
bo = simulate_block_oriented(Tp * P, 1 / fs, f, A, ph, branches)
lg = simulate_legendre(Tp * P, 1 / fs, f, A, ph, bo["H_linear"])
print(np.abs(bo["voltage_V"] - lg["voltage_V"]).max() / np.abs(lg["voltage_V"]).max())
# -> ~4.5e-14

# 2. A loop is first order: the correction V1 = C g' Zp^3 I reproduces ODE - frozen
Tp, P, fs = 20.0, 20, 1000.0
f = np.array([1, 3, 5, 9, 13, 17, 27, 39, 59, 89, 133, 201, 301]) / Tp
A = np.full(f.size, 0.01)
ph = np.random.default_rng(3).uniform(0, 2 * np.pi, f.size)
T, C = Tp * P, 1e-3
s = simulate_r_rc_transistor(T, 1 / fs, f, A, ph, 10.0, 30.0, C, 1.645, 0.81 / T, 70.0)
t, R = s["time_s"], s["R_ct_t"]
g, gdot = 1 / R, np.gradient(1 / R, t)
Zp = 1.0 / (g[None, :] + 1j * 2 * np.pi * f[:, None] * C)
V1 = (A[:, None] * np.real(C * gdot * Zp**3
      * np.exp(1j * (2 * np.pi * f[:, None] * t + ph[:, None])))).sum(0)
resid = s["voltage_V"] - s["voltage_quasistatic_V"]
m = slice(t.size // 20, 19 * t.size // 20)          # skip the record edges
print(np.std(resid[m]) / np.std(s["voltage_V"][m]),  # -> ~1.2e-4  (0.012 %)
      np.std(resid[m] - V1[m]) / np.std(resid[m]))   # -> ~0: V1 is the whole error
```

---

## 9. References

Each entry is marked with how far it was checked in October 2026.

- **Hallemans, N., Pintelon, R., *et al.*, Lataire, J.** (2021). Best linear
  time-varying approximation of a general class of nonlinear time-varying
  systems. *IEEE Trans. Instrum. Meas.* **70**.
  doi:[10.1109/TIM.2021.3086891](https://doi.org/10.1109/TIM.2021.3086891).
  *Title, volume and DOI confirmed.* The BLTVA model; the basis of
  `legendre` and `block_oriented`.
- **Hallemans, N., Pintelon, R., Zhu, X., Collet, T., Dabiri Havigh, M.,
  Wouters, B., Revilla, R. I., Claessens, R., Ramharter, K., Hubin, A., &
  Lataire, J.** (2022). Trend removal in measurements of best linear
  time-varying approximations—with application to operando electrochemical
  impedance spectroscopy. *IEEE Trans. Instrum. Meas.* **71**, 1–11.
  *Title, authors and volume confirmed; article number not.* The source of
  `relaxation_trend`.
- **Breugelmans, T., Lataire, J., Muselle, T., Tourwé, E., Pintelon, R., &
  Hubin, A.** (2012). Odd random phase multisine electrochemical impedance
  spectroscopy to quantify a non-stationary behaviour: theory and validation
  by calculating an instantaneous impedance value. *Electrochim. Acta* **76**,
  375–382. doi:[10.1016/j.electacta.2012.05.051](https://doi.org/10.1016/j.electacta.2012.05.051).
  *Confirmed; the figure number in §6 is not.* Cited for the transistor circuit.
- **Pintelon, R., & Schoukens, J.** (2012). *System Identification: A Frequency
  Domain Approach*, 2nd ed. Wiley–IEEE Press. *Standard reference, not
  re-checked.* The best linear approximation and random-phase multisines.

---

## Adding a model

Decide the engine with §3, before writing any code:

1. Write the system so that every time-varying parameter is visible. If each
   one scales the **output** of a time-invariant subsystem, frozen time is
   exact: write an `impedance_fn(f, t) -> (n_freq, N)` and call
   `engine.simulate`.
2. If a varying parameter sits **inside a loop**, estimate
   $\tau_\text{loop}\,|d\ln p/dt|$. If it is far below the precision you need,
   use the engine anyway and *state the estimate*. Otherwise integrate the
   equation, as `r_rc_transistor` does, and return the frozen-time synthesis
   alongside so the gap stays visible.
3. If the system is nonlinear, the target is a BLA, not an impedance you
   prescribe. Follow `block_oriented`.
4. Export the new `simulate_*` from `models/__init__.py` and `simulation/__init__.py`.
