# deistools.simulation

Synthetic signal generator for testing non-stationary impedance estimation algorithms (BLTVA, DMFA, LIA-EIS).

## Structure

```
simulation/
├── engine.py          — multisine generator and voltage synthesis
├── trends.py          — additive drifts, for testing trend removal
└── models/
    ├── r_rc.py            — stationary R + RC (reference baseline)
    ├── r_rc_transistor.py — non-stationary R + (R+T)||C
    └── legendre.py        — Legendre polynomial oracle model
```

## Systems against disturbances

The models below are the *system* -- what an estimator is trying to measure.
`trends.py` is the opposite: additive drift laid on top of the response, the
disturbance that detrending has to remove. Three shapes, chosen to fail a
polynomial fit in different ways:

| | source | removable by a polynomial? |
|---|---|---|
| `polynomial_trend` | this package | exactly, at its own order |
| `relaxation_trend` | Hallemans 2022 eq. (31)-(32), its coefficients | only if little of its ringing fits the record |
| `logit_trend` | built here, Nernst relation | no, at any order or record length |

Every one takes its shape parameters by keyword, so the published defaults
can be changed one at a time, and `peak_to_peak=` puts them all at the same
severity for a sweep.

Worth knowing before trusting the second row: at the published coefficients
that trend rings with a period near 1000 s, so a 160 s record sees only its
smooth leading edge and a degree-8 polynomial removes it to 8.7e-11. On a
2000 s record the same trend leaves 3.2e-02. The difficulty is a property of
the record as much as of the trend.

## How it works

The **engine** (`engine.py`) handles the three steps common to the frozen-time models:

1. Build the multisine current from the specified frequencies, amplitudes, and phases.
2. Compute the voltage response tone-by-tone using the frozen-time impedance `Z(f, t)`.
3. Add Gaussian noise to voltage and current if requested.

Each such **model** provides an `impedance_fn(f, t)` returning a `(n_freq, N)` complex array. The engine does not know or care how the impedance varies with time — that is entirely the model's concern.

### When frozen-time synthesis is exact, and when it is not

For a **linear time-varying** system expanded in *real* basis functions, the frozen-time synthesis is not an approximation — it is algebraically identical to the block-diagram model of Hallemans TIM 2021 Fig. 1. Because `b_p(t)` is real the two sums commute:

```
Σ_p b_p(t)·Σ_k A_k Re{G_p(jω_k)e^{jθ}}  ==  Σ_k A_k Re{(Σ_p G_p(jω_k)b_p(t))·e^{jθ}}
```

so `legendre.py` reproduces the exact LTV response with no ODE solve. This is what makes it usable as an oracle.

It does **not** hold when a time-varying parameter sits inside a feedback loop. In `r_rc_transistor.py`, `Z` is a nonlinear function of `R_ct(t)` and `R_ct` sits inside the C loop, so the frozen-time response is merely a quasi-static approximation of the governing equation — wrong by 1–2% rms at realistic gate-ramp rates. That model therefore bypasses the engine and integrates the circuit ODE directly.

Between those two sits `randles_diffusion.py`, and it is worth knowing why, because the intuition that long memory makes frozen time worse turns out to be wrong. Both of its varying parameters are in branch form, so the diffusion element's memory — seconds to hours, against milliseconds for an RC — never enters the error at all. What is left is only the double-layer loop:

```
residual ~ tau_loop * |d(ln p)/dt|,     tau_loop = C_dl * (R_ct + R_d/3)
```

measured to be first order in each factor separately. On the hardest record its tests use, a 20C charge across the OCV inflection, that is 3.5e-03 rms; a realistic 0.5C charge is of order 1e-04. So it uses the engine, and its error is documented rather than integrated away.

The rule of thumb that follows: ask *where* a varying parameter sits, not how much memory the system has. A parameter multiplying a subsystem's output costs nothing however slow that subsystem is; a parameter inside a loop costs the loop's time constant times the drift rate, however fast the loop is.

## Models

### `r_rc` — stationary reference

Series resistance `R_s` plus a parallel RC element with constant parameters:

```
Z(f) = R_s + R_p / (1 + j·2π·f·R_p·C)
```

Use this to verify that the estimator recovers a known, time-invariant impedance spectrum correctly.

### `r_rc_transistor` — transistor-gated R_ct

The parallel branch contains `R_p` in series with a transistor `T`:

```
R_s ---+--- R_p --- T ---+
       |                 |
       +------- C -------+
```

The transistor is modelled as a voltage-controlled resistance:

- `V_gate >= V_on (2.5 V)`: T is a short circuit → `R_ct = R_p`
- `V_gate <= V_off (1.6 V)`: T is fully blocking → `R_ct = R_p + R_T_max`
- Between `V_off` and `V_on`: linear interpolation

Driving `V_gate` with a linear ramp (`V_gate(t) = V_gate_start + V_gate_rate * t`) produces a continuously time-varying `R_ct(t)` without any artificial parameter forcing — the variation comes from the circuit topology.

This is the circuit of Breugelmans et al., Electrochim. Acta 76 (2012) 375, Fig. 7, and it is the only model here that can be **built on the bench as well as simulated**.

**It is solved as a genuine time-varying ODE**, not by frozen-time synthesis. Under galvanostatic excitation, with `v_C` across the parallel branch:

```
dv_C/dt = (i(t) − v_C/R_ct(t)) / C
v(t)    = R_s·i(t) + v_C(t)
```

a scalar linear ODE with a time-varying coefficient, integrated with fixed-step RK4 from the quasi-static initial state (so no startup transient is injected). The step size must stay well below `τ = R_ct·C`; a warning is raised otherwise.

Two extra keys come back:

- `voltage_quasistatic_V` — the frozen-time synthesis, so `voltage_V − voltage_quasistatic_V` is the quasi-static modelling error directly.
- `Z_true` — still the **frozen-time** impedance. That is the quantity an estimator is asked to recover, so it stays the ground truth even though it is not the response the circuit produces.

Because the response does not satisfy the estimators' model assumptions, this model is the one to use when asking *how far* a method degrades on a physically real system. Note `noise_std_A` models a noisy ammeter: it perturbs the recorded current, not the current driving the circuit.

### `randles_diffusion` — charging insertion electrode

The first system here with any diffusion in it, and the one most
electrochemists would want to validate against:

```
R_s ---+--- R_ct(t) --- Z_d(f,t) ---+
       |                            |
       +---------- C_dl ------------+

Z_d(f,t) = R_d(t) * coth(sqrt(j*w*tau_d)) / sqrt(j*w*tau_d),   tau_d = L^2/D
```

`Z_d` is finite-space diffusion with a **blocking** far boundary — a battery
particle, which stores lithium rather than passing it on. That gives a
45-degree Warburg tail which turns up into the intercalation capacitance
(`Z_d -> R_d/3 + R_d/(j w tau_d)`), so unlike every other model here `|Z|`
grows without limit towards DC instead of settling on a resistance.

A DC current charges the electrode, so SOC drifts through the record and takes
two parameters with it:

- `R_d` is proportional to the OCV slope `dU/dSOC`, since
  `R_d = (dU/dc) L / (n F D)` and everything but the slope is constant. On a
  plateau the slope is small, so the diffusion resistance is small and the
  intercalation capacitance large — correct physics, and it makes `R_d(t)`
  **non-monotone** when the record crosses a slope minimum.

  This makes the OCV curve the most load-bearing choice in the model: it sets
  the trend *and*, through its slope, the whole impedance variation. The
  built-in `default_ocv` is **synthetic, not fitted to any material** — one
  plateau and one step, in a graphite-like voltage window. Real graphite
  stages, with three or four plateaus, and substituting such a curve via
  `ocv_fn` swings `R_d` by ~90x with features narrower than the lowest tone's
  period, which breaks the slowly-varying premise the estimators need. The
  smooth default keeps the system inside that premise on purpose; see
  `default_ocv`'s docstring.
- `R_ct` follows the Butler-Volmer concentration factor `i_0 ~ sqrt(x(1-x))`.

The capacity is not a free parameter: the low-frequency limit fixes
`Q = tau_d |dU/dSOC| / R_d`, which is the same at every SOC precisely because
`R_d` was defined proportional to the slope.

Two things it gives that nothing else here does. The **impedance is not
RC-shaped**, and the time variation is strongly frequency-dependent — about
6x at the bottom tone and 1.0x at the top on the example record, so it is a
low-frequency problem by construction. And the **trend is a charge curve**,
not a polynomial: `legendre.py`'s drift is written in BLTVA's own trend basis
and so costs nothing, while this one has to be bought in `Nq`.

It uses the frozen-time engine. See the previous section for why that is
legitimate here and not for the transistor, and
`tests/test_randles_diffusion.py` for the measurement.

### `legendre` — Legendre polynomial oracle

The transfer function at each excited frequency is a Legendre polynomial expansion in time:

```
H(f_k, t) = Σ_{p=0}^{P}  a_p(f_k) · L_p(2t/T - 1)
```

An optional additive voltage drift follows its own Legendre expansion:

```
v_drift(t) = Σ_{q=0}^{Q}  b_q · L_q(2t/T - 1)
```

This is exactly the signal model that BLTVA assumes. Running BLTVA on a signal from this model is an oracle test: the estimated coefficients should match the input `impedance_coefficients` and `drift_coefficients` exactly (up to noise).

## Usage

```python
import numpy as np
from deistools.simulation import (simulate_r_rc, simulate_r_rc_transistor,
                                  simulate_randles_diffusion, simulate_legendre)

frequencies = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
amplitudes  = np.full(5, 0.01)
phases      = np.zeros(5)

# Stationary reference
result = simulate_r_rc(
    duration_s=100, sampling_time_s=1e-3,
    frequencies=frequencies, amplitudes=amplitudes, phases=phases,
    R_s=0.01, R_p=0.05, C=1.0,
    noise_std_V=1e-5,
)

# Transistor-gated non-stationary
result = simulate_r_rc_transistor(
    duration_s=100, sampling_time_s=1e-3,
    frequencies=frequencies, amplitudes=amplitudes, phases=phases,
    R_s=0.01, R_p=0.05, C=1.0,
    V_gate_start=1.4, V_gate_rate=0.012,   # ramps from blocking to conducting
    R_T_max=10.0,
    noise_std_V=1e-5,
)

# Charging insertion electrode, with diffusion
result = simulate_randles_diffusion(
    duration_s=2000, sampling_time_s=0.02,
    frequencies=frequencies, amplitudes=amplitudes, phases=phases,
    R_s=0.02, C_dl=1.0, R_ct_ref=0.015, R_d_ref=0.08, tau_d=20.0,
    soc_start=0.2, c_rate=0.5,       # 0.5C, so SOC runs 0.20 -> 0.48
    include_dc=True,                 # the charging current and the charge curve
    noise_std_V=2e-5,
)

# Legendre oracle (order 2 in time)
coeffs = np.array([
    [0.06 + 0j,   0.01 + 0j,  0.002 + 0j],   # f=0.1 Hz
    [0.05 + 0j,   0.008 + 0j, 0.001 + 0j],   # f=0.5 Hz
    # ...one row per frequency
])
result = simulate_legendre(
    duration_s=100, sampling_time_s=1e-3,
    frequencies=frequencies, amplitudes=amplitudes, phases=phases,
    impedance_coefficients=coeffs,
    drift_coefficients=[0.0, 0.002, 0.0],   # linear drift
    noise_std_V=1e-5,
)
```

All `simulate_*` functions return a dict with at minimum:
- `time_s` — sample times, shape `(N,)`
- `voltage_V` — simulated voltage, shape `(N,)`
- `current_A` — multisine current, shape `(N,)`
- `Z_true` — ground-truth (frozen-time) impedance, shape `(n_freq, N)`
- `frequencies_Hz` — excited frequencies

Model-specific keys are added by each model: `R_ct_t`, `V_gate_t`, `voltage_quasistatic_V` (transistor); `soc_t`, `R_ct_t`, `R_d_t`, `drift_V`, `current_dc_A`, `capacity_As` (diffusion); `drift_V`, `impedance_coefficients`, `drift_coefficients` (Legendre); `R_s` and the remaining circuit parameters (all three circuit models).

## Adding a new model

1. Create `models/your_model.py`.
2. Write an `impedance_fn(f, t)` that returns `(n_freq, N)` complex.
3. Call `from ..engine import simulate` and pass your function.
4. Export `simulate_your_model` from `models/__init__.py` and `simulation/__init__.py`.
