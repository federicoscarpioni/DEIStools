# deistools.simulation

Synthetic signal generator for testing non-stationary impedance estimation algorithms (BLTVA, DMFA, LIA-EIS).

## Structure

```
simulation/
├── engine.py          — multisine generator and voltage synthesis
└── models/
    ├── r_rc.py            — stationary R + RC (reference baseline)
    ├── r_rc_transistor.py — non-stationary R + (R+T)||C
    └── legendre.py        — Legendre polynomial oracle model
```

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

It does **not** hold when the impedance depends nonlinearly on a time-varying parameter. In `r_rc_transistor.py`, `Z` is a nonlinear function of `R_ct(t)`, so the frozen-time response is merely a quasi-static approximation of the governing equation — wrong by around 1% rms at realistic gate-ramp rates. That model therefore bypasses the engine and integrates the circuit ODE directly.

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
from deistools.simulation import simulate_r_rc, simulate_r_rc_transistor, simulate_legendre

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

Model-specific keys are added by each model: `R_ct_t`, `V_gate_t`, `voltage_quasistatic_V` (transistor); `drift_V`, `impedance_coefficients`, `drift_coefficients` (Legendre); `R_s`, `R_p`, `C` (both circuit models).

## Adding a new model

1. Create `models/your_model.py`.
2. Write an `impedance_fn(f, t)` that returns `(n_freq, N)` complex.
3. Call `from ..engine import simulate` and pass your function.
4. Export `simulate_your_model` from `models/__init__.py` and `simulation/__init__.py`.
