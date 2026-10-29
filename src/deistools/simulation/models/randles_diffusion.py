"""
Randles circuit with finite-space solid diffusion, on a charging electrode.

    R_s ---+--- R_ct(t) --- Z_d(f,t) ---+
           |                            |
           +---------- C_dl ------------+

``Z_d`` is the diffusion impedance of an insertion electrode of thickness L
with a *blocking* far boundary -- a battery particle, which stores lithium
rather than passing it on:

    Z_d(f, t) = R_d(t) * coth(sqrt(j*w*tau_d)) / sqrt(j*w*tau_d),  tau_d = L^2/D

Low-frequency limit ``Z_d -> R_d/3 + R_d/(j*w*tau_d)``: the 45-degree Warburg
tail turns up into the intercalation capacitance. Jacobsen and West,
Electrochim. Acta 40 (1995) 255 give this and the cylindrical and spherical
cases; Ho, Raistrick and Huggins, J. Electrochem. Soc. 127 (1980) 343 derive
the planar one used here.

A DC current charges the electrode, so the state of charge drifts through the
record and takes two of the parameters with it:

* ``R_d`` is proportional to the OCV slope ``dU/dSOC``. From
  ``R_d = (dU/dc) L / (n F D)`` with ``dU/dc = (dU/dSOC)/c_max``, everything
  except the slope is constant, so ``R_d(t)`` is just ``R_d_ref`` rescaled by
  the slope at the present SOC.
* ``R_ct`` follows the Butler-Volmer concentration factor
  ``i_0 ~ sqrt(x (1-x))``, so ``R_ct ~ 1/sqrt(x (1-x))``: shallowest in the
  middle of the window, rising towards both ends.

``tau_d``, ``R_s`` and ``C_dl`` are held constant. That is a modelling choice,
and the reason for it is the next section.

Why this model needs no ODE solve
---------------------------------
``r_rc_transistor`` cannot use the frozen-time engine: its ``Z`` depends
nonlinearly on a drifting ``R_ct`` that sits inside the C loop, so the
quasi-static response is wrong by about 1-2% rms and the model has to
integrate its circuit equation instead. One might expect diffusion to be worse
still, since its memory ``tau_d`` is seconds to hours rather than
milliseconds.

It is not, and that is worth being precise about. The rule is the one in
``simulation/README.md``: frozen time is exact when the time-varying
parameters multiply the *output* of time-invariant subsystems -- the
block-diagram form of Hallemans TIM 2021 Fig. 1 -- and approximate when one
sits inside a feedback loop. Both parameters here are in branch form:

* ``R_d(t)`` is a pure output gain on a diffusion kernel that does not itself
  change, because ``tau_d`` is constant. The surface concentration responds to
  current through a fixed operator and the OCV slope converts it to a voltage
  afterwards, so the two sums commute exactly as they do in ``legendre.py``.
  Diffusion's long memory never enters, which is the part that surprises.
* ``R_ct(t)`` is memoryless.

What is left is the double-layer loop, which both parameters do sit inside, so
the quasi-static error is not zero -- it is of order ``tau_loop`` times the
rate at which the parameters drift, with
``tau_loop = C_dl (R_ct + R_d/3)``. ``test_randles_diffusion.py`` measures it
by integrating the circuit as a genuine time-varying ODE, through the modal
expansion of the element,

    coth(x)/x = 1/x^2 + sum_{m>=1} 2/(x^2 + m^2 pi^2),   x = sqrt(s tau_d)

in which every mode is one scalar ODE, and confirms that it is first order in
the drift rate and first order in ``C_dl``, as that estimate requires. On the
hardest record the tests use -- a 20C charge across the OCV inflection, where
``R_d`` swings by a factor of eight and ``tau_loop`` is 0.14 s -- it is
3.5e-03 rms. Divide by the ratio of C rates for anything gentler: a 0.5C
charge of the same cell is of order 1e-04, which is two to three orders below
the precision of the estimators under test and the reason this model can skip
the ODE that ``r_rc_transistor`` cannot.

So ``voltage_V`` here *is* the frozen-time synthesis, and ``Z_true`` is exactly
the impedance it was built from. There is no ``voltage_quasistatic_V`` to
compare against, because there is no second response: if you need the true
one at a rate where the difference matters, the integrator in the test file is
the worked example.

The other thing that would break it is letting ``tau_d`` drift, which
modulates the kernel rather than its gain. That is deliberately not offered,
and it is a milder effect than it looks -- measured at 0.05% rms for a
threefold change in D across a record -- because the blocking boundary's
infinite-memory part is the ``m = 0`` mode, whose pole is zero and therefore
independent of D. The D-sensitive modes are the fast ones, and they track D
adiabatically. Long memory and parameter sensitivity live in different modes
and never bite at once.

The drift
---------
With ``include_dc`` the DC charging current appears in ``current_A`` and its
consequences appear in ``voltage_V``: the open-circuit voltage ``U(SOC(t))``
and the DC polarisation ``i_dc (R_s + R_ct + R_d/3)``. This is the
nonzero-mean multisine of operando practice.

It matters because it is the first trend in this package that was not written
down by someone who knew what the estimator assumes. ``legendre.py``'s drift
is a Legendre polynomial, which is BLTVA's own trend basis; a charge curve has
a plateau and an inflection and belongs to no polynomial of low order. Use
``include_dc=False`` to keep the same parameter trajectories with no trend, so
that the cost of the trend can be separated from the cost of the variation.
"""
import numpy as np

from ..engine import simulate as _simulate


def default_ocv(soc):
    """A synthetic open-circuit voltage curve: falling, one plateau, one step.

    **This is not any real material.** The functional form -- a constant, a
    decaying exponential, a linear term and a tanh step -- is the ansatz that
    published OCV fits use (for instance the graphite OCP in the PyBaMM
    Chen2020 parameter set, which is the same form with three tanh steps), but
    the coefficients here were chosen by eye to be smooth and analytically
    differentiable, not fitted to data. It lands in a graphite-like voltage
    window, 0.04-0.18 V over 0.1 < SOC < 0.9, and that is the end of the
    resemblance.

    What a real graphite curve has and this one does not is **staging**: three
    or four plateaus rather than one, so ``|dU/dSOC|`` has several sharp
    minima instead of a single broad one. That is not a cosmetic difference
    here, because ``R_d`` is proportional to ``|dU/dSOC|``. On the record of
    example 24 the curve below swings ``R_d`` by a factor of 7 with features
    400 s wide; the same ansatz with three steps swings it by a factor of 90
    with features 247 s wide, against a 200 s period for the lowest tone. The
    second case puts most of its variation *inside one period of the tone that
    is supposed to measure it*, which is the assumption every estimator in this
    package rests on.

    So the smooth curve is the default deliberately: it keeps the system inside
    the regime the estimators claim to handle. A staged curve is the right
    thing to pass as ``ocv_fn`` when the question is what happens at the edge
    of that regime -- and it is a question worth asking, because real graphite
    does stage and real operando measurements do run across the transitions.
    """
    soc = np.asarray(soc, dtype=float)
    return (0.12 + 0.30 * np.exp(-25.0 * soc) - 0.05 * soc
            - 0.035 * np.tanh((soc - 0.52) / 0.07))


def default_ocv_slope(soc):
    """``d(default_ocv)/d(SOC)``, analytically."""
    soc = np.asarray(soc, dtype=float)
    return (-7.5 * np.exp(-25.0 * soc) - 0.05
            - 0.035 / 0.07 / np.cosh((soc - 0.52) / 0.07) ** 2)


def _slope(ocv_fn, ocv_slope_fn, soc):
    if ocv_slope_fn is not None:
        return np.asarray(ocv_slope_fn(soc), dtype=float)
    h = 1e-6
    return (np.asarray(ocv_fn(soc + h), dtype=float)
            - np.asarray(ocv_fn(soc - h), dtype=float)) / (2 * h)


def diffusion_impedance(f, R_d, tau_d):
    """Finite-space diffusion impedance, blocking far boundary.

    ``f`` is 1-D of length n_freq. ``R_d`` may be a scalar or an array of
    shape (N,), giving a result of shape (n_freq, 1) or (n_freq, N). ``coth``
    is evaluated as ``1/tanh``, which saturates cleanly to 1 at high
    frequency, so no asymptotic branch is needed.
    """
    u = np.sqrt(1j * 2 * np.pi * np.atleast_1d(f)[:, None] * tau_d)
    return np.atleast_1d(R_d)[None, :] * (1.0 / np.tanh(u)) / u


def impedance(f, R_s, C_dl, R_ct, R_d, tau_d):
    """Frozen-time Randles impedance, shape (n_freq, N) for array R_ct, R_d."""
    Z_branch = np.atleast_1d(R_ct)[None, :] + diffusion_impedance(f, R_d, tau_d)
    jw = 1j * 2 * np.pi * np.atleast_1d(f)[:, None]
    return R_s + 1.0 / (jw * C_dl + 1.0 / Z_branch)


def simulate_randles_diffusion(duration_s, sampling_time_s, frequencies,
                               amplitudes, phases,
                               R_s, C_dl, R_ct_ref, R_d_ref, tau_d,
                               soc_start=0.2, c_rate=1.0,
                               ocv_fn=None, ocv_slope_fn=None,
                               include_dc=True,
                               noise_std_V=0.0, noise_std_A=0.0, seed=None):
    """Simulate a charging insertion electrode under multisine excitation.

    ``R_ct_ref`` and ``R_d_ref`` are the values at ``soc_start``; both are
    rescaled over the record as described in the module docstring. ``c_rate``
    is in units of 1/h, so SOC advances by ``c_rate * duration_s / 3600``.

    Returns the usual ``time_s``, ``voltage_V``, ``current_A``, ``Z_true``
    and ``frequencies_Hz``, plus:

    ``soc_t``, ``R_ct_t``, ``R_d_t``
        the state of charge and the two parameters it drives.
    ``drift_V``
        the DC part of the voltage: ``U(SOC(t)) - U(SOC(0))`` plus the DC
        polarisation. Zero when ``include_dc`` is False.
    ``current_dc_A``, ``capacity_As``
        the DC charging current, a scalar, and the capacity it is derived
        from. ``current_dc_A`` is zero when ``include_dc`` is False;
        ``capacity_As`` is reported either way.

    ``Z_true`` is the frozen-time impedance, which for this model is also the
    exact impedance of the response -- see the module docstring.
    """
    if ocv_fn is None and ocv_slope_fn is None:
        ocv_fn, ocv_slope_fn = default_ocv, default_ocv_slope
    elif ocv_fn is None:
        raise ValueError("ocv_slope_fn was given without ocv_fn")

    N = int(round(duration_s / sampling_time_s))
    t = np.arange(N) * sampling_time_s
    frequencies = np.asarray(frequencies)

    soc_t = soc_start + c_rate * t / 3600.0
    if soc_t.min() <= 0.0 or soc_t.max() >= 1.0:
        raise ValueError(
            f"SOC runs from {soc_t.min():.3f} to {soc_t.max():.3f}, outside "
            "(0, 1); reduce c_rate, duration_s or soc_start. R_ct diverges at "
            "both ends of the window and the OCV curve is not defined there.")

    # R_d tracks the OCV slope; R_ct tracks the Butler-Volmer concentration
    # factor. Both are normalised to their reference value at soc_start, so
    # only the *shape* of the curves enters and the constants cancel.
    slope = _slope(ocv_fn, ocv_slope_fn, soc_t)
    slope_ref = float(_slope(ocv_fn, ocv_slope_fn, np.array(soc_start)))
    R_d_t = R_d_ref * np.abs(slope / slope_ref)

    conc = np.sqrt(soc_t * (1.0 - soc_t))
    R_ct_t = R_ct_ref * np.sqrt(soc_start * (1.0 - soc_start)) / conc

    def impedance_fn(f, _t):
        return impedance(f, R_s, C_dl, R_ct_t, R_d_t, tau_d)

    result = _simulate(duration_s, sampling_time_s, frequencies,
                       amplitudes, phases, impedance_fn,
                       noise_std_V, noise_std_A, seed)

    # The capacity is not a free parameter: the low-frequency limit of Z_d
    # carries an intercalation capacitance tau_d/R_d = dQ/dU, and
    # dQ/dU = (dQ/dSOC)/(dU/dSOC) = Q/slope, so Q = tau_d*|slope|/R_d. Because
    # R_d was defined proportional to the slope, that ratio is the same at
    # every SOC, as a capacity should be.
    capacity_As = tau_d * abs(slope_ref) / R_d_ref

    if include_dc:
        i_dc = c_rate * capacity_As / 3600.0
        ocv = np.asarray(ocv_fn(soc_t), dtype=float)
        polarisation = i_dc * (R_ct_t + R_d_t / 3.0)
        drift_V = (ocv - ocv[0]) + (polarisation - polarisation[0])
        result['voltage_V'] = result['voltage_V'] + drift_V
        result['current_A'] = result['current_A'] + i_dc
    else:
        i_dc = 0.0
        drift_V = np.zeros(N)

    result.update({
        'soc_t': soc_t,
        'R_ct_t': R_ct_t,
        'R_d_t': R_d_t,
        'drift_V': drift_V,
        'current_dc_A': i_dc,
        'capacity_As': capacity_As,
        'R_s': R_s,
        'C_dl': C_dl,
        'tau_d': tau_d,
    })
    return result
