# Lock-in amplifier

The classical instrument, implemented offline. For each excited tone the
record is multiplied by a cosine and a sine at that frequency, and the
products are low-pass filtered. What survives is that tone's slowly varying
complex amplitude, and the ratio of the voltage and current amplitudes is
$Z(f,t)$.

It is the most familiar of the four methods to an experimentalist, and — as
shown below — it turns out to be the continuous-time generalisation of
[STFT](stft.md) rather than a separate idea.

## The mathematics

Write the response near the $k$-th tone as a carrier with a slowly varying
phasor:

\[
v(t) \;=\; A_v(t)\,\cos\!\big(\omega_k t + \varphi_v(t)\big),
\qquad \omega_k = 2\pi f_k
\]

Mixing with the in-phase and quadrature references gives

\[
\begin{aligned}
v(t)\cos(\omega_k t) &= \tfrac{A_v}{2}\cos\varphi_v
   \;+\; \tfrac{A_v}{2}\cos(2\omega_k t + \varphi_v) \\
v(t)\sin(\omega_k t) &= -\tfrac{A_v}{2}\sin\varphi_v
   \;+\; \tfrac{A_v}{2}\sin(2\omega_k t + \varphi_v)
\end{aligned}
\]

Each product has a baseband term carrying the information and an image at
$2\omega_k$. A low-pass filter $h$ that rejects the image leaves

\[
X = \tfrac{A_v}{2}\cos\varphi_v, \qquad
Y = -\tfrac{A_v}{2}\sin\varphi_v
\]

and twice the complex combination is the phasor itself:

\[
V(t) \;=\; 2\big(X - \mathrm{j}Y\big) \;=\; A_v(t)\,e^{\mathrm{j}\varphi_v(t)}
\]

The same is done for the current, and

\[
Z(f_k, t) \;=\; \frac{V(t)}{I(t)}
\]

The factor of two and the one-sided convention both cancel in that ratio, so
they matter only if you want $V$ and $I$ to be meaningful on their own.

## Choosing the integrator — the whole method

The low-pass filter has two jobs. Rejecting the image at $2f_k$ is easy. The
binding requirement is the other one: **every other tone of the multisine**,
after mixing at $f_k$, lands at the difference frequency $|f_j - f_k|$, and
must be rejected too.

For a multisine whose tones are all harmonics of a base frequency $f_b$,
every such difference is a multiple of $f_b$. A **boxcar average over exactly
an integer number of periods of $f_b$** has spectral nulls at every multiple
of $1/(N_w \Delta t)$ — that is, at every one of those difference frequencies.
It rejects them all, exactly.

| integrator | worst rejection of the other 12 tones |
|---|---|
| boxcar, 1 base period | 315 dB (arithmetic limit) |
| boxcar, 2 base periods | 315 dB |
| Hann, 2 base periods | 116 dB |
| Hann, 4 base periods | 134 dB |

!!! warning "The window must span whole base periods of $f_b$, not of $f_k$"
    The nulls sit at multiples of $1/(N_w\Delta t)$ and only land on the
    harmonics if $N_w \Delta t$ is a whole number of base periods. Rounding
    $N_w$ to an odd number to centre the kernel on a sample detunes them: on a
    20 000-sample period that one extra sample costs about **230 dB** of
    rejection. `integrator_length` therefore returns the exact length and
    refuses a non-integer one. The resulting half-sample group delay is common
    to voltage and current and cancels in their ratio.

A Hann window has its nulls in the same places but a mainlobe twice as wide,
so over a single base period the nearest harmonic falls *inside* the mainlobe
and is not rejected at all. `integrator_length` refuses `window='hann'` with
`n_periods=1` for that reason.

The integrator length is also the **time resolution**: a window of $m$ base
periods cannot follow anything faster than $m/f_b$ seconds. That is the same
trade STFT makes with its window and DMFA with its bandwidth.

## Two demodulators

### Mixing (`method='mix'`)

The textbook route described above. Two FFT convolutions per frequency, so
the whole record at every sample costs a couple of seconds.

### Local least squares (`method='ls'`)

Instead of mixing, fit

\[
x(t) \approx a\cos\omega_k\tau + b\sin\omega_k\tau + c + d\,\tau,
\qquad \tau = t - t_0
\]

in a sliding window, and take $a - \mathrm{j}b$ as the phasor. Carrying the
offset $c$ and the **local slope $d$** is the point: the fit tracks drift
*within* the window instead of averaging over it.

That matters more than it sounds. On the transistor system:

| demodulator | median relative error |
|---|---|
| mixing, 1-period boxcar | 1.16e-03 |
| local least squares, same window | **1.09e-04** |

An order of magnitude, from the same data and the same window, because the
within-window averaging bias is what limits the mixing route on a drifting
system. The cost is one least-squares solve per reported instant, so it must
be evaluated on a decimated grid.

!!! note "The phasor is referenced to the window centre"
    `local_ls` centres $\tau$ on the window, so the phase it returns is
    relative to $t_0$, not to $t=0$. The reference cancels in $V/I$ because
    both are fitted on the same window — which is why `lia_impedance_ls` is
    correct — but a caller using `local_ls` on its own must undo the
    $e^{-\mathrm{j}\omega t_0}$ itself.

## Calling it

```python
from deistools.processing import MultiFrequencyAnalysis

mfa = MultiFrequencyAnalysis(frequencies, voltage, current, sampling_time)

Z, t = mfa.run_lia(n_periods=1, decimate=200)        # mixing
Z, t = mfa.run_lia(n_periods=1, method="ls", stride=20000)   # local LS

mfa.lia_tone_rejection(1)     # dB, one row per demodulated tone
mfa.lia_valid_mask()          # False where the window ran past the record
```

Or directly:

```python
from deistools.processing.lock_in_amp import integrator_length, lia_eis

samples_window = integrator_length(sampling_time, base_frequency=1/Tp,
                                   n_periods=1, window="boxcar")
Z, t = lia_eis(time, voltage, current, sampling_time, frequencies,
               samples_window, decimate=200)
```

## Edges

The output within half an integrator length of either end rests on a window
that extends past the data. This implementation reflects the signal at the
boundaries, which is exact for an envelope with zero slope there and good
otherwise, and `valid_mask` marks the affected samples so a fit can exclude
them.

This is the lock-in's version of [DMFA's edge
problem](dmfa.md#the-record-edges-are-unusable), but it is milder and
explicitly controllable, because the filtering happens in the time domain
where the boundary can be handled directly instead of being wrapped around by
a circular DFT.

## It is STFT, evaluated everywhere

A boxcar lock-in over exactly one base period and a rectangular STFT over the
same window are **the same estimator**. Verified on the transistor record:

```
max |Z_lia − Z_stft| / |Z_stft|  =  9e-14      (at the STFT window centres)
```

STFT is that estimator decimated to one point per window; the lock-in reports
at every sample. So the lock-in buys no accuracy over STFT at the instants
STFT reports — its median error over *all* instants is higher (1.2e-03 against
7.1e-04) precisely because it also reports between the centres, where the
within-window bias is larger.

What it buys is a continuous output, an explicit and adjustable integrator,
and — through `method='ls'` — a way to remove the within-window bias that
neither STFT nor plain mixing can.

## When to use it

- **You want the familiar instrument.** This is what a hardware lock-in does,
  so the parameters mean what an experimentalist expects them to mean.
- **You want a continuous $Z(f,t)$** rather than one value per window.
- **Your system drifts within a window** — use `method='ls'`, which is the
  only estimator here that removes that bias without assuming a global model
  the way BLTVA does.
- **You are heading towards online processing.** The mixing route is the one
  that becomes a streaming filter, though a causal integrator would replace
  the zero-phase one used here.

## Limits

- Time resolution is the integrator length, with the same floor as STFT: one
  base period.
- No uncertainty is reported, as with STFT and DMFA.
- Cost grows linearly with the number of tones, since each is demodulated
  separately. DMFA transforms once for all of them.
