# DMFA implementation notes

!!! warning "Verification status"
    This implementation has **not** been checked against the original
    description of the method in the literature. It behaves as one expects —
    the parameter studies in example 12 show both predicted failure modes, and
    it agrees with STFT and BLTVA to within an order of magnitude on four
    simulated systems — but *behaving plausibly is not the same as being
    correct*.

    Checking it is an open item on the [roadmap](../roadmap.md). Until then,
    cross-check DMFA results against STFT and do not use DMFA alone for a
    published number.

## The filter belongs on the band's local axis

`elaboration_frequency_axis(n_points, T)` returns the frequency axis of an
extracted band: `n_points` bins of width $1/T$, centred on zero. That spans
`n_points`/$T$ Hz — in the worked example, 0.2 Hz.

The full spectrum axis spans $f_s$, which is 1000 Hz in the same example.

Build the passband on the wrong axis and its bandwidth is scaled wrongly by
$f_s T / n_{\text{points}}$, a factor of 2500 there. The result is not an
error, just a filter that does something entirely different from what was
asked. This is the easiest mistake to make with the method and the reason
`elaboration_frequency_axis` exists as a named function rather than being
inlined.

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
cost of representing a *changing* quantity with a finite band. No parameter
setting removes it, and the only correct response is to discard the affected
samples. This package uses the first and last tenth throughout.

## Band width is chosen from the bandwidth

`max_elaboration_points` reports the widest band that fits inside the spectrum
at every tone, which on a long record is almost the whole spectrum — 387 960
bins in the worked example. Using it would produce that many output instants
while the filter still limits the real resolution to far less.

The examples instead size the band to about eight bandwidths:

```python
n_points = int(round(8 * bandwidth * T))
n_points = min(n_points - n_points % 2, max_elaboration_points(indexes, N))
```

This is a choice of this package, not a rule from a paper. The band only needs
to be wide enough to contain the filter's skirts; making it wider costs output
points without adding information.

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
