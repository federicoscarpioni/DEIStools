# DMFA

Demodulated multi-frequency analysis. Transform the whole record once; around
each excited tone, cut out a band with a passband filter; transform that band
back. What comes out is that tone's complex amplitude as a function of time,
and voltage over current is $Z(f,t)$.

Nothing is windowed, so the time resolution is not tied to the multisine
period — it is set by the filter bandwidth instead.

!!! warning "Provisional"
    This implementation reproduces the behaviour one expects of the method,
    but it has **not** been checked against the original description in the
    literature. Verifying it is an open item on the [roadmap](../roadmap.md).
    Treat DMFA results as provisional and cross-check them against STFT.

## Calling it

DMFA is assembled from four functions rather than called as one.

```python
import numpy as np
from numpy.fft import fft, fftshift, fftfreq
from deistools.processing.dmfa_functions import (
    extract_impedance, max_elaboration_points,
    elaboration_frequency_axis, elaboration_time_axis)
from deistools.processing.passband_filters import create_fermi_dirac_fun

# 1. transform the whole record, zero frequency in the middle
ft_voltage = fftshift(fft(voltage)) / N
ft_current = fftshift(fft(current)) / N
spectrum_axis = fftshift(fftfreq(N, 1.0 / fs))

# 2. the bin of each excited tone
index_f0 = int(np.argmin(np.abs(spectrum_axis)))
indexes = (np.round(frequencies * T).astype(int) + index_f0).tolist()

# 3. the passband, on the band's LOCAL frequency axis
bandwidth = 0.5 * frequencies.min()
n_points = min(int(round(8 * bandwidth * T)),
               max_elaboration_points(indexes, N))
band_axis = elaboration_frequency_axis(n_points, T)
band_filter = create_fermi_dirac_fun(band_axis, 0, bandwidth, 2)

# 4. extract and divide
Z = extract_impedance(ft_voltage, ft_current, indexes, band_filter, n_points)
t = elaboration_time_axis(n_points, T)
```

!!! danger "Build the filter on the local axis"
    `elaboration_frequency_axis(n_points, T)` spans `n_points` bins of width
    $1/T$ — about 0.2 Hz in the example above. The full spectrum axis spans
    $f_s$, which is 1000 Hz. Build the filter on the wrong one and its
    bandwidth is wrong by a factor of $f_s T / n_{\text{points}}$ — 2500 here.
    This is the easiest mistake to make with this method.

## The two parameters

### Bandwidth

Wider bands give finer time resolution and admit more of the neighbouring
tone. The useful unit is $f_{\min}$, because in DFT bins $f_{\min}$ is exactly
$P$ bins, and the closest pair of tones on a quasi-log odd grid sits
$2f_{\min}$ apart.

So a passband of $2f_{\min}$ is already reaching its neighbour.

| bandwidth | instants | median | worst (edges removed) |
|---|---|---|---|
| 0.5 $f_{\min}$ | 80 | 1.35e-04 | 2.20e-02 |
| 1.0 $f_{\min}$ | 160 | 1.01e-04 | 2.66e-03 |
| 2.0 $f_{\min}$ | 320 | 9.31e-05 | 2.19e-01 |
| 4.0 $f_{\min}$ | 640 | 1.04e-04 | 3.39e-01 |

*(transistor system, sharpness $n = 8$; example 12)*

The median barely moves while the worst case moves by two orders of
magnitude. **Judge DMFA settings by the worst case, not the median** — the
leakage shows up as a large error at a few instants, not a small error
everywhere. Selecting on the median here would pick $2f_{\min}$, whose worst
case is 158× poorer than the best setting.

### Sharpness

`n` is the Fermi-Dirac exponent; large `n` approaches a brick wall. It has to
be matched to the bandwidth, because there are two opposite failure modes:

- **Wide band, gentle filter** — the passband reaches the neighbouring tone
  and leakage dominates.
- **Narrow band, sharp filter** — the band is truncated abruptly, so its
  inverse transform rings in time. At $0.5f_{\min}$ the worst case grows from
  1.4e-03 at $n=2$ to 4.5e-02 at $n=16$, with no neighbouring tone anywhere
  near the passband to blame.

An ideal passband is the $n \to \infty$ corner, and that corner is a bad place
to be. This is why the filter is parameterised with an adjustable `n`.

![The filter shape and the bandwidth/sharpness grid](../images/12_dmfa_sharpness.png)

*Left: the Fermi-Dirac passband at four sharpnesses, with the nearest tone
marked. Right: worst relative error over the grid — the best settings lie on a
diagonal, not at a corner.*

## The record edges are unusable

The band-limited inverse transform rings at the record boundaries. **Discard
the first and last tenth of every DMFA result.** This is not optional and no
setting removes it:

| | worst, edges removed | worst, edges kept |
|---|---|---|
| transistor | 1.39e-03 | 4.13e-01 |
| legendre | 1.33e-03 | 8.85e-01 |

The stationary system shows **no difference at all**, which is the proof of
what causes the rings: a constant impedance occupies a single bin, so
truncating the band removes nothing and there is nothing to ring. The artefact
is the cost of representing a *changing* quantity with a finite band, not an
artefact of the record ending.

![Where DMFA fails](../images/13_dmfa_error_structure.png)

*Right: error against position in the record. The three time-varying systems
show the same U shape; the stationary one is flat to both ends.*

## Inspecting a fit

```python
from deistools.processing.dmfa_functions import visualize_dmfa_filtering

visualize_dmfa_filtering(band_filter, ft_voltage, ft_current,
                         spectrum_axis, list(frequencies), indexes)
```

A slider pages through the excited tones, showing the raw band, the filtered
band and the filter itself, for voltage and for current. It is the first thing
to open when a DMFA result looks wrong: a band that is visibly clipped, or one
with a neighbouring tone inside the passband, explains more than any residual
number.

## When to use it

- You need finer time resolution than one multisine period — DMFA reached 5 s
  against STFT's 20 s on the same record.
- You want a method that, like STFT, assumes nothing about the *form* of the
  variation.
- As a cross-check on BLTVA, whose accuracy depends on an assumption DMFA does
  not make.

Given the provisional status above, do not use DMFA alone for a result you
intend to publish.
