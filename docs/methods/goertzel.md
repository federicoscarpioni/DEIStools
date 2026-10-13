# Goertzel filter

The Goertzel algorithm evaluates a single DFT bin with a second-order
recurrence instead of a transform. A multisine measurement wants exactly a
handful of known bins, which is the case the algorithm was made for.

It computes the same quantity as a rectangular-window [STFT](stft.md), by
different arithmetic — verified here to 1e-10 — so it is best understood as
an independent route to that estimator rather than as a fourth method.

## The mathematics

For a block of $N$ samples and a bin $k$ — the frequency in units of the
block's own resolution, $k = f N \Delta t$ — define

\[
\omega = \frac{2\pi k}{N}, \qquad \text{coeff} = 2\cos\omega
\]

and run, from $s_{-1} = s_{-2} = 0$,

\[
s[n] \;=\; x[n] + \text{coeff}\cdot s[n-1] - s[n-2],
\qquad n = 0 \ldots N-1
\]

Only the final two states are needed:

\[
X[k] \;=\; \Big(e^{\mathrm{j}\omega}\,s[N-1] - s[N-2]\Big)\,e^{-\mathrm{j}2\pi k}
\]

!!! warning "The output formula is asymmetric, and the tempting form is wrong"
    $s[N-1] - e^{-\mathrm{j}\omega}s[N-2]$ looks like the natural
    rearrangement and appears in some write-ups. It has the **right magnitude**
    and is wrong by exactly $e^{-\mathrm{j}\omega}$ in phase. That error
    cancels in any ratio — including $V/I$ — so an impedance test will not
    catch it. Only a comparison against the DFT itself will, which is what
    `test_output_formula_is_not_the_tempting_rearrangement` does.

    The correct combination is what you get by running one further step with a
    zero input and forming $s[N] - e^{-\mathrm{j}\omega}s[N-1]$; writing it
    closed-form avoids the extra iteration.

The trailing $e^{-\mathrm{j}2\pi k}$ is unity whenever $k$ is an integer,
which is why most presentations omit it. Keeping it is the **generalised**
Goertzel of Sysel and Rajmic, and it makes the result correct for a
non-integer $k$ — a tone that does not land exactly on a bin of the block you
chose, which is the ordinary case rather than an exotic one.

## Calling it

```python
from deistools.processing import MultiFrequencyAnalysis

mfa = MultiFrequencyAnalysis(frequencies, voltage, current, sampling_time)
Z, t = mfa.run_goertzel(period=Tp)       # one spectrum per block
```

Or directly:

```python
from deistools.processing.goertzel import goertzel_dft, goertzel_eis

Z, t = goertzel_eis(voltage, current, frequencies, period, sampling_time)

# or one block, one set of bins, with no transform anywhere
values = goertzel_dft(block, bins=frequencies * period)
```

`period` must hold a whole number of periods of every tone, for the same
reason a rectangular STFT window must: otherwise each bin picks up the others.

## On a stream

The recurrence carries two numbers. Nothing is buffered, so it runs during an
acquisition rather than after it — which is the one thing none of the other
estimators in this package can do.

```python
from deistools.processing.goertzel import GoertzelFilter

filt = GoertzelFilter(frequency=0.05, sampling_time=1e-3, block=20000)
for sample in stream:
    value = filt.push(sample)
    if value is not None:
        ...            # one block completed; the filter has reset itself
```

`push` returns `None` until a block is complete, then that block's complex DFT
value.

## Cost: it is slower here, and that is expected

Goertzel costs about $N$ operations per bin; an FFT costs about
$N\log_2 N$ for every bin at once. The crossover is at roughly
$n_{\text{bins}} \approx \log_2 N$:

\[
\text{ratio} = \frac{n_{\text{bins}}}{\log_2 N}
\]

For 13 tones in a 20 000-sample block that is 0.91 — nominally a wash, and in
practice a loss, because a compiled FFT beats a Python recurrence at equal
operation count by a wide margin. Measured: **0.17 s against 0.05 s**.

`cost_ratio(n_bins, block)` reports the number. Below one favours Goertzel on
operations alone; the real crossover sits well below that.

So it is not here for speed on recorded files. It is here because it is a
standard estimator, because it checks the STFT path independently, and because
its recurrence is the one that also runs live.

## Accuracy

Identical to STFT, necessarily — same estimator. On the transistor system with
a one-period block:

| | median relative error | worst |
|---|---|---|
| Goertzel | 7.13e-04 | 8.96e-03 |
| STFT | 7.13e-04 | 8.96e-03 |

Its limits are STFT's limits: time resolution is the block length, with a
floor of one multisine period; the skirts are not used; no uncertainty is
reported. See [STFT](stft.md).

## Worked example

[`examples/07_goertzel_first_run.py`](https://github.com/federicoscarpioni/DEIStools/blob/main/examples/07_goertzel_first_run.py)
checks the recurrence against `np.fft`, reproduces the STFT estimate, times
both, and feeds the streaming `GoertzelFilter` sample by sample.

## When to use it

- **Live processing**, where no FFT is possible because the record has not
  finished.
- **Very few bins out of very many** — a handful of tones in a long block,
  where $n_{\text{bins}} \ll \log_2 N$.
- **As a cross-check.** Agreement to 1e-10 with a completely different
  arithmetic path is a meaningful check on an FFT-based implementation,
  including on the bin indexing that is easy to get subtly wrong.
