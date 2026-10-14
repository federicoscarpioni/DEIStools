# Implementation notes

The papers this package implements leave some things ambiguous, and in a few
places this code makes a choice they do not dictate. Those choices change the
numbers, so they are recorded here with the evidence rather than left in the
source for a reader to reverse-engineer.

This section exists because the package crosses electrochemistry and system
theory, where a reader is likely to be expert in one and not the other. If you
are going to cite a result from DEIStools, these are the pages to read first.

## What is on these pages

- **[BLTVA](bltva.md)** — the ambiguous reading of the rank threshold, why the
  solver deliberately does not equilibrate, the Legendre basis convention, why
  eq. (49) is necessary but not sufficient, and why $N_p$ and the band width
  compete.
- **[DMFA](dmfa.md)** — the local filter axis, the edge artefact, and the
  current verification status.
- **[BLA](bla.md)** — where the frequency derivatives of eq. (15) come from,
  and why the central differences of step 3 cannot be transcribed onto a
  quasi-logarithmic grid: 156× on the first derivative, and a second-difference
  formula that *diverges* under refinement. Includes the replacement.

## How to read a claim here

Every entry states what was tested and what the number was. Where a choice was
made between two readings of a paper, both were implemented and measured; the
one that is used is the one that was unbiased, not the one that looked
plausible.

Where something is *not* verified, it says so. The DMFA page is mostly such a
statement at present, and the BLA page is a design note for work not yet done
rather than a record of work finished -- it says so at the end.

## A general convention

Where a paper gives a rule and this package also allows a manual override, the
rule is the default and the override exists because real frequency grids are
not always what the papers assumed. Two cases:

| quantity | paper rule | override | why the override exists |
|---|---|---|---|
| $N_a$ | eq. (22) of the 2022 paper | `Na=` | none needed in practice; the rule beat a hand-picked value by 19× |
| band width | eq. (55) of the 2021 paper | `band_halfwidth=` | the rule produces enormous bands at the top of a sparse quasi-log grid |

The papers use linearly spaced odd harmonics. Electrochemistry uses
logarithmic grids with fifteen tones over three decades. Several rules that
are natural on the first are impractical on the second, and where that is the
case it is noted on the method's page.

The sharpest instance is on the [BLA page](bla.md): a numerical
differentiation that is second-order accurate on an equispaced grid is not
merely less accurate on a logarithmic one but, for the second derivative,
inconsistent -- its error grows as the grid is refined. The grid this package
uses throughout is

```
1  3  5  9  13  17  27  39  59  89  133  201  301
```

whose gaps between neighbouring tones span a factor of 50.
