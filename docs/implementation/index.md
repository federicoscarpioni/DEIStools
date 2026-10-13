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

## How to read a claim here

Every entry states what was tested and what the number was. Where a choice was
made between two readings of a paper, both were implemented and measured; the
one that is used is the one that was unbiased, not the one that looked
plausible.

Where something is *not* verified, it says so. The DMFA page is mostly such a
statement at present.

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
