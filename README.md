# DEIStools

DEIStools estimates the (possibly time-varying) multi-frequency impedance of an electrochemical system from voltage/current signals, and visualises the results. It is hardware-agnostic: it works on any suitably sampled signal, not just signals acquired with a specific lab setup.

## Subpackages

- `deistools.processing` — signal processing and impedance/transfer-function estimation from voltage/current signals.
- `deistools.visualise` — plotting helpers for raw signals, spectra, and impedance results.

## History note

Before `v0.1.0`, this repo also contained lab-hardware acquisition code. That has moved to a separate package, [`elma`](https://github.com/federicoscarpioni/elma) (Electrochemistry Lab Multi-frequency Acquisition), which depends on `deistools` for impedance estimation. The last combined snapshot remains available at this repo's `v0.1.0` tag.
