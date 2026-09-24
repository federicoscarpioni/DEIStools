# DEIStools

DEIStools is a package for multi-frequency electrochemical impedance measurement: acquiring multi-frequency excitation/response signals from a lab setup, processing them to estimate the time-varyingimpedance, and visualising the results.

## Subpackages

- `deistools.acquisition` — automate the lab hardware (oscilloscope, waveform generator, potentiostat) to acquire multi-frequency signals from an electrochemical cell.
- `deistools.processing` — signal processing and impedance estimation from acquired (or externally supplied) signals; hardware-agnostic.
- `deistools.visualise` — plotting helpers for raw signals, spectra, and impedance results.

Note: as of `v0.1.0`, acquisition and processing/visualise live together in this repo. They are being split into two separate packages — `deistools` (processing/visualise) and `elma` (lab acquisition) since the acquisition code is tied to one specific lab setup while processing/visualise is general-purpose.
