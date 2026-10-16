# `deistools.simulation`

Synthetic measurements with known ground truth, for testing and benchmarking
estimators. Each model gives a different *kind* of ground truth — see the
[concepts](../concepts/when-meaningful.md) pages for what that means in
practice.

## Engine

::: deistools.simulation.engine

## Standard drifts

Additive disturbances for testing trend removal: the thing
[detrending](../methods/detrending.md) has to get rid of. The models that
follow are the opposite -- the system an estimator is trying to measure.

::: deistools.simulation.trends

## Stationary R + (R‖C)

::: deistools.simulation.models.r_rc

## Transistor-gated R + (R_ct(t)‖C)

::: deistools.simulation.models.r_rc_transistor

## Charging insertion electrode with diffusion

::: deistools.simulation.models.randles_diffusion

## Prescribed Legendre expansion

::: deistools.simulation.models.legendre

## Block-oriented nonlinear system

::: deistools.simulation.models.block_oriented
