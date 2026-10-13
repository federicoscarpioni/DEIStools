"""Oracle tests for BLTVA against the Legendre model.

The Legendre simulation prescribes the transfer function as

    Z(f_k, t) = sum_p  a_p(f_k) * L_p(2t/T - 1)

which is exactly the signal model BLTVA assumes, so the estimator must return
``H_p == a_p``. Ground truth and estimator output are the same mathematical
objects, which is what makes this an oracle rather than a regression check.

Every recovery test runs on both grid families (see ``conftest``): a dense
odd-linear grid, where bin spacing is uniform, and a sparse quasi-log grid,
where it varies by orders of magnitude. They exercise different paths through
``_band_bins`` and the two do not fail together.

Tolerances sit a few times above the O(1/N) floor documented in ``conftest``.
"""
import warnings

import numpy as np
import pytest

from deistools.processing.bltva import bltva, legendre_basis
from tests.conftest import (GRIDS, ORACLE_P, odd_linear_grid, quasi_log_odd_grid)

GRID_NAMES = list(GRIDS)

# Shared fit settings. band_halfwidth is explicit because the
# neighbour-distance rule collapses the low bands of a quasi-log grid.
# Na is left to eq. (22) rather than hand-picked. Pinning Na=6 here cost a
# factor of 100 in noiseless recovery (9.6e-05 against 7.9e-07), because
# the transient polynomial could not absorb the band's out-of-band content.
FIT = dict(Nq=5, harmonics="excited", band_halfwidth=5 * ORACLE_P)

TOL_P0 = 1e-4      # with noise; observed worst ~4e-5
TOL_PK = 5e-4      # with noise; observed worst ~1.5e-4 for p >= 1
TOL_EXACT = 5e-6   # noiseless; observed ~8e-7, limited by arithmetic alone


# --------------------------------------------------------------------------- #
# Coefficient recovery
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("grid", GRID_NAMES)
def test_recovers_prescribed_coefficients(make_oracle, grid):
    """Noiseless: H_p must reproduce the prescribed a_p at every order.

    The tolerance is far tighter than the noisy cases below because with
    simulator and estimator sharing a basis there is nothing left between
    them but arithmetic. It was 100x looser while the simulator used the raw
    Legendre polynomials, and this assertion is what stops that returning.
    """
    o = make_oracle(grid=grid, variation=(0.30, 0.10))
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=2, **FIT)

    err = o.relative_error(r.H_p)
    assert err.max() < TOL_EXACT, f"noiseless recovery only {err.max():.2e}"


@pytest.mark.parametrize("grid", GRID_NAMES)
@pytest.mark.parametrize("noise", [1e-4, 1e-3])
def test_recovers_coefficients_under_noise(make_oracle, grid, noise):
    """Additive output noise must not bias the coefficients at this SNR."""
    o = make_oracle(grid=grid, variation=(0.30, 0.10), noise_std_V=noise)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=2, **FIT)

    err = o.relative_error(r.H_p)
    assert err[:, 0].max() < TOL_P0, f"H_0 error {err[:, 0].max():.2e}"
    assert err[:, 1:].max() < TOL_PK, f"H_p>=1 error {err[:, 1:].max():.2e}"


def test_static_system_has_no_time_variation(make_oracle):
    """variation=() prescribes a time-invariant Z; orders p>=1 must vanish."""
    o = make_oracle(grid="quasi_log_15", variation=())
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=3, **FIT)

    leak = np.abs(r.H_p[:, 1:]).max(axis=1) / np.abs(r.H_p[:, 0])
    assert leak.max() < TOL_PK, f"spurious time variation {leak.max():.2e}"


@pytest.mark.parametrize("Np", [3, 4, 5])
def test_overspecified_np_preserves_low_orders(make_oracle, Np):
    """Asking for more orders than the system has must not corrupt p = 0,1,2.

    The unsupported orders are not driven to zero here — that is Algorithm 2's
    pruning step (Hallemans TIM 2021, Algorithm 2 steps 6-7), which is not yet
    implemented. This test only pins the weaker property that over-specifying
    Np does not damage the orders that are genuinely present.
    """
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-4)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=Np, **FIT)

    err = o.relative_error(r.H_p)
    assert err[:, 0].max() < TOL_P0
    assert err[:, 1:].max() < TOL_PK


# --------------------------------------------------------------------------- #
# Internal consistency
# --------------------------------------------------------------------------- #

def test_time_average_of_h_wt_equals_h0(make_oracle):
    """b_0 = 1 and the basis is mean-removed for p >= 1, so the time average
    of the reconstructed H(w,t) is exactly H_0."""
    o = make_oracle(grid="quasi_log_15")
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=3, **FIT)

    dev = np.abs(r.H_wt.mean(axis=1) - r.H_p[:, 0]) / np.abs(r.H_p[:, 0])
    assert dev.max() < 1e-12, f"time average deviates by {dev.max():.2e}"


def test_h_wt_is_the_basis_reconstruction(make_oracle):
    """H_wt must equal H_p projected back onto the Legendre basis."""
    o = make_oracle(grid="quasi_log_15")
    Np = 3
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=Np, **FIT)

    b = legendre_basis(o.N, Np)
    assert np.allclose(r.H_wt, r.H_p @ b.T, rtol=0, atol=1e-12)


def test_reported_frequencies_match_the_grid(make_oracle):
    o = make_oracle(grid="quasi_log_15")
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=2, **FIT)
    assert np.allclose(r.f_exc, o.H_exc / o.Tp)


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #

DRIFT = [0.5, 0.20, 0.05]       # Legendre orders 0..2, well inside Nq = 5


def test_coefficients_survive_a_drift(make_oracle):
    """A trend must not bias H_p.

    eq. (58) carries a trend block in every band for exactly this reason.
    This previously failed: the ``drift_bands`` parameter truncated the block
    to the lowest bands, and the first band *without* one absorbed the trend
    skirt into its target coefficients -- a 4.4e-03 spike at h=5 against
    ~5e-06 either side. The block is now present everywhere, and Algorithm 2
    step 6 removes it where it is not supported.
    """
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10),
                    drift=DRIFT, noise_std_V=1e-4)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Np=2, **FIT)

    err = o.relative_error(r.H_p)
    assert err[:, 0].max() < TOL_P0, f"H_0 error {err[:, 0].max():.2e}"
    assert err[:, 1:].max() < TOL_PK, f"H_p>=1 error {err[:, 1:].max():.2e}"


def test_no_band_is_an_outlier_under_drift(make_oracle):
    """The old failure was a spike at one band, not a general loss of accuracy,
    so guard the shape of the error across bands and not just its maximum."""
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10),
                    drift=DRIFT, noise_std_V=1e-4)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Np=2, **FIT)

    per_band = o.relative_error(r.H_p)[:, 0]
    assert per_band.max() < 20 * np.median(per_band), (
        f"one band stands out: {per_band}")


def test_drift_is_not_reported(make_oracle):
    """theta_q is a nuisance parameter, so the result must not offer it.

    It was previously reconstructed from band 0 alone, where the trend's 1/k
    tail is nearly collinear with the transient polynomial. The published
    value was wrong by ~50% of the drift range and carried a ~26% imaginary
    part, which a real signal's drift cannot have. Recovering the trend
    belongs to a detrending step.
    """
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), drift=DRIFT)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Np=2, **FIT)

    assert not hasattr(r, "drift")
    assert not hasattr(r, "theta_q")


# --------------------------------------------------------------------------- #
# Band construction
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("grid", GRID_NAMES)
@pytest.mark.filterwarnings("ignore:the eq. .55. band reaches")
def test_both_band_rules_work_on_well_formed_grids(make_oracle, grid):
    """On a grid whose spacing changes smoothly, the neighbour-distance rule
    and an explicit halfwidth both recover the coefficients.

    Log spacing alone does not break the neighbour rule, despite the bands it
    produces differing by more than an order of magnitude in width across the
    spectrum.
    """
    o = make_oracle(grid=grid, variation=(0.30, 0.10), noise_std_V=1e-4)
    common = dict(Np=2, Nq=5, Na=6, harmonics="excited")

    fixed = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
                  band_halfwidth=5 * ORACLE_P, **common)
    neighbour = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
                      band_halfwidth=None, **common)

    assert o.relative_error(fixed.H_p).max() < TOL_PK
    assert o.relative_error(neighbour.H_p).max() < TOL_PK


def test_eq55_band_spans_three_excited_frequencies():
    """eq. (55): the band holds k_e^-, k_e and k_e^+, and reaches half way
    towards the next excited bin beyond each of the outer two."""
    import deistools.processing.bltva as M

    k = np.array([20, 40, 60, 80, 100])
    band = M._band_bins(k, 2, 100_000, None)

    assert (band[0], band[-1]) == (30, 90)
    assert sorted(int(x) for x in k if band[0] <= x <= band[-1]) == [40, 60, 80]


def test_eq55_band_is_mirrored_at_the_edges():
    """The outermost excited frequencies have no outer neighbour, so the
    margin is reflected rather than truncated -- otherwise the first and last
    estimates would come from a differently shaped band than the rest."""
    import deistools.processing.bltva as M

    k = np.array([20, 40, 60, 80, 100])
    widths = [M._band_bins(k, e, 100_000, None).size for e in range(k.size)]

    assert widths[1:4] == [61, 61, 61]
    assert widths[-1] == 61


@pytest.mark.filterwarnings("ignore:the eq. .55. band reaches")
def test_eq55_band_repairs_the_abrupt_spacing_weakness(make_oracle):
    """Widening to eq. (55) largely fixes the abrupt-spacing failure.

    Two harmonics close together at the top of an otherwise sparse grid used
    to give that band a half-distance of ~P bins on one side and hundreds on
    the other, truncating the skirt and costing 3.3e-03. With both neighbours
    now inside the band the error is 5.3e-05, a 60x improvement.

    A fixed halfwidth is still better here (1.2e-05), being the only rule that
    does not scale with the local spacing at all.
    """
    abrupt = [1, 3, 5, 7, 11, 19, 31, 51, 85, 139, 227, 373, 611, 997, 999]
    o = make_oracle(grid=abrupt, variation=(0.30, 0.10), noise_std_V=1e-4)
    common = dict(Np=2, Nq=5, Na=6, harmonics="excited")

    fixed = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
                  band_halfwidth=5 * ORACLE_P, **common)
    eq55 = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
                 band_halfwidth=None, **common)

    assert o.relative_error(fixed.H_p).max() < TOL_PK
    assert o.relative_error(eq55.H_p).max() < 1e-3, "no better than the old band"


def test_huge_eq55_bands_are_flagged(make_oracle):
    """On a sparse quasi-log grid the eq. (55) band reaches ~1000 harmonics at
    the top -- 332 s against 0.6 s for the same answer. Warn before spending
    the time, and name the remedy."""
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10))

    with pytest.warns(UserWarning, match=r"eq\. \(55\) band reaches"):
        bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=2, Nq=5, Na=6, harmonics="excited", band_halfwidth=None)


def test_no_warning_for_a_dense_grid(make_oracle):
    o = make_oracle(grid="odd_linear_L5", variation=(0.30, 0.10))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=2, Nq=5, Na=6, harmonics="excited", band_halfwidth=None)


# --------------------------------------------------------------------------- #
# Excitation grid properties
# --------------------------------------------------------------------------- #

def test_odd_linear_grid_is_odd_and_omits_one_per_group():
    """The harmonic-grid half of the ORP construction (TIM 2021, Definition 1)."""
    L, h_max = 5, 59
    H = odd_linear_grid(h_max=h_max, L=L, seed=0)

    assert np.all(H % 2 == 1), "even harmonics must not be excited"
    assert np.all(np.diff(H) > 0), "harmonics must be strictly increasing"

    odd = np.arange(1, h_max + 1, 2)
    for i in range(0, odd.size - L + 1, L):
        group = odd[i:i + L]
        assert np.isin(group, H).sum() == L - 1, (
            f"exactly one harmonic must be omitted from group {group.tolist()}"
        )


def test_odd_linear_grid_is_reproducible_under_seed():
    a = odd_linear_grid(h_max=59, L=5, seed=7)
    b = odd_linear_grid(h_max=59, L=5, seed=7)
    c = odd_linear_grid(h_max=59, L=5, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_quasi_log_grid_is_odd_and_strictly_increasing():
    """Strict increase matters: two tones on the same or adjacent bins would
    put overlapping skirts inside a single band."""
    H = quasi_log_odd_grid(n=15, h_max=999)

    assert H.size == 15
    assert np.all(H % 2 == 1), "even harmonics must not be excited"
    assert np.all(np.diff(H) > 0), "harmonics must be strictly increasing"
    assert H[0] == 1


@pytest.mark.parametrize("n,h_max", [(10, 99), (15, 999), (25, 2001)])
def test_quasi_log_grid_spans_requested_range(n, h_max):
    H = quasi_log_odd_grid(n=n, h_max=h_max)
    assert H.size == n
    assert np.all(np.diff(H) > 0)
    assert H[-1] <= h_max + 2 * n          # strict-increase nudging may overshoot


@pytest.mark.parametrize("grid", GRID_NAMES)
def test_diagnostics_are_well_posed(make_oracle, grid):
    """Every band must be overdetermined and reported consistently.

    Rank deficiency is expected and not an error: the drift skirt and the
    low-order transient polynomial are genuinely near-collinear in a narrow
    band, and eq. (64) resolves that with the Moore-Penrose pseudo-inverse,
    which is well defined regardless. What must hold is that the band has
    more equations than unknowns and that pruning only ever removes
    regressors.
    """
    o = make_oracle(grid=grid)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc,
              Np=2, **FIT)

    assert len(r.diagnostics) == o.H_exc.size
    for d in r.diagnostics:
        assert d["n_theta"] < d["N_w"], "band is underdetermined"
        assert 0 < d["rank"] <= d["n_theta"]
        assert d["n_theta"] <= d["n_theta_full"], "pruning added regressors"
        assert (d["n_theta_full"] - d["n_theta"]
                == d["n_transient_dropped"] + d["n_unsupported_dropped"])
        assert np.isfinite(d["cond"])


def test_pruning_removes_the_drift_block_at_high_frequency(make_oracle):
    """The drift skirt B_q(k) ~ 1/k carries nothing in the upper bands, so
    Algorithm 2 step 6 must drop it there.

    This is the principled replacement for the old ``drift_bands`` parameter:
    the block is included everywhere per eq. (58) and removed where the data
    do not support it, on statistical grounds rather than by band index.

    Only the quasi-log grid is used. The odd-linear grid tops out at h = 59,
    i.e. bin 1180, where |B_q| ~ 8e-4 is still well above the noise and the
    block is legitimately kept -- the drift becomes negligible because of how
    far up the spectrum you go, not because of the band index.
    """
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-4)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Np=2, **FIT)

    dropped = [d["n_unsupported_dropped"] for d in r.diagnostics]
    half = len(dropped) // 2
    assert dropped[-1] > 0, f"nothing pruned in the highest band: {dropped}"
    assert sum(dropped[half:]) > sum(dropped[:half]), (
        f"pruning should concentrate at high frequency: {dropped}")


def test_pruning_zeroes_unsupported_legendre_orders(make_oracle):
    """With Np over-specified, the orders the system does not have must be
    driven to zero rather than left to absorb noise.

    This is Algorithm 2 step 6 doing the job the hyperparameter guidance
    cannot: before it was implemented these orders carried ~1.5e-04 of
    spurious magnitude.
    """
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-4)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Np=6, **FIT)

    spurious = np.abs(r.H_p[:, 3:]).max() / np.abs(o.a[:, 0]).max()
    assert spurious == 0.0, f"unsupported orders not pruned: {spurious:.2e}"

    err = o.relative_error(r.H_p)
    assert err[:, 0].max() < TOL_P0
    assert err[:, 1:].max() < TOL_PK


def test_prune_can_be_disabled(make_oracle):
    """prune=False must give the plain least-squares fit of eq. (58), which
    leaves the unsupported orders populated."""
    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-4)
    r = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Np=6, prune=False, **FIT)

    assert np.abs(r.H_p[:, 3:]).max() > 0.0
    assert all(d["n_unsupported_dropped"] == 0 for d in r.diagnostics)


def test_supplied_noise_estimate_is_reused(make_oracle):
    """Passing a NoiseEstimate must reproduce the internally computed result,
    so it can be shared across calls without changing the answer."""
    from deistools.processing.bltva import estimate_noise_variance

    o = make_oracle(grid="quasi_log_15", variation=(0.30, 0.10), noise_std_V=1e-4)
    ne = estimate_noise_variance(o.voltage, o.current, o.fs, o.P, o.H_exc,
                                 Np=2, Nq=5, Na=6)

    auto = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Np=2, **FIT)
    given = bltva(o.voltage, o.current, o.fs, o.P, o.H_exc, Np=2, noise=ne, **FIT)

    assert np.allclose(auto.H_p, given.H_p)
