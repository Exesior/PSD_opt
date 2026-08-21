"""Guards for two classes of defect, each of which had a real instance on 2026-08-17.

1. **Batch path vs. scalar path.** Several kernels have a vectorised variant
   that replaces a loop of individual calls, and whose docstring claims bit
   equality. That was never checked. `PorosityCompressionKernel.compute_array`
   wrote its result through a chained fancy-index assignment
   (``out[finite][can_compress] = updated``) into a throwaway copy, making it a
   complete no-op -- compression changed not a single porosity value for
   months. Wherever a fast variant replaces a slow one, a test belongs between
   them that compares the two; the slow one is the reference.

2. **Visibility of new particles to the sampler.** Nucleation creates particles
   and shifts weights, but never triggered a recompute of the
   agglomeration/breakage propensities. `_append_particle_column` appends a new
   particle with ``_r_agg[idx]`` at whatever the fresh slot holds -- 0. Such
   particles were unreachable for pair selection; at ``INITIAL_POROSITY = 0``
   this produced a self-lock (no breakage => no refresh => no agglomeration =>
   no refresh) that pinned 13 of 14 parameter variants at exactly zero
   agglomerations.

   The existing test ``test_sampler_totals_match_arrays`` could not see this:
   it runs without active nucleation.

See ``mcpbe/docs/historical/Kompression_und_Parametervalidierung.md`` and
``mcpbe/docs/historical/Nucleation_Propensity_Blockade.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmcpbe.kernels.continuous_processes.liquid_internalization import (
    LiquidInternalizationKernel,
)
from wmcpbe.kernels.continuous_processes.porosity_compression import (
    PorosityCompressionKernel,
)

from .bench.scenarios import make_solver


# ===========================================================================
# 1) Batch path vs. scalar path
# ===========================================================================

#: Time steps over which bit equality must hold. dt = 0 and a tiny dt are
#: deliberately included: for those, a correct kernel returns the input
#: unchanged -- and so does a broken one. They alone would NOT have caught the
#: no-op; only the large dt values expose it.
DTS = [0.0, 1e-6, 0.5, 100.0]


def _porosity_samples(seed: int = 0) -> np.ndarray:
    """Edge cases plus random spread: NaN (Vollkoerper), 0.0 (poreless), eps_min, 1.0."""
    rng = np.random.default_rng(seed)
    edge = np.array([0.7, 0.5, 0.3, 0.2, 0.1, 0.0, 1.0, np.nan])
    return np.concatenate([edge, rng.random(50)])


@pytest.mark.parametrize("dt", DTS)
def test_compression_batch_matches_scalar_loop(dt: float) -> None:
    """``compute_array`` must match a loop of ``compute`` calls exactly."""
    kernel = PorosityCompressionKernel(rate=0.2, min_porosity=0.2)
    poro = _porosity_samples()

    expected = np.array([kernel.compute(float(p), dt) for p in poro])
    actual = np.asarray(kernel.compute_array(poro, dt), dtype=float)

    assert np.allclose(expected, actual, rtol=0.0, atol=0.0, equal_nan=True), (
        f"Batch and scalar path disagree (dt={dt}): "
        f"max |delta| = {np.nanmax(np.abs(expected - actual)):.3e}"
    )


def test_compression_actually_reduces_porosity() -> None:
    """The kernel must actually change porosity, not just run through formally.

    This claim was missing: a no-op is consistent with any parity check, as
    long as BOTH paths do nothing. Here the effect itself is pinned down --
    against the analytic solution eps(t) = eps_min + (eps_0 - eps_min)*exp(-k*t).
    """
    kernel = PorosityCompressionKernel(rate=0.2, min_porosity=0.2)
    poro = np.array([0.7, 0.7, 0.5])
    dt = 10.0

    out = np.asarray(kernel.compute_array(poro, dt), dtype=float)
    analytic = 0.2 + (poro - 0.2) * np.exp(-0.2 * dt)

    assert np.allclose(out, analytic), f"expected {analytic}, got {out}"
    assert np.all(out < poro), "compression did not lower porosity"


@pytest.mark.parametrize("dt", DTS)
def test_internalization_batch_matches_scalar_loop(dt: float) -> None:
    """Same parity requirement for the internalisation kernel.

    ``k_int`` is deliberately 1e12: below that, ``exp(alpha*dt)`` rounds to
    exactly 1.0 for pore volumes of this size, and the analytic solution
    algebraically returns the input unchanged -- the test would then run
    against a standstill and be worthless.
    """
    kernel = LiquidInternalizationKernel(k_int=1e12)
    rng = np.random.default_rng(1)
    n = 40

    v_pore = np.concatenate([[0.0, 1e-14], rng.random(n) * 1e-14 + 1e-16])
    l_total = np.concatenate([[1e-15, 0.0], rng.random(n) * 1e-14])
    sat = np.concatenate([[0.0, 1.0], rng.random(n)])

    expected = np.array([
        kernel.compute(float(s), float(v), float(l), dt)
        for s, v, l in zip(sat, v_pore, l_total)
    ])
    actual = np.asarray(kernel.compute_array(sat, v_pore, l_total, dt), dtype=float)

    assert np.allclose(expected, actual, rtol=1e-12, atol=0.0, equal_nan=True), (
        f"Batch and scalar path disagree (dt={dt}): "
        f"max |delta| = {np.nanmax(np.abs(expected - actual)):.3e}"
    )


# ===========================================================================
# 2) Nucleation particles must be visible to the sampler
# ===========================================================================

def _solve_granulation(maxiter: int = 400, block_agglomeration: bool = False):
    """Run granulation (agglomeration + nucleation) for up to ``maxiter`` events.

    ``block_agglomeration`` attaches an acceptance kernel that rejects EVERY
    collision (``fittable`` with ``u_acc = 0``). That is essential to make the
    original defect visible at all: as long as agglomeration or breakage
    events go through, the solver already calls
    ``_refresh_samplers_after_agg`` and keeps the propensities current -- the
    run self-heals and a test against it stays green even with the bug
    present.

    Only once nothing is accepted does nucleation remain the sole process that
    changes the population. That is exactly the situation that occurred in
    production at ``INITIAL_POROSITY = 0`` (poreless particles do not break,
    and the Stokes criterion rejected every collision as "both dry").
    """
    solver = make_solver("granulation_1d")
    if block_agglomeration:
        from wmcpbe.kernels.agglomeration_acceptance import (
            get_agglomeration_acceptance_kernel,
        )
        solver.kernel_manager.agglomeration_acceptance_kernel = (
            get_agglomeration_acceptance_kernel("fittable", u_acc=0.0)
        )
    solver.solve(maxiter=maxiter)
    return solver


def test_nucleation_keeps_agg_propensities_current() -> None:
    """Stored propensities must match a freshly computed rebuild.

    This is the invariant that was previously violated: nucleation created
    particles and shifted weights without triggering
    ``_rebuild_all_propensities``. Measured, ``_r_agg`` deviated from the
    correct value by up to 6.9 (absolute), and the propensity of wetted
    particles stayed exactly 0 while their weight grew to 77 % of the
    population.
    """
    solver = _solve_granulation(block_agglomeration=True)
    a = solver.a_tot
    assert a > 0

    stored = np.asarray(solver._r_agg[:a], dtype=float).copy()
    solver._rebuild_all_propensities()
    fresh = np.asarray(solver._r_agg[:a], dtype=float).copy()

    scale = max(float(np.max(np.abs(fresh))), 1e-300)
    max_dev = float(np.max(np.abs(stored - fresh))) / scale
    assert max_dev < 1e-12, (
        f"Agglomeration propensities are stale: relative deviation {max_dev:.3e}. "
        "A process changed the population without triggering the rebuild "
        "(see Nucleation_Propensity_Blockade.md)."
    )


def test_compression_keeps_break_rates_current() -> None:
    """Same invariant for the breakage rates.

    The test above only checks ``_r_agg``. The second cause of the same gap --
    the continuous processes run after every event and change porosity,
    V_dry and saturation without triggering a rebuild -- hits ``_break_rate``
    even harder: measured 2.5e-2 relative deviation against 5.5e-3 for the
    agglomeration propensities.

    ``powerlaw_rumpf`` reads both porosity and saturation for its strength
    model, so internalisation also shows up here in addition -- something that
    leaves the agglomeration kernels unaffected (their beta depends only on
    radii).
    """
    solver = make_solver("granulation_1d")
    solver.process_type = "mix"
    solver.BREAKFVAL = 2
    solver.break_dW_max = 50.0
    solver.kernel_manager.break_kernel_name = "powerlaw_rumpf"
    solver.kernel_manager.break_kernel_params = dict(
        p1=1e-6, p2=1.0, g=1000.0, breakrval=4, k=2.5, alpha=1.0,
        gamma=0.072, delta=0.0, x_s=1e-6,
    )
    solver.kernel_manager.initialize_kernels(solver)

    # Reject every collision: otherwise an agglomeration event keeps the
    # propensities current anyway and the defect stays invisible.
    from wmcpbe.kernels.agglomeration_acceptance import (
        get_agglomeration_acceptance_kernel,
    )
    solver.kernel_manager.agglomeration_acceptance_kernel = (
        get_agglomeration_acceptance_kernel("fittable", u_acc=0.0)
    )
    solver._initialize_samplers()
    solver.solve(maxiter=400)

    a = solver.a_tot
    assert a > 0

    stored = np.asarray(solver._break_rate[:a], dtype=float).copy()
    solver._calc_break_rates_full()
    fresh = np.asarray(solver._break_rate[:a], dtype=float).copy()

    scale = max(float(np.max(np.abs(fresh))), 1e-300)
    max_dev = float(np.max(np.abs(stored - fresh))) / scale
    assert max_dev < 1e-12, (
        f"Breakage rates are stale: relative deviation {max_dev:.3e}. "
        "The continuous processes changed porosity or saturation without "
        "triggering the rebuild (see Nucleation_Propensity_Blockade.md, section 8)."
    )


def test_nucleated_particles_are_reachable_by_the_sampler() -> None:
    """No particle carrying weight may have propensity 0.

    Propensity 0 means: the Fenwick sampler can never draw it. That is exactly
    the state every particle created by nucleation ended up in.
    """
    solver = _solve_granulation(block_agglomeration=True)
    a = solver.a_tot

    weights = np.asarray(solver.W[:a], dtype=float)
    prop = np.asarray(solver._r_agg[:a], dtype=float)

    alive = weights > 0.0
    assert np.any(alive), "run ended with no particle carrying positive weight"

    invisible = alive & ~(prop > 0.0)
    assert not np.any(invisible), (
        f"{int(np.count_nonzero(invisible))} of {int(np.count_nonzero(alive))} "
        "particles with weight > 0 have propensity 0 and are unreachable for "
        "pair selection."
    )


def test_agg_sampler_covers_the_whole_population() -> None:
    """Sampler length and total must match the active population.

    Complements ``test_sampler_totals_match_arrays`` with the case WITH
    nucleation -- there, particles arise outside the agglomeration and
    breakage paths.

    NOTE: deliberately a weak test -- this check alone does NOT catch the
    original defect. Counter-check with the bug reinstated: the two tests
    above fail, this one stays green. Reason: length and total stay consistent
    when a particle with propensity 0 is appended -- 0 adds cleanly. That is
    exactly what the older ``test_sampler_totals_match_arrays`` walked past
    too. The value here lies elsewhere: it catches a sampler whose LENGTH
    falls behind.
    """
    solver = _solve_granulation(block_agglomeration=True)
    a = solver.a_tot
    sampler = solver._agg_sampler
    assert sampler is not None

    assert sampler.size() == a, (
        f"sampler knows {sampler.size()} particles, population has {a}"
    )

    expected_total = float(np.sum(solver._r_agg[:a]))
    assert np.isclose(sampler.total(), expected_total, rtol=1e-9, atol=0.0), (
        f"sampler total {sampler.total():.6e} != array total {expected_total:.6e}"
    )
