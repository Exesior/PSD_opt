"""Mass-conservation checks for the solid and the liquid phase.

The two invariants the weighted DSMC solver must satisfy are

    solid:   sum_i W_i * V_solid_i           is constant
    liquid:  sum_i W_i * liquid_volume_i     changes only by what nucleation adds

with ``V_solid_i = V_dry_i * (1 - porosity_i)`` and ``V_solid_i = V_dry_i`` for
non-porous primary particles (``porosity`` is NaN).

Both quantities are *extensive*: they carry the weight ``W``. Everything stored
in the state arrays is *intensive* (per physical particle), which is the single
most common source of conservation bugs in this code base - see
``mcpbe/docs/historical/REFACTORING_FINDINGS.md``.

Run standalone for a readable report::

    python -m tests.test_conservation

or under pytest::

    pytest tests/test_conservation.py -v
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from .bench.scenarios import SCENARIOS, build, warmup

# Relative tolerance for "conserved". Machine-precision accumulation over a few
# hundred events lands around 1e-15; 1e-9 leaves generous headroom while still
# catching any real leak.
CONSERVATION_RTOL = 1e-9

#: Scenarios that must conserve solid volume exactly.
SOLID_SCENARIOS = [
    "agg_shear_1d",
    "agg_shear_1d_large",
    "agg_constant_1d",
    "agg_sum_1d",
    "agg_shear_2d",
    "break_powerlaw_1d",
    "mix_1d",
    "granulation_1d",
    # The production wet-granulation configuration. Costs ~4 minutes across the
    # five checks that run over this list, and it is worth it: it is the only
    # scenario exercising the EKE kernel, `stokes_dynamik`,
    # `powerlaw_rumpf_dynamic` and `cone_model`, and the only one where a
    # breakage event fires while the nucleation and continuous-process handlers
    # are live. Everything else here runs `shear_chin1998`, so without it the
    # suite never touched the non-separable O(n^2) propensity path at all --
    # only the golden-reference fingerprint did, and a hash says nothing about
    # mass conservation, sampler consistency or state bounds.
    "granulation_rumpf_dynamic_1d",
    # Same configuration with the mixer-speed compression kernels. V_solid must
    # stay exactly constant whatever porosity the kernel returns -- it only
    # supplies `porosity`, and `_apply_compression` writes porosity and V_dry
    # over one mask (audit B-07). The `_rumpf` variant also exercises the
    # per-particle sigma(eps, S) path inside compute_array.
    "granulation_compression_dynamik_1d",
    "granulation_compression_dynamik_rumpf_1d",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def solid_volume(solver) -> float:
    """Total represented solid volume, sum_i W_i * V_solid_i [m^3]."""
    a = solver.a_tot
    if a <= 0:
        return 0.0
    v_dry = solver.V_flat[-1, :a]
    poro = solver.porosity[:a]
    v_solid = np.where(np.isnan(poro), v_dry, v_dry * (1.0 - np.nan_to_num(poro)))
    return float(np.sum(solver.W[:a] * v_solid))


def component_solid_volume(solver) -> float:
    """Same quantity via the component route, sum_i W_i * sum_d V_flat[d, i].

    ``V_flat[:dim]`` stores the per-component *solid* volume, so this must agree
    with :func:`solid_volume`. A mismatch means the two representations have
    drifted apart - that was the dim=2 defect fixed in this refactor.
    """
    a = solver.a_tot
    if a <= 0:
        return 0.0
    return float(np.sum(solver.W[:a] * np.sum(solver.V_flat[: solver.dim, :a], axis=0)))


def liquid_volume(solver) -> float:
    """Total represented liquid volume, sum_i W_i * liquid_volume_i [m^3]."""
    a = solver.a_tot
    if a <= 0:
        return 0.0
    return float(np.sum(solver.W[:a] * solver.liquid_volume[:a]))


def _rel(after: float, before: float) -> float:
    scale = max(abs(before), 1e-300)
    return abs(after - before) / scale


# ---------------------------------------------------------------------------
# solid phase
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", SOLID_SCENARIOS)
def test_solid_mass_is_conserved(name):
    """sum(W * V_solid) must not change over a whole run."""
    solver = build(SCENARIOS[name])
    before = solid_volume(solver)
    solver.solve(maxiter=SCENARIOS[name].maxiter)
    after = solid_volume(solver)

    assert before > 0.0, f"{name}: degenerate initial state"
    assert _rel(after, before) < CONSERVATION_RTOL, (
        f"{name}: solid volume drifted by {_rel(after, before):.3e} "
        f"({before:.6e} -> {after:.6e})"
    )


@pytest.mark.parametrize("name", SOLID_SCENARIOS)
def test_volume_representations_agree(name):
    """V_flat[:dim] (components) and V_dry*(1-porosity) must describe the same solid.

    Regression guard for the dim=2 defect where the merged solid volume was
    written as a scalar into every component, inflating the component route by a
    factor ``dim``.
    """
    solver = build(SCENARIOS[name])
    solver.solve(maxiter=SCENARIOS[name].maxiter)

    via_dry = solid_volume(solver)
    via_components = component_solid_volume(solver)
    assert _rel(via_components, via_dry) < 1e-9, (
        f"{name}: component route {via_components:.6e} disagrees with "
        f"V_dry route {via_dry:.6e}"
    )


def test_solid_mass_conserved_per_agglomeration_event():
    """Every single agglomeration event must be solid-mass neutral."""
    scenario = SCENARIOS["agg_shear_1d"]
    solver = build(scenario)

    worst = 0.0
    for _ in range(200):
        if solver.a_tot < 2:
            break
        before = solid_volume(solver)
        solver._do_one_agg()
        # `_do_one_agg` does not refresh the samplers -- `solve()` rebuilds once
        # per iteration instead. A bare loop like this one has to do it itself,
        # otherwise every draw after the first uses stale propensities.
        solver._refresh_samplers_after_agg()
        worst = max(worst, _rel(solid_volume(solver), before))

    assert worst < CONSERVATION_RTOL, f"worst per-event solid drift {worst:.3e}"


def test_solid_mass_conserved_per_breakage_event():
    """Every single breakage event must be solid-mass neutral."""
    scenario = SCENARIOS["break_powerlaw_1d"]
    solver = build(scenario)

    worst = 0.0
    for _ in range(200):
        before = solid_volume(solver)
        solver._do_one_break()
        worst = max(worst, _rel(solid_volume(solver), before))

    assert worst < CONSERVATION_RTOL, f"worst per-event solid drift {worst:.3e}"


# ---------------------------------------------------------------------------
# liquid phase
# ---------------------------------------------------------------------------
def test_liquid_conserved_by_agglomeration():
    """Agglomeration redistributes liquid but must not create or destroy it."""
    scenario = SCENARIOS["agg_shear_1d"]
    solver = build(scenario)

    # Seed a wet, porous population so the liquid paths are actually exercised.
    a = solver.a_tot
    rng = np.random.default_rng(4242)
    solver.porosity[:a] = 0.4
    solver.liquid_volume[:a] = rng.uniform(1e-22, 1e-21, a)
    solver.saturation[:a] = rng.uniform(0.1, 0.8, a)
    solver.V_flat[-1, :a] = solver.V_flat[0, :a] / (1.0 - 0.4)

    before = liquid_volume(solver)
    assert before > 0.0

    worst = 0.0
    for _ in range(200):
        if solver.a_tot < 2:
            break
        prev = liquid_volume(solver)
        solver._do_one_agg()
        solver._refresh_samplers_after_agg()  # see the note above
        worst = max(worst, _rel(liquid_volume(solver), prev))

    assert worst < CONSERVATION_RTOL, f"worst per-event liquid drift {worst:.3e}"


def test_liquid_conserved_by_breakage():
    """Fragments must share out exactly the parent's liquid."""
    scenario = SCENARIOS["break_powerlaw_1d"]
    solver = build(scenario)

    a = solver.a_tot
    rng = np.random.default_rng(99)
    solver.porosity[:a] = 0.35
    solver.liquid_volume[:a] = rng.uniform(1e-22, 1e-21, a)
    solver.saturation[:a] = 0.5
    solver.V_flat[-1, :a] = solver.V_flat[0, :a] / (1.0 - 0.35)

    worst = 0.0
    for _ in range(150):
        prev = liquid_volume(solver)
        solver._do_one_break()
        worst = max(worst, _rel(liquid_volume(solver), prev))

    assert worst < CONSERVATION_RTOL, f"worst per-event liquid drift {worst:.3e}"


def test_liquid_conserved_by_continuous_processes():
    """Internalisation and compression only move liquid between the internal and
    external reservoir; the per-particle total must not change."""
    scenario = SCENARIOS["granulation_1d"]
    solver = build(scenario)

    a = solver.a_tot
    rng = np.random.default_rng(7)
    solver.porosity[:a] = rng.uniform(0.35, 0.6, a)
    solver.liquid_volume[:a] = rng.uniform(1e-22, 1e-21, a)
    solver.saturation[:a] = rng.uniform(0.0, 0.9, a)
    solver.V_flat[-1, :a] = solver.V_flat[0, :a] / (1.0 - solver.porosity[:a])

    handler = solver.continuous_processes
    before = liquid_volume(solver)
    for _ in range(50):
        handler.step(0.0, 1e-3)
    after = liquid_volume(solver)

    assert _rel(after, before) < CONSERVATION_RTOL, (
        f"continuous processes changed total liquid by {_rel(after, before):.3e}"
    )


def test_saturation_never_implies_more_liquid_than_stored():
    """Internal liquid (V_pore * S) must never exceed the stored total liquid.

    Otherwise ``get_V_liquid_external`` clamps a negative value to zero and the
    accounting quietly invents liquid.
    """
    scenario = SCENARIOS["granulation_1d"]
    solver = build(scenario)
    solver.solve(maxiter=scenario.maxiter)

    a = solver.a_tot
    poro = solver.porosity[:a]
    porous = ~np.isnan(poro)
    if not np.any(porous):
        pytest.skip("no porous particles were produced")

    v_pore = solver.V_flat[-1, :a][porous] * poro[porous]
    v_int = v_pore * solver.saturation[:a][porous]
    v_total = solver.liquid_volume[:a][porous]

    # Allow a small relative slack for accumulated rounding.
    excess = v_int - v_total
    worst = float(np.max(excess / np.maximum(v_total, 1e-300)))
    assert worst < 1e-9, f"internal liquid exceeds stored liquid by {worst:.3e} (relative)"


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", SOLID_SCENARIOS)
def test_state_bounds(name):
    """Weights positive, porosity in [0,1) or NaN, saturation in [0,1], volumes > 0."""
    solver = build(SCENARIOS[name])
    solver.solve(maxiter=SCENARIOS[name].maxiter)

    a = solver.a_tot
    W = solver.W[:a]
    poro = solver.porosity[:a]
    sat = solver.saturation[:a]
    v_dry = solver.V_flat[-1, :a]

    assert np.all(W > 0.0), f"{name}: non-positive weights"
    assert np.all(np.isfinite(W)), f"{name}: non-finite weights"
    assert np.all(v_dry > 0.0), f"{name}: non-positive dry volumes"

    poro_ok = np.isnan(poro) | ((poro >= 0.0) & (poro < 1.0))
    assert np.all(poro_ok), f"{name}: porosity out of bounds"

    porous = ~np.isnan(poro)
    if np.any(porous):
        assert np.all(sat[porous] >= -1e-12), f"{name}: negative saturation"
        assert np.all(sat[porous] <= 1.0 + 1e-12), f"{name}: saturation above 1"


@pytest.mark.parametrize("name", SOLID_SCENARIOS)
def test_sampler_totals_match_arrays(name):
    """Fenwick totals must agree with a direct sum of the propensity arrays."""
    solver = build(SCENARIOS[name])
    solver.solve(maxiter=SCENARIOS[name].maxiter)

    a = solver.a_tot
    if solver._agg_sampler is not None:
        direct = float(np.sum(solver._r_agg[:a]))
        assert solver._agg_sampler.size() == a, f"{name}: agg sampler size mismatch"
        assert _rel(solver._agg_sampler.total(), direct) < 1e-9, (
            f"{name}: agg sampler total {solver._agg_sampler.total():.6e} != {direct:.6e}"
        )
    if solver._break_sampler is not None:
        direct = float(np.sum(solver._break_rate[:a]))
        assert solver._break_sampler.size() == a, f"{name}: break sampler size mismatch"
        assert _rel(solver._break_sampler.total(), direct) < 1e-9, (
            f"{name}: break sampler total {solver._break_sampler.total():.6e} != {direct:.6e}"
        )


# ---------------------------------------------------------------------------
# standalone report
# ---------------------------------------------------------------------------
def _report() -> int:
    warmup()
    print("\nSolid- and liquid-phase conservation over a full run")
    print(
        f"{'scenario':<22} {'solid before':>14} {'solid after':>14} {'rel drift':>11} "
        f"{'components':>11} {'liquid drift':>13}"
    )
    print("-" * 92)

    failures = 0
    for name in SOLID_SCENARIOS:
        scenario = SCENARIOS[name]
        solver = build(scenario)
        s0, l0 = solid_volume(solver), liquid_volume(solver)
        solver.solve(maxiter=scenario.maxiter)
        s1, l1 = solid_volume(solver), liquid_volume(solver)

        comp = _rel(component_solid_volume(solver), s1)
        drift = _rel(s1, s0)

        nucleation = getattr(solver, "nucleation", None)
        if nucleation is not None:
            expected_liquid = l0 + float(nucleation._liquid_volume_added_total)
        else:
            expected_liquid = l0
        if expected_liquid > 0.0:
            liq_drift = _rel(l1, expected_liquid)
        else:
            liq_drift = 0.0 if l1 == 0.0 else float('inf')

        flag = "" if drift < CONSERVATION_RTOL and comp < 1e-9 else "  <-- FAIL"
        if flag:
            failures += 1
        print(
            f"{name:<22} {s0:14.6e} {s1:14.6e} {drift:11.3e} {comp:11.3e} "
            f"{liq_drift:13.3e}{flag}"
        )

    print(
        "\n'components' compares sum(W*sum(V_flat[:dim])) against sum(W*V_dry*(1-poro)).\n"
        "'liquid drift' compares sum(W*liquid_volume) against initial + nucleation-reported input;\n"
        "a non-zero value there is a nucleation accounting issue, not a solver leak "
        "(see mcpbe/docs/historical/REFACTORING_FINDINGS.md, F-03)."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_report())
