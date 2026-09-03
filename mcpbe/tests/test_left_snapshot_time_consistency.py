"""Time-consistency of the ``*_save_left`` snapshots (regression guard for F-07).

The ``*_save_left`` arrays recorded by ``MCPBESolver.solve()`` are supposed to
hold the population state *before* the event that produced a given output
snapshot (state at ``t_left``), the counterpart to the post-event ``*_save``
arrays (state at ``t_right``). ``compute_psd_cdf_over_time`` uses this
left/right bracketing for ``time_scheme`` in ``left`` / ``nearest`` / ``interp``
(``"interp"`` is the default of the repeat-PSD pipeline).

F-07 (``mcpbe/docs/historical/REFACTORING_FINDINGS.md``): only ``V_save_left``
and ``W_save_left`` used to be genuine pre-event copies.
``liquid_volume_save_left`` / ``porosity_save_left`` / ``saturation_save_left``
were filled from the *current, post-event* arrays, so combining them with
``V_save_left`` mixed state from two different instants.

These tests run a small wet-granulation config (liquid internalisation +
porosity compression enabled, so pre- and post-event porosity/saturation
actually differ) and assert:

1. The left snapshots are no longer byte-identical to the right snapshots -
   i.e. they are genuinely pre-event, not a second copy of the post-event
   arrays.

2. A consistency identity that only holds if ``W_save_left[k]``,
   ``V_save_left[k]`` and ``porosity_save_left[k]`` are all from the same
   instant: the represented solid volume ``sum(W * V_dry * (1 - porosity))``
   reconstructed purely from the *left* snapshots equals the (conserved) t=0
   solid volume at every save index.

3. ``compute_psd_cdf_over_time(time_scheme="interp")`` still returns a CDF list
   of the expected length.
"""

from __future__ import annotations

import numpy as np
import pytest

from .bench.scenarios import SCENARIOS, build

# Solid volume is conserved to machine precision; a few hundred events of
# floating-point accumulation land near 1e-14. 1e-9 catches any real time-mixing
# (which shows up as a percent-level error) with generous headroom.
CONSISTENCY_RTOL = 1e-9


def _solid_volume_from_snapshot(V_snap, W_snap, poro_snap) -> float:
    """sum_i W_i * V_solid_i for one snapshot triple.

    Vollkoerper convention (matching the rest of the code base): a NaN porosity
    marks a non-porous primary particle, for which V_solid == V_dry.
    """
    V_snap = np.asarray(V_snap, dtype=float)
    W = np.asarray(W_snap, dtype=float)
    poro = np.asarray(poro_snap, dtype=float)
    v_dry = V_snap[-1, :]
    v_solid = np.where(np.isnan(poro), v_dry, v_dry * (1.0 - np.nan_to_num(poro)))
    return float(np.sum(W * v_solid))


@pytest.fixture(scope="module")
def solved_granulation_solver():
    """A solved wet-granulation run (agglomeration + nucleation + liquid
    internalisation + porosity compression).

    The initial population is seeded wet and porous (as in
    ``test_conservation.test_liquid_conserved_by_continuous_processes``) so the
    porosity-compression and liquid-internalisation kernels actually move the
    continuous-phase state between ``t_left`` and ``t_right`` - otherwise
    porosity would sit at 0 and pre/post snapshots would be trivially equal.
    """
    scenario = SCENARIOS["granulation_1d"]
    solver = build(scenario)

    # Sanity: the continuous-process handler that makes pre/post state differ
    # must actually be attached, enabled, and doing porosity compression.
    handler = solver.continuous_processes
    assert handler is not None
    assert handler.config.enabled and handler.config.compression_enabled

    a = solver.a_tot
    rng = np.random.default_rng(20240607)
    solver.porosity[:a] = rng.uniform(0.40, 0.60, a)
    solver.liquid_volume[:a] = rng.uniform(1e-22, 1e-21, a)
    solver.saturation[:a] = rng.uniform(0.05, 0.85, a)
    solver.V_flat[-1, :a] = solver.V_flat[0, :a] / (1.0 - solver.porosity[:a])

    # The t=0 snapshot containers were filled by _initialize_particles() inside
    # build(), i.e. before this reseeding. Refresh every t=0 entry (left == right
    # at t=0) plus the recorded initial state so the whole trajectory is
    # self-consistent.
    solver.V0 = solver.V_flat[:, :a].copy()
    solver.W0 = solver.W[:a].copy()
    solver.porosity0 = solver.porosity[:a].copy()
    for name in ("V_save", "V_save_left"):
        getattr(solver, name)[0] = solver.V_flat[:, :a].copy()
    for name in ("W_save", "W_save_left"):
        getattr(solver, name)[0] = solver.W[:a].copy()
    for name in ("porosity_save", "porosity_save_left"):
        getattr(solver, name)[0] = solver.porosity[:a].copy()
    for name in ("saturation_save", "saturation_save_left"):
        getattr(solver, name)[0] = solver.saturation[:a].copy()
    for name in ("liquid_volume_save", "liquid_volume_save_left"):
        getattr(solver, name)[0] = solver.liquid_volume[:a].copy()

    solver.solve(maxiter=scenario.maxiter)
    return solver


def test_left_snapshot_arrays_are_populated(solved_granulation_solver):
    solver = solved_granulation_solver
    n = len(solver.V_save)
    assert n >= 2, "run produced too few output snapshots to test bracketing"
    for name in (
        "V_save_left",
        "W_save_left",
        "liquid_volume_save_left",
        "porosity_save_left",
        "saturation_save_left",
        "t_left",
        "t_right",
    ):
        arr = getattr(solver, name)
        assert len(arr) == n, f"{name} has {len(arr)} entries, expected {n}"


def test_left_snapshots_are_genuinely_pre_event(solved_granulation_solver):
    """The continuous-phase left snapshots must not just be a second copy of the
    post-event arrays.

    Before the F-07 fix, ``porosity_save_left[k]`` / ``saturation_save_left[k]``
    / ``liquid_volume_save_left[k]`` were appended from the very same
    ``self.<arr>[:a_tot]`` slice as the ``*_save`` arrays in the same loop
    iteration, so they were byte-identical (same length, same values). After the
    fix they are pre-event copies and differ on every save index where a
    continuous process ran between ``t_left`` and ``t_right``.
    """
    solver = solved_granulation_solver
    n = len(solver.V_save)

    def _any_differs(left_name, right_name):
        left = getattr(solver, left_name)
        right = getattr(solver, right_name)
        for k in range(1, n):  # k=0 is t=0, where pre == post by construction
            lk = np.asarray(left[k], dtype=float)
            rk = np.asarray(right[k], dtype=float)
            if lk.shape != rk.shape:
                return True
            if not np.array_equal(lk, rk, equal_nan=True):
                return True
        return False

    assert _any_differs("porosity_save_left", "porosity_save"), (
        "porosity_save_left is identical to porosity_save at every save index - "
        "it is still being recorded from the post-event array (F-07 not fixed)."
    )
    # Compression drives porosity down, so on affected indices the pre-event
    # (left) mean porosity must be strictly higher than the post-event (right).
    higher_before = False
    for k in range(1, n):
        lk = np.asarray(solver.porosity_save_left[k], dtype=float)
        rk = np.asarray(solver.porosity_save[k], dtype=float)
        if np.nanmean(lk) > np.nanmean(rk) + 1e-6:
            higher_before = True
            break
    assert higher_before, (
        "pre-event porosity is never higher than post-event porosity - "
        "the left snapshot does not predate the compression step."
    )


def test_left_snapshots_are_time_consistent(solved_granulation_solver):
    """Solid volume reconstructed purely from the left snapshots must equal the
    conserved t=0 value at every save index.

    ``V_solid = W * V_dry * (1 - porosity)`` only comes out right if
    ``W_save_left[k]``, ``V_save_left[k][-1]`` and ``porosity_save_left[k]`` are
    all from the same instant. With the pre-F-07 code, ``V_save_left`` is
    pre-event but ``porosity_save_left`` is post-event, so this identity is
    violated (by a percent-level amount) on every index where compression acted
    between the two times.
    """
    solver = solved_granulation_solver
    n = len(solver.V_save)

    # The t=0 left snapshot is the (pre==post) initial state; total represented
    # solid volume is conserved by every process, so this is the reference value
    # for every later left snapshot.
    solid_0 = _solid_volume_from_snapshot(
        solver.V_save_left[0], solver.W_save_left[0], solver.porosity_save_left[0]
    )
    assert solid_0 > 0.0, "degenerate initial state"

    for k in range(n):
        V_l = np.asarray(solver.V_save_left[k], dtype=float)
        W_l = np.asarray(solver.W_save_left[k], dtype=float)
        poro_l = np.asarray(solver.porosity_save_left[k], dtype=float)
        # The three left arrays must describe the same particle set. With the
        # pre-F-07 code V_save_left is a pre-event copy while porosity_save_left
        # is post-event, so on indices where an agglomeration/nucleation event
        # changed the particle count they even have different lengths.
        assert V_l.shape[1] == W_l.shape[0] == poro_l.shape[0], (
            f"left snapshot {k}: V_save_left has {V_l.shape[1]} particles, "
            f"W_save_left {W_l.shape[0]}, porosity_save_left {poro_l.shape[0]} - "
            f"they are not from the same instant."
        )
        solid_k = _solid_volume_from_snapshot(V_l, W_l, poro_l)
        rel = abs(solid_k - solid_0) / solid_0
        assert rel < CONSISTENCY_RTOL, (
            f"left snapshot {k}: solid volume reconstructed from "
            f"(W_save_left, V_save_left, porosity_save_left) is {solid_k:.6e}, "
            f"expected the conserved t=0 value {solid_0:.6e} "
            f"(relative error {rel:.3e}). The left snapshots are not all from "
            f"the same instant."
        )


def test_interp_time_scheme_still_returns_full_cdf_list(solved_granulation_solver):
    solver = solved_granulation_solver
    cdf_list, t_vec = solver.compute_psd_cdf_over_time(
        psd_basis="volume", time_scheme="interp"
    )
    expected = min(len(solver.V_save), len(solver.t_vec))
    assert len(cdf_list) == expected
    assert len(t_vec) == expected
    assert expected >= 2
    # interp needs both brackets; with a few hundred particles present at every
    # save index the first and last entries must be usable (x, Q) pairs.
    assert cdf_list[0] is not None and cdf_list[-1] is not None
