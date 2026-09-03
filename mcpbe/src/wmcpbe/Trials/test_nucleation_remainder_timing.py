"""
Test: the sub-droplet liquid remainder is placed at window close, not at t_total.

Background
---------
During the liquid-addition window ``NucleationHandler.step()`` distributes only
whole droplets per MC event and carries the fractional part forward in
``_liquid_remainder``. Once the window closes, ``step()`` early-returns on every
later event and never touches the remainder again.

Before the fix, that leftover (< 1 droplet) was placed only in
``finalize_after_solve`` -> ``_distribute_remaining_liquid``, timestamped at the
END of the simulation. For a run whose window closes well before ``t_total`` the
liquid then sat "unadded" through the whole post-window granulation phase and
appeared at the very end, where it could no longer take part in agglomeration.

The fix: the MC event whose interval straddles ``window_end`` flushes the
remainder immediately, so it enters the population at window close.

What this checks
---------------
1. The remainder path is actually exercised (flow rate tuned so a sub-droplet
   fraction carries past the window).
2. ``_distribute_remaining_liquid`` fires at ``solver._elapsed ~ window_end``,
   nowhere near ``t_total``.
3. MC events still happen after the flush -- the liquid had a chance to act.
4. Mass conservation stays exact: total liquid == initial + reported input, and
   the reported input == flow_rate * duration to within one droplet volume.
5. The normal path (window covers the whole run) is untouched.

Run: PYTHONIOENCODING=utf-8 python Trials/test_nucleation_remainder_timing.py
(also runs under pytest: pytest Trials/test_nucleation_remainder_timing.py)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver


# =============================================================================
# Configuration
# =============================================================================

SEED = 20240607

T_TOTAL = 4.0                 # total simulated time [s]
N_SAVE = 9

# Solid phase: same shape as the `granulation_1d` benchmark scenario
# (mcpbe/tests/bench/scenarios.py) -- 1 um mono-disperse primaries at 1 % solid
# volume fraction, which gives a number density that actually produces
# agglomeration events.
PARTICLE_SIZE = 1e-6          # x [m]
SOLID_VOLUME_FRACTION = 1e-2  # c [-]
INITIAL_PARTICLES = 400       # a0
INITIAL_POROSITY = 0.4

DROPLET_DIAMETER = 1e-7       # m  -> v_droplet ~ 5.24e-22 m^3

# Flow rate deliberately low: the accumulation over the final in-window event is
# less than one droplet, so a real sub-droplet remainder is carried past the
# window instead of being flushed by the whole-droplet path inside
# _distribute_liquid_volume.
VOLUMETRIC_FLOW_RATE = 5e-22  # m^3/s

WINDOW_END = 1.0             # liquid_addition_duration for the main test [s]
MAXITER = 4000              # high enough that the run ends by time, not budget


# =============================================================================
# Helpers
# =============================================================================

def _liquid_total(solver) -> float:
    """Represented liquid volume, sum_i W_i * liquid_volume_i [m^3]."""
    a = solver.a_tot
    if a <= 0:
        return 0.0
    return float(np.sum(solver.W[:a] * solver.liquid_volume[:a]))


def _build_solver(addition_duration: float) -> MCPBESolver:
    """Granulation-style solver: agglomeration + nucleation + continuous processes.

    Solid-phase setup mirrors the ``granulation_1d`` benchmark scenario so the
    collision rate is in a regime that produces events.
    """
    t_vec = np.linspace(0.0, T_TOTAL, N_SAVE)

    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=False,
        load_attr=False,
        init=False,
        seed=SEED,
        agg_kernel_name="shear_chin1998",
        agg_kernel_params={"corr_beta": 1e-3, "g": 1000.0},
        porosity_growth_kernel_name="volume_mixing",
        porosity_growth_kernel_params={},
    )
    solver.process_type = "agglomeration"
    solver.c = np.full(1, SOLID_VOLUME_FRACTION)
    solver.x = np.full(1, PARTICLE_SIZE)
    solver.PGV = np.full(1, "mono")
    solver.SIG = np.full(1, 0.1)
    solver.a0 = INITIAL_PARTICLES
    solver.alpha_prim = np.ones(1)
    solver.maybe_double_control_volume = False

    solver._initialize_particles()
    solver._init_lmc()
    solver.porosity[:solver.a_tot] = INITIAL_POROSITY
    solver.V_flat[-1, :solver.a_tot] = solver.V_flat[0, :solver.a_tot] / (1.0 - INITIAL_POROSITY)
    solver.liquid_volume[:solver.a_tot] = 0.0
    solver.saturation[:solver.a_tot] = 0.0
    solver._initialize_samplers()

    solver.create_nucleation_handler(
        enabled=True,
        volumetric_flow_rate=VOLUMETRIC_FLOW_RATE,
        droplet_diameter=DROPLET_DIAMETER,
        liquid_addition_start=0.0,
        liquid_addition_duration=addition_duration,
        batch_size=25.0,
    )
    solver.create_continuous_processes_handler(
        enabled=True,
        k_int=1e12,
        compression_enabled=True,
        compression_rate=0.02,
        min_porosity=0.3,
    )
    return solver


def _instrument(handler):
    """Record every _distribute_remaining_liquid call with the live solver clock."""
    calls = []
    original = handler._distribute_remaining_liquid
    solver = handler.solver

    def spy(at_time):
        calls.append({
            "at_time": float(at_time),
            "solver_elapsed": float(getattr(solver, "_elapsed", float("nan"))),
            "iter_count": int(getattr(solver, "_iter_count", -1)),
            "remainder_before": float(handler._liquid_remainder),
        })
        return original(at_time)

    handler._distribute_remaining_liquid = spy
    return calls


def _assert_liquid_conserved(solver, handler, liquid_before):
    """sum(W * liquid) must equal initial + what nucleation reports adding."""
    liquid_after = _liquid_total(solver)
    added_total = float(handler._liquid_volume_added_total)
    expected = liquid_before + added_total
    rel = abs(liquid_after - expected) / max(abs(expected), 1e-300)
    assert rel < 1e-9, (
        f"liquid balance off by {rel:.3e}: have {liquid_after:.6e}, "
        f"expected initial+input = {expected:.6e}"
    )
    v_droplet = handler.config.droplet_volume
    assert handler._liquid_remainder <= 1e-3 * v_droplet, (
        f"{handler._liquid_remainder / v_droplet:.3f} droplets left unplaced in "
        f"_liquid_remainder"
    )
    return liquid_after, added_total


# =============================================================================
# Test 1: remainder lands at window_end
# =============================================================================

def check_remainder_flushed_at_window_end(verbose: bool = True) -> None:
    solver = _build_solver(addition_duration=WINDOW_END)
    handler = solver.nucleation
    calls = _instrument(handler)
    v_droplet = handler.config.droplet_volume
    window_end = handler.config.liquid_addition_end
    assert abs(window_end - WINDOW_END) < 1e-12

    liquid_before = _liquid_total(solver)
    solver.solve(maxiter=MAXITER)

    final_elapsed = float(solver._elapsed)
    final_events = int(solver._iter_count)

    if verbose:
        print(f"  window_end={window_end:.4f}s  t_total={T_TOTAL:.4f}s  "
              f"final _elapsed={final_elapsed:.4f}s  events={final_events}")
        for c in calls:
            print(f"  flush: at_time={c['at_time']:.4f}s  _elapsed={c['solver_elapsed']:.4f}s  "
                  f"iter={c['iter_count']}  remainder={c['remainder_before'] / v_droplet:.3f} droplets")

    # The run must end by simulated time, not the event budget -- otherwise the
    # window never closes during solve() and this test is meaningless.
    assert final_elapsed >= T_TOTAL, (
        f"run stopped early at {final_elapsed:.3f}s (events={final_events}); "
        f"raise MAXITER or adjust the scenario"
    )

    # 1. the remainder path was actually exercised, once, with a sub-droplet load
    assert len(calls) == 1, (
        f"expected exactly one _distribute_remaining_liquid call, got {len(calls)}; "
        f"re-tune VOLUMETRIC_FLOW_RATE so a sub-droplet fraction carries past the window"
    )
    flush = calls[0]
    frac = flush["remainder_before"] / v_droplet
    assert 0.02 < frac < 1.0, (
        f"flushed remainder is {frac:.3f} droplets, not a sub-droplet fraction; "
        f"scenario no longer covers the intended path"
    )

    # 2. it fired at window close, nowhere near t_total. The straddling event
    #    ends just past window_end; a quarter of the post-window span is generous
    #    head-room and still far from the end.
    latest_ok = window_end + 0.25 * (T_TOTAL - window_end)
    assert flush["solver_elapsed"] <= latest_ok, (
        f"flush at _elapsed={flush['solver_elapsed']:.4f}s, expected <= {latest_ok:.4f}s "
        f"(just after window_end={window_end}s, not near t_total={T_TOTAL}s)"
    )
    assert abs(flush["at_time"] - window_end) < 1e-12, (
        f"_distribute_remaining_liquid got at_time={flush['at_time']}, expected "
        f"window_end={window_end}"
    )

    # 3. MC events still happened after the flush -> the liquid could take part
    events_after = final_events - flush["iter_count"]
    assert events_after > 0, (
        f"no MC events after the flush (iter {flush['iter_count']} -> {final_events})"
    )

    # 4. mass conservation
    liquid_after, added_total = _assert_liquid_conserved(solver, handler, liquid_before)
    expected_input = VOLUMETRIC_FLOW_RATE * window_end
    assert abs(added_total - expected_input) <= v_droplet, (
        f"reported input {added_total:.6e} differs from flow*duration "
        f"{expected_input:.6e} by more than one droplet ({v_droplet:.3e})"
    )

    if verbose:
        print(f"  OK: flush at window close, {events_after} events followed, "
              f"liquid conserved, input == flow*duration")


# =============================================================================
# Test 2: regression guard -- window covers the whole run
# =============================================================================

def check_window_covers_full_run(verbose: bool = True) -> None:
    solver = _build_solver(addition_duration=T_TOTAL + 10.0)
    handler = solver.nucleation
    calls = _instrument(handler)
    window_end = handler.config.liquid_addition_end

    liquid_before = _liquid_total(solver)
    solver.solve(maxiter=MAXITER)

    if verbose:
        print(f"  final _elapsed={solver._elapsed:.4f}s  events={solver._iter_count}")
        for c in calls:
            print(f"  flush: at_time={c['at_time']:.4f}s  _elapsed={c['solver_elapsed']:.4f}s")

    # window never closes during the run -> the straddle branch must not fire
    for c in calls:
        assert abs(c["at_time"] - window_end) > 1e-9, (
            "straddle flush fired although the addition window never closes"
        )

    _assert_liquid_conserved(solver, handler, liquid_before)
    if verbose:
        print("  OK: normal end-of-run path intact, liquid conserved")


# =============================================================================
# Entry points
# =============================================================================

def test_remainder_flushed_at_window_end():
    check_remainder_flushed_at_window_end(verbose=False)


def test_window_covers_full_run():
    check_window_covers_full_run(verbose=False)


def main() -> bool:
    print("=" * 78)
    print("NUCLEATION REMAINDER TIMING")
    print("=" * 78)

    cases = [
        ("remainder is flushed at window close, not t_total",
         check_remainder_flushed_at_window_end),
        ("window spanning the whole run: normal path intact",
         check_window_covers_full_run),
    ]

    results = []
    for title, fn in cases:
        print(f"\n{title}")
        try:
            fn(verbose=True)
            print("  PASS")
            results.append(True)
        except AssertionError as exc:
            print(f"  FAIL: {exc}")
            results.append(False)

    print("\n" + "=" * 78)
    ok = all(results)
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
