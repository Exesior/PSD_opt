"""
Simple Monte Carlo Liquid Volume Test - Proof of Concept.

This is a minimal test to verify liquid volume behavior during
agglomeration and breakage events. Designed for small compute capacity.

Usage:
    cd mcpbe/src
    python -m wmcpbe.Trials.test_liquid_mc_simple
"""

from __future__ import annotations

import numpy as np
import sys


def print_separator(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def calculate_weighted_total(array: np.ndarray, weights: np.ndarray, n_active: int) -> float:
    """Calculate weighted total for active particles."""
    if n_active <= 0:
        return 0.0
    return float(np.sum(array[:n_active] * weights[:n_active]))


def run_agglomeration_mc():
    """
    Minimal agglomeration test with detailed tracking.
    
    Setup:
    - 20 initial particles (very small!)
    - Each has 10% liquid fraction
    - Run only 15 events (proof of concept)
    - Track each event's mass balance
    """
    print_separator("TEST 1: Agglomeration MC (Small Scale)")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=5.0,
        t_write=3,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    # Basic parameters
    solver.c = np.array([1e-3])
    solver.x = np.array([2e-6])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 20  # Very few particles!
    solver.process_type = "agglomeration"
    
    # Physics: constant kernel
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-9
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    # Initialize
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    print(f"\nInitial setup:")
    print(f"  Particles: {n_init}")
    print(f"  Total weight: {np.sum(solver.W[:n_init]):.2f}")
    
    # Assign liquid: 10% of total particle volume
    if not hasattr(solver, 'liquid_volume'):
        print("ERROR: liquid_volume attribute missing!")
        return False
    
    solver.liquid_volume[:n_init] = solver.V_flat[-1, :n_init] * 0.1
    
    # Initial state
    initial_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_init)
    initial_solid = calculate_weighted_total(solver.V_flat[-1], solver.W, n_init)
    
    print(f"\nInitial masses:")
    print(f"  Solid:  {initial_solid:.6e} m³")
    print(f"  Liquid: {initial_liquid:.6e} m³")
    print(f"  Ratio:  {initial_liquid/initial_solid*100:.1f}%")
    
    # Run very few events for proof of concept
    max_events = 15
    print(f"\nRunning {max_events} agglomeration events...")
    
    # Manual event loop with tracking
    event_count = 0
    history = []
    
    for evt in range(max_events):
        if solver.a_tot < 2:
            print(f"  Stopped early: only {solver.a_tot} particles left")
            break
        
        # Record state before event
        liq_before = calculate_weighted_total(solver.liquid_volume, solver.W, solver.a_tot)
        solid_before = calculate_weighted_total(solver.V_flat[-1], solver.W, solver.a_tot)
        
        # Execute one agglomeration event
        solver._do_one_agg()
        
        # Record state after event
        liq_after = calculate_weighted_total(solver.liquid_volume, solver.W, solver.a_tot)
        solid_after = calculate_weighted_total(solver.V_flat[-1], solver.W, solver.a_tot)
        
        event_count += 1
        
        # Track relative errors
        liq_err = abs(liq_after - liq_before) / max(abs(liq_before), 1e-300)
        solid_err = abs(solid_after - solid_before) / max(abs(solid_before), 1e-300)
        
        history.append({
            'event': evt,
            'particles': solver.a_tot,
            'liquid': liq_after,
            'solid': solid_after,
            'liq_error': liq_err,
            'solid_error': solid_err,
        })
        
        # Print every 5th event
        if evt % 5 == 0 or evt == max_events - 1:
            print(f"  Event {evt:3d}: particles={solver.a_tot:3d}, "
                  f"liquid={liq_after:.6e}, error={liq_err:.2e}")
    
    # Final check
    n_final = solver.a_tot
    final_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_final)
    final_solid = calculate_weighted_total(solver.V_flat[-1], solver.W, n_final)
    
    total_liq_error = abs(final_liquid - initial_liquid) / max(abs(initial_liquid), 1e-300)
    total_solid_error = abs(final_solid - initial_solid) / max(abs(initial_solid), 1e-300)
    
    print(f"\nFinal state (after {event_count} events):")
    print(f"  Particles: {n_final}")
    print(f"  Solid:  {final_solid:.6e} m³  (error: {total_solid_error:.2e})")
    print(f"  Liquid: {final_liquid:.6e} m³  (error: {total_liq_error:.2e})")
    
    # Check conservation
    solid_ok = total_solid_error < 1e-10
    liquid_ok = total_liq_error < 1e-10
    
    print(f"\nConservation check:")
    print(f"  Solid:  {'✅ PASS' if solid_ok else '❌ FAIL'}")
    print(f"  Liquid: {'✅ PASS' if liquid_ok else '❌ FAIL'}")
    
    return liquid_ok and solid_ok


def run_breakage_mc():
    """
    Minimal breakage test with detailed tracking.
    
    Setup:
    - 15 initial particles (very small!)
    - Each has 10% liquid fraction
    - Run only 10 breakage events
    - Binary breakage (2 fragments)
    """
    print_separator("TEST 2: Breakage MC (Small Scale)")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=3.0,
        t_write=2,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    solver.c = np.array([1e-3])
    solver.x = np.array([5e-6])  # Larger for breakage
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 15  # Very few particles!
    solver.process_type = "breakage"
    
    # Breakage physics - simple binary breakage
    solver.BREAKRVAL = 1
    solver.BREAKFVAL = 2  # Binary: 2 fragments
    solver.pl_v = 1.0
    solver.pl_q = 1.0
    solver.pl_P1 = 0.3  # Moderate breakage rate
    solver.pl_P2 = 1.0
    solver.G = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    print(f"\nInitial setup:")
    print(f"  Particles: {n_init}")
    print(f"  Total weight: {np.sum(solver.W[:n_init]):.2f}")
    
    # Assign liquid: 10% of total volume
    if not hasattr(solver, 'liquid_volume'):
        print("ERROR: liquid_volume attribute missing!")
        return False
    
    solver.liquid_volume[:n_init] = solver.V_flat[-1, :n_init] * 0.1
    
    initial_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_init)
    initial_solid = calculate_weighted_total(solver.V_flat[-1], solver.W, n_init)
    
    print(f"\nInitial masses:")
    print(f"  Solid:  {initial_solid:.6e} m³")
    print(f"  Liquid: {initial_liquid:.6e} m³")
    print(f"  Ratio:  {initial_liquid/initial_solid*100:.1f}%")
    
    # Run very few events
    max_events = 10
    print(f"\nRunning {max_events} breakage events...")
    
    event_count = 0
    
    for evt in range(max_events):
        if solver.a_tot < 1:
            print(f"  Stopped early: no particles left")
            break
        
        # Record before
        liq_before = calculate_weighted_total(solver.liquid_volume, solver.W, solver.a_tot)
        solid_before = calculate_weighted_total(solver.V_flat[-1], solver.W, solver.a_tot)
        
        # Execute one breakage event
        solver._do_one_break()
        
        # Record after
        liq_after = calculate_weighted_total(solver.liquid_volume, solver.W, solver.a_tot)
        solid_after = calculate_weighted_total(solver.V_flat[-1], solver.W, solver.a_tot)
        
        event_count += 1
        
        liq_err = abs(liq_after - liq_before) / max(abs(liq_before), 1e-300)
        solid_err = abs(solid_after - solid_before) / max(abs(solid_before), 1e-300)
        
        if evt % 3 == 0 or evt == max_events - 1:
            print(f"  Event {evt:3d}: particles={solver.a_tot:3d}, "
                  f"liquid={liq_after:.6e}, error={liq_err:.2e}")
    
    # Final check
    n_final = solver.a_tot
    final_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_final)
    final_solid = calculate_weighted_total(solver.V_flat[-1], solver.W, n_final)
    
    total_liq_error = abs(final_liquid - initial_liquid) / max(abs(initial_liquid), 1e-300)
    total_solid_error = abs(final_solid - initial_solid) / max(abs(initial_solid), 1e-300)
    
    print(f"\nFinal state (after {event_count} events):")
    print(f"  Particles: {n_final}")
    print(f"  Solid:  {final_solid:.6e} m³  (error: {total_solid_error:.2e})")
    print(f"  Liquid: {final_liquid:.6e} m³  (error: {total_liq_error:.2e})")
    
    solid_ok = total_solid_error < 1e-10
    liquid_ok = total_liq_error < 1e-10
    
    print(f"\nConservation check:")
    print(f"  Solid:  {'✅ PASS' if solid_ok else '❌ FAIL'}")
    print(f"  Liquid: {'✅ PASS' if liquid_ok else '❌ FAIL'}")
    
    return liquid_ok and solid_ok


def run_mix_mc():
    """
    Mixed agglomeration + breakage test.
    
    Setup:
    - 25 initial particles
    - 10% liquid fraction
    - Run 20 mixed events
    """
    print_separator("TEST 3: Mix MC (Agg+Break, Small Scale)")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=5.0,
        t_write=3,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    solver.c = np.array([1e-3])
    solver.x = np.array([3e-6])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 25
    solver.process_type = "mix"
    
    # Agglomeration
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-9
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    # Breakage
    solver.BREAKRVAL = 1
    solver.BREAKFVAL = 2
    solver.pl_v = 1.0
    solver.pl_q = 1.0
    solver.pl_P1 = 0.2
    solver.pl_P2 = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    
    if not hasattr(solver, 'liquid_volume'):
        print("ERROR: liquid_volume attribute missing!")
        return False
    
    solver.liquid_volume[:n_init] = solver.V_flat[-1, :n_init] * 0.1
    
    initial_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_init)
    initial_solid = calculate_weighted_total(solver.V_flat[-1], solver.W, n_init)
    
    print(f"\nInitial:")
    print(f"  Particles: {n_init}")
    print(f"  Solid:  {initial_solid:.6e} m³")
    print(f"  Liquid: {initial_liquid:.6e} m³")
    
    # Run mixed events
    max_events = 20
    print(f"\nRunning {max_events} mixed events...")
    
    for evt in range(max_events):
        if solver.a_tot < 1:
            break
        
        pt = getattr(solver, "process_type", "mix")
        
        # Choose event type based on propensities
        agg_prop = float(solver._agg_sampler.total()) if solver._agg_sampler else 0.0
        break_prop = float(solver._break_sampler.total()) if solver._break_sampler else 0.0
        total_prop = agg_prop + break_prop
        
        if total_prop <= 0:
            break
        
        # Stochastic choice
        if solver._rng.random() < (agg_prop / total_prop):
            solver._do_one_agg()
        else:
            solver._do_one_break()
        
        if evt % 5 == 0:
            liq = calculate_weighted_total(solver.liquid_volume, solver.W, solver.a_tot)
            print(f"  Event {evt:3d}: particles={solver.a_tot:3d}, liquid={liq:.6e}")
    
    n_final = solver.a_tot
    final_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_final)
    final_solid = calculate_weighted_total(solver.V_flat[-1], solver.W, n_final)
    
    total_liq_error = abs(final_liquid - initial_liquid) / max(abs(initial_liquid), 1e-300)
    total_solid_error = abs(final_solid - initial_solid) / max(abs(initial_solid), 1e-300)
    
    print(f"\nFinal (after events):")
    print(f"  Particles: {n_final}")
    print(f"  Solid error:  {total_solid_error:.2e}")
    print(f"  Liquid error: {total_liq_error:.2e}")
    
    solid_ok = total_solid_error < 1e-10
    liquid_ok = total_liq_error < 1e-10
    
    print(f"\nSolid:  {'✅ PASS' if solid_ok else '❌ FAIL'}")
    print(f"Liquid: {'✅ PASS' if liquid_ok else '❌ FAIL'}")
    
    return liquid_ok and solid_ok


def main():
    print("\n" + "#" * 70)
    print("# Monte Carlo Liquid Volume Tests - PROOF OF CONCEPT")
    print("# Small scale for limited compute capacity")
    print("#"*70)
    
    results = []
    
    # Test 1: Agglomeration
    try:
        passed = run_agglomeration_mc()
        results.append(("Agglomeration", passed))
    except Exception as e:
        print(f"\n❌ Agglomeration EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Agglomeration", False))
    
    # Test 2: Breakage
    try:
        passed = run_breakage_mc()
        results.append(("Breakage", passed))
    except Exception as e:
        print(f"\n❌ Breakage EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Breakage", False))
    
    # Test 3: Mix
    try:
        passed = run_mix_mc()
        results.append(("Mix", passed))
    except Exception as e:
        print(f"\n❌ Mix EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Mix", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED! 🎉")
    else:
        print("SOME TESTS FAILED! ⚠️")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
