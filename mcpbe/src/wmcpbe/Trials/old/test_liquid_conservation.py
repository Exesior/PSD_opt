"""
Liquid Volume Mass Conservation Test for WMCPBE Solver.

This test verifies that liquid volume is conserved during agglomeration 
and breakage events in the weighted Monte Carlo PBE solver.

Usage:
    cd mcpbe/src
    python -m wmcpbe.Trials.test_liquid_conservation

Or in IDE:
    runfile with wdir='.../mcpbe/src/wmcpbe/Trials'
"""

from __future__ import annotations

import numpy as np
import sys


def print_separator(title: str) -> None:
    """Print a visual separator."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def calculate_weighted_total(array: np.ndarray, weights: np.ndarray, n_active: int) -> float:
    """Calculate weighted total for active particles."""
    return float(np.sum(array[:n_active] * weights[:n_active]))


def test_agglomeration():
    """
    Test liquid conservation during pure agglomeration.
    
    Setup:
    - 100 initial particles
    - Each particle has 10% of its volume as liquid
    - Run 200 agglomeration events
    - Verify: total liquid mass before == total liquid mass after
    """
    print_separator("TEST 1: Agglomeration - Liquid Conservation")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    # Create solver with minimal config
    solver = MCPBESolver(
        dim=1,
        t_total=10.0,
        t_write=5,
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
    solver.a0 = 100
    solver.process_type = "agglomeration"
    
    # Physics: constant kernel
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-15
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    # Initialize
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    
    # Assign liquid: 10% of total particle volume
    if hasattr(solver, 'liquid_volume'):
        solver.liquid_volume[:n_init] = solver.V_flat[-1, :n_init] * 0.1
    else:
        print("ERROR: liquid_volume attribute not found!")
        return False
    
    # Initial totals (weighted!)
    initial_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_init)
    initial_solid = calculate_weighted_total(solver.V_flat[-1], solver.W, n_init)
    
    print(f"\nInitial state:")
    print(f"  Particles: {n_init}")
    print(f"  Total weight: {calculate_weighted_total(np.ones_like(solver.W), solver.W, n_init):.2f}")
    print(f"  Solid volume: {initial_solid:.6e} m³")
    print(f"  Liquid volume: {initial_liquid:.6e} m³")
    print(f"  Liquid fraction: {initial_liquid/initial_solid*100:.1f}%")
    
    # Run simulation (limited events for speed)
    solver.solve(maxiter=200)
    
    n_final = solver.a_tot
    final_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_final)
    final_solid = calculate_weighted_total(solver.V_flat[-1], solver.W, n_final)
    
    print(f"\nFinal state (after {solver._iter_count} events):")
    print(f"  Particles: {n_final}")
    print(f"  Total weight: {calculate_weighted_total(np.ones_like(solver.W), solver.W, n_final):.2f}")
    print(f"  Solid volume: {final_solid:.6e} m³")
    print(f"  Liquid volume: {final_liquid:.6e} m³")
    
    # Conservation check
    solid_error = abs(final_solid - initial_solid) / max(abs(initial_solid), 1e-300)
    liquid_error = abs(final_liquid - initial_liquid) / max(abs(initial_liquid), 1e-300)
    
    print(f"\nMass conservation:")
    print(f"  Solid error:  {solid_error:.6e} ({solid_error*100:.4f}%)")
    print(f"  Liquid error: {liquid_error:.6e} ({liquid_error*100:.4f}%)")
    
    solid_ok = solid_error < 1e-10
    liquid_ok = liquid_error < 1e-10
    
    print(f"\nSolid:  {'✅ PASS' if solid_ok else '❌ FAIL'}")
    print(f"Liquid: {'✅ PASS' if liquid_ok else '❌ FAIL'}")
    
    return liquid_ok and solid_ok


def test_breakage():
    """
    Test liquid conservation during pure breakage.
    
    Setup:
    - 50 initial particles (larger, for breakage)
    - Each particle has 10% liquid
    - Run 100 breakage events
    - Verify: total liquid mass conserved
    """
    print_separator("TEST 2: Breakage - Liquid Conservation")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=5.0,
        t_write=2,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    solver.c = np.array([1e-3])
    solver.x = np.array([5e-6])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 50
    solver.process_type = "breakage"
    
    # Breakage physics
    solver.BREAKRVAL = 1
    solver.BREAKFVAL = 2  # Binary breakage
    solver.pl_v = 1.0
    solver.pl_q = 1.0
    solver.pl_P1 = 0.5
    solver.pl_P2 = 1.0
    solver.G = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    
    # Assign liquid: 10% of total volume
    if hasattr(solver, 'liquid_volume'):
        solver.liquid_volume[:n_init] = solver.V_flat[-1, :n_init] * 0.1
    else:
        print("ERROR: liquid_volume attribute not found!")
        return False
    
    initial_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_init)
    initial_solid = calculate_weighted_total(solver.V_flat[-1], solver.W, n_init)
    
    print(f"\nInitial state:")
    print(f"  Particles: {n_init}")
    print(f"  Solid volume: {initial_solid:.6e} m³")
    print(f"  Liquid volume: {initial_liquid:.6e} m³")
    
    # Run limited events
    solver.solve(maxiter=100)
    
    n_final = solver.a_tot
    final_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_final)
    final_solid = calculate_weighted_total(solver.V_flat[-1], solver.W, n_final)
    
    print(f"\nFinal state (after {solver._iter_count} events):")
    print(f"  Particles: {n_final}")
    print(f"  Solid volume: {final_solid:.6e} m³")
    print(f"  Liquid volume: {final_liquid:.6e} m³")
    
    solid_error = abs(final_solid - initial_solid) / max(abs(initial_solid), 1e-300)
    liquid_error = abs(final_liquid - initial_liquid) / max(abs(initial_liquid), 1e-300)
    
    print(f"\nMass conservation:")
    print(f"  Solid error:  {solid_error:.6e} ({solid_error*100:.4f}%)")
    print(f"  Liquid error: {liquid_error:.6e} ({liquid_error*100:.4f}%)")
    
    solid_ok = solid_error < 1e-10
    liquid_ok = liquid_error < 1e-10
    
    print(f"\nSolid:  {'✅ PASS' if solid_ok else '❌ FAIL'}")
    print(f"Liquid: {'✅ PASS' if liquid_ok else '❌ FAIL'}")
    
    return liquid_ok and solid_ok


def test_mix():
    """
    Test liquid conservation during mixed agglomeration + breakage.
    """
    print_separator("TEST 3: Mix (Agg+Break) - Liquid Conservation")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=10.0,
        t_write=5,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    solver.c = np.array([1e-3])
    solver.x = np.array([3e-6])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 80
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
    solver.pl_P1 = 0.3
    solver.pl_P2 = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    if hasattr(solver, 'liquid_volume'):
        solver.liquid_volume[:n_init] = solver.V_flat[-1, :n_init] * 0.1
    else:
        print("ERROR: liquid_volume attribute not found!")
        return False
    
    initial_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_init)
    initial_solid = calculate_weighted_total(solver.V_flat[-1], solver.W, n_init)
    
    print(f"\nInitial state:")
    print(f"  Particles: {n_init}")
    print(f"  Solid volume: {initial_solid:.6e} m³")
    print(f"  Liquid volume: {initial_liquid:.6e} m³")
    
    solver.solve(maxiter=150)
    
    n_final = solver.a_tot
    final_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_final)
    final_solid = calculate_weighted_total(solver.V_flat[-1], solver.W, n_final)
    
    print(f"\nFinal state (after {solver._iter_count} events):")
    print(f"  Particles: {n_final}")
    print(f"  Solid volume: {final_solid:.6e} m³")
    print(f"  Liquid volume: {final_liquid:.6e} m³")
    
    solid_error = abs(final_solid - initial_solid) / max(abs(initial_solid), 1e-300)
    liquid_error = abs(final_liquid - initial_liquid) / max(abs(initial_liquid), 1e-300)
    
    print(f"\nMass conservation:")
    print(f"  Solid error:  {solid_error:.6e} ({solid_error*100:.4f}%)")
    print(f"  Liquid error: {liquid_error:.6e} ({liquid_error*100:.4f}%)")
    
    solid_ok = solid_error < 1e-10
    liquid_ok = liquid_error < 1e-10
    
    print(f"\nSolid:  {'✅ PASS' if solid_ok else '❌ FAIL'}")
    print(f"Liquid: {'✅ PASS' if liquid_ok else '❌ FAIL'}")
    
    return liquid_ok and solid_ok


def main():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("# WMCPBE Liquid Volume Mass Conservation Tests")
    print("#"*70)
    print("\nTesting: Σ(W_i × liquid_volume_i) = constant")
    
    results = []
    
    # Test 1
    try:
        passed = test_agglomeration()
        results.append(("Agglomeration", passed))
    except Exception as e:
        print(f"\n❌ Agglomeration EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Agglomeration", False))
    
    # Test 2
    try:
        passed = test_breakage()
        results.append(("Breakage", passed))
    except Exception as e:
        print(f"\n❌ Breakage EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Breakage", False))
    
    # Test 3
    try:
        passed = test_mix()
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
        print("\nDebug hints:")
        print("  - Check mcpbe_agg.py: liquid_volume update in _do_one_agg()")
        print("  - Check mcpbe_break.py: liquid_volume distribution in _break_apply_and_maintain()")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
