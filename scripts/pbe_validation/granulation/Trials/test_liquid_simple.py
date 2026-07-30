"""Simple liquid mass conservation test - direct solver access.

This bypasses the validation framework and directly tests the solver
to identify where liquid mass conservation might be failing.
"""

from __future__ import annotations

import numpy as np
import sys
import os

# Add wmcpbe to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'mcpbe', 'src'))

from wmcpbe.mcpbe import MCPBESolver


def print_separator(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_agglomeration_simple():
    """Minimal agglomeration test with liquid volume tracking."""
    print_separator("TEST 1: Simple Agglomeration with Liquid")
    
    # Create minimal solver
    solver = MCPBESolver(
        dim=1,
        t_total=10.0,
        t_write=5,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    # Set minimal parameters
    solver.c = np.array([1e-3])
    solver.x = np.array([2e-6])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 100  # Few particles for speed
    solver.process_type = "agglomeration"
    
    # Physics parameters
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-9
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    # Initialize
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    print(f"\nInitial state:")
    print(f"  Particles: {n_init}")
    print(f"  Total weight: {np.sum(solver.W[:n_init]):.2f}")
    
    # Assign liquid volume: 10% of particle volume
    solver.liquid_volume[:n_init] = solver.V_flat[-1, :n_init] * 0.1
    
    # Calculate initial weighted total liquid
    initial_liquid = np.sum(solver.liquid_volume[:n_init] * solver.W[:n_init])
    initial_solid = np.sum(solver.V_flat[-1, :n_init] * solver.W[:n_init])
    print(f"  Total solid volume: {initial_solid:.6e} m³")
    print(f"  Total liquid volume: {initial_liquid:.6e} m³")
    
    # Run a few events only
    solver.solve(maxiter=100)
    
    n_final = solver.a_tot
    final_liquid = np.sum(solver.liquid_volume[:n_final] * solver.W[:n_final])
    final_solid = np.sum(solver.V_flat[-1, :n_final] * solver.W[:n_final])
    
    print(f"\nFinal state (after {solver._iter_count} events):")
    print(f"  Particles: {n_final}")
    print(f"  Total weight: {np.sum(solver.W[:n_final]):.2f}")
    print(f"  Total solid volume: {final_solid:.6e} m³")
    print(f"  Total liquid volume: {final_liquid:.6e} m³")
    
    # Check conservation
    solid_error = abs(final_solid - initial_solid) / max(initial_solid, 1e-300)
    liquid_error = abs(final_liquid - initial_liquid) / max(initial_liquid, 1e-300)
    
    print(f"\nConservation check:")
    print(f"  Solid relative error: {solid_error:.6e} ({solid_error*100:.4f}%)")
    print(f"  Liquid relative error: {liquid_error:.6e} ({liquid_error*100:.4f}%)")
    
    solid_ok = solid_error < 1e-10
    liquid_ok = liquid_error < 1e-10
    
    print(f"\nSolid conservation: {'✅ PASS' if solid_ok else '❌ FAIL'}")
    print(f"Liquid conservation: {'✅ PASS' if liquid_ok else '❌ FAIL'}")
    
    return liquid_ok and solid_ok


def test_breakage_simple():
    """Minimal breakage test with liquid volume tracking."""
    print_separator("TEST 2: Simple Breakage with Liquid")
    
    # Create minimal solver
    solver = MCPBESolver(
        dim=1,
        t_total=5.0,  # Shorter time
        t_write=2,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    # Set minimal parameters
    solver.c = np.array([1e-3])
    solver.x = np.array([5e-6])  # Smaller than before
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 50  # Even fewer particles
    solver.process_type = "breakage"
    
    # Physics parameters for breakage
    solver.BREAKRVAL = 1
    solver.BREAKFVAL = 2  # Binary breakage
    solver.pl_v = 1.0
    solver.pl_q = 1.0
    solver.pl_P1 = 0.5  # Breakage rate
    solver.pl_P2 = 1.0
    solver.G = 1.0
    
    # Initialize
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    print(f"\nInitial state:")
    print(f"  Particles: {n_init}")
    print(f"  Total weight: {np.sum(solver.W[:n_init]):.2f}")
    
    # Assign liquid volume: 10% of particle volume
    solver.liquid_volume[:n_init] = solver.V_flat[-1, :n_init] * 0.1
    
    # Calculate initial weighted total liquid
    initial_liquid = np.sum(solver.liquid_volume[:n_init] * solver.W[:n_init])
    initial_solid = np.sum(solver.V_flat[-1, :n_init] * solver.W[:n_init])
    print(f"  Total solid volume: {initial_solid:.6e} m³")
    print(f"  Total liquid volume: {initial_liquid:.6e} m³")
    
    # Run very few events
    solver.solve(maxiter=50)
    
    n_final = solver.a_tot
    final_liquid = np.sum(solver.liquid_volume[:n_final] * solver.W[:n_final])
    final_solid = np.sum(solver.V_flat[-1, :n_final] * solver.W[:n_final])
    
    print(f"\nFinal state (after {solver._iter_count} events):")
    print(f"  Particles: {n_final}")
    print(f"  Total weight: {np.sum(solver.W[:n_final]):.2f}")
    print(f"  Total solid volume: {final_solid:.6e} m³")
    print(f"  Total liquid volume: {final_liquid:.6e} m³")
    
    # Check conservation
    solid_error = abs(final_solid - initial_solid) / max(initial_solid, 1e-300)
    liquid_error = abs(final_liquid - initial_liquid) / max(initial_liquid, 1e-300)
    
    print(f"\nConservation check:")
    print(f"  Solid relative error: {solid_error:.6e} ({solid_error*100:.4f}%)")
    print(f"  Liquid relative error: {liquid_error:.6e} ({liquid_error*100:.4f}%)")
    
    solid_ok = solid_error < 1e-10
    liquid_ok = liquid_error < 1e-10
    
    print(f"\nSolid conservation: {'✅ PASS' if solid_ok else '❌ FAIL'}")
    print(f"Liquid conservation: {'✅ PASS' if liquid_ok else '❌ FAIL'}")
    
    return liquid_ok and solid_ok


def main():
    print("\n" + "#" * 70)
    print("# Simple Liquid Mass Conservation Tests")
    print("#"*70)
    
    results = []
    
    # Test 1: Agglomeration
    try:
        passed = test_agglomeration_simple()
        results.append(("Agglomeration", passed))
    except Exception as e:
        print(f"\n❌ Agglomeration test EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Agglomeration", False))
    
    # Test 2: Breakage
    try:
        passed = test_breakage_simple()
        results.append(("Breakage", passed))
    except Exception as e:
        print(f"\n❌ Breakage test EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Breakage", False))
    
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
        print("\nNext steps:")
        print("  1. Check error messages above")
        print("  2. Verify liquid_volume updates in mcpbe_agg.py / mcpbe_break.py")
        print("  3. Add debug prints to track liquid during events")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
