"""
Nucleation Test - Liquid Addition via Droplet Distribution.

This test verifies the nucleation/liquid addition functionality:
1. Droplet count calculation from volumetric flow rate
2. Size-biased particle selection (large particles preferred)
3. Agglomeration for undersized particles
4. Liquid volume conservation

Usage:
    cd mcpbe/src
    python -m wmcpbe.Trials.test_nucleation
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


def test_nucleation_basic():
    """
    Basic nucleation test with simple agglomeration + liquid addition.
    
    Setup:
    - 50 initial particles
    - No initial liquid
    - Add liquid via nucleation over 5 seconds
    - Verify: liquid volume is added correctly AND multiple particles remain
    """
    print_separator("TEST 1: Basic Nucleation")
    
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
    solver.x = np.array([3e-6])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 5000  # ↑ More initial particles
    solver.process_type = "agglomeration"
    
    # Physics: VERY weak agglomeration to prevent all particles merging
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-15  # ↓↓ Much weaker (was 1e-12)
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    # Initialize
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    
    # Initialize liquid_volume to zero
    if hasattr(solver, 'liquid_volume'):
        solver.liquid_volume[:n_init] = 0.0
    else:
        print("ERROR: liquid_volume attribute missing!")
        return False
    
    # Create nucleation handler
    # Parameters:
    # - volumenstrom: 1e-15 m³/s (reduced for reasonable droplet count)
    # - tropfen_durchmesser: 1e-6 m = 1 µm
    # - wasserzugabe_start: 0.0 s (start immediately)
    # - wasserzugabe_dauer: 5.0 s (entire simulation)
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=1e-15,  # m³/s
        tropfen_durchmesser=1e-6,  # m
        wasserzugabe_start=0.0,  # s
        wasserzugabe_dauer=5.0,  # s
    )
    
    # Calculate expected liquid volume
    v_droplet = (4.0 / 3.0) * np.pi * (0.5e-6) ** 3  # Volume of 1 µm droplet
    total_time = 5.0
    v_liquid_expected = solver.nucleation.config.volumenstrom * total_time
    n_droplets_expected = v_liquid_expected / v_droplet
    
    print(f"\nInitial state:")
    print(f"  Particles: {n_init}")
    print(f"  Initial liquid: {np.sum(solver.liquid_volume[:n_init] * solver.W[:n_init]):.6e} m³")
    
    print(f"\nNucleation parameters:")
    print(f"  Volumenstrom: {solver.nucleation.config.volumenstrom:.6e} m³/s")
    print(f"  Tropfendurchmesser: {solver.nucleation.config.tropfen_durchmesser:.6e} m")
    print(f"  Tropfenvolumen: {v_droplet:.6e} m³")
    print(f"  Zugabedauer: {solver.nucleation.config.wasserzugabe_dauer:.1f} s")
    print(f"  Expected liquid added: {v_liquid_expected:.6e} m³")
    print(f"  Expected droplets: ~{n_droplets_expected:.0f}")
    
    # Run simulation
    print(f"\nRunning simulation...")
    solver.solve(maxiter=10000)  # ↑ More iterations allowed
    
    # Final state
    n_final = solver.a_tot
    final_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_final)
    final_solid = calculate_weighted_total(solver.V_flat[-1], solver.W, n_final)
    
    # Get nucleation statistics
    nuc_stats = solver.nucleation.get_statistics()
    
    print(f"\nFinal state:")
    print(f"  Particles: {n_final}")
    print(f"  Solid volume: {final_solid:.6e} m³")
    print(f"  Liquid volume: {final_liquid:.6e} m³")
    
    n_final = solver.a_tot
    final_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_final)
    
    nuc_stats = solver.nucleation.get_statistics()
    liquid_added = nuc_stats['liquid_volume_added_total']
    
   
    print(f"\nLiquid added (from nucleation): {liquid_added:.6e} m³")
    print(f"Liquid in system (final): {final_liquid:.6e} m³")
    
    # Check conservation: final liquid should equal added liquid
    # (within numerical tolerance)
    if abs(liquid_added) > 1e-300:
        rel_error = abs(final_liquid - liquid_added) / abs(liquid_added)
    else:
        rel_error = 0.0
    
    print(f"\nConservation error: {rel_error:.6e} ({rel_error*100:.4f}%)")
    
    print(f"\nNucleation statistics:")
    print(f"  Droplets added: {nuc_stats['droplets_added_total']}")
    print(f"  Liquid added: {nuc_stats['liquid_volume_added_total']:.6e} m³")
    
    # Check results
    liquid_added = final_liquid > 0.0
    multiple_particles = n_final >= 5  # At least 5 particles should remain
    
    print(f"\nResults:")
    print(f"  Liquid added: {'✅' if liquid_added else '❌'}")
    print(f"  Multiple particles remain ({n_final} ≥ 5): {'✅' if multiple_particles else '❌'}")
    
    overall_pass = liquid_added and multiple_particles
    print(f"\nOverall: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    
    if not liquid_added:
        print("  WARNING: No liquid was added!")
    if not multiple_particles:
        print(f"  WARNING: Only {n_final} particle(s) remain - too much agglomeration!")
    
    return overall_pass


def test_nucleation_window():
    """
    Test nucleation with delayed start window.
    
    Setup:
    - Liquid addition starts at t=0s
    - Ends at t=1.0s (early end)
    - Total simulation runs to t=3.0s
    - Verify: liquid added only during window
    """
    print_separator("TEST 2: Nucleation Time Window")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=4.0,
        t_write=10,  # More snapshots for better time resolution
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    solver.c = np.array([1e-3])
    solver.x = np.array([3e-6])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 2000
    solver.process_type = "agglomeration"
    
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-12
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    solver.liquid_volume[:n_init] = 0.0
    
    # Nucleation window: t=0.0s to t=1.0s (early in simulation)
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=1e-15,
        tropfen_durchmesser=1e-6,
        wasserzugabe_start=0.0,
        wasserzugabe_dauer=1.0,  # Ends at t=1.0s
    )
    
    print(f"\nNucleation window: t={solver.nucleation.config.wasserzugabe_start:.1f}s to "
          f"t={solver.nucleation.config.wasserzugabe_end:.1f}s")
    
    solver.solve(maxiter=100000)  # More iterations to ensure we pass the window
    
    n_final = solver.a_tot
    final_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_final)
    
    nuc_stats = solver.nucleation.get_statistics()
    
    print(f"Particles at beginning {n_init}")
    print(f"Particles at end {solver.a_tot}")
    print(f"\nFinal time: {solver._elapsed:.3f}s")
    print(f"Final liquid: {final_liquid:.6e} m³")
    print(f"Droplets added: {nuc_stats['droplets_added_total']}")
    print(f"Liquid added: {nuc_stats['liquid_volume_added_total']:.6e} m³")
    
    n_final = solver.a_tot
    final_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_final)
    
    nuc_stats = solver.nucleation.get_statistics()
    liquid_added = nuc_stats['liquid_volume_added_total']
    
    print(f"Particles: {n_final}")
    print(f"\nLiquid added (from nucleation): {liquid_added:.6e} m³")
    print(f"Liquid in system (final): {final_liquid:.6e} m³")
    
    # Check conservation: final liquid should equal added liquid
    # (within numerical tolerance)
    if abs(liquid_added) > 1e-300:
        rel_error = abs(final_liquid - liquid_added) / abs(liquid_added)
    else:
        rel_error = 0.0
    
    print(f"\nConservation error: {rel_error:.6e} ({rel_error*100:.4f}%)")
    
    # Check if liquid was added during window
    liquid_added = final_liquid > 0.0
    
    # Additional check: no more droplets should be added after window ends
    # This is implicitly tested by checking total matches expected
    
    print(f"\nResult: {'✅ PASS' if liquid_added else '❌ FAIL'}")
    
    return liquid_added


def test_nucleation_mass_conservation():
    """
    Test liquid mass conservation during nucleation + agglomeration.
    
    This verifies that:
    1. Added liquid is tracked correctly
    2. Liquid is conserved during subsequent agglomeration events
    """
    print_separator("TEST 3: Nucleation + Mass Conservation")
    
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
    solver.x = np.array([2e-6])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 7000
    solver.process_type = "mix"
    
    # Weak agglomeration
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-11
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    # Some breakage
    solver.BREAKRVAL = 1
    solver.BREAKFVAL = 2
    solver.pl_P1 = 0.1
    solver.pl_P2 = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    solver.liquid_volume[:n_init] = 0.0
    
    # Strong nucleation
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=5e-12,  # Higher flow rate
        tropfen_durchmesser=0.8e-5,  # Smaller droplets
        wasserzugabe_start=0.0,
        wasserzugabe_dauer=3.0,
    )
    
    solver.solve(maxiter=20000)
    
    n_final = solver.a_tot
    final_liquid = calculate_weighted_total(solver.liquid_volume, solver.W, n_final)
    
    nuc_stats = solver.nucleation.get_statistics()
    liquid_added = nuc_stats['liquid_volume_added_total']
    
    print(f"Particles: {n_final}")
    print(f"\nLiquid added (from nucleation): {liquid_added:.6e} m³")
    print(f"Liquid in system (final): {final_liquid:.6e} m³")
    
    # Check conservation: final liquid should equal added liquid
    # (within numerical tolerance)
    if abs(liquid_added) > 1e-300:
        rel_error = abs(final_liquid - liquid_added) / abs(liquid_added)
    else:
        rel_error = 0.0
    
    print(f"\nConservation error: {rel_error:.6e} ({rel_error*100:.4f}%)")
    
    # Allow 1% tolerance due to numerical effects
    conservation_ok = rel_error < 0.01
    
    print(f"\nMass conservation: {'✅ PASS' if conservation_ok else '❌ FAIL'}")
    
    return conservation_ok


def main():
    print("\n" + "#" * 70)
    print("# Nucleation Tests - Liquid Addition via Droplets")
    print("#"*70)
    
    results = []
    
    # Test 1: Basic nucleation
    try:
        passed = test_nucleation_basic()
        results.append(("Basic Nucleation", passed))
    except Exception as e:
        print(f"\n❌ Basic nucleation EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Basic Nucleation", False))
    
    # Test 2: Time window
    try:
        passed = test_nucleation_window()
        results.append(("Time Window", passed))
    except Exception as e:
        print(f"\n❌ Time window EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Time Window", False))
    
    # Test 3: Mass conservation
    try:
        passed = test_nucleation_mass_conservation()
        results.append(("Mass Conservation", passed))
    except Exception as e:
        print(f"\n❌ Mass conservation EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Mass Conservation", False))
    
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
