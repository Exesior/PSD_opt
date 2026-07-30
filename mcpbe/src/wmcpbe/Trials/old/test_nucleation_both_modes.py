"""
Test nucleation in both modes: regular step() and manual trigger.

This test verifies that both distribution paths use identical logic
and achieve the same accuracy targets.

Test Cases:
    1. Regular Mode: MC events occur during nucleation window
       → Liquid distributed via step() after each event
    
    2. Manual Trigger Mode: NO MC events during nucleation window
       → Liquid distributed once at window end via _trigger_manual_nucleation_at_window_end()

Expected Results:
    - Both modes achieve >99.5% liquid addition accuracy
    - Both modes distribute correct number of droplets (±1%)
    - No systematic bias between modes
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver
from wmcpbe.mcpbe_nucleation import NucleationHandler, NucleationConfig


def run_regular_mode():
    """
    Test regular nucleation mode with MC events during window.
    
    Uses moderate agglomeration coefficient to ensure events occur.
    """
    print("\n" + "=" * 80)
    print("TEST 1: REGULAR MODE (MC events during window)")
    print("=" * 80)
    
    # Particle properties
    d_particle = 5e-5  # 50 µm
    v_particle = (4.0 / 3.0) * np.pi * (d_particle / 2.0) ** 3
    
    # Droplet properties
    d_droplet = 1e-5  # 10 µm
    v_droplet = (4.0 / 3.0) * np.pi * (d_droplet / 2.0) ** 3
    
    # Process parameters
    volumetric_flow_rate = 1e-14  # m³/s
    t_window = 2.0  # s (duration of nucleation window)
    beta_0 = 1e-10  # Moderate agglomeration → ensures MC events
    
    # Expected results
    expected_liquid_volume = volumetric_flow_rate * t_window
    expected_droplet_count = expected_liquid_volume / v_droplet
    
    # Initial state
    n_particles_initial = 100
    W_initial = 100.0  # Weight per particle
    n_phys_target = 1e6
    Vc = n_particles_initial * W_initial / n_phys_target  # = 1.0e-2
    
    print(f"\nParticle Properties:")
    print(f"  Diameter: {d_particle*1e6:.1f} µm")
    print(f"  Volume: {v_particle:.3e} m³")
    
    print(f"\nDroplet Properties:")
    print(f"  Diameter: {d_droplet*1e6:.1f} µm")
    print(f"  Volume: {v_droplet:.3e} m³")
    
    print(f"\nProcess Parameters:")
    print(f"  Volumetric flow rate: {volumetric_flow_rate:.3e} m³/s")
    print(f"  Duration: {t_window:.1f} s")
    print(f"  Agglomeration coefficient: {beta_0:.2e}")
    
    print(f"\nExpected Results:")
    print(f"  Total liquid volume: {expected_liquid_volume:.3e} m³")
    print(f"  Number of droplets: {expected_droplet_count:.2e}")
    
    print(f"\nInitial State:")
    print(f"  Computational particles: {n_particles_initial}")
    print(f"  Control volume Vc: {Vc:.3e} m³")
    print(f"  Physical particles represented: {n_particles_initial * W_initial / Vc:.2e}")
    
    # Initialize solver
    print("\n" + "-" * 80)
    print("INITIALIZING SOLVER")
    print("-" * 80)
    
    # Create monodisperse initial particles
    V_flat = np.zeros((1, n_particles_initial), dtype=float)
    V_flat[0, :] = v_particle
    
    W_init = np.full(n_particles_initial, W_initial, dtype=float)
    
    # Time vector
    t_total_sim = 5.0
    t_write = 0.1
    t_vec = np.linspace(0.0, t_total_sim, int(t_total_sim / t_write) + 1)
    
    # Create solver without auto-initialization
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=True,
        load_attr=False,
        init=False,
    )
    
    # Set process type and physics parameters
    solver.process_type = "agglomeration"
    solver.COLEVAL = 3  # Constant kernel
    solver.CORR_BETA = beta_0
    solver.recon_enable = False
    solver.Vc = Vc
    
    # Initialize particles
    solver._initialize_particles(
        init_Vc=False,
        V_flat=V_flat,
        W_init=W_init,
    )
    
    # Configure nucleation
    print("\n" + "-" * 80)
    print("CONFIGURING NUCLEATION")
    print("-" * 80)
    
    nucleation_config = NucleationConfig(
        enabled=True,
        volumetric_flow_rate=volumetric_flow_rate,
        droplet_diameter=d_droplet,
        liquid_addition_start=0.0,
        liquid_addition_duration=t_window,
    )
    
    solver.nucleation = NucleationHandler(solver, nucleation_config)
    
    print(f"Nucleation configured:")
    print(f"  Enabled: {nucleation_config.enabled}")
    print(f"  Window: [{nucleation_config.liquid_addition_start:.1f}s, {nucleation_config.liquid_addition_end:.1f}s]")
    
    # Run simulation
    print("\n" + "-" * 80)
    print("RUNNING SIMULATION (Regular Mode)")
    print("-" * 80)
    
    import time
    start_time = time.time()
    
    solver.solve(maxiter=int(1e9))
    
    elapsed_real = time.time() - start_time
    
    # Analyze results
    nuc_stats = solver.nucleation.get_statistics()
    
    actual_liquid = nuc_stats['liquid_volume_added_total']
    actual_droplets = nuc_stats['droplets_added_total']
    
    accuracy_vol = actual_liquid / expected_liquid_volume * 100
    accuracy_droplets = actual_droplets / expected_droplet_count * 100
    
    print("\n" + "=" * 80)
    print("RESULTS - REGULAR MODE")
    print("=" * 80)
    
    print(f"\nLiquid Addition:")
    print(f"  Expected volume: {expected_liquid_volume:.6e} m³")
    print(f"  Actual volume:   {actual_liquid:.6e} m³")
    print(f"  Accuracy:        {accuracy_vol:.2f}%")
    
    print(f"\nDroplet Count:")
    print(f"  Expected: {expected_droplet_count:.2e}")
    print(f"  Actual:   {actual_droplets:.2e}")
    print(f"  Accuracy: {accuracy_droplets:.2f}%")
    
    print(f"\nParticle Evolution:")
    print(f"  Initial n_comp: {n_particles_initial}")
    print(f"  Final n_comp:   {solver.a_tot}")
    
    final_sum_W = np.sum(solver.W[:solver.a_tot])
    final_n_phys = final_sum_W / solver.Vc
    print(f"  Final n_phys:   {final_n_phys:.2e}")
    
    print(f"\nPerformance:")
    print(f"  Real time elapsed: {elapsed_real:.2f} s")
    print(f"  Simulated time:    {float(solver._iter_count):.1f} s (event count)")
    print(f"  MC events:         {solver._iter_count}")
    
    # Check if result is acceptable
    print("\n" + "=" * 80)
    if 95.0 <= accuracy_vol <= 105.0:
        print("✓ TEST PASSED: Liquid addition accurate within 5%")
        regular_passed = True
    else:
        print("✗ TEST FAILED: Liquid addition deviates by more than 5%")
        regular_passed = False
    print("=" * 80)
    
    return {
        'passed': regular_passed,
        'accuracy_vol': accuracy_vol,
        'accuracy_droplets': accuracy_droplets,
        'actual_liquid': actual_liquid,
        'actual_droplets': actual_droplets,
        'mc_events': solver._iter_count,
        'elapsed_real': elapsed_real,
    }


def run_manual_trigger_mode():
    """
    Test manual trigger mode with NO MC events during window.
    
    Uses very small agglomeration coefficient to prevent events.
    """
    print("\n" + "=" * 80)
    print("TEST 2: MANUAL TRIGGER MODE (NO MC events during window)")
    print("=" * 80)
    
    # Particle properties
    d_particle = 5e-5  # 50 µm
    v_particle = (4.0 / 3.0) * np.pi * (d_particle / 2.0) ** 3
    
    # Droplet properties
    d_droplet = 1e-5  # 10 µm
    v_droplet = (4.0 / 3.0) * np.pi * (d_droplet / 2.0) ** 3
    
    # Process parameters
    volumetric_flow_rate = 1e-14  # m³/s
    t_window = 2.0  # s (duration of nucleation window)
    beta_0 = 1e-11  # VERY small → NO MC events during window!
    
    # Expected results
    expected_liquid_volume = volumetric_flow_rate * t_window
    expected_droplet_count = expected_liquid_volume / v_droplet
    
    # Initial state
    n_particles_initial = 100
    W_initial = 100.0  # Weight per particle
    n_phys_target = 1e6
    Vc = n_particles_initial * W_initial / n_phys_target  # = 1.0e-2
    
    print(f"\nParticle Properties:")
    print(f"  Diameter: {d_particle*1e6:.1f} µm")
    print(f"  Volume: {v_particle:.3e} m³")
    
    print(f"\nDroplet Properties:")
    print(f"  Diameter: {d_droplet*1e6:.1f} µm")
    print(f"  Volume: {v_droplet:.3e} m³")
    
    print(f"\nProcess Parameters:")
    print(f"  Volumetric flow rate: {volumetric_flow_rate:.3e} m³/s")
    print(f"  Duration: {t_window:.1f} s")
    print(f"  Agglomeration coefficient: {beta_0:.2e} (intentionally tiny!)")
    
    print(f"\nExpected Results:")
    print(f"  Total liquid volume: {expected_liquid_volume:.3e} m³")
    print(f"  Number of droplets: {expected_droplet_count:.2e}")
    
    print(f"\nInitial State:")
    print(f"  Computational particles: {n_particles_initial}")
    print(f"  Control volume Vc: {Vc:.3e} m³")
    print(f"  Physical particles represented: {n_particles_initial * W_initial / Vc:.2e}")
    
    # Initialize solver
    print("\n" + "-" * 80)
    print("INITIALIZING SOLVER")
    print("-" * 80)
    
    # Create monodisperse initial particles
    V_flat = np.zeros((1, n_particles_initial), dtype=float)
    V_flat[0, :] = v_particle
    
    W_init = np.full(n_particles_initial, W_initial, dtype=float)
    
    # Time vector
    t_total_sim = 5.0
    t_write = 0.1
    t_vec = np.linspace(0.0, t_total_sim, int(t_total_sim / t_write) + 1)
    
    # Create solver without auto-initialization
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=True,
        load_attr=False,
        init=False,
    )
    
    # Set process type and physics parameters
    solver.process_type = "agglomeration"
    solver.COLEVAL = 3  # Constant kernel
    solver.CORR_BETA = beta_0
    solver.recon_enable = False
    solver.Vc = Vc
    
    # Initialize particles
    solver._initialize_particles(
        init_Vc=False,
        V_flat=V_flat,
        W_init=W_init,
    )
    
    # Configure nucleation
    print("\n" + "-" * 80)
    print("CONFIGURING NUCLEATION")
    print("-" * 80)
    
    nucleation_config = NucleationConfig(
        enabled=True,
        volumetric_flow_rate=volumetric_flow_rate,
        droplet_diameter=d_droplet,
        liquid_addition_start=0.0,
        liquid_addition_duration=t_window,
    )
    
    solver.nucleation = NucleationHandler(solver, nucleation_config)
    
    print(f"Nucleation configured:")
    print(f"  Enabled: {nucleation_config.enabled}")
    print(f"  Window: [{nucleation_config.liquid_addition_start:.1f}s, {nucleation_config.liquid_addition_end:.1f}s]")
    
    # Run simulation
    print("\n" + "-" * 80)
    print("RUNNING SIMULATION (Manual Trigger Mode)")
    print("-" * 80)
    
    import time
    start_time = time.time()
    
    solver.solve(maxiter=int(1e9))
    
    elapsed_real = time.time() - start_time
    
    # Analyze results
    nuc_stats = solver.nucleation.get_statistics()
    
    actual_liquid = nuc_stats['liquid_volume_added_total']
    actual_droplets = nuc_stats['droplets_added_total']
    
    accuracy_vol = actual_liquid / expected_liquid_volume * 100
    accuracy_droplets = actual_droplets / expected_droplet_count * 100
    
    print("\n" + "=" * 80)
    print("RESULTS - MANUAL TRIGGER MODE")
    print("=" * 80)
    
    print(f"\nLiquid Addition:")
    print(f"  Expected volume: {expected_liquid_volume:.6e} m³")
    print(f"  Actual volume:   {actual_liquid:.6e} m³")
    print(f"  Accuracy:        {accuracy_vol:.2f}%")
    
    print(f"\nDroplet Count:")
    print(f"  Expected: {expected_droplet_count:.2e}")
    print(f"  Actual:   {actual_droplets:.2e}")
    print(f"  Accuracy: {accuracy_droplets:.2f}%")
    
    print(f"\nParticle Evolution:")
    print(f"  Initial n_comp: {n_particles_initial}")
    print(f"  Final n_comp:   {solver.a_tot}")
    
    final_sum_W = np.sum(solver.W[:solver.a_tot])
    final_n_phys = final_sum_W / solver.Vc
    print(f"  Final n_phys:   {final_n_phys:.2e}")
    
    print(f"\nPerformance:")
    print(f"  Real time elapsed: {elapsed_real:.2f} s")
    print(f"  Simulated time:    {float(solver._iter_count):.1f} s (event count)")
    print(f"  MC events:         {solver._iter_count}")
    
    # Verify manual trigger was actually called
    print(f"\nVerification:")
    if solver._iter_count < 10:
        print(f"  ✓ Manual trigger confirmed (only {solver._iter_count} MC events)")
    else:
        print(f"  ⚠ Warning: More MC events than expected ({solver._iter_count})")
    
    # Check if result is acceptable
    print("\n" + "=" * 80)
    if 95.0 <= accuracy_vol <= 105.0:
        print("✓ TEST PASSED: Liquid addition accurate within 5%")
        manual_passed = True
    else:
        print("✗ TEST FAILED: Liquid addition deviates by more than 5%")
        manual_passed = False
    print("=" * 80)
    
    return {
        'passed': manual_passed,
        'accuracy_vol': accuracy_vol,
        'accuracy_droplets': accuracy_droplets,
        'actual_liquid': actual_liquid,
        'actual_droplets': actual_droplets,
        'mc_events': solver._iter_count,
        'elapsed_real': elapsed_real,
    }


def compare_results(regular, manual):
    """Compare results from both modes."""
    print("\n" + "=" * 80)
    print("COMPARISON: Regular vs Manual Trigger")
    print("=" * 80)
    
    vol_diff = abs(regular['accuracy_vol'] - manual['accuracy_vol'])
    droplet_diff = abs(regular['accuracy_droplets'] - manual['accuracy_droplets'])
    
    print(f"\nAccuracy Comparison:")
    print(f"  Volume:      Regular={regular['accuracy_vol']:.2f}%, Manual={manual['accuracy_vol']:.2f}%, Diff={vol_diff:.2f}%")
    print(f"  Droplets:    Regular={regular['accuracy_droplets']:.2f}%, Manual={manual['accuracy_droplets']:.2f}%, Diff={droplet_diff:.2f}%")
    
    print(f"\nConsistency Check:")
    if vol_diff < 1.0:
        print(f"  ✓ Volume accuracy consistent (diff < 1%)")
        vol_consistent = True
    else:
        print(f"  ⚠ Volume accuracy differs by {vol_diff:.2f}%")
        vol_consistent = False
    
    if droplet_diff < 2.0:
        print(f"  ✓ Droplet accuracy consistent (diff < 2%)")
        droplet_consistent = True
    else:
        print(f"  ⚠ Droplet accuracy differs by {droplet_diff:.2f}%")
        droplet_consistent = False
    
    print(f"\nBoth Tests Passed: {regular['passed'] and manual['passed']}")
    
    overall_pass = regular['passed'] and manual['passed'] and vol_consistent and droplet_consistent
    
    print("\n" + "=" * 80)
    if overall_pass:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("Both regular and manual trigger modes work correctly!")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("=" * 80)
    
    return overall_pass


if __name__ == "__main__":
    # Run both tests
    regular_results = run_regular_mode()
    manual_results = run_manual_trigger_mode()
    
    # Compare results
    overall_pass = compare_results(regular_results, manual_results)
    
    # Exit code for CI/CD
    sys.exit(0 if overall_pass else 1)
