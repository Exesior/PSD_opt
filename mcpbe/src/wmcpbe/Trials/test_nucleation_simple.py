"""
Simple test for nucleation + agglomeration only.

This test verifies that nucleation works correctly with the new kernel setup.
Uses standard particle initialization from MCPBEBase._initialize_particles().
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver


def run_simple_nucleation_test():
    """
    Run a minimal nucleation test with agglomeration.
    
    Physical parameters:
    - Particle diameter: 5 µm (5e-6 m)
    - Droplet diameter: 1 µm (1e-6 m)
    - Volumetric flow rate: 3e-12 m³/s
    - Agglomeration coefficient: 1e-4
    - Duration: 2 seconds nucleation, 5 seconds total
    """
    
    print("=" * 80)
    print("SIMPLE NUCLEATION TEST (Agglomeration + Nucleation only)")
    print("=" * 80)
    
    # =====================================================================
    # Physical Parameters
    # =====================================================================
    
    particle_diameter = 5e-3  # 5 µm in meters
    particle_radius = particle_diameter / 2.0
    particle_volume = (4.0 / 3.0) * np.pi * particle_radius ** 3
    
    print(f"\nParticle Properties:")
    print(f"  Diameter: {particle_diameter * 1e6:.1f} µm")
    print(f"  Volume: {particle_volume:.3e} m³")
    
    droplet_diameter = 1e-3  # 1 µm in meters
    droplet_radius = droplet_diameter / 2.0
    droplet_volume = (4.0 / 3.0) * np.pi * droplet_radius ** 3
    
    print(f"\nDroplet Properties:")
    print(f"  Diameter: {droplet_diameter * 1e6:.1f} µm")
    print(f"  Volume: {droplet_volume:.3e} m³")

    volumetric_flow_rate = 3e-10  # m³/s
    liquid_addition_duration = 2.0  # seconds
    agg_coefficient = 1e-4  # Agglomeration kernel coefficient
    
    print(f"\nProcess Parameters:")
    print(f"  Volumetric flow rate: {volumetric_flow_rate:.3e} m³/s")
    print(f"  Nucleation duration: {liquid_addition_duration:.1f} s")
    print(f"  Agglomeration coefficient: {agg_coefficient:.3e}")
    
    expected_liquid_volume = volumetric_flow_rate * liquid_addition_duration
    expected_droplet_count = expected_liquid_volume / droplet_volume
    
    print(f"\nExpected Results:")
    print(f"  Total liquid volume: {expected_liquid_volume:.3e} m³")
    print(f"  Number of droplets: {expected_droplet_count:.2e}")
    
    # =====================================================================
    # Solver Configuration (using standard initialization)
    # =====================================================================
    
    print("\n" + "=" * 80)
    print("INITIALIZING SOLVER")
    print("=" * 80)
    
    t_total = 4.0  # Total simulation time
    t_write = 0.1  # Write interval
    t_vec = np.linspace(0.0, t_total, int(t_total / t_write) + 1)
    
    # FIXED SEED for reproducibility
    seed = 42
    rng = np.random.default_rng(seed)
    
    # Create solver with explicit kernel configuration
    # init=True calls _initialize_particles() automatically
    print(f"\nCreating solver with kernel configuration (seed={seed})...")
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=True,
        load_attr=False,
        init=True,  # Standard initialization
        # Kernel configuration per README.md
        agg_kernel_name='constant',
        agg_kernel_params={'corr_beta': agg_coefficient},
        # Pass RNG for reproducibility
        rng=rng,
    )
    
    # Set process type
    solver.process_type = "agglomeration"
    
    # Disable reconstruction
    solver.recon_enable = False
    
    # Create initial particle state manually (since init=True already ran)
    # We need to override with our custom particle size and control volume
    n_particles_initial = 1000
    V_flat = np.zeros((2, n_particles_initial), dtype=float)
    V_flat[0, :] = particle_volume
    V_flat[1, :] = particle_volume
    
    W_init = np.full(n_particles_initial, 100.0, dtype=float)
    
    # Set control volume BEFORE re-initialization
    solver.Vc = 1.0
    
    # Re-initialize particles with custom state
    print("\nRe-initializing particles with custom particle size...")
    solver._initialize_particles(
        init_Vc=False,
        V_flat=V_flat,
        W_init=W_init,
    )
    
    print(f"  After _initialize_particles:")
    print(f"    a_tot = {solver.a_tot}")
    print(f"    W[0:5] = {solver.W[:5]}")
    print(f"    Vc = {solver.Vc:.3e}")
    print(f"    x[0] = {solver.x[0]:.3e} m")
    print(f"    v[0] = {solver.v[0]:.3e} m³")
    
    # Initialize samplers
    print("\nInitializing samplers...")
    solver._initialize_samplers()
    
    print(f"  After _initialize_samplers:")
    print(f"    a_tot = {solver.a_tot}")
    print(f"    W[0:5] = {solver.W[:5]}")
    
    # Verify kernel is properly initialized
    print("\nVerifying kernel setup...")
    if hasattr(solver, 'kernel_manager') and solver.kernel_manager is not None:
        print(f"  kernel_manager: OK")
        if solver.kernel_manager.agg_kernel is not None:
            print(f"  agg_kernel: {solver.kernel_manager.agg_kernel.name}")
            print(f"  corr_beta: {solver.kernel_manager.agg_kernel.corr_beta}")
        else:
            print("  WARNING: agg_kernel is None!")
    else:
        print("  WARNING: kernel_manager is None!")
    
    # =====================================================================
    # Nucleation Configuration
    # =====================================================================
    
    print("\n" + "=" * 80)
    print("CONFIGURING NUCLEATION")
    print("=" * 80)
    
    solver.create_nucleation_handler(
        enabled=True,
        volumetric_flow_rate=volumetric_flow_rate,
        droplet_diameter=droplet_diameter,
        liquid_addition_start=0.0,
        liquid_addition_duration=liquid_addition_duration,
    )
    
    print(f"Nucleation configured:")
    print(f"  Enabled: {solver.nucleation.config.enabled}")
    print(f"  Window: [{solver.nucleation.config.liquid_addition_start:.1f}s, "
          f"{solver.nucleation.config.liquid_addition_end:.1f}s]")
    print(f"  Flow rate: {solver.nucleation.config.volumetric_flow_rate:.3e} m³/s")
    
    # =====================================================================
    # Run Simulation
    # =====================================================================
    
    print("\n" + "=" * 80)
    print("RUNNING SIMULATION")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        solver.solve(maxiter=int(1e9))
    except Exception as e:
        print(f"\nERROR during solve: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False
    
    elapsed_real = time.time() - start_time
    
    # =====================================================================
    # Results Analysis
    # =====================================================================
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    nuc_stats = solver.nucleation.get_statistics()
    
    actual_liquid = nuc_stats['liquid_volume_added_total']
    actual_droplets = nuc_stats['droplets_added_total']
    
    v_liquid_in_system = 0.0
    n_comp = solver.a_tot
    n_active = 0
    
    if hasattr(solver, 'liquid_volume') and solver.liquid_volume is not None:
        for k in range(n_comp):
            if solver.W[k] > 0:
                v_liquid_in_system += solver.liquid_volume[k] * solver.W[k]
                n_active += 1
    
    print(f"\n  Active particles: {n_active} / {n_comp}")
    
    error_added = (actual_liquid - expected_liquid_volume) / expected_liquid_volume
    error_in_system = (v_liquid_in_system - expected_liquid_volume) / expected_liquid_volume
    
    print(f"\nLiquid Addition:")
    print(f"  Expected volume:      {expected_liquid_volume:.6e} m³")
    print(f"  Added (statistics):   {actual_liquid:.6e} m³")
    print(f"  In system (calculated): {v_liquid_in_system:.6e} m³")
    print(f"  Error (added):        {error_added:+.6f}  ({error_added*100:+.4f}%)")
    print(f"  Error (in system):    {error_in_system:+.6f}  ({error_in_system*100:+.4f}%)")
    
    print(f"\nDroplet Count:")
    print(f"  Expected: {expected_droplet_count:.2e}")
    print(f"  Actual:   {actual_droplets:.2e}")
    print(f"  Accuracy: {actual_droplets/expected_droplet_count*100:.2f}%")
    
    print(f"\nParticle Evolution:")
    print(f"  Initial n_comp: {solver.a_tot}")
    
    final_sum_W = np.sum(solver.W[:solver.a_tot])
    final_n_phys = final_sum_W / solver.Vc
    print(f"  Final n_phys:   {final_n_phys:.2e}")
    
    print(f"\nPerformance:")
    print(f"  Real time elapsed: {elapsed_real:.2f} s")
    print(f"  Simulated time:    {t_total:.1f} s")
    print(f"  MC events:         {solver._iter_count}")
    
    accuracy = actual_liquid / expected_liquid_volume * 100
    
    # Additional debug info
    print(f"\nVc Tracking:")
    print(f"  Initial Vc: 1.000e+00 m³")
    print(f"  Final Vc:   {solver.Vc:.6e} m³")
    
    print(f"\nRemainder Status:")
    if hasattr(solver.nucleation, '_liquid_remainder'):
        remainder = solver.nucleation._liquid_remainder
        print(f"  Final _liquid_remainder: {remainder:.6e} m³ ({remainder/droplet_volume:.6f} droplets)")
    else:
        print(f"  _liquid_remainder: N/A")
    
    print(f"\n" + "=" * 80)
    if 99.9 <= accuracy <= 100.1:
        print("✓ TEST PASSED: Liquid addition accurate within 0.1%")
        success = True
    else:
        print("✗ TEST FAILED: Liquid addition deviates by more than 0.1%")
        success = False
    print("=" * 80)
    
    return solver, nuc_stats, success


if __name__ == "__main__":
    solver, stats, success = run_simple_nucleation_test()
    sys.exit(0 if success else 1)
