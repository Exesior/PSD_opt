"""
Debug test for nucleation with controlled parameters.

Test conditions:
- 2 seconds liquid addition
- Particles: 5 µm diameter
- Droplets: 1 µm diameter  
- Volumetric flow rate: 1e-15 m³/s
- Agglomeration coefficient: 1e-10
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver
from wmcpbe.mcpbe_nucleation import NucleationHandler, NucleationConfig


def run_nucleation_test():
    """
    Run a simple nucleation test with debug output.
    
    Physical parameters:
    - Particle diameter: 5 µm (5e-6 m)
    - Droplet diameter: 1 µm (1e-6 m)
    - Volumetric flow rate: 1e-15 m³/s
    - Agglomeration coefficient: 1e-10
    - Duration: 2 seconds
    """
    
    print("=" * 80)
    print("NUCLEATION DEBUG TEST")
    print("=" * 80)
    
    # =====================================================================
    # Physical Parameters
    # =====================================================================
    
    # Particle properties
    particle_diameter = 5e-6 # 5 µm in meters
    particle_radius = particle_diameter / 2.0
    particle_volume = (4.0 / 3.0) * np.pi * particle_radius ** 3
    
    print(f"\nParticle Properties:")
    print(f"  Diameter: {particle_diameter * 1e6:.1f} µm")
    print(f"  Volume: {particle_volume:.3e} m³")
    
    # Droplet properties
    droplet_diameter = 1e-6 # 1 µm in meters
    droplet_radius = droplet_diameter / 2.0
    droplet_volume = (4.0 / 3.0) * np.pi * droplet_radius ** 3
    
    print(f"\nDroplet Properties:")
    print(f"  Diameter: {droplet_diameter * 1e6:.1f} µm")
    print(f"  Volume: {droplet_volume:.3e} m³")
    
    # Process parameters
    volumetric_flow_rate = 3e-12  # m³/s
    liquid_addition_duration = 1.0  # seconds
    agg_coefficient = 1e-4  # Agglomeration kernel coefficient
    
    print(f"\nProcess Parameters:")
    print(f"  Volumetric flow rate: {volumetric_flow_rate:.3e} m³/s")
    print(f"  Duration: {liquid_addition_duration:.1f} s")
    print(f"  Agglomeration coefficient: {agg_coefficient:.3e}")
    
    # Expected liquid addition
    expected_liquid_volume = volumetric_flow_rate * liquid_addition_duration
    expected_droplet_count = expected_liquid_volume / droplet_volume
    
    print(f"\nExpected Results:")
    print(f"  Total liquid volume: {expected_liquid_volume:.3e} m³")
    print(f"  Number of droplets: {expected_droplet_count:.2e}")
    
    # =====================================================================
    # Initial Particle Setup
    # =====================================================================
    
    # Create monodisperse initial particles
    n_particles_initial = 100  # Number of computational particles
    V_flat = np.zeros((2, n_particles_initial), dtype=float)
    V_flat[0, :] = particle_volume  # Component volume
    V_flat[1, :] = particle_volume  # Total volume (same for 1D)
    
    # Weights: each computational particle represents equal number of physical particles
    # OPTIMIZED: High enough to avoid capacity explosion, low enough for reasonable dW
    # Test different values to find sweet spot
    n_phys_per_computational = 100.0  # Physical particles per computational particle
    W_init = np.full(n_particles_initial, n_phys_per_computational, dtype=float)
    
    # Control volume: DEFAULT TO 1.0 for nucleation testing
    # Vc should only be adjusted for agglomeration/breakage domination to control statistics/runtime
    # For pure nucleation tests, Vc=1 avoids scaling confusion
    Vc = 1.0  # Standard value - no scaling
    
    print(f"\nInitial State:")
    print(f"  Computational particles: {n_particles_initial}")
    print(f"  Control volume Vc: {Vc:.3e} m³")
    print(f"  Physical particles represented: {np.sum(W_init)/Vc:.2e}")
    
    # =====================================================================
    # Solver Configuration
    # =====================================================================
    
    print("\n" + "=" * 80)
    print("INITIALIZING SOLVER")
    print("=" * 80)
    
    # Time vector
    t_total = 5.0  # Total simulation time (longer than nucleation window)
    t_write = 0.1  # Write interval
    t_vec = np.linspace(0.0, t_total, int(t_total / t_write) + 1)
    
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
    solver.CORR_BETA = agg_coefficient
    
    # Disable reconstruction for cleaner debugging
    solver.recon_enable = False
    
    # Set control volume directly before initialization
    solver.Vc = Vc
    
    # Initialize solver with explicit particle state
    solver._initialize_particles(
        init_Vc=False,
        V_flat=V_flat,
        W_init=W_init,
    )
    
    # DEBUG: Check weights AFTER initialization
    print(f"  [DEBUG] After init: W[0:5]={solver.W[:5]}, Vc={solver.Vc:.3e}")
    
    solver._init_lmc()
    solver._initialize_samplers()
    
    print(f"  [DEBUG] After _init_lmc: W[0:5]={solver.W[:5]}, Vc={solver.Vc:.3e}")
    
    # =====================================================================
    # Nucleation Configuration
    # =====================================================================
    
    print("\n" + "=" * 80)
    print("CONFIGURING NUCLEATION")
    print("=" * 80)
    
    print(f"  [DEBUG] Before nucleation config: W[0:5]={solver.W[:5]}, Vc={solver.Vc:.3e}")
    
    nucleation_config = NucleationConfig(
        enabled=True,
        volumetric_flow_rate=volumetric_flow_rate,
        droplet_diameter=droplet_diameter,
        liquid_addition_start=0.0,  # Start immediately
        liquid_addition_duration=liquid_addition_duration,
    )
    
    solver.nucleation = NucleationHandler(solver, nucleation_config)
    
    print(f"Nucleation configured:")
    print(f"  Enabled: {nucleation_config.enabled}")
    print(f"  Window: [{nucleation_config.liquid_addition_start:.1f}s, {nucleation_config.liquid_addition_end:.1f}s]")
    
    # =====================================================================
    # Run Simulation
    # =====================================================================
    
    print("\n" + "=" * 80)
    print("RUNNING SIMULATION")
    print("=" * 80)
    
    import time
    start_time = time.time()
    
    solver.solve(maxiter=int(1e9))
    
    elapsed_real = time.time() - start_time
    
    # =====================================================================
    # Results Analysis
    # =====================================================================
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # Get nucleation statistics
    nuc_stats = solver.nucleation.get_statistics()
    
    actual_liquid = nuc_stats['liquid_volume_added_total']
    actual_droplets = nuc_stats['droplets_added_total']
    
    # Calculate liquid currently in system
    # IMPORTANT: liquid_volume[k] is liquid PER PHYSICAL PARTICLE (intensive property).
    # To get total physical liquid, we weight by W[k] (number of physical particles):
    #   total_liquid = Sum_k (liquid_volume[k] × W[k])
    # NOTE: NO division by Vc! Vc is only for number densities, not extensive quantities.
    # IMPORTANT: Only count ACTIVE particles (W > 0)
    v_liquid_in_system = 0.0
    n_comp = solver.a_tot
    
    # Check if solver tracks active particles (W > 0 means active)
    n_active = 0
    if hasattr(solver, 'liquid_volume') and solver.liquid_volume is not None:
        for k in range(n_comp):
            if solver.W[k] > 0:  # Only count active particles
                v_liquid_in_system += solver.liquid_volume[k] * solver.W[k]
                n_active += 1
    
    print(f"\n  Active particles: {n_active} / {n_comp}")
    
    # Relative errors (Ist-Soll)/Soll
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
    print(f"  Initial n_comp: {n_particles_initial}")
    print(f"  Final n_comp:   {solver.a_tot}")
    
    final_sum_W = np.sum(solver.W[:solver.a_tot])
    final_n_phys = final_sum_W / solver.Vc
    print(f"  Final n_phys:   {final_n_phys:.2e}")
    
    print(f"\nPerformance:")
    print(f"  Real time elapsed: {elapsed_real:.2f} s")
    print(f"  Simulated time:    {t_total:.1f} s")
    print(f"  MC events:         {solver._iter_count}")
    
    # Check if result is acceptable
    accuracy = actual_liquid / expected_liquid_volume * 100
    print(f"\n" + "=" * 80)
    if 95.0 <= accuracy <= 105.0:
        print("✓ TEST PASSED: Liquid addition accurate within 5%")
    else:
        print("✗ TEST FAILED: Liquid addition deviates by more than 5%")
    print("=" * 80)
    
    return solver, nuc_stats


if __name__ == "__main__":
    solver, stats = run_nucleation_test()
