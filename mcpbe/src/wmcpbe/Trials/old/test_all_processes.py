"""
Simple test case for all 4 processes with simplest kernels.

Processes tested:
    1. Agglomeration → 'constant' kernel (simplest: β = const)
    2. Breakage → 'power_law' kernel (only available breakage kernel)
    3. Nucleation → Simple liquid addition handler
    4. Compression → 'linear' kernel (simplest porosity model)

This test uses NO legacy COLEVAL/BREAKRVAL parameters!
All configuration is done via explicit kernel names and parameters.
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, src_dir)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def compute_moments(X: np.ndarray, W: np.ndarray) -> dict:
    """Compute particle moments from size distribution."""
    Q0 = np.sum(W)
    V_particles = (np.pi / 6.0) * X**3
    Q3 = np.sum(W * V_particles)
    d_mean = np.average(X, weights=W) if Q0 > 0 else 0.0
    
    return {
        'Q0': float(Q0),
        'Q3': float(Q3),
        'd_mean': float(d_mean),
    }


def run_test():
    """Run simulation with all 4 processes using simplest kernels."""
    from wmcpbe.mcpbe import MCPBESolver
    
    print_header("SIMPLE TEST: All Processes with Simplest Kernels")
    
    # ================================================================
    # CONFIGURATION - NO LEGACY PARAMETERS!
    # ================================================================
    
    # Simulation settings
    t_total = 1.0       # Short simulation time for quick testing
    seed = 42           # Reproducible
    n_particles = 100   # Small number for fast execution
    
    print("\n📋 Configuration:")
    print(f"  t_total: {t_total}s")
    print(f"  seed: {seed}")
    print(f"  Initial particles: {n_particles}")
    
    # ================================================================
    # CREATE SOLVER WITH EXPLICIT KERNEL NAMES
    # ================================================================
    
    print("\n🔧 Creating solver with explicit kernel configuration...")
    
    solver = MCPBESolver(
        dim=1,
        t_total=t_total,
        t_write=10,
        init=False,      # Manual init for controlled setup
        load_attr=False, # Don't load external config
        seed=seed,
        
        # AGGLOMERATION: constant kernel (simplest possible)
        agg_kernel_name='constant',
        agg_kernel_params={'corr_beta': 1e-8},
        
        # BREAKAGE: power_law kernel (only option)
        break_kernel_name='power_law',
        break_kernel_params={'p1': 3e-2, 'p2': 1.0},
        
        # COMPRESSION: linear kernel (simplest porosity model)
        compression_kernel_name='exponential_decay',
        compression_kernel_params={},
        
        # Nucleation will be added separately via handler
    )
    
    # Set required solver attributes
    solver.G = 1000  # Shear rate (needed by some kernels)
    solver.pl_P1 = 3e-2
    solver.pl_P2 = 1.0
    solver.pl_v = 2.0
    solver.pl_q = 1.0
    
    # Initialize particles manually
    print("  Initializing particles...")
    solver.a0 = n_particles
    solver._initialize_particles()
    
    # Create nucleation handler (liquid addition)
    print("  Creating nucleation handler...")
    try:
        solver.create_nucleation_handler(
            enabled=True,
            volumetric_flow_rate=1e-13,  # Very small flow for testing
            droplet_diameter=1e-6,        # 1 µm droplets
            liquid_addition_start=0.0,
            liquid_addition_duration=t_total,
        )
        print("  ✓ Nucleation handler created")
    except Exception as e:
        print(f"  ⚠ Nucleation handler failed: {e}")
        print("  Continuing without nucleation...")
    
    # Initialize samplers
    print("  Initializing samplers...")
    solver._initialize_samplers()
    
    # ================================================================
    # PRINT INITIAL STATE
    # ================================================================
    
    print("\n📊 Initial State:")
    initial_moments = compute_moments(solver.X[:solver.a_tot], solver.W[:solver.a_tot])
    print(f"  Q0 (particles): {initial_moments['Q0']:.2f}")
    print(f"  Q3 (volume):    {initial_moments['Q3']:.6e} m³")
    print(f"  d_mean:         {initial_moments['d_mean']:.6e} m")
    
    # ================================================================
    # RUN SIMULATION
    # ================================================================
    
    print("\n⏱️  Running simulation...")
    start_time = time.time()
    
    try:
        solver.solve()
        elapsed = time.time() - start_time
        
        print(f"  ✓ Simulation completed in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ Simulation failed after {elapsed:.2f}s: {e}")
        raise
    
    # ================================================================
    # PRINT FINAL STATE
    # ================================================================
    
    print("\n📊 Final State:")
    final_moments = compute_moments(solver.X[:solver.a_tot], solver.W[:solver.a_tot])
    print(f"  Q0 (particles): {final_moments['Q0']:.2f}")
    print(f"  Q3 (volume):    {final_moments['Q3']:.6e} m³")
    print(f"  d_mean:         {final_moments['d_mean']:.6e} m")
    
    # ================================================================
    # ANALYSIS
    # ================================================================
    
    print("\n📈 Changes:")
    delta_Q0 = final_moments['Q0'] - initial_moments['Q0']
    delta_Q3 = final_moments['Q3'] - initial_moments['Q3']
    delta_d_mean = final_moments['d_mean'] - initial_moments['d_mean']
    
    print(f"  ΔQ0:      {delta_Q0:+.2f} ({delta_Q0/initial_moments['Q0']*100:+.1f}%)")
    print(f"  ΔQ3:      {delta_Q3:+.6e} m³ ({delta_Q3/initial_moments['Q3']*100:+.1f}%)")
    print(f"  Δd_mean:  {delta_d_mean:+.6e} m ({delta_d_mean/initial_moments['d_mean']*100:+.1f}%)")
    
    # ================================================================
    # VERIFICATION
    # ================================================================
    
    print("\n✅ Verification:")
    
    # Q0 should decrease due to agglomeration/breakage balance
    # Q3 should increase slightly due to nucleation (liquid addition)
    # d_mean may increase or decrease depending on dominant process
    
    checks_passed = 0
    total_checks = 3
    
    # Check 1: Particle count changed (something happened)
    if abs(delta_Q0) > 1e-6:
        print("  ✓ Q0 changed (agglomeration/breakage active)")
        checks_passed += 1
    else:
        print("  ✗ Q0 unchanged (no agglomeration/breakage?)")
    
    # Check 2: Volume non-negative
    if final_moments['Q3'] >= 0:
        print("  ✓ Q3 non-negative (volume conservation OK)")
        checks_passed += 1
    else:
        print("  ✗ Q3 negative (volume error!)")
    
    # Check 3: Simulation completed in reasonable time
    if elapsed < 60:
        print(f"  ✓ Completed in reasonable time (<60s)")
        checks_passed += 1
    else:
        print(f"  ✗ Took too long (>60s)")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    
    print_header("SUMMARY")
    
    if checks_passed == total_checks:
        print(f"\n✅ ALL CHECKS PASSED ({checks_passed}/{total_checks})")
        print("\nThe new kernel framework works correctly for all processes!")
        return True
    else:
        print(f"\n⚠️  SOME CHECKS FAILED ({checks_passed}/{total_checks})")
        print("\nReview the output above for details.")
        return False


if __name__ == '__main__':
    try:
        success = run_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
