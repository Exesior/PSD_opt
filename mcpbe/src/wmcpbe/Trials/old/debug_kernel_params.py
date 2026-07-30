"""
Debug Script: Compare Kernel Parameters between Legacy and New.

This script verifies that the new kernel framework receives the SAME parameters
as the legacy implementation.

Quick test - should complete in < 30 seconds.
"""

import sys
import os
import time
import numpy as np

# Add src directory to path for imports
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, src_dir)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def run_legacy_case():
    """Run simulation with LEGACY approach and print parameters."""
    from wmcpbe.mcpbe import MCPBESolver
    
    print_header("LEGACY CASE")
    
    # Test parameters (same as validate_old_vs_new.py)
    coleval = 1
    corr_beta = 1e-2
    g = 1000
    pl_p1 = 3e-4
    pl_p2 = 1.0
    seed = 42
    t_total = 5.0
    
    print(f"\nInput Parameters:")
    print(f"  COLEVAL={coleval}, CORR_BETA={corr_beta}, G={g}")
    print(f"  pl_P1={pl_p1}, pl_P2={pl_p2}")
    
    # Create solver WITHOUT auto-init
    solver = MCPBESolver(
        dim=1,
        t_total=t_total,
        t_write=10,
        init=False,
        load_attr=False,
        seed=seed,
    )
    
    # Set parameters BEFORE initialization
    solver.COLEVAL = coleval
    solver.CORR_BETA = corr_beta
    solver.G = g
    solver.pl_P1 = pl_p1
    solver.pl_P2 = pl_p2
    solver.pl_v = 2.0
    solver.pl_q = 0.5
    solver.alpha_prim = 1.0
    solver.process_type = 'agglomeration'
    
    print(f"\nAfter setting solver attributes:")
    print(f"  solver.COLEVAL={getattr(solver, 'COLEVAL', 'NOT SET')}")
    print(f"  solver.CORR_BETA={getattr(solver, 'CORR_BETA', 'NOT SET')}")
    print(f"  solver.G={getattr(solver, 'G', 'NOT SET')}")
    
    # Initialize kernels
    solver._initialize_kernels()
    
    print(f"\nAfter _initialize_kernels():")
    if hasattr(solver, 'kernel_manager') and solver.kernel_manager is not None:
        km = solver.kernel_manager
        print(f"  kernel_manager exists: YES")
        print(f"  km.COLEVAL={km.COLEVAL}")
        print(f"  km.CORR_BETA={km.CORR_BETA}")
        print(f"  km.G={km.G}")
        
        if km.agg_kernel is not None:
            print(f"\n  Aggregation Kernel:")
            print(f"    Type: {type(km.agg_kernel).__name__}")
            print(f"    Name: {km.agg_kernel.name}")
            print(f"    corr_beta: {km.agg_kernel.corr_beta}")
            print(f"    g: {km.agg_kernel.g}")
            print(f"    Expected: corr_beta={corr_beta}, g={g}")
            
            # Check if values match
            if abs(km.agg_kernel.corr_beta - corr_beta) > 1e-10:
                print(f"    ⚠️  MISMATCH! Expected {corr_beta}, got {km.agg_kernel.corr_beta}")
            else:
                print(f"    ✓  MATCH!")
                
            if abs(km.agg_kernel.g - g) > 1e-10:
                print(f"    ⚠️  MISMATCH! Expected {g}, got {km.agg_kernel.g}")
            else:
                print(f"    ✓  MATCH!")
        else:
            print(f"  ⚠️  agg_kernel is None!")
    else:
        print(f"  ⚠️  kernel_manager is None!")
    
    # Initialize particles and samplers
    solver._initialize_particles()
    solver._initialize_samplers()
    
    print(f"\nAfter _initialize_particles():")
    print(f"  a_tot (initial particle count)={solver.a_tot}")
    print(f"  W[:5] (first 5 weights)={solver.W[:5]}")
    print(f"  X[:5] (first 5 diameters)={solver.X[:5] * 1e6} µm")
    
    # Run simulation
    start_time = time.time()
    try:
        solver.solve()
        elapsed = time.time() - start_time
        
        a_tot = solver.a_tot
        V_ges = solver.V_flat[-1, :a_tot].copy()
        X_final = (6.0 * V_ges / np.pi) ** (1.0/3.0)
        W_final = solver.W[:a_tot].copy()
        
        Q0 = np.sum(W_final)
        Q3 = np.sum(W_final * (np.pi / 6.0) * X_final**3)
        d_mean = np.average(X_final, weights=W_final)
        
        print(f"\nResults after solve ({elapsed:.2f}s):")
        print(f"  a_tot={a_tot}")
        print(f"  Q0={Q0:.6e}")
        print(f"  Q3={Q3:.6e}")
        print(f"  d_mean={d_mean:.6e} m")
        
        return {
            'a_tot': a_tot,
            'Q0': Q0,
            'Q3': Q3,
            'd_mean': d_mean,
            'X': X_final,
            'W': W_final,
            'time': elapsed,
        }
        
    except Exception as e:
        print(f"\n⚠️  ERROR during solve: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_new_case():
    """Run simulation with NEW kernel framework and print parameters."""
    from wmcpbe.mcpbe import MCPBESolver
    
    print_header("NEW CASE (Kernel Framework)")
    
    # Same test parameters
    corr_beta = 1e-2
    g = 1000
    pl_p1 = 3e-4
    pl_p2 = 1.0
    seed = 42
    t_total = 5.0
    
    print(f"\nInput Parameters:")
    print(f"  corr_beta={corr_beta}, G={g}")
    print(f"  pl_P1={pl_p1}, pl_P2={pl_p2}")
    
    # Create solver WITHOUT auto-init
    solver = MCPBESolver(
        dim=1,
        t_total=t_total,
        t_write=10,
        init=False,
        load_attr=False,
        seed=seed,
    )
    
    # Set kernel parameters BEFORE initialization
    solver.agg_kernel_name = 'shear_chin1998'
    solver.agg_kernel_params = {'corr_beta': corr_beta, 'g': g}
    
    solver.alpha_prim = 1.0
    solver.process_type = 'agglomeration'
    
    print(f"\nAfter setting solver attributes:")
    print(f"  solver.agg_kernel_name={getattr(solver, 'agg_kernel_name', 'NOT SET')}")
    print(f"  solver.agg_kernel_params={getattr(solver, 'agg_kernel_params', 'NOT SET')}")
    
    # Initialize kernels
    solver._initialize_kernels()
    
    print(f"\nAfter _initialize_kernels():")
    if hasattr(solver, 'kernel_manager') and solver.kernel_manager is not None:
        km = solver.kernel_manager
        print(f"  kernel_manager exists: YES")
        print(f"  km.agg_kernel_name={km.agg_kernel_name}")
        print(f"  km.agg_kernel_params={km.agg_kernel_params}")
        
        if km.agg_kernel is not None:
            print(f"\n  Aggregation Kernel:")
            print(f"    Type: {type(km.agg_kernel).__name__}")
            print(f"    Name: {km.agg_kernel.name}")
            print(f"    corr_beta: {km.agg_kernel.corr_beta}")
            print(f"    g: {km.agg_kernel.g}")
            print(f"    Expected: corr_beta={corr_beta}, g={g}")
            
            # Check if values match
            if abs(km.agg_kernel.corr_beta - corr_beta) > 1e-10:
                print(f"    ⚠️  MISMATCH! Expected {corr_beta}, got {km.agg_kernel.corr_beta}")
            else:
                print(f"    ✓  MATCH!")
                
            if abs(km.agg_kernel.g - g) > 1e-10:
                print(f"    ⚠️  MISMATCH! Expected {g}, got {km.agg_kernel.g}")
            else:
                print(f"    ✓  MATCH!")
        else:
            print(f"  ⚠️  agg_kernel is None!")
    else:
        print(f"  ⚠️  kernel_manager is None!")
    
    # Initialize particles and samplers
    solver._initialize_particles()
    solver._initialize_samplers()
    
    print(f"\nAfter _initialize_particles():")
    print(f"  a_tot (initial particle count)={solver.a_tot}")
    print(f"  W[:5] (first 5 weights)={solver.W[:5]}")
    print(f"  X[:5] (first 5 diameters)={solver.X[:5] * 1e6} µm")
    
    # Run simulation
    start_time = time.time()
    try:
        solver.solve()
        elapsed = time.time() - start_time
        
        a_tot = solver.a_tot
        V_ges = solver.V_flat[-1, :a_tot].copy()
        X_final = (6.0 * V_ges / np.pi) ** (1.0/3.0)
        W_final = solver.W[:a_tot].copy()
        
        Q0 = np.sum(W_final)
        Q3 = np.sum(W_final * (np.pi / 6.0) * X_final**3)
        d_mean = np.average(X_final, weights=W_final)
        
        print(f"\nResults after solve ({elapsed:.2f}s):")
        print(f"  a_tot={a_tot}")
        print(f"  Q0={Q0:.6e}")
        print(f"  Q3={Q3:.6e}")
        print(f"  d_mean={d_mean:.6e} m")
        
        return {
            'a_tot': a_tot,
            'Q0': Q0,
            'Q3': Q3,
            'd_mean': d_mean,
            'X': X_final,
            'W': W_final,
            'time': elapsed,
        }
        
    except Exception as e:
        print(f"\n⚠️  ERROR during solve: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(legacy: dict, new: dict):
    """Compare results from both cases."""
    print_header("COMPARISON")
    
    if legacy is None or new is None:
        print("Cannot compare: One or both results are None")
        return
    
    print(f"\n  Particle Count:")
    print(f"    Legacy: {legacy['a_tot']}")
    print(f"    New:    {new['a_tot']}")
    if legacy['a_tot'] != new['a_tot']:
        print(f"    ⚠️  DIFFERENT! Difference: {abs(new['a_tot'] - legacy['a_tot'])}")
    else:
        print(f"    ✓  SAME!")
    
    print(f"\n  Moments:")
    for key in ['Q0', 'Q3', 'd_mean']:
        val_legacy = legacy[key]
        val_new = new[key]
        
        if val_legacy != 0:
            rel_error = abs(val_new - val_legacy) / abs(val_legacy) * 100
        else:
            rel_error = abs(val_new - val_legacy) * 100
        
        status = "✓" if rel_error < 0.1 else "⚠️"
        print(f"    {status} {key}: Legacy={val_legacy:.6e}, New={val_new:.6e}, Error={rel_error:.4f}%")
    
    print(f"\n  Performance:")
    print(f"    Legacy: {legacy['time']:.2f}s")
    print(f"    New:    {new['time']:.2f}s")
    if new['time'] > 0:
        speedup = legacy['time'] / new['time']
        print(f"    Speedup: {speedup:.1f}x")


def main():
    """Run debug comparison."""
    print("\n" + "=" * 70)
    print(" KERNEL PARAMETER DEBUG")
    print(" Comparing Legacy vs New Kernel Framework")
    print("=" * 70)
    
    legacy_result = run_legacy_case()
    new_result = run_new_case()
    
    compare_results(legacy_result, new_result)
    
    print_header("SUMMARY")
    if legacy_result and new_result:
        q0_error = abs(new_result['Q0'] - legacy_result['Q0']) / legacy_result['Q0'] * 100
        if q0_error < 0.1:
            print("  ✓ Parameters seem to be transferred correctly!")
            print("  The issue may be elsewhere (e.g., propensity calculation).")
        else:
            print(f"  ⚠️  Q0 differs by {q0_error:.2f}%")
            print("  This indicates parameter transfer issue OR algorithm difference.")
    else:
        print("  Could not determine - check errors above.")


if __name__ == "__main__":
    main()
