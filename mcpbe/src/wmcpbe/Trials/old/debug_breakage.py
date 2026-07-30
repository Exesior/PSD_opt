"""
Debug Breakage Rate Discrepancy.

Compare breakage rates between legacy JIT and new kernel framework
particle-by-particle to find the source of the 0.2% error.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wmcpbe.mcpbe import MCPBESolver


def compare_breakage_rates():
    """Compare breakage rates for identical particles."""
    
    print("=" * 70)
    print(" DEBUG: Breakage Rate Comparison")
    print("=" * 70)
    
    # Create two solvers with identical configuration
    seed = 42
    pl_p1 = 3e-2  # Lower rate for testing (0.0003)
    pl_p2 = 1.0
    g = 1000
    breakrval = 1
    n_particles = 10000  # Increased from 1000 for better statistics
    
    print(f"\nConfiguration:")
    print(f"  pl_P1 = {pl_p1}")
    print(f"  pl_P2 = {pl_p2}")
    print(f"  G = {g}")
    print(f"  BREAKRVAL = {breakrval}")
    
    # LEGACY solver
    print("\n[LEGACY] Creating solver with COLEVAL/BREAKRVAL...")
    
    # Create WITHOUT auto-init to control order exactly
    legacy_solver = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=seed,
    )
    
    # Set ALL parameters BEFORE any initialization
    legacy_solver.a0 = n_particles
    legacy_solver.COLEVAL = 1
    legacy_solver.BREAKRVAL = breakrval
    legacy_solver.CORR_BETA = 1e-2
    legacy_solver.G = g
    legacy_solver.pl_P1 = pl_p1
    legacy_solver.pl_P2 = pl_p2
    legacy_solver.pl_v = 2.0
    legacy_solver.pl_q = 0.5
    legacy_solver.alpha_prim = 1.0
    legacy_solver.process_type = 'breakage'
    # Manual init order (particles → LMC → kernels → samplers)
    # Kernels MUST be initialized AFTER setting COLEVAL/BREAKRVAL
    legacy_solver._initialize_particles()
    legacy_solver._init_lmc()
    legacy_solver._initialize_kernels()  # Uses COLEVAL/BREAKRVAL set above
    legacy_solver._initialize_samplers()
    
    # NEW solver
    print("[NEW] Creating solver with kernel framework...")
    
    # CRITICAL: For NEW solver, we must pass kernel params to __init__ because
    # _initialize_kernels() is called INSIDE __init__ BEFORE we can set attributes.
    # This is DIFFERENT from Legacy where COLEVAL/BREAKRVAL are set AFTER __init__.
    # 
    # To match Legacy RNG sequence, we need:
    # 1. __init__(init=False) - creates RNG only
    # 2. Set params
    # 3. _initialize_particles()
    # 4. _init_lmc()
    # 5. _initialize_kernels() - NOW with correct params
    # 6. _initialize_samplers()
    #
    # BUT: __init__ calls _initialize_kernels() even with init=False!
    # Solution: Pass kernel params to __init__ so first call is correct.
    
    new_solver = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=seed,
        break_kernel_name='power_law',
        break_kernel_params={
            'p1': pl_p1, 
            'p2': pl_p2, 
            'g': g, 
            'breakrval': breakrval,
            'pl_v': 2.0,  # Match legacy
            'pl_q': 0.5,  # Match legacy
        },
        porosity_growth_kernel_name='volume_mixing',
        compression_kernel_name='exponential_decay',
        compression_kernel_params={'rate': 0.02, 'min_porosity': 0.3},
        liquid_dist_kernel_name='uniform_weighted',
    )
    new_solver.a0 = n_particles
    new_solver.alpha_prim = 1.0
    new_solver.process_type = 'breakage'
    # Manual init order (same as legacy)
    new_solver._initialize_particles()
    new_solver._init_lmc()
    # DON'T call _initialize_kernels() again - already done correctly in __init__!
    # Just refresh samplers to match particle state
    new_solver._initialize_samplers()
    
    # Compare initial state
    print(f"\nInitial State:")
    print(f"  Legacy particles: {legacy_solver.a_tot}")
    print(f"  New particles: {new_solver.a_tot}")
    
    # Check if particles are identical
    legacy_V = legacy_solver.V_flat[-1, :legacy_solver.a_tot].copy()
    new_V = new_solver.V_flat[-1, :new_solver.a_tot].copy()
    legacy_W = legacy_solver.W[:legacy_solver.a_tot].copy()
    new_W = new_solver.W[:new_solver.a_tot].copy()
    
    print(f"\nParticle Comparison (first 5):")
    for i in range(min(5, legacy_solver.a_tot)):
        print(f"  Particle {i}: V_legacy={legacy_V[i]:.6e}, V_new={new_V[i]:.6e}, "
              f"W_legacy={legacy_W[i]:.6f}, W_new={new_W[i]:.6f}")
    
    # Compute breakage rates for both
    a = legacy_solver.a_tot
    
    print(f"\nBreakage Rate Comparison (first 10 particles):")
    print(f"  {'Idx':>4} | {'Volume':>14} | {'Legacy Rate':>14} | {'New Rate':>14} | {'Rel Error':>10}")
    print(f"  {'-'*4}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}")
    
    max_rel_error = 0.0
    max_abs_error = 0.0
    
    for i in range(min(10, a)):
        v_particle = float(legacy_V[i])
        
        # Legacy: Use JIT function directly
        from pbe_core.func.jit_kernel_break import calc_break_rate_1d as _kb_br1_single
        legacy_rate = float(_kb_br1_single(
            legacy_V,
            legacy_solver._break_pl_P1,
            legacy_solver._break_pl_P2,
            legacy_solver._break_G,
            legacy_solver._break_BREAKRVAL,
            i,
        ))
        
        # New: Use kernel
        new_rate = new_solver.kernel_manager.compute_break_rate(
            v_particle,
            particle_idx=i,
            solver=new_solver
        )
        
        if legacy_rate > 0:
            rel_error = abs(new_rate - legacy_rate) / legacy_rate
        else:
            rel_error = abs(new_rate - legacy_rate)
        
        max_rel_error = max(max_rel_error, rel_error)
        max_abs_error = max(max_abs_error, abs(new_rate - legacy_rate))
        
        status = "✓" if rel_error < 1e-10 else "✗"
        print(f"  {i:>4} | {v_particle:>14.6e} | {legacy_rate:>14.6e} | {new_rate:>14.6e} | {rel_error*100:>9.4f}% {status}")
    
    print(f"\nMax Relative Error: {max_rel_error*100:.6f}%")
    print(f"Max Absolute Error: {max_abs_error:.6e}")
    
    # Now check what the kernel is actually computing
    print(f"\nKernel Details:")
    kernel = new_solver.kernel_manager.break_kernel
    print(f"  Kernel type: {type(kernel).__name__}")
    print(f"  Kernel params: {kernel.params}")
    print(f"  breakrval attribute: {kernel.breakrval}")
    
    # Manual calculation
    print(f"\nManual Calculation (for particle 0):")
    v0 = float(legacy_V[0])
    print(f"  Volume: {v0:.6e}")
    
    if breakrval == 1:
        expected = pl_p1
        formula = "S = P1 (constant)"
    elif breakrval == 2:
        expected = pl_p1 * (v0 ** (1.0/3.0))
        formula = "S = P1 × V^(1/3)"
    else:
        alpha = pl_p2 / 3.0
        expected = pl_p1 * g * (v0 ** alpha)
        formula = f"S = P1 × G × V^(P2/3) = {pl_p1} × {g} × V^({pl_p2/3:.4f})"
    
    print(f"  Formula: {formula}")
    print(f"  Expected rate: {expected:.6e}")
    print(f"  Kernel rate: {new_rate:.6e}")
    print(f"  JIT rate: {legacy_rate:.6e}")
    
    # Run short simulation to see divergence
    print(f"\n" + "=" * 70)
    print(" Short Simulation Test")
    print("=" * 70)
    
    print("\nRunning legacy simulation (t=0.5s)...")
    legacy_solver.solve()
    legacy_Q0 = np.sum(legacy_solver.W[:legacy_solver.a_tot])
    legacy_a = legacy_solver.a_tot
    
    print("Running new simulation (t=0.5s)...")
    new_solver.solve()
    new_Q0 = np.sum(new_solver.W[:new_solver.a_tot])
    new_a = new_solver.a_tot
    
    print(f"\nResults:")
    print(f"  Legacy: Q0={legacy_Q0:.6e}, particles={legacy_a}")
    print(f"  New:    Q0={new_Q0:.6e}, particles={new_a}")
    
    if legacy_Q0 > 0:
        q0_error = abs(new_Q0 - legacy_Q0) / legacy_Q0 * 100
        print(f"  Q0 Error: {q0_error:.4f}%")
    
    print(f"  Particle count diff: {abs(new_a - legacy_a)}")


if __name__ == "__main__":
    compare_breakage_rates()
