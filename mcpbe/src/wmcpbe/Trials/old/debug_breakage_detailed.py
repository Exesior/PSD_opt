"""
DETAILED BREAKAGE DEBUG - Compare rates and sampler state BEFORE solve
"""
import numpy as np
import time

print_header = lambda title: print("\n" + "=" * 70 + f"\n {title}\n" + "=" * 70)


def run_legacy_case(verbose=True):
    """Run Legacy case and return detailed state."""
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=42)
    
    # Set parameters
    solver.COLEVAL = 1
    solver.BREAKRVAL = 1
    solver.G = 1000
    solver.pl_P1 = 3e-4
    solver.pl_P2 = 1.0
    solver.pl_v = 2.0
    solver.pl_q = 0.5
    solver.alpha_prim = 1.0
    solver.process_type = 'breakage'
    
    # Init
    solver._initialize_kernels()
    solver._initialize_particles()
    solver._initialize_samplers()
    
    if verbose:
        print("\n=== LEGACY CASE ===")
        print(f"a_tot: {solver.a_tot}")
        print(f"W[:5]: {solver.W[:5]}")
        print(f"V_flat[-1,:5]: {solver.V_flat[-1, :5]}")
        
        # Check cached break config
        print(f"\nCached break config:")
        print(f"  _break_pl_P1: {getattr(solver, '_break_pl_P1', 'N/A')}")
        print(f"  _break_pl_P2: {getattr(solver, '_break_pl_P2', 'N/A')}")
        print(f"  _break_G: {getattr(solver, '_break_G', 'N/A')}")
        print(f"  _break_BREAKRVAL: {getattr(solver, '_break_BREAKRVAL', 'N/A')}")
        
        # Check break rates
        print(f"\nBreak rates (_break_rate[:5]): {solver._break_rate[:5]}")
        print(f"Break sampler total: {solver._break_sampler.total()}")
        
        # Manually compute what JIT would give for first particle
        from pbe_core.func.jit_kernel_break import calc_break_rate_1d as jit_single
        V_arr = solver.V_flat[-1, :1]
        jit_rate = jit_single(V_arr, solver._break_pl_P1, solver._break_pl_P2, solver._break_G, solver._break_BREAKRVAL, 0)
        print(f"\nJIT calc_break_rate_1d for particle 0:")
        print(f"  Input: V={V_arr[0]:.6e}, pl_P1={solver._break_pl_P1}, pl_P2={solver._break_pl_P2}, G={solver._break_G}, BREAKRVAL={solver._break_BREAKRVAL}")
        print(f"  Output: S_i = {jit_rate:.6e}")
        print(f"  Expected propensity: W[0] * S_i / delta = {solver.W[0]:.1f} * {jit_rate:.6e} / min(50, {solver.W[0]:.1f}) = {solver.W[0] * jit_rate / min(50.0, solver.W[0]):.6e}")
        print(f"  Actual _break_rate[0]: {solver._break_rate[0]:.6e}")
    
    return solver


def run_new_case(verbose=True):
    """Run New case and return detailed state."""
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=42)
    
    # Set parameters
    solver.break_kernel_name = 'power_law'
    solver.break_kernel_params = {'p1': 3e-4, 'p2': 1.0, 'g': 1000, 'breakrval': 1}
    solver.porosity_growth_kernel_name = 'volume_mixing'
    solver.compression_kernel_name = 'exponential_decay'
    solver.compression_kernel_params = {'rate': 0.02, 'min_porosity': 0.3}
    solver.liquid_dist_kernel_name = 'uniform_weighted'
    solver.alpha_prim = 1.0
    solver.process_type = 'breakage'
    
    # Init
    solver._initialize_kernels()
    solver._initialize_particles()
    solver._initialize_samplers()
    
    if verbose:
        print("\n=== NEW CASE ===")
        print(f"a_tot: {solver.a_tot}")
        print(f"W[:5]: {solver.W[:5]}")
        print(f"V_flat[-1,:5]: {solver.V_flat[-1, :5]}")
        
        # Check kernel params
        km = solver.kernel_manager
        if km and km.break_kernel:
            bk = km.break_kernel
            print(f"\nKernel params:")
            print(f"  p1: {bk.p1}")
            print(f"  p2: {bk.p2}")
            print(f"  g: {bk.g}")
            print(f"  breakrval: {bk.breakrval}")
            print(f"  pl_v: {getattr(bk, 'pl_v', 'N/A')}")
            print(f"  pl_q: {getattr(bk, 'pl_q', 'N/A')}")
            
            # Compute rate via kernel
            V_particle = float(solver.V_flat[-1, 0])
            kernel_rate = bk.compute_rate(V_particle, particle_idx=0, solver=solver)
            print(f"\nKernel compute_rate() for particle 0:")
            print(f"  Input: V={V_particle:.6e}")
            print(f"  Output: S_i = {kernel_rate:.6e}")
        
        # Check cached break config (should still have legacy attrs for backward compat)
        print(f"\nCached break config (legacy attrs):")
        print(f"  _break_pl_P1: {getattr(solver, '_break_pl_P1', 'N/A')}")
        print(f"  _break_pl_P2: {getattr(solver, '_break_pl_P2', 'N/A')}")
        print(f"  _break_G: {getattr(solver, '_break_G', 'N/A')}")
        print(f"  _break_BREAKRVAL: {getattr(solver, '_break_BREAKRVAL', 'N/A')}")
        
        # Check break rates
        print(f"\nBreak rates (_break_rate[:5]): {solver._break_rate[:5]}")
        print(f"Break sampler total: {solver._break_sampler.total()}")
    
    return solver


# ======================================================================
# MAIN
# ======================================================================

print_header("DETAILED BREAKAGE STATE COMPARISON")
print("Comparing internal state BEFORE solve()\n")

legacy = run_legacy_case(verbose=True)
new = run_new_case(verbose=True)

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

print(f"\nBreak sampler total:")
print(f"  Legacy: {legacy._break_sampler.total():.6e}")
print(f"  New:    {new._break_sampler.total():.6e}")
if abs(legacy._break_sampler.total() - new._break_sampler.total()) > 1e-15:
    print(f"  ⚠️  MISMATCH!")
else:
    print(f"  ✓ MATCH")

print(f"\nBreak rates (first 5 particles):")
for i in range(5):
    leg_rate = legacy._break_rate[i]
    new_rate = new._break_rate[i]
    match = "✓" if abs(leg_rate - new_rate) < 1e-15 else "⚠️"
    print(f"  {match} Particle {i}: Legacy={leg_rate:.6e}, New={new_rate:.6e}")

print(f"\n_first_break_dt (initial time step):")
# Compute what dt would be
from wmcpbe.mcpbe_break import MCPBEBreak
break_initial_dt, _ = legacy._build_break_dt_strategy()
dt_legacy = break_initial_dt(lambda: float(legacy._break_sampler.total()))
dt_new = break_initial_dt(lambda: float(new._break_sampler.total()))
print(f"  Legacy: {dt_legacy:.6e}")
print(f"  New:    {dt_new:.6e}")
