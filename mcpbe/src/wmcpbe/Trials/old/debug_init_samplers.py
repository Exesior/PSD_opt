"""
DEBUG _initialize_samplers() - Check if it adds particles incorrectly
"""
import numpy as np


def debug_init_samplers(seed=42, label=""):
    """Debug sampler initialization step by step."""
    from wmcpbe.mcpbe import MCPBESolver
    
    print(f"\n{'='*70}")
    print(f"{label} CASE")
    print(f"{'='*70}")
    
    solver = MCPBESolver(dim=1, t_total=0.05, t_write=10, init=False, load_attr=False, seed=seed)
    solver.COLEVAL = 1
    solver.BREAKRVAL = 1
    solver.G = 1000
    solver.pl_P1 = 3e-4
    solver.pl_P2 = 1.0
    solver.pl_v = 2.0
    solver.pl_q = 0.5
    solver.alpha_prim = 1.0
    solver.process_type = 'breakage'
    
    solver._initialize_kernels()
    solver._initialize_particles()
    
    print(f"After _initialize_particles():")
    print(f"  a_tot: {solver.a_tot}")
    print(f"  W[:3]: {solver.W[:3]}")
    print(f"  V[:3]: {solver.V_flat[-1, :3]}")
    
    # Check if _init_lmc() changes anything
    print(f"\nCalling _init_lmc()...")
    solver._init_lmc()
    print(f"  a_tot after _init_lmc(): {solver.a_tot}")
    
    # Now _initialize_samplers()
    print(f"\nCalling _initialize_samplers()...")
    solver._initialize_samplers()
    print(f"  a_tot after _initialize_samplers(): {solver.a_tot}")
    
    # Check samplers
    if hasattr(solver, '_break_sampler') and solver._break_sampler:
        print(f"  Break sampler exists: YES")
        print(f"  Break sampler total: {solver._break_sampler.total():.6e}")
        print(f"  _break_rate[:3]: {solver._break_rate[:3]}")
    
    if hasattr(solver, '_agg_sampler') and solver._agg_sampler:
        print(f"  Agg sampler exists: YES")
        print(f"  Agg sampler total: {solver._agg_sampler.total():.6e}")
    else:
        print(f"  Agg sampler exists: NO (correct for breakage-only)")
    
    return solver


# Run Legacy
legacy = debug_init_samplers(seed=42, label="LEGACY")

# Run New
from wmcpbe.mcpbe import MCPBESolver

print(f"\n{'='*70}")
print(f"NEW CASE")
print(f"{'='*70}")

new = MCPBESolver(dim=1, t_total=0.05, t_write=10, init=False, load_attr=False, seed=42)
new.break_kernel_name = 'power_law'
new.break_kernel_params = {'p1': 3e-4, 'p2': 1.0, 'g': 1000, 'breakrval': 1}
new.porosity_growth_kernel_name = 'volume_mixing'
new.compression_kernel_name = 'exponential_decay'
new.compression_kernel_params = {'rate': 0.02, 'min_porosity': 0.3}
new.liquid_dist_kernel_name = 'uniform_weighted'
new.alpha_prim = 1.0
new.process_type = 'breakage'

new._initialize_kernels()
new._initialize_particles()

print(f"After _initialize_particles():")
print(f"  a_tot: {new.a_tot}")
print(f"  W[:3]: {new.W[:3]}")
print(f"  V[:3]: {new.V_flat[-1, :3]}")

print(f"\nCalling _init_lmc()...")
new._init_lmc()
print(f"  a_tot after _init_lmc(): {new.a_tot}")

print(f"\nCalling _initialize_samplers()...")
new._initialize_samplers()
print(f"  a_tot after _initialize_samplers(): {new.a_tot}")

if hasattr(new, '_break_sampler') and new._break_sampler:
    print(f"  Break sampler exists: YES")
    print(f"  Break sampler total: {new._break_sampler.total():.6e}")
    print(f"  _break_rate[:3]: {new._break_rate[:3]}")

if hasattr(new, '_agg_sampler') and new._agg_sampler:
    print(f"  Agg sampler exists: YES")
    print(f"  Agg sampler total: {new._agg_sampler.total():.6e}")
else:
    print(f"  Agg sampler exists: NO")

# Comparison
print(f"\n{'='*70}")
print(f"COMPARISON")
print(f"{'='*70}")
print(f"\na_tot timeline:")
print(f"  After _initialize_particles():")
print(f"    Legacy: {legacy.a_tot}")
print(f"    New:    {new.a_tot}")
print(f"  After _init_lmc():")
print(f"    Legacy: {getattr(legacy, 'a_tot', 'N/A')}")
print(f"    New:    {getattr(new, 'a_tot', 'N/A')}")
print(f"  After _initialize_samplers():")
print(f"    Legacy: {legacy.a_tot}")
print(f"    New:    {new.a_tot}")

if legacy.a_tot != new.a_tot:
    print(f"\n⚠️  MISMATCH FOUND! Investigating...")
    # Check which method changed a_tot
    print(f"\nChecking _cap (capacity):")
    print(f"  Legacy _cap: {legacy._cap}")
    print(f"  New _cap: {new._cap}")
