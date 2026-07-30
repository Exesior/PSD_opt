"""
DEBUG FRAGMENT CREATION - Compare how fragments are built

The bug: Both select k=773, but New creates +1 particle while Legacy doesn't.
This means fragment creation differs!
"""
import numpy as np


def debug_fragment_creation(seed=42, label=""):
    """Debug fragment creation for first breakage event."""
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
    solver._initialize_samplers()
    
    # Get cached break config
    solver._prepare_break_config()
    
    print(f"Break config:")
    print(f"  _break_pl_v: {solver._break_pl_v}")
    print(f"  _break_pl_q: {solver._break_pl_q}")
    print(f"  _break_BREAKFVAL: {solver._break_BREAKFVAL}")
    print(f"  frag_num: {getattr(solver, 'frag_num', 'N/A')}")
    
    # Sample particle k=773 (we know this is selected from previous debug)
    k = 773
    
    print(f"\nParticle k={k}:")
    print(f"  W[k]: {solver.W[k]}")
    print(f"  V[k]: {solver.V_flat[-1, k]}")
    
    # Get fragment tables
    Vrem_k = np.array([solver.V_flat[0, k]], dtype=float)
    print(f"\nVrem_k: {Vrem_k}")
    
    # Call _get_break_tables_for_state
    mode, rA, rB, rowsum_cdf, row_cdf, zmin1, zmin3, pexp = solver._get_break_tables_for_state(Vrem_k)
    print(f"\nBreak tables:")
    print(f"  mode: {mode}")
    print(f"  pexp (expected fragments): {pexp}")
    print(f"  rA (rel array) shape: {rA.shape if hasattr(rA, 'shape') else len(rA)}")
    print(f"  rB (cdf array) shape: {rB.shape if hasattr(rB, 'shape') else len(rB)}")
    
    # Check CDF at key points
    print(f"\nCDF values:")
    print(f"  CDF[0]: {rB[0]:.6f}")
    print(f"  CDF[-1]: {rB[-1]:.6f}")
    print(f"  CDF[500]: {rB[500]:.6f}")
    
    # Simulate fragment sampling with same RNG
    print(f"\nSampling fragments:")
    u_values = [solver._rng.random() for _ in range(5)]
    print(f"  Random numbers: {u_values}")
    
    # Build fragments stepwise
    frags = solver._build_fragments_stepwise(Vrem_k)
    print(f"\nFragments created: {len(frags)}")
    for i, frag in enumerate(frags):
        print(f"  Frag {i}: V={frag[0]:.6e}, sum={np.sum(frag):.6e}")
    
    print(f"\nTotal fragment volume: {sum(np.sum(f) for f in frags):.6e}")
    print(f"Original volume: {Vrem_k[0]:.6e}")
    print(f"Volume conservation: {abs(sum(np.sum(f) for f in frags) - Vrem_k[0]):.6e}")
    
    return solver, len(frags)


# Run both cases
legacy_solver, legacy_n_frags = debug_fragment_creation(seed=42, label="LEGACY")
new_solver, new_n_frags = debug_fragment_creation(seed=42, label="NEW")

# For new case
from wmcpbe.mcpbe import MCPBESolver

print(f"\n{'='*70}")
print(f"NEW CASE (with kernel framework)")
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
new._initialize_samplers()

# Get cached break config
new._prepare_break_config()

print(f"Break config:")
print(f"  _break_pl_v: {new._break_pl_v}")
print(f"  _break_pl_q: {new._break_pl_q}")
print(f"  _break_BREAKFVAL: {new._break_BREAKFVAL}")
print(f"  frag_num: {getattr(new, 'frag_num', 'N/A')}")

# Sample with same k
k = 773
Vrem_k = np.array([new.V_flat[0, k]], dtype=float)

# Get fragment tables
mode, rA, rB, rowsum_cdf, row_cdf, zmin1, zmin3, pexp = new._get_break_tables_for_state(Vrem_k)
print(f"\nBreak tables:")
print(f"  mode: {mode}")
print(f"  pexp (expected fragments): {pexp}")

# Build fragments
frags = new._build_fragments_stepwise(Vrem_k)
print(f"\nFragments created: {len(frags)}")
for i, frag in enumerate(frags):
    print(f"  Frag {i}: V={frag[0]:.6e}")

# Comparison
print(f"\n{'='*70}")
print(f"COMPARISON")
print(f"{'='*70}")
print(f"\nExpected fragments (pexp):")
print(f"  Legacy: {legacy_n_frags}")
print(f"  New:    {new_n_frags}")

if legacy_n_frags != new_n_frags:
    print(f"\n⚠️  MISMATCH! Different number of fragments!")
    print(f"  This explains the particle count difference!")
else:
    print(f"\n✓ Same number of fragments")
