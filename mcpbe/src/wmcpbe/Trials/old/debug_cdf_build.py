"""Debug: Why does LEGACY build NaN CDF table?"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wmcpbe.mcpbe import MCPBESolver
from pbe_core.func.jit_mcpbe import _build_table_1d_jit

print("="*80)
print(" DEBUG CDF TABLE BUILDING")
print("="*80)

# Test the JIT function directly
print("\n=== Testing _build_table_1d_jit() ===")
rel = np.linspace(0.0, 1.0, 1000).astype(np.float64)
v = 2.0
q = 0.5
bf = 1  # BREAKFVAL

print(f"Input: rel.size={rel.size}, v={v}, q={q}, bf={bf}")
cdf = _build_table_1d_jit(rel, v, q, bf)
print(f"Output: cdf.size={cdf.size}")
print(f"cdf[:10] = {cdf[:10]}")
print(f"cdf has NaN: {np.any(np.isnan(cdf))}")
print(f"cdf has Inf: {np.any(np.isinf(cdf))}")

# LEGACY
print("\n=== LEGACY ===")
legacy = MCPBESolver(dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None)
legacy.a0 = 100
legacy.COLEVAL = 1
legacy.BREAKRVAL = 1
legacy.CORR_BETA = 1e-2
legacy.G = 1000
legacy.pl_P1 = 3e-4
legacy.pl_P2 = 1.0
legacy.pl_v = 2.0
legacy.pl_q = 0.5
legacy.alpha_prim = 1.0
legacy.process_type = 'breakage'
legacy._initialize_particles()
legacy._init_lmc()
legacy._initialize_kernels()
legacy._initialize_samplers()

print(f"_break_pl_v = {legacy._break_pl_v}")
print(f"_break_pl_q = {legacy._break_pl_q}")
print(f"_break_BREAKFVAL = {legacy._break_BREAKFVAL}")
print(f"_bf_ready = {legacy._bf_ready}")
print(f"_bf_cache = {legacy._bf_cache}")

# Force build
if not legacy._bf_ready:
    print("\nBuilding CDF table...")
    legacy._build_break_function()
    print(f"After build: _bf_ready = {legacy._bf_ready}")

print(f"\n_bf1_rel[:10] = {legacy._bf1_rel[:10] if hasattr(legacy, '_bf1_rel') else 'N/A'}")
print(f"_bf1_cdf[:10] = {legacy._bf1_cdf[:10] if hasattr(legacy, '_bf1_cdf') else 'N/A'}")
print(f"_bf1_cdf has NaN: {np.any(np.isnan(legacy._bf1_cdf)) if hasattr(legacy, '_bf1_cdf') else 'N/A'}")

# Check what _get_break_tables_for_state returns
Vrem = np.array([legacy.V_flat[0, 0]], dtype=float)
mode, rA, rB, *_ = legacy._get_break_tables_for_state(Vrem)
print(f"\n_get_break_tables_for_state() returns:")
print(f"  mode = '{mode}'")
print(f"  rA is _bf1_rel: {rA is legacy._bf1_rel if hasattr(legacy, '_bf1_rel') else 'N/A'}")
print(f"  rB is _bf1_cdf: {rB is legacy._bf1_cdf if hasattr(legacy, '_bf1_cdf') else 'N/A'}")
print(f"  rB[:10] = {rB[:10] if rB is not None else 'N/A'}")

# NEW
print("\n\n=== NEW ===")
new = MCPBESolver(
    dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
    break_kernel_name='power_law',
    break_kernel_params={'p1': 3e-4, 'p2': 1.0, 'g': 1000, 'breakrval': 1, 'pl_v': 2.0, 'pl_q': 0.5},
    porosity_growth_kernel_name='volume_mixing',
    compression_kernel_name='exponential_decay',
    compression_kernel_params={'rate': 0.02, 'min_porosity': 0.3},
    liquid_dist_kernel_name='uniform_weighted',
)
new.a0 = 100
new.alpha_prim = 1.0
new.process_type = 'breakage'
new._initialize_particles()
new._init_lmc()
new._initialize_samplers()

print(f"_break_pl_v = {new._break_pl_v}")
print(f"_break_pl_q = {new._break_pl_q}")
print(f"_break_BREAKFVAL = {new._break_BREAKFVAL}")

if not new._bf_ready:
    print("\nBuilding CDF table...")
    new._build_break_function()

print(f"\n_bf1_rel[:10] = {new._bf1_rel[:10] if hasattr(new, '_bf1_rel') else 'N/A'}")
print(f"_bf1_cdf[:10] = {new._bf1_cdf[:10] if hasattr(new, '_bf1_cdf') else 'N/A'}")
print(f"_bf1_cdf has NaN: {np.any(np.isnan(new._bf1_cdf)) if hasattr(new, '_bf1_cdf') else 'N/A'}")

Vrem_new = np.array([new.V_flat[0, 0]], dtype=float)
mode_new, rA_new, rB_new, *_ = new._get_break_tables_for_state(Vrem_new)
print(f"\n_get_break_tables_for_state() returns:")
print(f"  mode = '{mode_new}'")
print(f"  rB[:10] = {rB_new[:10] if rB_new is not None else 'N/A'}")
