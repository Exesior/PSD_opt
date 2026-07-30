"""Test that the fix resolves the 2003 RNG calls issue."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class CounterRNG:
    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)
        self.count = 0
    def random(self, size=None):
        self.count += 1
        return self._rng.random(size)
    @property
    def rng(self): return self._rng

from wmcpbe.mcpbe import MCPBESolver

print("="*80)
print(" TEST FIX FOR 2003 RNG CALLS BUG")
print("="*80)

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

legacy_cnt = CounterRNG(42)
legacy._rng = legacy_cnt
legacy._do_one_break()
print(f"LEGACY: {legacy_cnt.count} calls, {legacy.a_tot} particles")

# NEW
print("\n=== NEW ===")
new = MCPBESolver(
    dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
    break_kernel_name='power_law',
    break_kernel_params={
        'p1': 3e-4, 
        'p2': 1.0, 
        'g': 1000, 
        'breakrval': 1,
        'pl_v': 2.0,
        'pl_q': 0.5,
    },
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

# Check that solver attributes were set correctly
print(f"\nNEW solver attributes after kernel init:")
print(f"  pl_P1 = {new.pl_P1} (expected: 0.0003)")
print(f"  pl_P2 = {new.pl_P2} (expected: 1.0)")
print(f"  pl_v  = {new.pl_v}  (expected: 2.0)")
print(f"  pl_q  = {new.pl_q}  (expected: 0.5)")

new_cnt = CounterRNG(42)
new._rng = new_cnt
new._do_one_break()
print(f"\nNEW: {new_cnt.count} calls, {new.a_tot} particles")

# Verdict
print("\n" + "="*80)
print(" VERDICT")
print("="*80)
if legacy_cnt.count == new_cnt.count == 3 and legacy.a_tot == new.a_tot == 101:
    print("\n✅ FIX SUCCESSFUL! Both solvers now behave identically.")
else:
    print(f"\n❌ Fix incomplete:")
    print(f"   LEGACY: {legacy_cnt.count} calls, {legacy.a_tot} particles")
    print(f"   NEW:    {new_cnt.count} calls, {new.a_tot} particles")
