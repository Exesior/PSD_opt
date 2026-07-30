"""Reproduce 2003 RNG calls - FIXED VERSION with all parameters"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class CounterRNG:
    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)
        self.count = 0
    def random(self, size=None):
        self.count += 1
        return self._rng.random(size)
    def choice(self, a, size=None, replace=True, p=None):
        self.count += 1
        return self._rng.choice(a, size, replace, p)
    @property
    def rng(self): return self._rng

from wmcpbe.mcpbe import MCPBESolver

print("="*80)
print(" REPRODUCE 2003 RNG CALLS - FIXED")
print("="*80)

# LEGACY - NOW WITH ALL PARAMETERS
legacy = MCPBESolver(dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None)
legacy.a0 = 100
legacy.COLEVAL = 1
legacy.BREAKRVAL = 1
legacy.CORR_BETA = 1e-2
legacy.G = 1000
legacy.pl_P1 = 3e-4
legacy.pl_P2 = 1.0       # ← ADDED
legacy.pl_v = 2.0        # ← ADDED
legacy.pl_q = 0.5        # ← ADDED
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

# NEW - NOW WITH ALL PARAMETERS
new = MCPBESolver(
    dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
    break_kernel_name='power_law', 
    break_kernel_params={
        'p1': 3e-4, 
        'p2': 1.0, 
        'g': 1000, 
        'breakrval': 1,
        'pl_v': 2.0,     # ← ADDED
        'pl_q': 0.5,     # ← ADDED
    },
    porosity_growth_kernel_name='volume_mixing',           # ← ADDED
    compression_kernel_name='exponential_decay',           # ← ADDED
    compression_kernel_params={'rate': 0.02, 'min_porosity': 0.3},  # ← ADDED
    liquid_dist_kernel_name='uniform_weighted',            # ← ADDED
)
new.a0 = 100
new.alpha_prim = 1.0
new.process_type = 'breakage'
new._initialize_particles()
new._init_lmc()
new._initialize_samplers()

new_cnt = CounterRNG(42)
new._rng = new_cnt
new._do_one_break()
print(f"NEW:    {new_cnt.count} calls, {new.a_tot} particles")

if legacy_cnt.count > 100:
    print(f"\n🔴 REPRODUCED! {legacy_cnt.count} calls!")
else:
    print(f"\n✓ Both ~3 calls (bug not reproduced)")
