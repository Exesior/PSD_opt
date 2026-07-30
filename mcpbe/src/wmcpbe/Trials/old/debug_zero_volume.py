"""Debug: Are fragments really zero-volume in LEGACY?"""
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
print(" DEBUG ZERO-VOLUME FRAGMENTS")
print("="*80)

# LEGACY
legacy = MCPBESolver(dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None)
legacy.a0 = 100
legacy.COLEVAL = 1
legacy.BREAKRVAL = 1
legacy.CORR_BETA = 1e-2
legacy.G = 1000
legacy.pl_P1 = 3e-4
legacy.alpha_prim = 1.0
legacy.process_type = 'breakage'
legacy._initialize_particles()
legacy._init_lmc()
legacy._initialize_kernels()
legacy._initialize_samplers()

cnt = CounterRNG(42)
legacy._rng = cnt

# Manually trace through _do_one_break()
print("\n=== LEGACY Manual Trace ===")
k = legacy._break_sampler.sample(cnt)
print(f"Sampled k={k}, RNG calls={cnt.count}")

Vrem_k = np.array([legacy.V_flat[0, k]], dtype=float)
print(f"Parent volume: {Vrem_k}")

print("\nCalling _break_build_fragments()...")
status, frags = legacy._break_build_fragments(Vrem_k)
print(f"status={status}, len(frags)={len(frags)}, RNG calls={cnt.count}")

for i, f in enumerate(frags):
    vol = float(np.sum(f))
    print(f"  frag[{i}]: V={vol:.6e}, zero={vol <= 0}")

has_zero = any(float(np.sum(f)) <= 0 for f in frags)
print(f"\nHas zero-volume: {has_zero}")

if has_zero:
    print("\n🔴 ZERO-VOLUME DETECTED! Entering resample loop...")
    for attempt in range(min(5, 1000)):
        old_count = cnt.count
        status_retry, frags_retry = legacy._break_build_fragments(Vrem_k.copy())
        new_count = cnt.count
        
        has_zero_retry = any(float(np.sum(f)) <= 0 for f in frags_retry)
        print(f"Attempt {attempt+1}: RNG={new_count-old_count}, zero={has_zero_retry}, calls_total={cnt.count}")
        
        if not has_zero_retry:
            print("  → Success! Breaking loop.")
            break

print(f"\nTotal RNG calls: {cnt.count}")

# NEW for comparison
print("\n\n=== NEW Manual Trace ===")
new = MCPBESolver(dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
    break_kernel_name='power_law', break_kernel_params={'p1': 3e-4, 'p2': 1.0, 'g': 1000, 'breakrval': 1})
new.a0 = 100
new.alpha_prim = 1.0
new.process_type = 'breakage'
new._initialize_particles()
new._init_lmc()
new._initialize_samplers()

cnt_new = CounterRNG(42)
new._rng = cnt_new

k_new = new._break_sampler.sample(cnt_new)
Vrem_k_new = np.array([new.V_flat[0, k_new]], dtype=float)
print(f"Parent volume: {Vrem_k_new}")

status_new, frags_new = new._break_build_fragments(Vrem_k_new)
print(f"status={status_new}, len(frags)={len(frags_new)}")

for i, f in enumerate(frags_new):
    vol = float(np.sum(f))
    print(f"  frag[{i}]: V={vol:.6e}, zero={vol <= 0}")

print(f"\nTotal RNG calls: {cnt_new.count}")
