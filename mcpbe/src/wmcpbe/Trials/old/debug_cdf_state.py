"""Debug: Check CDF table state and fragment volumes"""
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
print(" DEBUG CDF STATE AND FRAGMENT VOLUMES")
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

cnt = CounterRNG(42)
legacy._rng = cnt

# Get parent volume
Vrem = np.array([legacy.V_flat[0, 0]], dtype=float)
print(f"Parent volume: V={Vrem}")

# Check CDF tables
mode, rA, rB, _, _, _, _, pexp = legacy._get_break_tables_for_state(Vrem)
print(f"\nCDF Table State:")
print(f"  mode = '{mode}'")
print(f"  rA.size = {rA.size if rA is not None else 'N/A'}")
print(f"  rB.size = {rB.size if rB is not None else 'N/A'}")
print(f"  pexp = {pexp}")
print(f"  frag_num = {legacy.frag_num}")
print(f"  use_lmc_pre_model = {legacy.use_lmc_pre_model}")

# Check CDF values
if rA is not None and rA.size > 0:
    print(f"\n  rA[:10] = {rA[:10]}")
if rB is not None and rB.size > 0:
    print(f"  rB[:10] = {rB[:10]}")

# Generate fragments manually
print(f"\nGenerating fragments...")
status, frags = legacy._break_build_fragments(Vrem)
print(f"status='{status}', len(frags)={len(frags)}")

for i, f in enumerate(frags):
    vol = float(np.sum(f))
    print(f"  frag[{i}]: V={vol:.6e}, zero={vol <= 0}")

has_zero = any(float(np.sum(f)) <= 0 for f in frags)
print(f"\nHas zero-volume: {has_zero}")
print(f"RNG calls so far: {cnt.count}")

# If zero-volume, check resample loop
if has_zero:
    print(f"\n🔴 ZERO-VOLUME! Checking resample attempts...")
    for attempt in range(min(10, 1000)):
        old_cnt = cnt.count
        status_r, frags_r = legacy._break_build_fragments(Vrem.copy())
        has_zero_r = any(float(np.sum(f)) <= 0 for f in frags_r)
        
        vol_str = ", ".join([f"{float(np.sum(f)):.2e}" for f in frags_r])
        print(f"  Attempt {attempt+1}: RNG+={cnt.count-old_cnt}, total={cnt.count}, volumes=[{vol_str}], zero={has_zero_r}")
        
        if not has_zero_r:
            print(f"  → Success at attempt {attempt+1}!")
            break

print(f"\nTotal RNG calls: {cnt.count}")

# NEW for comparison
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

cnt_new = CounterRNG(42)
new._rng = cnt_new

Vrem_new = np.array([new.V_flat[0, 0]], dtype=float)
print(f"Parent volume: V={Vrem_new}")

mode_new, rA_new, rB_new, _, _, _, _, pexp_new = new._get_break_tables_for_state(Vrem_new)
print(f"\nCDF Table State:")
print(f"  mode = '{mode_new}'")
print(f"  rA.size = {rA_new.size if rA_new is not None else 'N/A'}")
print(f"  pexp = {pexp_new}")
print(f"  frag_num = {new.frag_num}")

if rA_new is not None and rA_new.size > 0:
    print(f"  rA[:10] = {rA_new[:10]}")

status_new, frags_new = new._break_build_fragments(Vrem_new)
print(f"\nstatus='{status_new}', len(frags)={len(frags_new)}")

for i, f in enumerate(frags_new):
    vol = float(np.sum(f))
    print(f"  frag[{i}]: V={vol:.6e}, zero={vol <= 0}")

print(f"\nTotal RNG calls: {cnt_new.count}")
