"""
Trace _do_one_break() step-by-step to find where 2000 RNG calls come from.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class CountingRNG:
    """RNG that counts ALL calls."""
    
    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)
        self.count = 0
    
    def random(self, size=None):
        self.count += 1
        return self._rng.random(size)
    
    def choice(self, a, size=None, replace=True, p=None):
        self.count += 1
        return self._rng.choice(a, size, replace, p)
    
    def shuffle(self, x):
        self.count += 1
        return self._rng.shuffle(x)
    
    def permutation(self, x):
        self.count += 1
        return self._rng.permutation(x)
    
    @property
    def rng(self):
        return self._rng


def trace_do_one_break(solver, label):
    """Trace every step of _do_one_break()."""
    
    print(f"\n{'='*80}")
    print(f" {label}: Tracing _do_one_break()")
    print(f"{'='*80}")
    
    # Replace RNG with counter
    counter = CountingRNG(42)
    solver._rng = counter
    
    particles_before = solver.a_tot
    print(f"\nParticles before: {particles_before}")
    
    # Manually trace through _do_one_break() logic
    print(f"\nStep 1: Check a_tot >= 1")
    if solver.a_tot < 1:
        print(f"  → Return early (a_tot={solver.a_tot})")
        return counter.count, particles_before
    
    print(f"  → Continue (a_tot={solver.a_tot})")
    
    print(f"\nStep 2: Ensure break sampler")
    solver._ensure_break_sampler()
    print(f"  → Sampler total: {solver._break_sampler.total():.6e}")
    
    if solver._break_sampler.total() <= 0.0:
        print(f"  → Return early (no breakable weight)")
        return counter.count, particles_before
    
    print(f"\nStep 3: Sample particle k")
    old_count = counter.count
    k = solver._break_sampler.sample(counter)
    sample_calls = counter.count - old_count
    print(f"  → Sampled k={k}, RNG calls={sample_calls}")
    
    print(f"\nStep 4: Check W[k]")
    Wk0 = float(solver.W[k])
    print(f"  → W[{k}] = {Wk0:.6e}")
    
    if Wk0 <= 0.0:
        print(f"  → Mark unbreakable and continue")
        solver._mark_unbreakable(k)
        return counter.count, particles_before
    
    print(f"\nStep 5: Compute dW_packet")
    old_count = counter.count
    dW_total = solver._compute_dW_packet(k)
    dW_calls = counter.count - old_count
    print(f"  → dW = {dW_total:.6e}, RNG calls={dW_calls}")
    
    if dW_total <= 0.0:
        print(f"  → Mark unbreakable and continue")
        solver._mark_unbreakable(k)
        return counter.count, particles_before
    
    print(f"\nStep 6: Store dW for dt update")
    solver._last_break_dW = float(dW_total)
    
    print(f"\nStep 7: Check k < a_tot")
    if k >= solver.a_tot:
        print(f"  → Return (k={k} >= a_tot={solver.a_tot})")
        return counter.count, particles_before
    
    print(f"\nStep 8: Re-check W[k]")
    Wk_now = float(solver.W[k])
    print(f"  → W[{k}] now = {Wk_now:.6e}")
    
    if Wk_now <= 0.0:
        print(f"  → Return (weight now zero)")
        return counter.count, particles_before
    
    dW = min(float(dW_total), Wk_now)
    print(f"  → dW = min({dW_total:.6e}, {Wk_now:.6e}) = {dW:.6e}")
    
    if dW <= 0.0:
        print(f"  → Return (dW <= 0)")
        return counter.count, particles_before
    
    print(f"\nStep 9: Get parent volume Vrem_k")
    if solver.dim == 1:
        Vrem_k = np.array([solver.V_flat[0, k]], dtype=float)
    else:
        Vrem_k = np.array([solver.V_flat[0, k], solver.V_flat[1, k]], dtype=float)
    print(f"  → Vrem_k = {Vrem_k}")
    
    print(f"\nStep 10: Call _break_build_fragments() 🔴")
    old_count = counter.count
    status, frags = solver._break_build_fragments(Vrem_k)
    build_calls = counter.count - old_count
    print(f"  → status='{status}', len(frags)={len(frags)}, RNG calls={build_calls}")
    
    if status == "disable":
        print(f"  → Mark unbreakable and return")
        solver._mark_unbreakable(k)
        return counter.count, particles_before
    
    print(f"\nStep 11: Check fragment volumes")
    for i, f in enumerate(frags):
        vol = float(np.sum(f))
        print(f"  → frag[{i}]: V={vol:.6e}")
    
    has_zero = any(float(np.sum(f)) <= 0 for f in frags)
    print(f"  → Has zero-volume: {has_zero}")
    
    if has_zero:
        print(f"\n⚠️  ZERO-VOLUME DETECTED! Entering resample loop...")
        # This is where the 2000 calls could come from!
    
    print(f"\nStep 12: Call _break_apply_and_maintain() 🔴")
    old_count = counter.count
    solver._break_apply_and_maintain(k, frags, dW, Vrem_k)
    apply_calls = counter.count - old_count
    print(f"  → RNG calls={apply_calls}")
    
    particles_after = solver.a_tot
    print(f"\nParticles after: {particles_after} (Δ={particles_after - particles_before})")
    
    total_calls = counter.count
    print(f"\nTotal RNG calls: {total_calls}")
    print(f"  - sampler.sample():     {sample_calls}")
    print(f"  - _compute_dW_packet(): {dW_calls}")
    print(f"  - _break_build_fragments(): {build_calls}")
    print(f"  - _break_apply_and_maintain(): {apply_calls}")
    print(f"  - Other/unaccounted:    {total_calls - sample_calls - dW_calls - build_calls - apply_calls}")
    
    return total_calls, particles_after


def main():
    print("=" * 80)
    print(" TRACE _do_one_break() STEP-BY-STEP")
    print("=" * 80)
    
    from wmcpbe.mcpbe import MCPBESolver
    
    pl_p1 = 3e-4
    n_particles = 100
    
    # LEGACY
    print("\n" + "="*80)
    print(" CREATING LEGACY SOLVER")
    print("="*80)
    
    legacy = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
    )
    legacy.a0 = n_particles
    legacy.COLEVAL = 1
    legacy.BREAKRVAL = 1
    legacy.CORR_BETA = 1e-2
    legacy.G = 1000
    legacy.pl_P1 = pl_p1
    legacy.pl_P2 = 1.0
    legacy.alpha_prim = 1.0
    legacy.process_type = 'breakage'
    legacy._initialize_particles()
    legacy._init_lmc()
    legacy._initialize_kernels()
    legacy._initialize_samplers()
    
    # NEW
    print("\n" + "="*80)
    print(" CREATING NEW SOLVER")
    print("="*80)
    
    new = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
        break_kernel_name='power_law',
        break_kernel_params={'p1': pl_p1, 'p2': 1.0, 'g': 1000, 'breakrval': 1},
        porosity_growth_kernel_name='volume_mixing',
        compression_kernel_name='exponential_decay',
        compression_kernel_params={'rate': 0.02, 'min_porosity': 0.3},
        liquid_dist_kernel_name='uniform_weighted',
    )
    new.a0 = n_particles
    new.alpha_prim = 1.0
    new.process_type = 'breakage'
    new._initialize_particles()
    new._init_lmc()
    new._initialize_samplers()
    
    # Trace both
    legacy_calls, legacy_particles = trace_do_one_break(legacy, "LEGACY")
    new_calls, new_particles = trace_do_one_break(new, "NEW")
    
    # Verdict
    print(f"\n{'='*80}")
    print(" FINAL VERDICT")
    print(f"{'='*80}")
    
    print(f"\nLEGACY: {legacy_calls} RNG calls, {legacy_particles} particles")
    print(f"NEW:    {new_calls} RNG calls, {new_particles} particles")
    print(f"\nDifference: {legacy_calls - new_calls} RNG calls")
    
    if legacy_calls > 100:
        print(f"\n🔴 LEGACY has MANY extra RNG calls!")
        print(f"   Check the step-by-step output above to find where they occur.")
    
    if legacy_particles == 100:
        print(f"\n⚠️  LEGACY did NOT create new particles!")
        print(f"   Breakage event was aborted or fragments filtered out.")


if __name__ == "__main__":
    main()
