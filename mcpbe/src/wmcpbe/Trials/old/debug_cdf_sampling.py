"""
Debug CDF Sampling - finds WHY LEGACY gets zero-volume fragments.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TracingRNG:
    """RNG wrapper that logs every call."""
    
    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)
        self.count = 0
        self.log = []
    
    def random(self, size=None):
        self.count += 1
        val = self._rng.random(size)
        self.log.append(f"  RNG #{self.count}: random() = {val}")
        return val
    
    def choice(self, a, size=None, replace=True, p=None):
        self.count += 1
        val = self._rng.choice(a, size, replace, p)
        self.log.append(f"  RNG #{self.count}: choice() = {val}")
        return val
    
    @property
    def rng(self):
        return self._rng


def trace_fragment_generation(solver, label):
    """Trace fragment generation step-by-step."""
    
    print(f"\n{'='*80}")
    print(f" {label}: Fragment Generation Trace")
    print(f"{'='*80}")
    
    # Replace RNG with tracer
    tracer = TracingRNG(42)
    solver._rng = tracer
    
    # Get parent volume
    Vrem_k = np.array([solver.V_flat[0, 0]], dtype=float)
    print(f"\nParent volume: V = {Vrem_k}")
    
    # Call _get_break_tables_for_state
    print(f"\nCalling _get_break_tables_for_state()...")
    mode, rA, rB, rowsum_cdf, row_cdf, zmin1d, zmin3d, pexp = solver._get_break_tables_for_state(Vrem_k)
    
    print(f"  mode = '{mode}'")
    print(f"  rA.size = {rA.size if rA is not None else 'N/A'}")
    print(f"  pexp = {pexp}")
    print(f"  frag_num = {solver.frag_num}")
    
    # Determine fragment count
    use_lmc = bool(getattr(solver, "use_lmc_pre_model", False))
    p = float(pexp) if (pexp is not None and use_lmc) else float(getattr(solver, "frag_num", 2))
    print(f"\nExpected fragment count: p = {p} (use_lmc={use_lmc})")
    
    import math
    fl = int(math.floor(p))
    ce = int(math.ceil(p))
    if fl <= 1:
        fl = 2
        ce = 2
    
    old_count = tracer.count
    u_stoch = float(tracer.random())
    n = ce if (u_stoch < (p - fl)) else fl
    print(f"Stochastic rounding: u={u_stoch:.4f}, p-fl={p-fl:.4f} → n={n}")
    stoch_calls = tracer.count - old_count
    
    # Generate fragments
    print(f"\nGenerating {n-1} fragments via _produce_one_frag_from_remaining():")
    Vrem = Vrem_k.copy()
    frags = []
    
    for i in range(n - 1):
        old_count = tracer.count
        frag = solver._produce_one_frag_from_remaining(Vrem)
        calls = tracer.count - old_count
        
        vol = float(np.sum(frag))
        print(f"  Fragment {i+1}: V={vol:.6e}, RNG calls={calls}")
        if vol <= 0:
            print(f"    ⚠️ ZERO VOLUME!")
            # Show what happened
            print(f"    CDF sampling details:")
            print(f"      rA[:5] = {rA[:5]}")
            print(f"      rB[:5] = {rB[:5] if rB is not None else 'N/A'}")
        
        frags.append(frag)
        Vrem -= frag
    
    # Last fragment
    last_vol = float(np.sum(Vrem))
    print(f"  Last fragment: V={last_vol:.6e}")
    frags.append(Vrem)
    
    # Check for zero volume
    has_zero = any(float(np.sum(f)) <= 0 for f in frags)
    print(f"\nZero-volume fragments: {has_zero}")
    
    # Print full RNG log
    print(f"\nRNG Call Log ({tracer.count} total):")
    for entry in tracer.log[:20]:  # First 20
        print(entry)
    if len(tracer.log) > 20:
        print(f"  ... and {len(tracer.log) - 20} more")
    
    return has_zero, tracer.count


def main():
    print("=" * 80)
    print(" CDF SAMPLING DEBUG")
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
    legacy_has_zero, legacy_calls = trace_fragment_generation(legacy, "LEGACY")
    new_has_zero, new_calls = trace_fragment_generation(new, "NEW")
    
    # Verdict
    print(f"\n{'='*80}")
    print(" VERDICT")
    print(f"{'='*80}")
    
    print(f"\nLEGACY: {legacy_calls} RNG calls, zero-volume={legacy_has_zero}")
    print(f"NEW:    {new_calls} RNG calls, zero-volume={new_has_zero}")
    
    if legacy_has_zero and not new_has_zero:
        print(f"\n🔴 FOUND IT! LEGACY produces zero-volume fragments!")
        print(f"   This triggers the resample loop (max 1000 attempts)")
        print(f"   Each attempt consumes RNG calls → explains 2003 calls!")
    elif legacy_calls > new_calls:
        print(f"\n⚠️  Different RNG consumption but no zero-volume detected")
        print(f"   Need to investigate further...")


if __name__ == "__main__":
    main()
