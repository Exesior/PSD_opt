"""
Simple RNG Tracker - counts total RNG calls per breakage event.
No method wrapping, just simple counting.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class SimpleRNGCounter:
    """Simple wrapper that counts ALL rng.random() calls."""
    
    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)
        self.count = 0
    
    def random(self, size=None):
        self.count += 1
        return self._rng.random(size)
    
    def standard_normal(self, size=None):
        self.count += 1
        return self._rng.standard_normal(size)
    
    def integers(self, low, high=None, size=None):
        self.count += 1
        return self._rng.integers(low, high, size)
    
    def choice(self, a, size=None, replace=True, p=None):
        self.count += 1
        return self._rng.choice(a, size, replace, p)
    
    @property
    def rng(self):
        return self._rng


def compare_simple():
    """Simple comparison of RNG consumption."""
    
    print("=" * 80)
    print(" SIMPLE RNG TRACKER")
    print("=" * 80)
    
    seed = 42
    pl_p1 = 3e-4
    n_particles = 100
    
    from wmcpbe.mcpbe import MCPBESolver
    
    # =========================================================================
    # Create fresh solvers with counted RNG
    # =========================================================================
    print("\nCreating LEGACY solver...")
    legacy_solver = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
    )
    legacy_solver.a0 = n_particles
    legacy_solver.COLEVAL = 1
    legacy_solver.BREAKRVAL = 1
    legacy_solver.CORR_BETA = 1e-2
    legacy_solver.G = 1000
    legacy_solver.pl_P1 = pl_p1
    legacy_solver.pl_P2 = 1.0
    legacy_solver.pl_v = 2.0
    legacy_solver.pl_q = 0.5
    legacy_solver.alpha_prim = 1.0
    legacy_solver.process_type = 'breakage'
    legacy_solver._initialize_particles()
    legacy_solver._init_lmc()
    legacy_solver._initialize_kernels()
    legacy_solver._initialize_samplers()
    
    # Replace RNG with counter
    legacy_counter = SimpleRNGCounter(seed)
    legacy_solver._rng = legacy_counter
    
    print("Creating NEW solver...")
    new_solver = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
        break_kernel_name='power_law',
        break_kernel_params={
            'p1': pl_p1, 
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
    new_solver.a0 = n_particles
    new_solver.alpha_prim = 1.0
    new_solver.process_type = 'breakage'
    new_solver._initialize_particles()
    new_solver._init_lmc()
    new_solver._initialize_samplers()
    
    # Replace RNG with counter
    new_counter = SimpleRNGCounter(seed)
    new_solver._rng = new_counter
    
    # =========================================================================
    # Print LMC config
    # =========================================================================
    print("\n" + "=" * 80)
    print(" LMC Configuration")
    print("=" * 80)
    
    print(f"\nLEGACY:")
    print(f"  use_lmc_pre_model = {legacy_solver.use_lmc_pre_model}")
    print(f"  use_lmc_live = {legacy_solver.use_lmc_live}")
    print(f"  frag_num = {legacy_solver.frag_num}")
    print(f"  lmc_NO_FRAG = {legacy_solver.lmc_NO_FRAG}")
    
    # Check tables
    Vrem = np.array([legacy_solver.V_flat[0, 0]], dtype=float)
    mode, rA, rB, *_ = legacy_solver._get_break_tables_for_state(Vrem)
    print(f"  CDF table size (rA.size) = {rA.size if rA is not None else 'N/A'}")
    
    print(f"\nNEW:")
    print(f"  use_lmc_pre_model = {new_solver.use_lmc_pre_model}")
    print(f"  use_lmc_live = {new_solver.use_lmc_live}")
    print(f"  frag_num = {new_solver.frag_num}")
    print(f"  lmc_NO_FRAG = {new_solver.lmc_NO_FRAG}")
    
    Vrem_new = np.array([new_solver.V_flat[0, 0]], dtype=float)
    mode_new, rA_new, rB_new, *_ = new_solver._get_break_tables_for_state(Vrem_new)
    print(f"  CDF table size (rA.size) = {rA_new.size if rA_new is not None else 'N/A'}")
    
    # =========================================================================
    # Run single breakage events
    # =========================================================================
    print("\n" + "=" * 80)
    print(" Single Breakage Event Analysis")
    print("=" * 80)
    
    print("\nRunning ONE breakage event in LEGACY...")
    old_count = legacy_counter.count
    legacy_solver._do_one_break()
    new_count = legacy_counter.count
    legacy_calls = new_count - old_count
    print(f"  RNG calls: {legacy_calls}")
    print(f"  Particles before: {n_particles}")
    print(f"  Particles after: {legacy_solver.a_tot}")
    print(f"  Fragments created: {legacy_solver.a_tot - n_particles + 1}")  # +1 because parent loses weight
    
    print("\nRunning ONE breakage event in NEW...")
    old_count = new_counter.count
    new_solver._do_one_break()
    new_count = new_counter.count
    new_calls = new_count - old_count
    print(f"  RNG calls: {new_calls}")
    print(f"  Particles before: {n_particles}")
    print(f"  Particles after: {new_solver.a_tot}")
    print(f"  Fragments created: {new_solver.a_tot - n_particles + 1}")
    
    # =========================================================================
    # Analyze fragment count
    # =========================================================================
    print("\n" + "=" * 80)
    print(" Fragment Analysis")
    print("=" * 80)
    
    # Infer fragment count from RNG calls
    # Each fragment needs ~1-2 RNG calls (1 for u, maybe 1 more for 2D)
    # Plus 1 call for stochastic rounding
    legacy_frags_est = (legacy_calls - 1) // 1  # Subtract 1 for rounding, divide by 1 for 1D
    new_frags_est = (new_calls - 1) // 1
    
    print(f"\nEstimated fragments per break:")
    print(f"  LEGACY: ~{legacy_frags_est} fragments ({legacy_calls} RNG calls)")
    print(f"  NEW:    ~{new_frags_est} fragments ({new_calls} RNG calls)")
    
    # Check actual particle increase
    print(f"\nActual particle increase:")
    print(f"  LEGACY: +{legacy_solver.a_tot - n_particles} particles")
    print(f"  NEW:    +{new_solver.a_tot - n_particles} particles")
    
    # =========================================================================
    # Verdict
    # =========================================================================
    print("\n" + "=" * 80)
    print(" VERDICT")
    print("=" * 80)
    
    if legacy_calls > 100:
        print(f"\n  🔴 LEGACY uses MANY more RNG calls ({legacy_calls} vs {new_calls})")
        print(f"     This suggests LEGACY generates many more fragments!")
        print(f"\n  Possible causes:")
        print(f"    1. frag_num is different between solvers")
        print(f"    2. CDF table sampling is different")
        print(f"    3. Something else consumes RNG in LEGACY path")
    else:
        print(f"\n  ✓ Both solvers use similar RNG calls")
        print(f"     Problem must be elsewhere (sampler, dt calculation, etc.)")


if __name__ == "__main__":
    compare_simple()
