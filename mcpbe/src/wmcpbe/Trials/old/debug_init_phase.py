"""
Debug: Check if RNG calls happen during INITIALIZATION phase.

Maybe the 2000 calls come from init, not from _do_one_break()?
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
    
    @property
    def rng(self):
        return self._rng


def test_with_counting_from_start():
    """Use counting RNG from the VERY BEGINNING of initialization."""
    
    from wmcpbe.mcpbe import MCPBESolver
    
    pl_p1 = 3e-4
    n_particles = 100
    
    print("=" * 80)
    print(" COUNTING RNG CALLS FROM START OF INITIALIZATION")
    print("=" * 80)
    
    # =========================================================================
    # LEGACY: Use counting RNG from start
    # =========================================================================
    print("\n\n=== LEGACY Solver ===")
    
    # Create solver WITHOUT seed, then inject counting RNG BEFORE any init
    legacy = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
    )
    
    # Inject counting RNG BEFORE any initialization!
    counter_legacy = CountingRNG(42)
    legacy._rng = counter_legacy
    
    print(f"Phase 0 (after RNG injection): {counter_legacy.count} calls")
    
    legacy.a0 = n_particles
    legacy.COLEVAL = 1
    legacy.BREAKRVAL = 1
    legacy.CORR_BETA = 1e-2
    legacy.G = 1000
    legacy.pl_P1 = pl_p1
    legacy.pl_P2 = 1.0
    legacy.alpha_prim = 1.0
    legacy.process_type = 'breakage'
    
    print(f"Phase 1 (after setting attributes): {counter_legacy.count} calls")
    
    legacy._initialize_particles()
    print(f"Phase 2 (_initialize_particles): {counter_legacy.count} calls")
    
    legacy._init_lmc()
    print(f"Phase 3 (_init_lmc): {counter_legacy.count} calls")
    
    legacy._initialize_kernels()
    print(f"Phase 4 (_initialize_kernels): {counter_legacy.count} calls")
    
    legacy._initialize_samplers()
    print(f"Phase 5 (_initialize_samplers): {counter_legacy.count} calls")
    
    init_total = counter_legacy.count
    print(f"\nTOTAL INITIALIZATION: {init_total} RNG calls")
    
    # Now run breakage
    print(f"\nRunning _do_one_break()...")
    old_count = counter_legacy.count
    legacy._do_one_break()
    break_calls = counter_legacy.count - old_count
    
    print(f"Breakage event: {break_calls} RNG calls")
    print(f"Particles: {legacy.a_tot}")
    print(f"CUMULATIVE TOTAL: {counter_legacy.count} RNG calls")
    
    # =========================================================================
    # NEW: Use counting RNG from start
    # =========================================================================
    print("\n\n=== NEW Solver ===")
    
    new = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
        break_kernel_name='power_law',
        break_kernel_params={'p1': pl_p1, 'p2': 1.0, 'g': 1000, 'breakrval': 1},
        porosity_growth_kernel_name='volume_mixing',
        compression_kernel_name='exponential_decay',
        compression_kernel_params={'rate': 0.02, 'min_porosity': 0.3},
        liquid_dist_kernel_name='uniform_weighted',
    )
    
    # Inject counting RNG BEFORE any initialization!
    counter_new = CountingRNG(42)
    new._rng = counter_new
    
    print(f"Phase 0 (after RNG injection): {counter_new.count} calls")
    
    new.a0 = n_particles
    new.alpha_prim = 1.0
    new.process_type = 'breakage'
    
    print(f"Phase 1 (after setting attributes): {counter_new.count} calls")
    
    new._initialize_particles()
    print(f"Phase 2 (_initialize_particles): {counter_new.count} calls")
    
    new._init_lmc()
    print(f"Phase 3 (_init_lmc): {counter_new.count} calls")
    
    # _initialize_kernels() already done in __init__
    print(f"Phase 4 (_initialize_kernels in __init__): {counter_new.count} calls")
    
    new._initialize_samplers()
    print(f"Phase 5 (_initialize_samplers): {counter_new.count} calls")
    
    init_total_new = counter_new.count
    print(f"\nTOTAL INITIALIZATION: {init_total_new} RNG calls")
    
    # Now run breakage
    print(f"\nRunning _do_one_break()...")
    old_count = counter_new.count
    new._do_one_break()
    break_calls_new = counter_new.count - old_count
    
    print(f"Breakage event: {break_calls_new} RNG calls")
    print(f"Particles: {new.a_tot}")
    print(f"CUMULATIVE TOTAL: {counter_new.count} RNG calls")
    
    # =========================================================================
    # Verdict
    # =========================================================================
    print("\n\n" + "=" * 80)
    print(" VERDICT")
    print("=" * 80)
    
    print(f"\nLEGACY:")
    print(f"  Initialization: {init_total} RNG calls")
    print(f"  Breakage:       {break_calls} RNG calls")
    print(f"  Total:          {counter_legacy.count} RNG calls")
    print(f"  Particles:      {legacy.a_tot}")
    
    print(f"\nNEW:")
    print(f"  Initialization: {init_total_new} RNG calls")
    print(f"  Breakage:       {break_calls_new} RNG calls")
    print(f"  Total:          {counter_new.count} RNG calls")
    print(f"  Particles:      {new.a_tot}")
    
    print(f"\nDifference:")
    print(f"  Initialization: {init_total - init_total_new} calls")
    print(f"  Breakage:       {break_calls - break_calls_new} calls")
    
    if init_total > 1000:
        print(f"\n🔴 FOUND IT! LEGACY initialization consumes {init_total} RNG calls!")
        print(f"   This explains the 2000 call difference!")
        print(f"   Check which init function calls RNG...")


if __name__ == "__main__":
    test_with_counting_from_start()
