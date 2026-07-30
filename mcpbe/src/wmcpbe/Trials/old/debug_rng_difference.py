"""
Debug: Why does simple RNG tracker show 2003 calls but step-by-step shows 3?

The difference must be in HOW we replace the RNG.
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


def test_replacement_strategy():
    """Test different RNG replacement strategies."""
    
    from wmcpbe.mcpbe import MCPBESolver
    
    pl_p1 = 3e-4
    n_particles = 100
    
    print("=" * 80)
    print(" TESTING RNG REPLACEMENT STRATEGIES")
    print("=" * 80)
    
    # =========================================================================
    # Strategy 1: Replace BEFORE _initialize_samplers()
    # =========================================================================
    print("\n\n=== Strategy 1: Replace RNG BEFORE _initialize_samplers() ===")
    
    solver1 = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
    )
    solver1.a0 = n_particles
    solver1.COLEVAL = 1
    solver1.BREAKRVAL = 1
    solver1.CORR_BETA = 1e-2
    solver1.G = 1000
    solver1.pl_P1 = pl_p1
    solver1.pl_P2 = 1.0
    solver1.alpha_prim = 1.0
    solver1.process_type = 'breakage'
    solver1._initialize_particles()
    solver1._init_lmc()
    solver1._initialize_kernels()
    
    # Replace RNG BEFORE creating samplers
    counter1 = CountingRNG(42)
    solver1._rng = counter1
    
    solver1._initialize_samplers()
    
    # Now run breakage
    print(f"Running _do_one_break()...")
    solver1._do_one_break()
    
    print(f"RNG calls: {counter1.count}")
    print(f"Particles: {solver1.a_tot}")
    
    # Check if sampler uses the same RNG
    print(f"solver._rng is counter1: {solver1._rng is counter1}")
    print(f"solver._break_sampler: {solver1._break_sampler}")
    
    # =========================================================================
    # Strategy 2: Replace AFTER _initialize_samplers() (like debug_simple_rng.py)
    # =========================================================================
    print("\n\n=== Strategy 2: Replace RNG AFTER _initialize_samplers() ===")
    
    solver2 = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
    )
    solver2.a0 = n_particles
    solver2.COLEVAL = 1
    solver2.BREAKRVAL = 1
    solver2.CORR_BETA = 1e-2
    solver2.G = 1000
    solver2.pl_P1 = pl_p1
    solver2.pl_P2 = 1.0
    solver2.alpha_prim = 1.0
    solver2.process_type = 'breakage'
    solver2._initialize_particles()
    solver2._init_lmc()
    solver2._initialize_kernels()
    solver2._initialize_samplers()
    
    # Replace RNG AFTER creating samplers
    counter2 = CountingRNG(42)
    solver2._rng = counter2
    
    # Now run breakage
    print(f"Running _do_one_break()...")
    solver2._do_one_break()
    
    print(f"RNG calls: {counter2.count}")
    print(f"Particles: {solver2.a_tot}")
    
    # =========================================================================
    # Strategy 3: Use seed parameter (like original debug_simple_rng.py intended)
    # =========================================================================
    print("\n\n=== Strategy 3: Use seed=42 in constructor ===")
    
    solver3 = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=42,
    )
    solver3.a0 = n_particles
    solver3.COLEVAL = 1
    solver3.BREAKRVAL = 1
    solver3.CORR_BETA = 1e-2
    solver3.G = 1000
    solver3.pl_P1 = pl_p1
    solver3.pl_P2 = 1.0
    solver3.alpha_prim = 1.0
    solver3.process_type = 'breakage'
    solver3._initialize_particles()
    solver3._init_lmc()
    solver3._initialize_kernels()
    solver3._initialize_samplers()
    
    # Don't replace RNG - use the one created with seed=42
    print(f"Running _do_one_break()...")
    old_count_before = 0  # Can't count without wrapper
    
    # Let's monkey-patch to count
    original_random = solver3._rng.random
    call_count = [0]
    
    def counting_random(*args, **kwargs):
        call_count[0] += 1
        return original_random(*args, **kwargs)
    
    solver3._rng.random = counting_random
    
    solver3._do_one_break()
    
    print(f"RNG calls: {call_count[0]}")
    print(f"Particles: {solver3.a_tot}")
    
    # =========================================================================
    # Verdict
    # =========================================================================
    print("\n\n" + "=" * 80)
    print(" VERDICT")
    print("=" * 80)
    
    print(f"\nStrategy 1 (replace before samplers): {counter1.count} calls")
    print(f"Strategy 2 (replace after samplers):  {counter2.count} calls")
    print(f"Strategy 3 (monkey-patch):            {call_count[0]} calls")
    
    if counter2.count > 100 or call_count[0] > 100:
        print(f"\n🔴 High RNG count detected!")
        print(f"   This suggests something OTHER than _do_one_break() consumes RNG.")
        print(f"   Maybe initialization? Or sampler creation?")


if __name__ == "__main__":
    test_replacement_strategy()
