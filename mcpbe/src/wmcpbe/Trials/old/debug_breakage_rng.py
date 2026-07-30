"""
BREAKAGE RNG DEBUG - Compare random number consumption BEFORE solve()

This checks if Legacy and New draw the SAME random numbers during initialization.
If they differ, solve() will produce different results even with identical params.
"""
import numpy as np
import time


def compare_rng_sequence(seed=42, n_draws=10):
    """Compare RNG sequences for both cases."""
    
    # Legacy case setup
    from wmcpbe.mcpbe import MCPBESolver
    
    print("=" * 70)
    print("RNG CONSUMPTION COMPARISON")
    print("=" * 70)
    
    # === LEGACY ===
    print("\n--- LEGACY CASE ---")
    legacy = MCPBESolver(dim=1, t_total=0.05, t_write=10, init=False, load_attr=False, seed=seed)
    legacy.COLEVAL = 1
    legacy.BREAKRVAL = 1
    legacy.G = 1000
    legacy.pl_P1 = 3e-4
    legacy.pl_P2 = 1.0
    legacy.pl_v = 2.0
    legacy.pl_q = 0.5
    legacy.alpha_prim = 1.0
    legacy.process_type = 'breakage'
    
    legacy._initialize_kernels()
    legacy._initialize_particles()
    legacy._initialize_samplers()
    
    # Draw test random numbers from final state
    print(f"First {n_draws} random numbers (before solve):")
    legacy_test_draws = [legacy._rng.random() for _ in range(n_draws)]
    print(f"  {legacy_test_draws}")
    
    # === NEW ===
    print("\n--- NEW CASE ---")
    new = MCPBESolver(dim=1, t_total=0.05, t_write=10, init=False, load_attr=False, seed=seed)
    new.break_kernel_name = 'power_law'
    new.break_kernel_params = {'p1': 3e-4, 'p2': 1.0, 'g': 1000, 'breakrval': 1}
    new.porosity_growth_kernel_name = 'volume_mixing'
    new.compression_kernel_name = 'exponential_decay'
    new.compression_kernel_params = {'rate': 0.02, 'min_porosity': 0.3}
    new.liquid_dist_kernel_name = 'uniform_weighted'
    new.alpha_prim = 1.0
    new.process_type = 'breakage'
    
    new._initialize_kernels()
    new._initialize_particles()
    new._initialize_samplers()
    
    # Draw same test random numbers
    print(f"First {n_draws} random numbers (before solve):")
    new_test_draws = [new._rng.random() for _ in range(n_draws)]
    print(f"  {new_test_draws}")
    
    # === COMPARISON ===
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    print(f"\nTest draws match:")
    if legacy_test_draws == new_test_draws:
        print(f"  ✓ YES! RNG states are synchronized")
    else:
        print(f"  ❌ NO! RNG states differ!")
        for i, (leg, nw) in enumerate(zip(legacy_test_draws, new_test_draws)):
            match = "✓" if leg == nw else "❌"
            print(f"    {match} Draw {i}: Legacy={leg:.6f}, New={nw:.6f}, diff={abs(nw-leg):.6e}")


if __name__ == "__main__":
    compare_rng_sequence(seed=42, n_draws=5)
