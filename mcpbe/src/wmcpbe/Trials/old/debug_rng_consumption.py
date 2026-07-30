"""
RNG Consumption Audit.

Count exact number of random number generator calls during:
1. Solver initialization
2. First 10 breakage events
3. Full simulation

This helps identify if Legacy and NEW solvers consume different numbers
of random numbers, which would cause divergence even with same seed.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class RNGCounter:
    """Wrapper around numpy RNG that counts random() calls."""
    
    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)
        self.count = 0
        self.history = []  # Optional: store actual values for verification
    
    def random(self, size=None):
        """Counted random number generation."""
        self.count += 1
        value = self._rng.random(size)
        self.history.append((self.count, float(value) if size is None else value.sum()))
        return value
    
    def standard_normal(self, size=None):
        """Counted standard normal generation."""
        self.count += 1
        value = self._rng.standard_normal(size)
        self.history.append((self.count, float(value.sum()) if size is not None else float(value)))
        return value
    
    def integers(self, low, high=None, size=None):
        """Counted integer generation."""
        self.count += 1
        value = self._rng.integers(low, high, size)
        self.history.append((self.count, int(value.sum()) if size is not None else int(value)))
        return value
    
    def choice(self, a, size=None, replace=True, p=None):
        """Counted choice."""
        self.count += 1
        value = self._rng.choice(a, size, replace, p)
        self.history.append((self.count, int(value.sum()) if hasattr(value, 'sum') else int(value)))
        return value
    
    def shuffle(self, x):
        """Counted shuffle."""
        self.count += 1
        self._rng.shuffle(x)
        self.history.append((self.count, f"shuffle({len(x)})"))
    
    def permutation(self, x):
        """Counted permutation."""
        self.count += 1
        return self._rng.permutation(x)
    
    @property
    def rng(self):
        """Return underlying RNG for direct access (uncounted!)."""
        return self._rng


def wrap_solver_rng(solver, counter):
    """Replace solver._rng with counted wrapper."""
    original_rng = solver._rng
    solver._rng = counter
    return original_rng


def compare_rng_consumption():
    """Compare RNG consumption between Legacy and NEW solvers."""
    
    print("=" * 80)
    print(" RNG CONSUMPTION AUDIT")
    print("=" * 80)
    
    seed = 42
    pl_p1 = 3e-4
    g = 1000
    breakrval = 1
    n_particles = 1000
    
    print(f"\nConfiguration:")
    print(f"  seed = {seed}")
    print(f"  pl_P1 = {pl_p1}")
    print(f"  G = {g}")
    print(f"  BREAKRVAL = {breakrval}")
    print(f"  n_particles = {n_particles}")
    
    # =========================================================================
    # PHASE 1: Initialization
    # =========================================================================
    print("\n" + "=" * 80)
    print(" PHASE 1: RNG Consumption During Initialization")
    print("=" * 80)
    
    # LEGACY solver
    from wmcpbe.mcpbe import MCPBESolver
    
    print("\n[LEGACY] Creating solver...")
    legacy_counter_init = RNGCounter(seed)
    legacy_solver = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
    )
    wrap_solver_rng(legacy_solver, legacy_counter_init)
    
    # Set parameters
    legacy_solver.a0 = n_particles
    legacy_solver.COLEVAL = 1
    legacy_solver.BREAKRVAL = breakrval
    legacy_solver.CORR_BETA = 1e-2
    legacy_solver.G = g
    legacy_solver.pl_P1 = pl_p1
    legacy_solver.pl_P2 = 1.0
    legacy_solver.pl_v = 2.0
    legacy_solver.pl_q = 0.5
    legacy_solver.alpha_prim = 1.0
    legacy_solver.process_type = 'breakage'
    
    # Initialize in order
    legacy_solver._initialize_particles()
    count_after_particles = legacy_counter_init.count
    
    legacy_solver._init_lmc()
    count_after_lmc = legacy_counter_init.count
    
    legacy_solver._initialize_kernels()
    count_after_kernels = legacy_counter_init.count
    
    legacy_solver._initialize_samplers()
    count_after_samplers = legacy_counter_init.count
    
    print(f"\n[LEGACY] RNG consumption during init:")
    print(f"  After _initialize_particles():  {count_after_particles:>6} calls")
    print(f"  After _init_lmc():              {count_after_lmc - count_after_particles:>6} calls")
    print(f"  After _initialize_kernels():    {count_after_kernels - count_after_lmc:>6} calls")
    print(f"  After _initialize_samplers():   {count_after_samplers - count_after_kernels:>6} calls")
    print(f"  TOTAL:                          {count_after_samplers:>6} calls")
    
    # NEW solver
    print("\n[NEW] Creating solver...")
    new_counter_init = RNGCounter(seed)
    new_solver = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=None,
        break_kernel_name='power_law',
        break_kernel_params={
            'p1': pl_p1, 
            'p2': 1.0, 
            'g': g, 
            'breakrval': breakrval,
            'pl_v': 2.0,
            'pl_q': 0.5,
        },
        porosity_growth_kernel_name='volume_mixing',
        compression_kernel_name='exponential_decay',
        compression_kernel_params={'rate': 0.02, 'min_porosity': 0.3},
        liquid_dist_kernel_name='uniform_weighted',
    )
    wrap_solver_rng(new_solver, new_counter_init)
    
    # Set parameters
    new_solver.a0 = n_particles
    new_solver.alpha_prim = 1.0
    new_solver.process_type = 'breakage'
    
    # Initialize in order (same as legacy)
    new_solver._initialize_particles()
    count_after_particles_new = new_counter_init.count
    
    new_solver._init_lmc()
    count_after_lmc_new = new_counter_init.count
    
    # NOTE: _initialize_kernels() already called in __init__, skip here!
    count_after_kernels_new = count_after_lmc_new  # No change
    
    new_solver._initialize_samplers()
    count_after_samplers_new = new_counter_init.count
    
    print(f"\n[NEW] RNG consumption during init:")
    print(f"  After _initialize_particles():  {count_after_particles_new:>6} calls")
    print(f"  After _init_lmc():              {count_after_lmc_new - count_after_particles_new:>6} calls")
    print(f"  After _initialize_kernels():    {count_after_kernels_new - count_after_lmc_new:>6} calls (skipped)")
    print(f"  After _initialize_samplers():   {count_after_samplers_new - count_after_kernels_new:>6} calls")
    print(f"  TOTAL:                          {count_after_samplers_new:>6} calls")
    
    # Compare
    print(f"\n{'='*80}")
    print(f" INITIALIZATION COMPARISON")
    print(f"{'='*80}")
    print(f"  LEGACY total: {count_after_samplers:>6} RNG calls")
    print(f"  NEW total:    {count_after_samplers_new:>6} RNG calls")
    print(f"  DIFFERENCE:   {count_after_samplers_new - count_after_samplers:>6} RNG calls")
    
    if count_after_samplers != count_after_samplers_new:
        print(f"\n  ⚠️  WARNING: Different RNG consumption during init!")
        print(f"      This will cause divergence even with same seed!")
    else:
        print(f"\n  ✓ OK: Same RNG consumption during init")
    
    # =========================================================================
    # PHASE 2: First Breakage Events
    # =========================================================================
    print("\n" + "=" * 80)
    print(" PHASE 2: RNG Consumption During First 10 Breakage Events")
    print("=" * 80)
    
    # Reset counters for solve phase
    legacy_counter_solve = RNGCounter(None)
    legacy_counter_solve._rng = legacy_solver._rng._rng if hasattr(legacy_solver._rng, '_rng') else legacy_solver._rng
    legacy_solver._rng = legacy_counter_solve
    
    new_counter_solve = RNGCounter(None)
    new_counter_solve._rng = new_solver._rng._rng if hasattr(new_solver._rng, '_rng') else new_solver._rng
    new_solver._rng = new_counter_solve
    
    # Run 10 breakage events manually
    print("\nRunning 10 breakage events...")
    
    for i in range(10):
        old_count_legacy = legacy_counter_solve.count
        legacy_solver._do_one_break()
        new_count_legacy = legacy_counter_solve.count
        
        old_count_new = new_counter_solve.count
        new_solver._do_one_break()
        new_count_new = new_counter_solve.count
        
        diff_legacy = new_count_legacy - old_count_legacy
        diff_new = new_count_new - old_count_new
        
        print(f"  Event {i+1:2d}: LEGACY={diff_legacy:>3} calls, NEW={diff_new:>3} calls, "
              f"delta={diff_new - diff_legacy:>+3}")
        
        # Early exit if no more breakable particles
        if legacy_solver._break_sampler.total() <= 0:
            print(f"  [LEGACY] No more breakable particles after event {i+1}")
            break
        if new_solver._break_sampler.total() <= 0:
            print(f"  [NEW] No more breakable particles after event {i+1}")
            break
    
    print(f"\n{'='*80}")
    print(f" BREAKAGE EVENTS COMPARISON (first 10 events)")
    print(f"{'='*80}")
    print(f"  LEGACY total: {legacy_counter_solve.count:>6} RNG calls")
    print(f"  NEW total:    {new_counter_solve.count:>6} RNG calls")
    print(f"  DIFFERENCE:   {new_counter_solve.count - legacy_counter_solve.count:>6} RNG calls")
    
    # =========================================================================
    # PHASE 3: Full Simulation
    # =========================================================================
    print("\n" + "=" * 80)
    print(" PHASE 3: RNG Consumption During Full Simulation (t=0.5s)")
    print("=" * 80)
    
    # Create fresh solvers for full simulation test
    print("\nCreating fresh solvers for full simulation...")
    
    # LEGACY
    legacy_full = MCPBESolver(
        dim=1, t_total=0.5, t_write=10, init=False, load_attr=False, seed=seed,
    )
    legacy_full.a0 = n_particles
    legacy_full.COLEVAL = 1
    legacy_full.BREAKRVAL = breakrval
    legacy_full.CORR_BETA = 1e-2
    legacy_full.G = g
    legacy_full.pl_P1 = pl_p1
    legacy_full.pl_P2 = 1.0
    legacy_full.pl_v = 2.0
    legacy_full.pl_q = 0.5
    legacy_full.alpha_prim = 1.0
    legacy_full.process_type = 'breakage'
    legacy_full._initialize_particles()
    legacy_full._init_lmc()
    legacy_full._initialize_kernels()
    legacy_full._initialize_samplers()
    
    # NEW
    new_full = MCPBESolver(
        dim=1, t_total=0.5, t_write=10, init=False, load_attr=False, seed=seed,
        break_kernel_name='power_law',
        break_kernel_params={
            'p1': pl_p1, 
            'p2': 1.0, 
            'g': g, 
            'breakrval': breakrval,
            'pl_v': 2.0,
            'pl_q': 0.5,
        },
        porosity_growth_kernel_name='volume_mixing',
        compression_kernel_name='exponential_decay',
        compression_kernel_params={'rate': 0.02, 'min_porosity': 0.3},
        liquid_dist_kernel_name='uniform_weighted',
    )
    new_full.a0 = n_particles
    new_full.alpha_prim = 1.0
    new_full.process_type = 'breakage'
    new_full._initialize_particles()
    new_full._init_lmc()
    # _initialize_kernels() already done in __init__
    new_full._initialize_samplers()
    
    # Wrap with counters
    legacy_full_counter = RNGCounter(None)
    legacy_full_counter._rng = legacy_full._rng
    legacy_full._rng = legacy_full_counter
    
    new_full_counter = RNGCounter(None)
    new_full_counter._rng = new_full._rng
    new_full._rng = new_full_counter
    
    # Run simulation
    print("Running LEGACY simulation...")
    legacy_full.solve()
    
    print("Running NEW simulation...")
    new_full.solve()
    
    print(f"\n{'='*80}")
    print(f" FULL SIMULATION COMPARISON")
    print(f"{'='*80}")
    print(f"  LEGACY total: {legacy_full_counter.count:>8} RNG calls")
    print(f"  NEW total:    {new_full_counter.count:>8} RNG calls")
    print(f"  DIFFERENCE:   {new_full_counter.count - legacy_full_counter.count:>8} RNG calls")
    
    print(f"\n  Final state:")
    print(f"    LEGACY: Q0={np.sum(legacy_full.W[:legacy_full.a_tot]):.6e}, particles={legacy_full.a_tot}")
    print(f"    NEW:    Q0={np.sum(new_full.W[:new_full.a_tot]):.6e}, particles={new_full.a_tot}")
    print(f"    Diff:   {abs(new_full.a_tot - legacy_full.a_tot)} particles")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    
    init_diff = count_after_samplers_new - count_after_samplers
    solve_diff = new_full_counter.count - legacy_full_counter.count
    total_diff = init_diff + solve_diff
    
    print(f"  Init phase difference:     {init_diff:>+8} RNG calls")
    print(f"  Solve phase difference:    {solve_diff:>+8} RNG calls")
    print(f"  TOTAL difference:          {total_diff:>+8} RNG calls")
    
    if total_diff != 0:
        print(f"\n  🔴 CONFIRMED: RNG sequences differ!")
        print(f"      This explains the simulation divergence.")
        print(f"      Even with seed=42, different # of RNG calls → different sequence.")
    else:
        print(f"\n  ✓ RNG sequences match - problem is elsewhere!")


if __name__ == "__main__":
    compare_rng_consumption()
