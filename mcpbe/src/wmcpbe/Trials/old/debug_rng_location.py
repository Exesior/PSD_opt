"""
RNG Location Tracker.

Tracks EXACTLY where random number generator calls occur during breakage events.
This will identify which methods consume RNG calls and help find the source
of the 2000 RNG call difference between Legacy and NEW solvers.
"""

import sys
import os
import numpy as np
from functools import wraps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class MethodRNGCounter:
    """Wrapper that counts RNG calls within a specific method."""
    
    def __init__(self):
        self.count = 0
        self.original_rng = None
        self.wrapped_rng = None
    
    def create_wrapped_rng(self, original_rng):
        """Create a wrapped RNG that counts calls."""
        self.original_rng = original_rng
        
        class CountingRNG:
            def __init__(self, counter, rng):
                self._counter = counter
                self._rng = rng
            
            def random(self, size=None):
                self._counter.count += 1
                return self._rng.random(size)
            
            def standard_normal(self, size=None):
                self._counter.count += 1
                return self._rng.standard_normal(size)
            
            def integers(self, low, high=None, size=None):
                self._counter.count += 1
                return self._rng.integers(low, high, size)
            
            def choice(self, a, size=None, replace=True, p=None):
                self._counter.count += 1
                return self._rng.choice(a, size, replace, p)
            
            def shuffle(self, x):
                self._counter.count += 1
                return self._rng.shuffle(x)
            
            def permutation(self, x):
                self._counter.count += 1
                return self._rng.permutation(x)
            
            @property
            def rng(self):
                return self._rng
        
        self.wrapped_rng = CountingRNG(self, original_rng)
        return self.wrapped_rng
    
    def reset(self):
        """Reset counter to zero."""
        self.count = 0


def trace_method(counter, method_name):
    """Decorator to trace RNG calls within a method."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get solver instance (first argument)
            if len(args) > 0 and hasattr(args[0], '_rng'):
                solver = args[0]
                old_rng = solver._rng
                solver._rng = counter.create_wrapped_rng(old_rng)
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    solver._rng = old_rng
            else:
                # Can't wrap, just call normally
                return func(*args, **kwargs)
        return wrapper
    return decorator


def compare_rng_locations():
    """Compare WHERE RNG calls occur in Legacy vs NEW solvers."""
    
    print("=" * 80)
    print(" RNG LOCATION TRACKER")
    print("=" * 80)
    
    seed = 42
    pl_p1 = 3e-4
    g = 1000
    breakrval = 1
    n_particles = 100
    
    print(f"\nConfiguration:")
    print(f"  seed = {seed}")
    print(f"  pl_P1 = {pl_p1}")
    print(f"  G = {g}")
    print(f"  BREAKRVAL = {breakrval}")
    print(f"  n_particles = {n_particles}")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    # =========================================================================
    # PHASE 1: Check LMC Configuration
    # =========================================================================
    print("\n" + "=" * 80)
    print(" PHASE 1: LMC Configuration Check")
    print("=" * 80)
    
    # LEGACY solver
    print("\n[LEGACY] Creating solver...")
    legacy_solver = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=seed,
    )
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
    legacy_solver._initialize_particles()
    legacy_solver._init_lmc()
    legacy_solver._initialize_kernels()
    legacy_solver._initialize_samplers()
    
    print(f"\n[LEGACY] LMC Configuration:")
    print(f"  use_lmc_pre_model = {getattr(legacy_solver, 'use_lmc_pre_model', 'NOT SET')}")
    print(f"  use_lmc_live = {getattr(legacy_solver, 'use_lmc_live', 'NOT SET')}")
    print(f"  lmc_adapter = {type(getattr(legacy_solver, 'lmc_adapter', None)).__name__}")
    print(f"  frag_num = {getattr(legacy_solver, 'frag_num', 'NOT SET')}")
    print(f"  lmc_NO_FRAG = {getattr(legacy_solver, 'lmc_NO_FRAG', 'NOT SET')}")
    
    # Check what _get_break_tables_for_state returns
    Vrem_test = np.array([legacy_solver.V_flat[0, 0]], dtype=float)
    tables_result = legacy_solver._get_break_tables_for_state(Vrem_test)
    mode, rA, rB, rowsum_cdf, row_cdf, zmin1d, zmin3d, pexp = tables_result
    print(f"\n[LEGACY] _get_break_tables_for_state() returns:")
    print(f"  mode = '{mode}'")
    print(f"  rA.size = {rA.size if rA is not None else 'N/A'}")
    print(f"  pexp = {pexp}")
    
    # NEW solver
    print("\n[NEW] Creating solver...")
    new_solver = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=seed,
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
    new_solver.a0 = n_particles
    new_solver.alpha_prim = 1.0
    new_solver.process_type = 'breakage'
    new_solver._initialize_particles()
    new_solver._init_lmc()
    # _initialize_kernels() already done in __init__
    new_solver._initialize_samplers()
    
    print(f"\n[NEW] LMC Configuration:")
    print(f"  use_lmc_pre_model = {getattr(new_solver, 'use_lmc_pre_model', 'NOT SET')}")
    print(f"  use_lmc_live = {getattr(new_solver, 'use_lmc_live', 'NOT SET')}")
    print(f"  lmc_adapter = {type(getattr(new_solver, 'lmc_adapter', None)).__name__}")
    print(f"  frag_num = {getattr(new_solver, 'frag_num', 'NOT SET')}")
    print(f"  lmc_NO_FRAG = {getattr(new_solver, 'lmc_NO_FRAG', 'NOT SET')}")
    
    # Check what _get_break_tables_for_state returns
    Vrem_test_new = np.array([new_solver.V_flat[0, 0]], dtype=float)
    tables_result_new = new_solver._get_break_tables_for_state(Vrem_test_new)
    mode_new, rA_new, rB_new, rowsum_cdf_new, row_cdf_new, zmin1d_new, zmin3d_new, pexp_new = tables_result_new
    print(f"\n[NEW] _get_break_tables_for_state() returns:")
    print(f"  mode = '{mode_new}'")
    print(f"  rA.size = {rA_new.size if rA_new is not None else 'N/A'}")
    print(f"  pexp = {pexp_new}")
    
    # =========================================================================
    # PHASE 2: Trace Individual Methods
    # =========================================================================
    print("\n" + "=" * 80)
    print(" PHASE 2: Method-Level RNG Tracking")
    print("=" * 80)
    
    # Create counters for each method we want to trace
    counters = {
        '_break_sampler.sample': MethodRNGCounter(),
        '_build_fragments_stepwise': MethodRNGCounter(),
        '_produce_one_frag_from_remaining': MethodRNGCounter(),
        '_break_build_fragments': MethodRNGCounter(),
        '_do_one_break': MethodRNGCounter(),
    }
    
    # Wrap methods in LEGACY solver
    print("\nWrapping LEGACY solver methods...")
    
    # Save original methods
    legacy_originals = {}
    
    for method_name, counter in counters.items():
        # Navigate to the method
        obj = legacy_solver
        parts = method_name.split('.')
        
        if len(parts) == 1:
            # Simple method like '_do_one_break'
            if hasattr(obj, parts[0]):
                legacy_originals[method_name] = getattr(obj, parts[0])
                
                # Create wrapped version
                original_method = legacy_originals[method_name]
                
                def make_wrapper(orig, ctr):
                    @wraps(orig)
                    def wrapper(*args, **kwargs):
                        old_rng = obj._rng
                        obj._rng = counter.create_wrapped_rng(old_rng)
                        try:
                            return orig(*args, **kwargs)
                        finally:
                            obj._rng = old_rng
                    return wrapper
                
                wrapped = make_wrapper(original_method, counter)
                setattr(obj, parts[0], wrapped)
                print(f"  ✓ Wrapped {method_name}")
        else:
            # Attribute access like '_break_sampler.sample'
            attr_name = parts[0]
            method_name_only = parts[1]
            
            if hasattr(obj, attr_name):
                sub_obj = getattr(obj, attr_name)
                if sub_obj is not None and hasattr(sub_obj, method_name_only):
                    legacy_originals[method_name] = getattr(sub_obj, method_name_only)
                    
                    original_method = legacy_originals[method_name]
                    
                    def make_wrapper_sub(orig, ctr, sub):
                        @wraps(orig)
                        def wrapper(*args, **kwargs):
                            old_rng = obj._rng
                            obj._rng = counter.create_wrapped_rng(old_rng)
                            try:
                                return orig(*args, **kwargs)
                            finally:
                                obj._rng = old_rng
                        return wrapper
                    
                    wrapped = make_wrapper_sub(original_method, counter, sub_obj)
                    setattr(sub_obj, method_name_only, wrapped)
                    print(f"  ✓ Wrapped {method_name}")
    
    # Wrap methods in NEW solver (same code)
    print("\nWrapping NEW solver methods...")
    
    new_originals = {}
    
    for method_name, counter in counters.items():
        obj = new_solver
        parts = method_name.split('.')
        
        if len(parts) == 1:
            if hasattr(obj, parts[0]):
                new_originals[method_name] = getattr(obj, parts[0])
                original_method = new_originals[method_name]
                
                def make_wrapper(orig, ctr):
                    @wraps(orig)
                    def wrapper(*args, **kwargs):
                        old_rng = obj._rng
                        obj._rng = counter.create_wrapped_rng(old_rng)
                        try:
                            return orig(*args, **kwargs)
                        finally:
                            obj._rng = old_rng
                    return wrapper
                
                wrapped = make_wrapper(original_method, counter)
                setattr(obj, parts[0], wrapped)
        else:
            attr_name = parts[0]
            method_name_only = parts[1]
            
            if hasattr(obj, attr_name):
                sub_obj = getattr(obj, attr_name)
                if sub_obj is not None and hasattr(sub_obj, method_name_only):
                    new_originals[method_name] = getattr(sub_obj, method_name_only)
                    original_method = new_originals[method_name]
                    
                    def make_wrapper_sub(orig, ctr, sub):
                        @wraps(orig)
                        def wrapper(*args, **kwargs):
                            old_rng = obj._rng
                            obj._rng = counter.create_wrapped_rng(old_rng)
                            try:
                                return orig(*args, **kwargs)
                            finally:
                                obj._rng = old_rng
                        return wrapper
                    
                    wrapped = make_wrapper_sub(original_method, counter, sub_obj)
                    setattr(sub_obj, method_name_only, wrapped)
    
    # =========================================================================
    # PHASE 3: Run Single Breakage Event
    # =========================================================================
    print("\n" + "=" * 80)
    print(" PHASE 3: Single Breakage Event Analysis")
    print("=" * 80)
    
    # Reset all counters
    for counter in counters.values():
        counter.reset()
    
    print("\nRunning ONE breakage event in LEGACY...")
    legacy_solver._do_one_break()
    
    print("Running ONE breakage event in NEW...")
    new_solver._do_one_break()
    
    # Print results
    print(f"\n{'='*80}")
    print(f" RNG CALLS BY METHOD (per breakage event)")
    print(f"{'='*80}")
    print(f"  {'Method':<40} | {'LEGACY':>10} | {'NEW':>10} | {'Diff':>10}")
    print(f"  {'-'*40}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    for method_name in counters.keys():
        counter = counters[method_name]
        # Note: We can't easily separate LEGACY vs NEW counters with this approach
        # So we'll run them separately below
    
    # Actually, let's redo this more cleanly - run each solver separately
    print(f"\n  [Note: Running solvers separately for accurate counts...]")
    
    # =========================================================================
    # PHASE 4: Clean Separate Runs
    # =========================================================================
    print("\n" + "=" * 80)
    print(" PHASE 4: Clean Separate Runs (Most Accurate)")
    print("=" * 80)
    
    # Create fresh solvers
    print("\nCreating fresh LEGACY solver...")
    legacy_fresh = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=seed,
    )
    legacy_fresh.a0 = n_particles
    legacy_fresh.COLEVAL = 1
    legacy_fresh.BREAKRVAL = breakrval
    legacy_fresh.CORR_BETA = 1e-2
    legacy_fresh.G = g
    legacy_fresh.pl_P1 = pl_p1
    legacy_fresh.pl_P2 = 1.0
    legacy_fresh.pl_v = 2.0
    legacy_fresh.pl_q = 0.5
    legacy_fresh.alpha_prim = 1.0
    legacy_fresh.process_type = 'breakage'
    legacy_fresh._initialize_particles()
    legacy_fresh._init_lmc()
    legacy_fresh._initialize_kernels()
    legacy_fresh._initialize_samplers()
    
    # Wrap _build_fragments_stepwise specifically
    legacy_frag_counter = MethodRNGCounter()
    legacy_orig_build_frags = legacy_fresh._build_fragments_stepwise
    
    def legacy_wrapped_build_frags(*args, **kwargs):
        old_rng = legacy_fresh._rng
        legacy_fresh._rng = legacy_frag_counter.create_wrapped_rng(old_rng)
        try:
            result = legacy_orig_build_frags(*args, **kwargs)
            return result
        finally:
            legacy_fresh._rng = old_rng
    
    legacy_fresh._build_fragments_stepwise = legacy_wrapped_build_frags
    
    # Also wrap _produce_one_frag_from_remaining
    legacy_produce_counter = MethodRNGCounter()
    legacy_orig_produce = legacy_fresh._produce_one_frag_from_remaining
    
    def legacy_wrapped_produce(*args, **kwargs):
        old_rng = legacy_fresh._rng
        legacy_fresh._rng = legacy_produce_counter.create_wrapped_rng(old_rng)
        try:
            result = legacy_orig_produce(*args, **kwargs)
            return result
        finally:
            legacy_fresh._rng = old_rng
    
    legacy_fresh._produce_one_frag_from_remaining = legacy_wrapped_produce
    
    # Wrap sampler
    legacy_sampler_counter = MethodRNGCounter()
    legacy_orig_sample = legacy_fresh._break_sampler.sample
    
    def legacy_wrapped_sample(*args, **kwargs):
        old_rng = legacy_fresh._rng
        legacy_fresh._rng = legacy_sampler_counter.create_wrapped_rng(old_rng)
        try:
            result = legacy_orig_sample(*args, **kwargs)
            return result
        finally:
            legacy_fresh._rng = old_rng
    
    legacy_fresh._break_sampler.sample = legacy_wrapped_sample
    
    print("Creating fresh NEW solver...")
    new_fresh = MCPBESolver(
        dim=1, t_total=0.1, t_write=10, init=False, load_attr=False, seed=seed,
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
    new_fresh.a0 = n_particles
    new_fresh.alpha_prim = 1.0
    new_fresh.process_type = 'breakage'
    new_fresh._initialize_particles()
    new_fresh._init_lmc()
    new_fresh._initialize_samplers()
    
    # Wrap same methods in NEW
    new_frag_counter = MethodRNGCounter()
    new_orig_build_frags = new_fresh._build_fragments_stepwise
    
    def new_wrapped_build_frags(*args, **kwargs):
        old_rng = new_fresh._rng
        new_fresh._rng = new_frag_counter.create_wrapped_rng(old_rng)
        try:
            result = new_orig_build_frags(*args, **kwargs)
            return result
        finally:
            new_fresh._rng = old_rng
    
    new_fresh._build_fragments_stepwise = new_wrapped_build_frags
    
    new_produce_counter = MethodRNGCounter()
    new_orig_produce = new_fresh._produce_one_frag_from_remaining
    
    def new_wrapped_produce(*args, **kwargs):
        old_rng = new_fresh._rng
        new_fresh._rng = new_produce_counter.create_wrapped_rng(old_rng)
        try:
            result = new_orig_produce(*args, **kwargs)
            return result
        finally:
            new_fresh._rng = old_rng
    
    new_fresh._produce_one_frag_from_remaining = new_wrapped_produce
    
    new_sampler_counter = MethodRNGCounter()
    new_orig_sample = new_fresh._break_sampler.sample
    
    def new_wrapped_sample(*args, **kwargs):
        old_rng = new_fresh._rng
        new_fresh._rng = new_sampler_counter.create_wrapped_rng(old_rng)
        try:
            result = new_orig_sample(*args, **kwargs)
            return result
        finally:
            new_fresh._rng = old_rng
    
    new_fresh._break_sampler.sample = new_wrapped_sample
    
    # Run one breakage event each
    print("\nRunning ONE breakage event in LEGACY (fresh)...")
    legacy_fresh._do_one_break()
    
    print("Running ONE breakage event in NEW (fresh)...")
    new_fresh._do_one_break()
    
    # Print detailed results
    print(f"\n{'='*80}")
    print(f" DETAILED RNG ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\n  LEGACY Solver:")
    print(f"    _break_sampler.sample():          {legacy_sampler_counter.count:>6} RNG calls")
    print(f"    _build_fragments_stepwise():      {legacy_frag_counter.count:>6} RNG calls")
    print(f"    _produce_one_frag_from_remaining(): {legacy_produce_counter.count:>6} RNG calls")
    print(f"    TOTAL (tracked methods):          {legacy_sampler_counter.count + legacy_frag_counter.count + legacy_produce_counter.count:>6} RNG calls")
    
    print(f"\n  NEW Solver:")
    print(f"    _break_sampler.sample():          {new_sampler_counter.count:>6} RNG calls")
    print(f"    _build_fragments_stepwise():      {new_frag_counter.count:>6} RNG calls")
    print(f"    _produce_one_frag_from_remaining(): {new_produce_counter.count:>6} RNG calls")
    print(f"    TOTAL (tracked methods):          {new_sampler_counter.count + new_frag_counter.count + new_produce_counter.count:>6} RNG calls")
    
    # Calculate fragments produced
    print(f"\n  Fragment Analysis:")
    print(f"    (Inferred from _produce_one_frag_from_remaining calls)")
    print(f"    LEGACY: ~{legacy_produce_counter.count} fragments per break")
    print(f"    NEW:    ~{new_produce_counter.count} fragments per break")
    
    # Final verdict
    print(f"\n{'='*80}")
    print(f" VERDICT")
    print(f"{'='*80}")
    
    if legacy_produce_counter.count > 100:
        print(f"\n  🔴 CONFIRMED: LEGACY uses LMC Pre-Model!")
        print(f"      {legacy_produce_counter.count} fragments per break (vs. expected 4)")
        print(f"      This explains the {legacy_produce_counter.count - new_produce_counter.count} RNG call difference!")
        print(f"\n  FIX: Set both solvers to use_lmc_pre_model=False")
    else:
        print(f"\n  ✓ Both solvers use classical fragmentation (~4 fragments)")
        print(f"      Difference must be elsewhere.")


if __name__ == "__main__":
    compare_rng_locations()
