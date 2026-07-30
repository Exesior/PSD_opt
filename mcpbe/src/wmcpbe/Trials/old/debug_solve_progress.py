"""
Debug script to track solve() progress for COLEVAL=3.
"""

import sys
import os
import time
import threading

# Add src directory to path for imports
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, src_dir)


def debug_solve_with_progress():
    """Run solve() with progress tracking."""
    from wmcpbe.mcpbe import MCPBESolver
    
    print("=" * 60)
    print("DEBUG: Tracking solve() progress for COLEVAL=3")
    print("=" * 60)
    
    t_total = 0.05
    seed = 42
    corr_beta = 1e-2
    
    solver = MCPBESolver(
        dim=1,
        t_total=t_total,
        t_write=10,
        init=False,
        load_attr=False,
        seed=seed,
    )
    
    solver.COLEVAL = 3
    solver.BREAKRVAL = 1
    solver.CORR_BETA = corr_beta
    solver.G = 1000
    solver.pl_P1 = 3e-4
    solver.pl_P2 = 1.0
    solver.pl_v = 2.0
    solver.pl_q = 1.0
    solver.alpha_prim = 1.0
    solver.process_type = 'agglomeration'
    
    solver._initialize_kernels()
    solver._initialize_particles()
    solver._initialize_samplers()
    
    print(f"Initial state:")
    print(f"  a_tot: {solver.a_tot}")
    print(f"  t_total: {t_total}")
    print(f"  corr_beta: {corr_beta}")
    
    # Monkey-patch _rebuild_all_propensities to track calls
    original_rebuild = solver._rebuild_all_propensities
    rebuild_count = [0]
    last_print = [time.time()]
    
    def tracked_rebuild():
        rebuild_count[0] += 1
        now = time.time()
        if now - last_print[0] >= 1.0:  # Print every second
            elapsed = now - start_time
            print(f"  [{elapsed:.1f}s] Event #{rebuild_count[0]:,}, a_tot={solver.a_tot}, t={getattr(solver, '_elapsed', 0):.4f}")
            last_print[0] = now
        return original_rebuild()
    
    solver._rebuild_all_propensities = tracked_rebuild
    
    print(f"\nStarting solve()...")
    start_time = time.time()
    
    try:
        solver.solve(max_particles=50000)
        elapsed = time.time() - start_time
        print(f"\n✓ Solve completed in {elapsed:.1f}s")
        print(f"  Total events: {solver._iter_count:,}")
        print(f"  Final a_tot: {solver.a_tot}")
        print(f"  Rebuild calls: {rebuild_count[0]:,}")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Solve failed after {elapsed:.1f}s: {e}")
        print(f"  Events so far: {getattr(solver, '_iter_count', 0):,}")


if __name__ == "__main__":
    debug_solve_with_progress()
