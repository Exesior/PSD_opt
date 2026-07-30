"""
Debug script to find where COLEVAL=3 hangs.
"""

import sys
import os
import time
import traceback

# Add src directory to path for imports
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, src_dir)


def print_step(step_name: str):
    """Print current step with timestamp."""
    print(f"[{time.time():.2f}s] {step_name}")


def debug_coleval3():
    """Debug COLEVAL=3 constant kernel initialization and solve."""
    from wmcpbe.mcpbe import MCPBESolver
    
    print("=" * 60)
    print("DEBUG: COLEVAL=3 (constant kernel) hang investigation")
    print("=" * 60)
    
    t_total = 0.05
    seed = 42
    corr_beta = 1e-2
    g = 1000
    
    print_step("Creating solver...")
    start = time.time()
    
    solver = MCPBESolver(
        dim=1,
        t_total=t_total,
        t_write=10,
        init=False,
        load_attr=False,
        seed=seed,
    )
    
    print_step(f"Solver created ({time.time() - start:.3f}s)")
    print_step("Setting parameters...")
    
    solver.COLEVAL = 3
    solver.BREAKRVAL = 1
    solver.CORR_BETA = corr_beta
    solver.G = g
    solver.pl_P1 = 3e-4
    solver.pl_P2 = 1.0
    solver.pl_v = 2.0
    solver.pl_q = 1.0
    solver.alpha_prim = 1.0
    solver.process_type = 'agglomeration'
    
    print_step(f"Parameters set ({time.time() - start:.3f}s)")
    print_step("Initializing kernels...")
    
    solver._initialize_kernels()
    
    print_step(f"Kernels initialized ({time.time() - start:.3f}s)")
    print(f"  Kernel name: {solver.kernel_manager.agg_kernel.name}")
    print(f"  corr_beta: {solver.kernel_manager.agg_kernel.corr_beta}")
    
    print_step("Initializing particles...")
    solver._initialize_particles()
    
    print_step(f"Particles initialized ({time.time() - start:.3f}s)")
    print(f"  a_tot: {solver.a_tot}")
    print(f"  X[:5]: {solver.X[:5]}")
    
    print_step("Initializing samplers (THIS IS WHERE IT MIGHT HANG)...")
    
    # This is the critical step - _rebuild_all_propensities is called here
    try:
        solver._initialize_samplers()
        print_step(f"Samplers initialized ({time.time() - start:.3f}s)")
        print(f"  _r_agg[:5]: {solver._r_agg[:5]}")
    except Exception as e:
        print_step(f"ERROR in _initialize_samplers: {e}")
        traceback.print_exc()
        return
    
    print_step("Starting solve...")
    
    try:
        solver.solve(max_particles=50000)
        print_step(f"Solve completed ({time.time() - start:.3f}s)")
        print(f"  Final a_tot: {solver.a_tot}")
        print(f"  Events: {solver._iter_count}")
    except Exception as e:
        print_step(f"ERROR in solve: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    debug_coleval3()
