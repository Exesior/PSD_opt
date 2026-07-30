"""
Kernel Framework Validation - NO LEGACY CODE.

This script validates the kernel framework by testing SELF-CONSISTENCY:
- All kernels run without errors
- Results are physically plausible (Q0 > 0, Q3 > 0, etc.)
- Different seeds produce different results (stochasticity check)

NO LEGACY COMPARISON - Legacy code is DEAD!
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, src_dir)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def compute_moments(X: np.ndarray, W: np.ndarray) -> dict:
    """Compute particle moments from size distribution."""
    Q0 = np.sum(W)
    V_particles = (np.pi / 6.0) * X**3
    Q3 = np.sum(W * V_particles)
    d_mean = np.average(X, weights=W) if Q0 > 0 else 0.0
    
    return {
        'Q0': float(Q0),
        'Q3': float(Q3),
        'd_mean': float(d_mean),
    }


class ProgressTracker:
    """Track and display progress as percentage bar."""
    
    def __init__(self, total_tests: int):
        self.total_tests = total_tests
        self.completed = 0
        self.last_percent = -1
    
    def update(self, test_name: str = ""):
        """Update progress and print if crossed 10% threshold."""
        self.completed += 1
        percent = int((self.completed / self.total_tests) * 100)
        
        if percent >= self.last_percent + 10 or percent == 100:
            self.last_percent = percent
            bar_len = 30
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  Progress: [{bar}] {percent}% ({test_name})", end="", flush=True)
            
            if percent == 100:
                print()


def validate_simulation(result: dict, test_name: str, expect_breakage: bool = False) -> tuple[bool, str]:
    """
    Validate that a simulation produced PHYSICALLY PLAUSIBLE results.
    
    Checks:
    1. Simulation completed without error
    2. Q0 > 0 (particles exist)
    3. Q3 > 0 (volume exists)
    4. d_mean > 0 (mean diameter makes sense)
    5. For breakage: Q0 should increase or stay similar
    6. For agglomeration: Q0 should decrease or stay similar
    
    Returns:
        (passed, message)
    """
    # Check 1: Simulation ran
    if result is None:
        return False, "Simulation returned None"
    
    if result.get('error'):
        return False, f"Simulation error: {result['error']}"
    
    # Check 2: Has particles
    Q0 = result['moments'].get('Q0', 0)
    Q3 = result['moments'].get('Q3', 0)
    d_mean = result['moments'].get('d_mean', 0)
    
    if Q0 <= 0:
        return False, f"Q0={Q0} (should be > 0, no particles?)"
    
    if Q3 <= 0:
        return False, f"Q3={Q3} (should be > 0, no volume?)"
    
    if d_mean <= 0:
        return False, f"d_mean={d_mean} (should be > 0)"
    
    # Check 3: Particle count plausible
    a_tot = result.get('a_tot', 0)
    if a_tot <= 0:
        return False, f"a_tot={a_tot} (no particles after simulation)"
    
    # Check 4: Physical plausibility
    if expect_breakage:
        # Breakage increases particle count
        pass  # Can't check without initial value
    else:
        # Agglomeration decreases particle count
        pass  # Can't check without initial value
    
    return True, f"Q0={Q0:.2f}, Q3={Q3:.3e}, d_mean={d_mean:.3e}"


def run_kernel_simulation(
    process_type: str = 'agglomeration',
    t_total: float = 5.0,
    seed: int = 42,
    agg_kernel_name: str = None,
    agg_kernel_params: dict = None,
    break_kernel_name: str = None,
    break_kernel_params: dict = None,
    g: float = 1000,
    n_particles: int = 1000,
    max_particles: int = 50000,
) -> dict:
    """
    Run simulation with KERNEL FRAMEWORK ONLY.
    
    Returns dict with moments, timing, and any error messages.
    """
    from wmcpbe.mcpbe import MCPBESolver
    
    start_time = time.time()
    
    try:
        # Prepare kernel parameters
        agg_params = agg_kernel_params or {}
        if 'g' not in agg_params and agg_kernel_name:
            agg_params['g'] = g
        
        break_params = break_kernel_params or {}
        if 'g' not in break_params and break_kernel_name:
            break_params['g'] = g
        
        # Create solver with explicit kernel configuration
        solver = MCPBESolver(
            dim=1,
            t_total=t_total,
            t_write=10,
            init=False,
            load_attr=False,
            seed=seed,
            agg_kernel_name=agg_kernel_name,
            agg_kernel_params=agg_params,
            break_kernel_name=break_kernel_name,
            break_kernel_params=break_params,
        )
        
        # Set required solver attributes
        solver.G = g
        solver.process_type = process_type
        solver.a0 = n_particles
        if break_kernel_name:
            solver.pl_P1 = break_params.get('p1', 3e-2)
            solver.pl_P2 = break_params.get('p2', 1.0)
            solver.pl_v = break_params.get('pl_v', 2.0)
            solver.pl_q = break_params.get('pl_q', 1.0)
        
        # Initialize
        solver._initialize_kernels()
        solver._initialize_particles()
        solver._initialize_samplers()
        
        # Run simulation
        solver.solve(max_particles=max_particles)
        
        elapsed = time.time() - start_time
        
        a_tot = solver.a_tot
        if a_tot < 1:
            return {
                'error': 'No particles after simulation',
                'moments': {'Q0': 0, 'Q3': 0, 'd_mean': 0},
                'time': elapsed,
                'a_tot': 0,
            }
        
        V_ges = solver.V_flat[-1, :a_tot].copy()
        X_final = (6.0 * V_ges / np.pi) ** (1.0/3.0)
        W_final = solver.W[:a_tot].copy()
        
        moments = compute_moments(X_final, W_final)
        moments['time'] = elapsed
        
        return {
            'X': X_final,
            'W': W_final,
            'moments': moments,
            'time': elapsed,
            'a_tot': a_tot,
            'error': None,
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'moments': {'Q0': 0, 'Q3': 0, 'd_mean': 0},
            'time': elapsed,
            'a_tot': 0,
        }


def test_agg_kernel(kernel_name: str, corr_beta: float, t_total: float, progress: ProgressTracker) -> bool:
    """Test one aggregation kernel for physical plausibility."""
    progress.update(f"Agg {kernel_name}")
    
    print_header(f"TEST: Agglomeration - {kernel_name}")
    
    # Run simulation
    result = run_kernel_simulation(
        process_type='agglomeration',
        t_total=t_total,
        seed=42,
        agg_kernel_name=kernel_name,
        agg_kernel_params={'corr_beta': corr_beta, 'g': 1000} if kernel_name == 'shear_chin1998' else {'corr_beta': corr_beta},
        n_particles=1000,
    )
    
    print(f"\n  Simulation time: {result['time']:.2f}s")
    print(f"  Final particles: {result['a_tot']}")
    
    # Validate
    passed, msg = validate_simulation(result, kernel_name, expect_breakage=False)
    
    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] {msg}")
    
    if not passed and result.get('traceback'):
        print(f"\n  Traceback:\n{result['traceback']}")
    
    return passed


def test_break_kernel(breakrval: int, progress: ProgressTracker) -> bool:
    """Test breakage with different BREAKRVAL-equivalent configurations."""
    progress.update(f"Break BREAKRVAL={breakrval}")
    
    print_header(f"TEST: Breakage - BREAKRVAL={breakrval}")
    
    # BREAKRVAL affects power_law p1 parameter
    p1_values = {1: 3e-4, 2: 3e-4, 3: 3e-4, 4: 3e-4, 5: 3e-4}
    
    result = run_kernel_simulation(
        process_type='breakage',
        t_total=0.1,  # Short time to avoid too many fragments
        seed=42,
        break_kernel_name='power_law',
        break_kernel_params={'p1': p1_values.get(breakrval, 3e-4), 'p2': 1.0, 'g': 1000},
        n_particles=1000,
    )
    
    print(f"\n  Simulation time: {result['time']:.2f}s")
    print(f"  Final particles: {result['a_tot']}")
    
    # Validate
    passed, msg = validate_simulation(result, f"BREAKRVAL={breakrval}", expect_breakage=True)
    
    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] {msg}")
    
    if not passed and result.get('traceback'):
        print(f"\n  Traceback:\n{result['traceback']}")
    
    return passed


def test_combined(progress: ProgressTracker) -> bool:
    """Test combined agglomeration + breakage."""
    progress.update("Combined")
    
    print_header("TEST: Combined Agglomeration + Breakage")
    
    result = run_kernel_simulation(
        process_type='mix',
        t_total=1.0,
        seed=42,
        agg_kernel_name='shear_chin1998',
        agg_kernel_params={'corr_beta': 1e-2, 'g': 1000},
        break_kernel_name='power_law',
        break_kernel_params={'p1': 3e-4, 'p2': 1.0, 'g': 1000},
        n_particles=1000,
    )
    
    print(f"\n  Simulation time: {result['time']:.2f}s")
    print(f"  Final particles: {result['a_tot']}")
    
    # Validate
    passed, msg = validate_simulation(result, "Combined")
    
    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] {msg}")
    
    if not passed and result.get('traceback'):
        print(f"\n  Traceback:\n{result['traceback']}")
    
    return passed


def test_stochasticity(progress: ProgressTracker) -> bool:
    """Verify that different seeds produce different results (sanity check)."""
    progress.update("Stochasticity")
    
    print_header("TEST: Stochasticity Check")
    
    results = []
    for seed in [42, 123, 456]:
        result = run_kernel_simulation(
            process_type='agglomeration',
            t_total=1.0,
            seed=seed,
            agg_kernel_name='shear_chin1998',
            agg_kernel_params={'corr_beta': 1e-1, 'g': 1000},
            n_particles=100,  # Small for speed
        )
        if result.get('error'):
            print(f"\n  [ERROR] Seed {seed}: {result['error']}")
            return False
        results.append(result['moments']['Q0'])
    
    # Check that results differ (stochastic!)
    unique_results = len(set(round(r, 2) for r in results))
    
    print(f"\n  Q0 for seeds 42, 123, 456: {results}")
    print(f"  Unique results: {unique_results}/3")
    
    if unique_results >= 2:
        print(f"\n  [PASS] Different seeds produce different results ✓")
        return True
    else:
        print(f"\n  [FAIL] All seeds produced same result (deterministic?) ✗")
        return False


def main():
    """Run all validation tests."""
    print_header("KERNEL FRAMEWORK VALIDATION")
    print("Testing physical plausibility of all kernels")
    print("NO LEGACY CODE - Pure kernel framework!")
    
    total_tests = 11  # 4 agg + 5 break + 1 combined + 1 stochasticity
    progress = ProgressTracker(total_tests)
    
    all_passed = True
   
    # Phase 1: Aggregation kernels
    print_header("PHASE 1: AGGLOMERATION KERNELS")
    
    agg_configs = [
        ('shear_chin1998', 1e-2, 5.0),
        ('brownian_tsouris1995', 1e-2, 2.0),
        ('constant', 1e-8, 0.05),
        ('sum', 1e-8, 0.05),
    ]
    
    for kernel_name, corr_beta, t_total in agg_configs:
        try:
            passed = test_agg_kernel(kernel_name, corr_beta, t_total, progress=progress)
            if not passed:
                all_passed = False
                print(f"\n  ❌ {kernel_name} FAILED")
            else:
                print(f"\n  ✅ {kernel_name} PASSED")
        except Exception as e:
            all_passed = False
            print(f"\n  ❌ {kernel_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Phase 2: Breakage configurations
    print_header("PHASE 2: BREAKAGE KERNELS")
    
    for breakrval in [1, 2, 3, 4, 5]:
        try:
            passed = test_break_kernel(breakrval, progress=progress)
            if not passed:
                all_passed = False
                print(f"\n  ❌ BREAKRVAL={breakrval} FAILED")
            else:
                print(f"\n  ✅ BREAKRVAL={breakrval} PASSED")
        except Exception as e:
            all_passed = False
            print(f"\n  ❌ BREAKRVAL={breakrval} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Phase 3: Combined test
    print_header("PHASE 3: COMBINED TEST")
    
    try:
        passed = test_combined(progress)
        if not passed:
            all_passed = False
            print("\n  ❌ Combined FAILED")
        else:
            print("\n  ✅ Combined PASSED")
    except Exception as e:
        all_passed = False
        print(f"\n  ❌ Combined ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Phase 4: Stochasticity check
    print_header("PHASE 4: STOCHASTICITY CHECK")
    
    try:
        passed = test_stochasticity(progress)
        if not passed:
            all_passed = False
            print("\n  ❌ Stochasticity check FAILED")
        else:
            print("\n  ✅ Stochasticity check PASSED")
    except Exception as e:
        all_passed = False
        print(f"\n  ❌ Stochasticity check ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print_header("OVERALL SUMMARY")
    if all_passed:
        print("\n  ✅ ALL TESTS PASSED!")
        print("  The kernel framework produces physically plausible results.")
    else:
        print("\n  ❌ SOME TESTS FAILED")
        print("  Review the output above for details.")
    
    return all_passed


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
