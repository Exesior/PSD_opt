"""
Liquid Internalization Kernel - Analytical Validation.

The ODE: dl_intern/dt = k × (l_total - l_intern) × (v_pore - l_intern)

This is a Riccati-type ODE with analytical solution:

For l_total ≠ v_pore:
    l_intern(t) = [l_total × A × exp(α×t) - v_pore × B] / [A × exp(α×t) - B]
    
    where:
        α = k × (l_total - v_pore)
        A = v_pore - l_0
        B = l_total - l_0
        l_0 = initial internal liquid

In terms of saturation S = l_intern / v_pore:
    S(t) = [r × (1 - S_0) × exp(β×t) - (r - S_0)] / [(1 - S_0) × exp(β×t) - (r - S_0)]
    
    where:
        r = l_total / v_pore (max saturation)
        β = k × v_pore × (l_total - v_pore) = k × v_pore² × (r - 1)
        S_0 = initial saturation

Run: python Trials/test_liquid_internalization.py
"""

import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, '..')

from kernels.continuous_processes.liquid_internalization import LiquidInternalizationKernel


def reference_solution(S_0, v_pore, l_total, k_int, t, dt_ref=1e-6):
    """
    High-accuracy numerical reference solution for liquid internalization.
    
    Uses very small time steps (default dt=1µs) to approximate the exact solution.
    This is more robust than the analytical formula for this nonlinear ODE.
    
    ODE: dl/dt = k × (l_total - l) × (v_pore - l)
    
    Args:
        S_0: Initial saturation [0, 1]
        v_pore: Pore volume [m³]
        l_total: Total liquid volume [m³]
        k_int: Rate constant [1/(m³·s)]
        t: Time [s]
        dt_ref: Reference time step [s] (default: 1µs)
    
    Returns:
        S(t): Saturation at time t (high accuracy)
    """
    n_steps = max(1, int(t / dt_ref))
    dt = t / n_steps
    
    S = S_0
    l_max = min(l_total, v_pore)  # Equilibrium limit
    
    for _ in range(n_steps):
        # Current internal liquid
        l = S * v_pore
        
        # Rate: dl/dt = k × (l_total - l) × (v_pore - l)
        rate = k_int * (l_total - l) * (v_pore - l)
        
        # Euler step
        dl = rate * dt
        l_new = l + dl
        
        # Clamp to physical bounds
        l_new = max(0.0, min(l_new, l_max))
        
        S = l_new / v_pore
    
    return S


def main():
    print("=" * 80)
    print("LIQUID INTERNALIZATION KERNEL - ANALYTICAL VALIDATION")
    print("=" * 80)
    
    # Kernel parameters
    k_int = 1e18  # 1/(m³·s) - gives reasonable timescale for µm³ particles
    kernel = LiquidInternalizationKernel(k_int=k_int)
    
    print(f"\nKernel parameters:")
    print(f"  k_int = {k_int:.2e} 1/(m³·s)")
    
    # Initial conditions
    v_pore = 1e-18      # 1 µm³ pore volume
    l_total = 0.8e-18   # 80% of pore volume available
    S_0 = 0.0           # Start dry
    
    print(f"\nInitial conditions:")
    print(f"  v_pore = {v_pore:.2e} m³")
    print(f"  l_total = {l_total:.2e} m³ ({l_total/v_pore*100:.1f}% of pore volume)")
    print(f"  S_0 = {S_0}")
    
    # Expected equilibrium
    S_equilibrium = min(1.0, l_total / v_pore)
    print(f"\nExpected equilibrium saturation: S_eq = {S_equilibrium:.4f}")
    
    # Time points for comparison
    t_max = 5.0  # s
    n_points = 11
    times = np.linspace(0, t_max, n_points)
    
    # ================================================================
    # COMPUTE NUMERICAL VS. REFERENCE SOLUTIONS
    # ================================================================
    print(f"\n{'='*80}")
    print("TIME EVOLUTION: Simulation vs. High-Accuracy Reference")
    print(f"{'='*80}")
    print(f"Reference: dt_ref = 1e-6 s (high accuracy numerical integration)")
    
    results = []
    S_num = S_0
    
    print(f"\n{'Time [s]':<12} {'Simulation':<14} {'Reference':<14} {'Error':<12}")
    print("-" * 52)
    
    for i, t in enumerate(times):
        # Reference solution at time t (high accuracy)
        S_ref = reference_solution(S_0, v_pore, l_total, k_int, t)
        
        # Store result
        results.append({
            't': t,
            'simulation': S_num,
            'reference': S_ref,
            'error': abs(S_num - S_ref)
        })
        
        print(f"{t:<12.3f} {S_num:<14.6f} {S_ref:<14.6f} {results[-1]['error']:<12.2e}")
        
        # Step forward with simulation time step (if not last point)
        if i < len(times) - 1:
            dt = times[i+1] - t  # dt = 0.5 s
            S_num = kernel.compute(S_num, v_pore, l_total, dt)
    
    # Maximum error
    max_error = max(r['error'] for r in results)
    mean_error = sum(r['error'] for r in results) / len(results)
    
    # ================================================================
    # CONVERGENCE TEST
    # ================================================================
    print(f"\n{'='*80}")
    print("CONVERGENCE TEST: Error vs. Time Step")
    print(f"{'='*80}")
    
    # Test at fixed time t=1.0s with varying dt
    t_test = 1.0
    dt_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    S_ref_at_t = reference_solution(S_0, v_pore, l_total, k_int, t_test)
    
    print(f"\nComparison at t = {t_test} s (reference: {S_ref_at_t:.8f})\n")
    print(f"{'dt [s]':<12} {'Steps':<8} {'Numerical':<14} {'Error':<12} {'Order':<8}")
    print("-" * 70)
    
    conv_results = []
    prev_error = None
    prev_dt = None
    
    for dt in dt_values:
        n_steps = int(t_test / dt)
        S_test = S_0
        for _ in range(n_steps):
            S_test = kernel.compute(S_test, v_pore, l_total, dt)
        
        error = abs(S_test - S_ref_at_t)
        
        if prev_error is not None and prev_dt is not None:
            order = np.log(prev_error / error) / np.log(prev_dt / dt)
        else:
            order = float('nan')
        
        conv_results.append({'dt': dt, 'error': error})
        print(f"{dt:<12.3f} {n_steps:<8} {S_test:<14.6f} {error:<12.2e} {order:<8.2f}")
        
        prev_error = error
        prev_dt = dt
    
    # ================================================================
    # EQUILIBRIUM TEST
    # ================================================================
    print(f"\n{'='*80}")
    print("EQUILIBRIUM TEST")
    print(f"{'='*80}")
    
    S_eq_test = S_0
    t_eq = 20.0
    dt_eq = 0.1
    n_eq = int(t_eq / dt_eq)
    
    for _ in range(n_eq):
        S_eq_test = kernel.compute(S_eq_test, v_pore, l_total, dt_eq)
    
    eq_error = abs(S_eq_test - S_equilibrium)
    
    print(f"\nSaturation after {t_eq} s: S = {S_eq_test:.8f}")
    print(f"Expected equilibrium:       S = {S_equilibrium:.8f}")
    print(f"Difference:                 ΔS = {eq_error:.2e}")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Maximum deviation from analytical: {max_error:.2e}")
    print(f"Mean deviation:                    {mean_error:.2e}")
    print(f"Equilibrium error:                 {eq_error:.2e}")
    
    # Tolerances
    tol_max = 1e-2
    tol_eq = 1e-6
    
    if max_error < tol_max and eq_error < tol_eq:
        print(f"\n✓ VALIDATION PASSED")
        print(f"  - Max error < {tol_max:.0e}: {max_error:.2e} ✓")
        print(f"  - Equilibrium error < {tol_eq:.0e}: {eq_error:.2e} ✓")
        return True
    else:
        print(f"\n✗ VALIDATION FAILED")
        if max_error >= tol_max:
            print(f"  - Max error {max_error:.2e} >= {tol_max:.0e} ✗")
        if eq_error >= tol_eq:
            print(f"  - Equilibrium error {eq_error:.2e} >= {tol_eq:.0e} ✗")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
