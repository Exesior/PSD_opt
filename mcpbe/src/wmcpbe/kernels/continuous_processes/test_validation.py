"""
Validation Test for Continuous Processes Kernels.

Run this file to validate both kernels against analytical solutions:
    python test_validation.py
"""

import numpy as np
import sys

# Add parent directory to path for imports
sys.path.insert(0, '..')

from porosity_compression import PorosityCompressionKernel
from liquid_internalization import LiquidInternalizationKernel


def test_porosity_compression():
    """
    Test porosity compression kernel against analytical solution.
    
    Analytical solution: ε(t) = ε_min + (ε_0 - ε_min) × exp(-k×t)
    """
    print("=" * 80)
    print("POROSITY COMPRESSION KERNEL - ANALYTICAL VALIDATION")
    print("=" * 80)
    
    # Kernel parameters
    rate = 0.02  # 1/s
    min_porosity = 0.3
    kernel = PorosityCompressionKernel(rate=rate, min_porosity=min_porosity)
    
    print(f"\nKernel parameters:")
    print(f"  rate = {rate} 1/s")
    print(f"  min_porosity = {min_porosity}")
    
    # Initial conditions
    eps_0 = 0.5
    t_max = 100.0  # s
    n_steps = 20
    
    print(f"\nInitial porosity: ε_0 = {eps_0}")
    print(f"Simulation time: t = 0 to {t_max} s ({n_steps} steps)")
    
    # Run simulation
    eps_sim = eps_0
    dt = t_max / n_steps
    
    results = []
    for i in range(n_steps + 1):
        t = i * dt
        
        # Analytical solution
        eps_analytical = min_porosity + (eps_0 - min_porosity) * np.exp(-rate * t)
        
        # Store results
        results.append({
            't': t,
            'simulated': eps_sim,
            'analytical': eps_analytical,
            'error': abs(eps_sim - eps_analytical)
        })
        
        # Step forward (if not last iteration)
        if i < n_steps:
            eps_sim = kernel.compute(eps_sim, dt)
    
    # Print table
    print(f"\n{'Time [s]':<12} {'Simulated':<14} {'Analytical':<14} {'Error':<12}")
    print("-" * 52)
    
    for r in results:
        print(f"{r['t']:<12.2f} {r['simulated']:<14.6f} {r['analytical']:<14.6f} {r['error']:<12.2e}")
    
    # Summary statistics
    max_error = max(r['error'] for r in results)
    mean_error = sum(r['error'] for r in results) / len(results)
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Maximum absolute error: {max_error:.2e}")
    print(f"Mean absolute error:    {mean_error:.2e}")
    
    if max_error < 1e-10:
        print("\n✓ VALIDATION PASSED: Simulation matches analytical solution")
        return True
    else:
        print("\n✗ VALIDATION FAILED: Error exceeds tolerance")
        return False


def test_liquid_internalization():
    """
    Test liquid internalization kernel.
    
    The ODE: dl_intern/dt = k × (l_total - l_intern) × (v_pore - l_intern)
    
    This is a Riccati-type equation. For validation, we compare:
    1. Single-step computation vs. multi-step integration (convergence test)
    2. Equilibrium states (S → l_total/v_pore when l_total < v_pore)
    """
    print("\n" + "=" * 80)
    print("LIQUID INTERNALIZATION KERNEL - VALIDATION")
    print("=" * 80)
    
    # Kernel parameters
    k_int = 1e12  # 1/(m³·s)
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
    
    # Expected equilibrium: all liquid internalized until saturation
    # Since l_total < v_pore, final saturation should be l_total/v_pore
    S_equilibrium = min(1.0, l_total / v_pore)
    print(f"\nExpected equilibrium saturation: S_eq = {S_equilibrium:.4f}")
    
    # Simulation parameters
    t_max = 1.0  # s
    n_steps_coarse = 10
    n_steps_fine = 1000
    
    print(f"\nSimulation time: t = 0 to {t_max} s")
    
    # Run coarse simulation
    print(f"\n{'='*60}")
    print(f"TIME EVOLUTION (coarse grid, {n_steps_coarse} steps)")
    print(f"{'='*60}")
    
    S_sim = S_0
    dt_coarse = t_max / n_steps_coarse
    
    results_coarse = []
    for i in range(n_steps_coarse + 1):
        t = i * dt_coarse
        results_coarse.append({'t': t, 'S': S_sim})
        
        if i < n_steps_coarse:
            S_sim = kernel.compute(S_sim, v_pore, l_total, dt_coarse)
    
    # Run fine simulation as reference
    S_ref = S_0
    dt_fine = t_max / n_steps_fine
    times_fine = [0.0]
    S_values_fine = [S_0]
    
    for i in range(n_steps_fine):
        S_ref = kernel.compute(S_ref, v_pore, l_total, dt_fine)
        times_fine.append((i+1) * dt_fine)
        S_values_fine.append(S_ref)
    
    # Interpolate reference to coarse times for comparison
    S_reference = np.interp([r['t'] for r in results_coarse], times_fine, S_values_fine)
    
    # Print table
    print(f"\n{'Time [s]':<12} {'Simulated':<14} {'Reference':<14} {'Error':<12}")
    print("-" * 52)
    
    errors = []
    for i, r in enumerate(results_coarse):
        error = abs(r['S'] - S_reference[i])
        errors.append(error)
        print(f"{r['t']:<12.3f} {r['S']:<14.6f} {S_reference[i]:<14.6f} {error:<12.2e}")
    
    # Convergence test
    print(f"\n{'='*60}")
    print("CONVERGENCE TEST")
    print(f"{'='*60}")
    
    # Test with different step sizes
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]
    S_final_values = []
    
    for dt_test in step_sizes:
        S_test = S_0
        n_test = int(t_max / dt_test)
        for _ in range(n_test):
            S_test = kernel.compute(S_test, v_pore, l_total, dt_test)
        S_final_values.append(S_test)
    
    # Use finest resolution as reference
    S_final_ref = S_final_values[-1]
    
    print(f"\n{'Step size [s]':<16} {'S(t='+str(t_max)+')':<14} {'Error':<12}")
    print("-" * 42)
    
    for i, dt_test in enumerate(step_sizes):
        error = abs(S_final_values[i] - S_final_ref)
        print(f"{dt_test:<16.4f} {S_final_values[i]:<14.6f} {error:<12.2e}")
    
    # Equilibrium test
    print(f"\n{'='*60}")
    print("EQUILIBRIUM TEST")
    print(f"{'='*60}")
    
    # Run until equilibrium (long time)
    S_eq_test = S_0
    t_eq = 10.0  # Long enough to reach equilibrium
    dt_eq = 0.1
    n_eq = int(t_eq / dt_eq)
    
    for _ in range(n_eq):
        S_eq_test = kernel.compute(S_eq_test, v_pore, l_total, dt_eq)
    
    print(f"\nSaturation after {t_eq} s: S = {S_eq_test:.6f}")
    print(f"Expected equilibrium:       S = {S_equilibrium:.6f}")
    print(f"Difference:                 ΔS = {abs(S_eq_test - S_equilibrium):.2e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    max_error = max(errors)
    print(f"Maximum deviation from reference: {max_error:.2e}")
    print(f"Equilibrium error:                {abs(S_eq_test - S_equilibrium):.2e}")
    
    if max_error < 1e-3 and abs(S_eq_test - S_equilibrium) < 1e-6:
        print("\n✓ VALIDATION PASSED: Kernel behaves correctly")
        return True
    else:
        print("\n✗ VALIDATION FAILED: Errors exceed tolerance")
        return False


if __name__ == '__main__':
    """Run all validation tests."""
    print("\n" + "=" * 80)
    print("CONTINUOUS PROCESSES KERNELS - COMPLETE VALIDATION SUITE")
    print("=" * 80)
    
    # Run tests
    test1_passed = test_porosity_compression()
    test2_passed = test_liquid_internalization()
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Porosity Compression:     {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Liquid Internalization:   {'PASSED' if test2_passed else 'FAILED'}")
    print("=" * 80)
    
    if test1_passed and test2_passed:
        print("\n✓ ALL VALIDATION TESTS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME VALIDATION TESTS FAILED")
        sys.exit(1)
