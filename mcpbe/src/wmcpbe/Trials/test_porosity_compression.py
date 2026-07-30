"""
Porosity Compression Kernel - Analytical Validation.

Analytical solution: ε(t) = ε_min + (ε_0 - ε_min) × exp(-k×t)

Run: python Trials/test_porosity_compression.py
"""

import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, '..')

from kernels.continuous_processes.porosity_compression import PorosityCompressionKernel


def main():
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


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
