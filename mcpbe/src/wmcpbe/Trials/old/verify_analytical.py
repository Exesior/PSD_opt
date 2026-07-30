"""
Verify analytical solution for liquid internalization ODE.

ODE: dx/dt = k × (l - x) × (p - x)

Analytical solution:
x(t) = [p(l-x₀) - l(p-x₀)e^{k(p-l)t}] / [(l-x₀) - (p-x₀)e^{k(p-l)t}]

Parameters:
    k = 0.5
    l = 5
    p = 10
    x₀ = 0
"""

import numpy as np
import matplotlib.pyplot as plt


def analytical_solution(t, x0, k, l, p):
    """
    Analytical solution for ODE: dx/dt = k × (l - x) × (p - x)
    
    x(t) = [p(l-x₀) - l(p-x₀)e^{k(p-l)t}] / [(l-x₀) - (p-x₀)e^{k(p-l)t}]
    """
    # Exponential term
    exp_term = np.exp(k * (p - l) * t)
    
    # Numerator and denominator
    numerator = p * (l - x0) - l * (p - x0) * exp_term
    denominator = (l - x0) - (p - x0) * exp_term
    
    # Avoid division by zero
    if np.abs(denominator) < 1e-15:
        return min(l, p)  # Equilibrium
    
    x_t = numerator / denominator
    return x_t


def step(x_current, dt, k, l, p):
    """
    Exact solution over time step dt starting from x_current.
    
    Same formula as analytical_solution but with x_current as initial condition.
    """
    exp_term = np.exp(k * (p - l) * dt)
    numerator = l * (p - x_current) * exp_term - p * (l - x_current)
    denominator = (p - x_current) * exp_term - (l - x_current)
    
    if np.abs(denominator) < 1e-15:
        return min(l, p)
    
    return numerator / denominator


def euler_step(x_current, dt, k, l, p):
    """Simple Euler forward step."""
    dx_dt = k * (l - x_current) * (p - x_current)
    x_new = x_current + dx_dt * dt
    return x_new


def main():
    print("="*80)
    print("ANALYTICAL SOLUTION VERIFICATION")
    print("="*80)
    
    # Parameters
    k = 0.5
    l = 5
    p = 10
    x0 = 0
    
    print(f"\nParameters:")
    print(f"  k = {k}")
    print(f"  l = {l}  (total liquid)")
    print(f"  p = {p}  (pore volume)")
    print(f"  x₀ = {x0}")
    
    # Time points
    t_max = 5.0
    n_points = 21
    times = np.linspace(0, t_max, n_points)
    
    # Compute analytical solution
    x_analytical = [analytical_solution(t, x0, k, l, p) for t in times]
    
    # Compute step-by-step using exact solution
    x_step = [x0]
    x_current = x0
    dt = times[1] - times[0]
    
    for i in range(1, len(times)):
        x_current = step(x_current, dt, k, l, p)
        x_step.append(x_current)
    
    # Compute Euler integration
    x_euler = [x0]
    x_current = x0
    
    for i in range(1, len(times)):
        x_current = euler_step(x_current, dt, k, l, p)
        x_euler.append(x_current)
    
    # Print results
    print(f"\n{'Time':<8} {'Analytical':<14} {'Step-by-Step':<14} {'Euler':<14} {'Error(step)':<12}")
    print("-"*70)
    
    for i, t in enumerate(times):
        error = abs(x_step[i] - x_analytical[i])
        print(f"{t:<8.2f} {x_analytical[i]:<14.6f} {x_step[i]:<14.6f} {x_euler[i]:<14.6f} {error:<12.2e}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(times, x_analytical, 'b-', linewidth=2, label='Analytical')
    plt.plot(times, x_step, 'ro', markersize=4, label='Step-by-step (exact)')
    plt.plot(times, x_euler, 'g--', linewidth=1, label=f'Euler (dt={dt:.2f})')
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.title('Liquid Internalization: Analytical vs. Step-by-Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Trials/analytical_verification.png', dpi=150)
    print(f"\nPlot saved to Trials/analytical_verification.png")
    
    # Check equilibrium
    x_eq = analytical_solution(100.0, x0, k, l, p)
    print(f"\nEquilibrium value (t=100): x = {x_eq:.6f}")
    print(f"Expected equilibrium: min(l, p) = {min(l, p)}")


if __name__ == '__main__':
    main()
