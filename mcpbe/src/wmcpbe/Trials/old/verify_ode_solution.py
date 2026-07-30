"""
Verification of ODE solution for liquid internalization.

ODE: dl/dt = k × (l_total - l) × (v_pore - l)

This script derives and verifies the analytical solution step by step.
"""

import numpy as np


def derive_solution():
    """
    Derivation of analytical solution.
    
    ODE: dl/dt = k × (l_total - l) × (v_pore - l)
    
    Step 1: Separation of variables
    ∫ dl / [(l_total - l)(v_pore - l)] = ∫ k dt
    
    Step 2: Partial fraction decomposition
    1/[(l_total-l)(v_pore-l)] = A/(l_total-l) + B/(v_pore-l)
    
    Solving: A = 1/(l_total-v_pore), B = -1/(l_total-v_pore)
    
    Step 3: Integration
    [1/(l_total-v_pore)] × ln|(l_total-l)/(v_pore-l)| = k×t + C
    
    Step 4: Solve for ratio R = (l_total-l)/(v_pore-l)
    ln(R) = k×(l_total-v_pore)×t + C'
    R(t) = R₀ × exp(k×(l_total-v_pore)×t)
    
    where R₀ = (l_total-l₀)/(v_pore-l₀)
    
    Step 5: Solve for l(t)
    R = (l_total-l)/(v_pore-l)
    R×(v_pore-l) = l_total-l
    R×v_pore - R×l = l_total - l
    R×v_pore - l_total = R×l - l = l×(R-1)
    l = (R×v_pore - l_total)/(R-1)
    """
    print("="*80)
    print("DERIVATION OF ANALYTICAL SOLUTION")
    print("="*80)
    print("""
    ODE: dl/dt = k × (l_total - l) × (v_pore - l)
    
    Solution:
    R(t) = R₀ × exp(k × (l_total - v_pore) × t)
    l(t) = (R(t) × v_pore - l_total) / (R(t) - 1)
    
    where R₀ = (l_total - l₀) / (v_pore - l₀)
    
    NOTE: The key is (l_total - v_pore), NOT (v_pore - l_total)!
    """)


def correct_analytical_solution(S_0, v_pore, l_total, k_int, t):
    """
    CORRECT analytical solution.
    
    CRITICAL: β = k × (l_total - v_pore), NOT k × (v_pore - l_total)!
    """
    if np.isclose(t, 0.0):
        return S_0
    
    l_0 = S_0 * v_pore
    
    # Special case: l_total == v_pore
    if np.isclose(l_total, v_pore):
        if np.isclose(l_0, v_pore):
            return 1.0
        denom = k_int * t + 1.0 / (v_pore - l_0)
        if np.abs(denom) < 1e-15:
            return 1.0
        l_t = v_pore - 1.0 / denom
        return max(0.0, min(1.0, l_t / v_pore))
    
    # CORRECT: β = k × (l_total - v_pore)
    beta = k_int * (l_total - v_pore)  # This is NEGATIVE when l_total < v_pore
    
    # Initial ratio R₀
    denom_R = v_pore - l_0
    if np.abs(denom_R) < 1e-30:
        return min(1.0, l_total / v_pore)
    
    R_0 = (l_total - l_0) / denom_R
    
    # Exponential decay (since beta < 0 when l_total < v_pore)
    exp_term = np.exp(beta * t)
    R_t = R_0 * exp_term
    
    # Solve for l(t)
    denom = R_t - 1.0
    if np.abs(denom) < 1e-15:
        return min(l_total, v_pore) / v_pore
    
    l_t = (R_t * v_pore - l_total) / denom
    
    S_t = l_t / v_pore
    S_t = max(0.0, min(1.0, S_t, l_total / v_pore))
    
    return S_t


def kernel_compute_correct(S_0, v_pore, l_total, k_int, dt):
    """
    CORRECT kernel implementation using exact ODE solution.
    """
    l_0 = S_0 * v_pore
    
    # Special case: l_total == v_pore
    if np.isclose(l_total, v_pore):
        if np.isclose(l_0, v_pore):
            return 1.0
        denom = k_int * dt + 1.0 / (v_pore - l_0)
        if np.abs(denom) < 1e-15:
            return 1.0
        l_new = v_pore - 1.0 / denom
        l_new = max(0.0, min(l_new, l_total, v_pore))
        return l_new / v_pore
    
    # CORRECT: beta = k × (l_total - v_pore)
    beta = k_int * (l_total - v_pore)
    
    # Initial ratio R₀ = (l_total - l₀) / (v_pore - l₀)
    denom_R = v_pore - l_0
    if np.abs(denom_R) < 1e-30:
        return min(1.0, l_total / v_pore)
    
    R_0 = (l_total - l_0) / denom_R
    
    # Exponential factor
    exp_factor = np.exp(beta * dt)
    R_new = R_0 * exp_factor
    
    # Solve for l_new
    denom = R_new - 1.0
    if np.abs(denom) < 1e-15:
        l_new = min(l_total, v_pore)
    else:
        l_new = (R_new * v_pore - l_total) / denom
    
    # Clamp to physical bounds
    l_max = min(l_total, v_pore)
    l_new = max(0.0, min(l_new, l_max))
    
    return l_new / v_pore


def test_step_by_step():
    """Test with explicit step-by-step calculation."""
    print("\n" + "="*80)
    print("STEP-BY-STEP VERIFICATION")
    print("="*80)
    
    # Parameters
    k_int = 1e18
    v_pore = 1e-18
    l_total = 0.8e-18
    S_0 = 0.0
    l_0 = S_0 * v_pore
    
    print(f"\nParameters:")
    print(f"  k_int = {k_int:.2e}")
    print(f"  v_pore = {v_pore:.2e}")
    print(f"  l_total = {l_total:.2e}")
    print(f"  S_0 = {S_0}, l_0 = {l_0:.2e}")
    
    # Compute beta
    beta = k_int * (l_total - v_pore)
    print(f"\nBeta = k × (l_total - v_pore) = {beta:.2f} s⁻¹")
    print(f"  Note: beta < 0 because l_total < v_pore")
    
    # Initial ratio R₀
    R_0 = (l_total - l_0) / (v_pore - l_0)
    print(f"\nR₀ = (l_total - l₀)/(v_pore - l₀) = {R_0:.6f}")
    
    # Test at t = 0.5 s
    t = 0.5
    exp_term = np.exp(beta * t)
    R_t = R_0 * exp_term
    
    print(f"\nAt t = {t} s:")
    print(f"  exp(β×t) = exp({beta}×{t}) = {exp_term:.6f}")
    print(f"  R(t) = R₀ × exp(β×t) = {R_0:.6f} × {exp_term:.6f} = {R_t:.6f}")
    
    # Compute l(t)
    l_t = (R_t * v_pore - l_total) / (R_t - 1.0)
    S_t = l_t / v_pore
    
    print(f"\n  l(t) = (R×v_pore - l_total)/(R-1)")
    print(f"       = ({R_t:.6f}×{v_pore:.2e} - {l_total:.2e})/({R_t:.6f}-1)")
    print(f"       = {l_t:.6e} m³")
    print(f"  S(t) = l(t)/v_pore = {S_t:.6f}")
    
    # Verify with numerical integration (small steps)
    print(f"\nNumerical reference (dt=1e-6 s):")
    dt_ref = 1e-6
    n_steps = int(t / dt_ref)
    S_num = S_0
    for _ in range(n_steps):
        l = S_num * v_pore
        rate = k_int * (l_total - l) * (v_pore - l)
        dl = rate * dt_ref
        l_new = l + dl
        l_new = max(0.0, min(l_new, l_total, v_pore))
        S_num = l_new / v_pore
    
    print(f"  S_numerical = {S_num:.6f}")
    print(f"  S_analytical = {S_t:.6f}")
    print(f"  Difference = {abs(S_num - S_t):.2e}")
    
    # Test kernel compute
    print(f"\nKernel compute (single step dt={t}):")
    S_kernel = kernel_compute_correct(S_0, v_pore, l_total, k_int, t)
    print(f"  S_kernel = {S_kernel:.6f}")


if __name__ == '__main__':
    derive_solution()
    test_step_by_step()
