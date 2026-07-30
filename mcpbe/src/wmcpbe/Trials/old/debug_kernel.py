"""
Deep debugging: Why does kernel jump to S=0.8 immediately?

This script traces through EVERY calculation in the kernel step-by-step.
"""

import numpy as np


def debug_kernel_compute(saturation, v_pore, l_total, k_int, dt):
    """
    Exact copy of kernel compute() with debug output.
    NO MODIFICATIONS - just printing intermediate values.
    """
    print("\n" + "="*80)
    print("KERNEL COMPUTE - STEP BY STEP")
    print("="*80)
    
    # Edge cases
    if v_pore <= 0 or not np.isfinite(v_pore):
        print(f"Edge case: v_pore invalid")
        return saturation
    if l_total <= 0 or not np.isfinite(l_total):
        print(f"Edge case: l_total invalid")
        return 0.0
    
    saturation = max(0.0, min(1.0, saturation))
    print(f"Input saturation (clamped): {saturation}")
    
    if saturation >= 1.0 or k_int <= 0:
        print(f"Edge case: saturated or k_int=0")
        return saturation
    
    # Convert saturation to internal liquid volume
    l_intern = saturation * v_pore
    print(f"\nl_intern = S × v_pore = {saturation} × {v_pore:.2e} = {l_intern:.6e}")
    
    # Rate factor: α = k × (v_pore - l_total)
    alpha = k_int * (v_pore - l_total)
    print(f"\nalpha = k_int × (v_pore - l_total)")
    print(f"      = {k_int:.2e} × ({v_pore:.2e} - {l_total:.2e})")
    print(f"      = {k_int:.2e} × {(v_pore - l_total):.2e}")
    print(f"      = {alpha:.6f}")
    
    # Exponential term
    exp_arg = alpha * dt
    exp_term = np.exp(exp_arg)
    print(f"\nexp_term = exp(alpha × dt)")
    print(f"         = exp({alpha:.6f} × {dt})")
    print(f"         = exp({exp_arg:.6f})")
    print(f"         = {exp_term:.6f}")
    
    # Coefficients from current state
    A = v_pore - l_intern
    B = l_total - l_intern
    print(f"\nCoefficients:")
    print(f"  A = v_pore - l_intern = {v_pore:.2e} - {l_intern:.2e} = {A:.6e}")
    print(f"  B = l_total - l_intern = {l_total:.2e} - {l_intern:.2e} = {B:.6e}")
    
    # Analytical solution
    numerator = v_pore * B - l_total * A * exp_term
    denominator = B - A * exp_term
    
    print(f"\nNumerator = v_pore × B - l_total × A × exp_term")
    print(f"          = {v_pore:.2e} × {B:.6e} - {l_total:.2e} × {A:.6e} × {exp_term:.6f}")
    print(f"          = {v_pore * B:.6e} - {l_total * A * exp_term:.6e}")
    print(f"          = {numerator:.6e}")
    
    print(f"\nDenominator = B - A × exp_term")
    print(f"            = {B:.6e} - {A:.6e} × {exp_term:.6f}")
    print(f"            = {B:.6e} - {A * exp_term:.6e}")
    print(f"            = {denominator:.6e}")
    
    # Handle division by zero
    if np.abs(denominator) < 1e-15:
        print(f"\nDenominator ≈ 0! Using equilibrium value")
        l_intern_new = min(l_total, v_pore)
    else:
        l_intern_new = numerator / denominator
    
    print(f"\nl_intern_new = numerator / denominator")
    print(f"             = {numerator:.6e} / {denominator:.6e}")
    print(f"             = {l_intern_new:.6e}")
    
    # Clamp to physical bounds
    l_max = min(l_total, v_pore)
    print(f"\nClamping:")
    print(f"  l_max = min(l_total, v_pore) = min({l_total:.2e}, {v_pore:.2e}) = {l_max:.2e}")
    print(f"  l_intern_new before clamp: {l_intern_new:.6e}")
    
    l_intern_new_clamped = max(0.0, min(l_intern_new, l_max))
    print(f"  l_intern_new after clamp:  {l_intern_new_clamped:.6e}")
    
    # Convert back to saturation
    saturation_new = l_intern_new_clamped / v_pore
    print(f"\nsaturation_new = l_intern_new / v_pore")
    print(f"               = {l_intern_new_clamped:.6e} / {v_pore:.2e}")
    print(f"               = {saturation_new:.6f}")
    
    return saturation_new


def reference_solution(S_0, v_pore, l_total, k_int, t, dt_ref=1e-6):
    """High-accuracy numerical reference."""
    n_steps = max(1, int(t / dt_ref))
    dt = t / n_steps
    
    S = S_0
    l_max = min(l_total, v_pore)
    
    for _ in range(n_steps):
        l = S * v_pore
        rate = k_int * (l_total - l) * (v_pore - l)
        dl = rate * dt
        l_new = l + dl
        l_new = max(0.0, min(l_new, l_max))
        S = l_new / v_pore
    
    return S


def main():
    print("="*80)
    print("DEBUG: WHY DOES KERNEL JUMP TO S=0.8?")
    print("="*80)
    
    # Test parameters
    k_int = 1e18
    v_pore = 1e-18
    l_total = 0.8e-18
    S_0 = 0.0
    dt = 0.5
    
    print(f"\nParameters:")
    print(f"  k_int = {k_int:.2e}")
    print(f"  v_pore = {v_pore:.2e}")
    print(f"  l_total = {l_total:.2e}")
    print(f"  S_0 = {S_0}")
    print(f"  dt = {dt}")
    
    # First step: S_0 = 0 → S_1
    print("\n\n" + "="*80)
    print("FIRST STEP: t=0 → t=0.5s")
    print("="*80)
    
    S_1 = debug_kernel_compute(S_0, v_pore, l_total, k_int, dt)
    
    # Reference solution at t=0.5
    S_ref = reference_solution(S_0, v_pore, l_total, k_int, 0.5)
    
    print(f"\n\nRESULT:")
    print(f"  Kernel:     S(0.5) = {S_1:.6f}")
    print(f"  Reference:  S(0.5) = {S_ref:.6f}")
    print(f"  Error:      {abs(S_1 - S_ref):.6f}")
    
    # What SHOULD happen mathematically?
    print("\n\n" + "="*80)
    print("WHAT SHOULD HAPPEN?")
    print("="*80)
    
    # From verify_analytical.py formula:
    # x(t) = [p(l-x₀) - l(p-x₀)e^{k(p-l)t}] / [(l-x₀) - (p-x₀)e^{k(p-l)t}]
    # Mapping: x→l_intern, p→v_pore, l→l_total
    
    l_0 = S_0 * v_pore
    
    # Expected formula
    alpha_correct = k_int * (v_pore - l_total)  # = 0.2
    exp_term_correct = np.exp(alpha_correct * dt)
    
    A_correct = v_pore - l_0
    B_correct = l_total - l_0
    
    # CORRECT formula from verify_analytical.py:
    # numerator = p × (l - x₀) - l × (p - x₀) × exp_term
    # denominator = (l - x₀) - (p - x₀) × exp_term
    
    num_correct = v_pore * B_correct - l_total * A_correct * exp_term_correct
    den_correct = B_correct - A_correct * exp_term_correct
    
    l_expected = num_correct / den_correct
    S_expected = l_expected / v_pore
    
    print(f"\nExpected calculation (from verify_analytical.py):")
    print(f"  alpha = k × (v_pore - l_total) = {alpha_correct:.6f}")
    print(f"  exp_term = exp(alpha × dt) = {exp_term_correct:.6f}")
    print(f"  A = v_pore - l_0 = {A_correct:.6e}")
    print(f"  B = l_total - l_0 = {B_correct:.6e}")
    print(f"  numerator = v_pore × B - l_total × A × exp = {num_correct:.6e}")
    print(f"  denominator = B - A × exp = {den_correct:.6e}")
    print(f"  l_expected = {l_expected:.6e}")
    print(f"  S_expected = {S_expected:.6f}")
    
    # Check sign of exponent
    print("\n\n" + "="*80)
    print("SIGN ANALYSIS")
    print("="*80)
    
    print(f"\nv_pore - l_total = {v_pore - l_total:.2e} (POSITIVE)")
    print(f"k × (v_pore - l_total) = {alpha_correct:.6f} (POSITIVE)")
    print(f"exp(k × (v_pore - l_total) × dt) = {exp_term_correct:.6f} (> 1)")
    
    print(f"\nFor t → ∞:")
    print(f"  exp_term → ∞ (grows exponentially)")
    print(f"  numerator ≈ -l_total × A × exp_term → -∞")
    print(f"  denominator ≈ -A × exp_term → -∞")
    print(f"  l → (-l_total × A × exp) / (-A × exp) = l_total ✓")
    
    # But wait... let's check the formula more carefully
    print("\n\n" + "="*80)
    print("FORMULA VERIFICATION")
    print("="*80)
    
    print("""
From verify_analytical.py:
    x(t) = [p(l-x₀) - l(p-x₀)e^{k(p-l)t}] / [(l-x₀) - (p-x₀)e^{k(p-l)t}]

With p=v_pore, l=l_total, x₀=l_intern:
    l(t) = [v_pore×(l_total-l) - l_total×(v_pore-l)×exp] / [(l_total-l) - (v_pore-l)×exp]

But this assumes l(t=0) = l₀. For step-by-step, we use l_current as initial condition.

At t=0, l=l₀=0:
    numerator = v_pore×l_total - l_total×v_pore×1 = 0 ✓
    denominator = l_total - v_pore×1 = l_total - v_pore ≠ 0

Wait! At t=0, exp_term should be exp(0)=1, so:
    numerator = v_pore×B - l_total×A×1 
              = v_pore×l_total - l_total×v_pore = 0 ✓
    denominator = B - A×1 = l_total - v_pore = -0.2e-18

So l(0) = 0 / (-0.2e-18) = 0 ✓

At t=0.5:
    exp_term = exp(0.2 × 0.5) = exp(0.1) = 1.105
    
    numerator = 1e-18 × 0.8e-18 - 0.8e-18 × 1e-18 × 1.105
              = 0.8e-36 - 0.884e-36
              = -0.084e-36
    
    denominator = 0.8e-18 - 1e-18 × 1.105
                = 0.8e-18 - 1.105e-18
                = -0.305e-18
    
    l(0.5) = -0.084e-36 / -0.305e-18 = 0.275e-18 ✓
    S(0.5) = 0.275e-18 / 1e-18 = 0.275 ✓

This matches the reference! So why is the kernel giving 0.8?
""")
    
    # Let's check what the kernel ACTUALLY computes
    print("\n\n" + "="*80)
    print("ACTUAL KERNEL COMPUTATION")
    print("="*80)
    
    l_0 = 0
    alpha = k_int * (v_pore - l_total)
    exp_term = np.exp(alpha * dt)
    A = v_pore - l_0
    B = l_total - l_0
    
    print(f"alpha = {alpha:.6f}")
    print(f"exp_term = {exp_term:.6f}")
    print(f"A = {A:.6e}")
    print(f"B = {B:.6e}")
    
    # Kernel formula (current implementation):
    num_kernel = v_pore * B - l_total * A * exp_term
    den_kernel = B - A * exp_term
    
    print(f"\nKernel formula:")
    print(f"  numerator = v_pore × B - l_total × A × exp_term")
    print(f"            = {v_pore:.2e} × {B:.6e} - {l_total:.2e} × {A:.6e} × {exp_term:.6f}")
    print(f"            = {v_pore * B:.6e} - {l_total * A * exp_term:.6e}")
    print(f"            = {num_kernel:.6e}")
    
    print(f"  denominator = B - A × exp_term")
    print(f"              = {B:.6e} - {A:.6e} × {exp_term:.6f}")
    print(f"              = {B:.6e} - {A * exp_term:.6e}")
    print(f"              = {den_kernel:.6e}")
    
    if abs(den_kernel) > 1e-30:
        l_kernel = num_kernel / den_kernel
        S_kernel = l_kernel / v_pore
        print(f"  l_kernel = {l_kernel:.6e}")
        print(f"  S_kernel = {S_kernel:.6f}")
        
        # After clamping
        l_max = min(l_total, v_pore)
        l_kernel_clamped = max(0.0, min(l_kernel, l_max))
        S_kernel_clamped = l_kernel_clamped / v_pore
        print(f"  l_kernel (clamped) = {l_kernel_clamped:.6e}")
        print(f"  S_kernel (clamped) = {S_kernel_clamped:.6f}")


if __name__ == '__main__':
    main()
