"""
Test Script for PowerLaw-Rumpf Breakage Kernel.

Tests the new powerlaw_rumpf breakage kernel which incorporates
particle strength σ as a function of porosity and saturation,
based on Rumpf's theory for granule strength.

Formula: S(V) = P1 × (1/σ) × G^P2 × V^alpha

Particle Strength Model (Rumpf Theory, 3 Regimes):
    - S < 0.3 (dry):   σ = (1-poro)/poro × k × γ / x_s
    - S > 0.8 (wet):   σ = 6×α × (1-poro)/poro × γ×cos(δ) / x_s × S
    - 0.3 ≤ S ≤ 0.8:   Linear interpolation

Run this script to verify the kernel before integration.

Usage:
    python -m wmcpbe.Trials.test_powerlaw_rumpf_kernel
    
Or from the wmcpbe directory:
    python Trials/test_powerlaw_rumpf_kernel.py
"""

import sys
import numpy as np

# Add parent directory to path for imports
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {name}")
    if details and not passed:
        print(f"         Details: {details}")


# =============================================================================
# Test 1: Import and Instantiation
# =============================================================================

def test_import_and_creation():
    """Test that the kernel can be imported and instantiated."""
    print_header("TEST 1: Import and Instantiation")
    
    all_passed = True
    
    try:
        from wmcpbe.kernels.breakage import get_breakage_kernel, list_breakage_kernels
        
        # Check kernel is in registry
        kernels = list_breakage_kernels()
        assert 'powerlaw_rumpf' in kernels, f"Kernel not in registry: {kernels}"
        print_test("Kernel in registry", True)
        
        # Create with default parameters
        kernel = get_breakage_kernel('powerlaw_rumpf')
        assert kernel.name == 'powerlaw_rumpf'
        assert kernel.category == 'breakage'
        print_test("Default instantiation", True)
        
        # Create with custom parameters
        kernel = get_breakage_kernel(
            'powerlaw_rumpf',
            p1=3e-2, p2=1.0, g=1000,
            k=2.5, alpha=1.15, gamma=0.072, delta=0.0
        )
        assert kernel.k == 2.5
        assert kernel.alpha == 1.15
        assert kernel.gamma == 0.072
        assert kernel.cos_delta == np.cos(0.0)
        print_test("Custom parameter instantiation", True)
        
    except Exception as e:
        print_test("Import and creation", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# Test 2: Parameter Validation
# =============================================================================

def test_parameter_validation():
    """Test that parameter validation works correctly."""
    print_header("TEST 2: Parameter Validation")
    
    all_passed = True
    
    from wmcpbe.kernels.breakage import get_breakage_kernel
    
    # Test k out of range (too low)
    try:
        kernel = get_breakage_kernel('powerlaw_rumpf', k=2.0)
        print_test("Reject k < 2.2", False, "Should have raised ValueError")
        all_passed = False
    except ValueError as e:
        if "2.2" in str(e) and "2.8" in str(e):
            print_test("Reject k < 2.2", True)
        else:
            print_test("Reject k < 2.2", False, f"Wrong error message: {e}")
            all_passed = False
    
    # Test k out of range (too high)
    try:
        kernel = get_breakage_kernel('powerlaw_rumpf', k=3.0)
        print_test("Reject k > 2.8", False, "Should have raised ValueError")
        all_passed = False
    except ValueError:
        print_test("Reject k > 2.8", True)
    
    # Test alpha out of range (too low)
    try:
        kernel = get_breakage_kernel('powerlaw_rumpf', alpha=0.9)
        print_test("Reject alpha < 1.0", False, "Should have raised ValueError")
        all_passed = False
    except ValueError:
        print_test("Reject alpha < 1.0", True)
    
    # Test alpha out of range (too high)
    try:
        kernel = get_breakage_kernel('powerlaw_rumpf', alpha=1.5)
        print_test("Reject alpha > 1.33", False, "Should have raised ValueError")
        all_passed = False
    except ValueError:
        print_test("Reject alpha > 1.33", True)
    
    # Test negative gamma
    try:
        kernel = get_breakage_kernel('powerlaw_rumpf', gamma=-0.072)
        print_test("Reject negative gamma", False, "Should have raised ValueError")
        all_passed = False
    except ValueError:
        print_test("Reject negative gamma", True)
    
    # Test invalid delta (negative)
    try:
        kernel = get_breakage_kernel('powerlaw_rumpf', delta=-0.1)
        print_test("Reject negative delta", False, "Should have raised ValueError")
        all_passed = False
    except ValueError:
        print_test("Reject negative delta", True)
    
    # Test invalid delta (too large)
    try:
        kernel = get_breakage_kernel('powerlaw_rumpf', delta=np.pi/2 + 0.1)
        print_test("Reject delta > π/2", False, "Should have raised ValueError")
        all_passed = False
    except ValueError:
        print_test("Reject delta > π/2", True)
    
    # Valid edge cases should work
    try:
        kernel = get_breakage_kernel('powerlaw_rumpf', k=2.2)
        kernel = get_breakage_kernel('powerlaw_rumpf', k=2.8)
        kernel = get_breakage_kernel('powerlaw_rumpf', alpha=1.0)
        kernel = get_breakage_kernel('powerlaw_rumpf', alpha=1.33)
        kernel = get_breakage_kernel('powerlaw_rumpf', delta=0.0)
        kernel = get_breakage_kernel('powerlaw_rumpf', delta=np.pi/2)
        print_test("Accept valid edge cases", True)
    except Exception as e:
        print_test("Accept valid edge cases", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# Test 3: Sigma Calculation (3 Regimes)
# =============================================================================

def test_sigma_calculation():
    """Test particle strength σ calculation in all three regimes."""
    print_header("TEST 3: Sigma Calculation (3 Regimes)")
    
    all_passed = True
    
    try:
        from wmcpbe.kernels.breakage import get_breakage_kernel
        
        kernel = get_breakage_kernel(
            'powerlaw_rumpf',
            k=2.5, alpha=1.15, gamma=0.072, delta=0.0
        )
        
        # Fixed parameters for sigma calculation
        poro = 0.4
        x_s = 10e-6  # 10 μm
        
        # Common factor
        poro_factor = (1.0 - poro) / poro  # = 1.5
        base_factor = poro_factor * kernel.gamma / x_s  # = 1.5 * 0.072 / 10e-6 = 10800
        
        # --- DRY REGIME (S < 0.3) ---
        sigma_dry = kernel._compute_sigma(poro, 0.1, x_s)
        expected_dry = base_factor * kernel.k  # = 10800 * 2.5 = 27000
        
        if np.isclose(sigma_dry, expected_dry, rtol=1e-10):
            print_test("Dry regime (S=0.1)", True)
        else:
            print_test("Dry regime (S=0.1)", False, 
                      f"Expected {expected_dry:.2f}, got {sigma_dry:.2f}")
            all_passed = False
        
        # Boundary: S = 0.3 (should still use dry formula)
        sigma_03 = kernel._compute_sigma(poro, 0.3, x_s)
        expected_03 = base_factor * kernel.k
        
        if np.isclose(sigma_03, expected_03, rtol=1e-10):
            print_test("Dry boundary (S=0.3)", True)
        else:
            print_test("Dry boundary (S=0.3)", False,
                      f"Expected {expected_03:.2f}, got {sigma_03:.2f}")
            all_passed = False
        
        # --- WET REGIME (S > 0.8) ---
        sigma_wet = kernel._compute_sigma(poro, 0.9, x_s)
        expected_wet = 6.0 * kernel.alpha * base_factor * kernel.cos_delta * 0.9
        # = 6 * 1.15 * 10800 * 1.0 * 0.9 = 67356
        
        if np.isclose(sigma_wet, expected_wet, rtol=1e-10):
            print_test("Wet regime (S=0.9)", True)
        else:
            print_test("Wet regime (S=0.9)", False,
                      f"Expected {expected_wet:.2f}, got {sigma_wet:.2f}")
            all_passed = False
        
        # Boundary: S = 0.8 (should use wet formula)
        sigma_08 = kernel._compute_sigma(poro, 0.8, x_s)
        expected_08 = 6.0 * kernel.alpha * base_factor * kernel.cos_delta * 0.8
        
        if np.isclose(sigma_08, expected_08, rtol=1e-10):
            print_test("Wet boundary (S=0.8)", True)
        else:
            print_test("Wet boundary (S=0.8)", False,
                      f"Expected {expected_08:.2f}, got {sigma_08:.2f}")
            all_passed = False
        
        # --- TRANSITION REGIME (0.3 ≤ S ≤ 0.8) ---
        # S = 0.55 (middle of transition)
        sigma_mid = kernel._compute_sigma(poro, 0.55, x_s)
        # t = (0.55 - 0.3) / 0.5 = 0.5
        # sigma = 0.5 * sigma_dry + 0.5 * sigma_wet(S=0.8)
        sigma_wet_08 = 6.0 * kernel.alpha * base_factor * kernel.cos_delta * 0.8
        expected_mid = 0.5 * expected_03 + 0.5 * sigma_wet_08
        
        if np.isclose(sigma_mid, expected_mid, rtol=1e-10):
            print_test("Transition regime (S=0.55)", True)
        else:
            print_test("Transition regime (S=0.55)", False,
                      f"Expected {expected_mid:.2f}, got {sigma_mid:.2f}")
            all_passed = False
        
        # Test linearity: S=0.3, 0.55, 0.8 should be collinear
        slope_1 = (sigma_mid - sigma_03) / (0.55 - 0.3)
        slope_2 = (sigma_08 - sigma_mid) / (0.8 - 0.55)
        if np.isclose(slope_1, slope_2, rtol=1e-10):
            print_test("Transition linearity", True)
        else:
            print_test("Transition linearity", False,
                      f"Slopes differ: {slope_1:.2f} vs {slope_2:.2f}")
            all_passed = False
        
        # --- NAN SATURATION (should treat as dry) ---
        sigma_nan = kernel._compute_sigma(poro, np.nan, x_s)
        if np.isclose(sigma_nan, expected_dry, rtol=1e-10):
            print_test("NaN saturation → dry", True)
        else:
            print_test("NaN saturation → dry", False,
                      f"Expected {expected_dry:.2f}, got {sigma_nan:.2f}")
            all_passed = False
        
    except Exception as e:
        print_test("Sigma calculation", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# Test 4: Rate Calculation (PowerLaw × 1/σ)
# =============================================================================

def test_rate_calculation():
    """Test breakage rate calculation with σ correction."""
    print_header("TEST 4: Rate Calculation (PowerLaw × 1/σ)")
    
    all_passed = True
    
    try:
        from wmcpbe.kernels.breakage import get_breakage_kernel
        
        kernel = get_breakage_kernel(
            'powerlaw_rumpf',
            p1=3e-2, p2=1.0, g=1000, breakrval=4, pl_v=2.0,
            k=2.5, alpha=1.15, gamma=0.072, delta=0.0, x_s=10e-6
        )
        
        v_particle = 1e-18  # 1 μm³
        
        # Compute base PowerLaw rate (without σ correction)
        base_rate = kernel._compute_base_rate_powerlaw(v_particle)
        # = p1 * g^p2 * v^(pl_v/3) = 0.03 * 1000^1 * (1e-18)^(2/3)
        # = 0.03 * 1000 * 1e-12 = 3e-11
        
        # Test without solver (should return base rate)
        rate_no_solver = kernel.compute_rate(v_particle)
        if np.isclose(rate_no_solver, base_rate, rtol=1e-10):
            print_test("No solver → base rate", True)
        else:
            print_test("No solver → base rate", False,
                      f"Expected {base_rate:.2e}, got {rate_no_solver:.2e}")
            all_passed = False
        
        # Create mock solver with porosity/saturation
        class MockSolver:
            def __init__(self, poro, sat, X0):
                self.porosity = np.array([poro])
                self.saturation = np.array([sat])
                self.X0 = np.array([X0])
                self.VERBOSE = True
        
        # Test with Vollkörper (poro = NaN) → should return base rate
        solver_vollkoerper = MockSolver(poro=np.nan, sat=0.5, X0=10e-6)
        rate_vollkoerper = kernel.compute_rate(v_particle, particle_idx=0, solver=solver_vollkoerper)
        if np.isclose(rate_vollkoerper, base_rate, rtol=1e-10):
            print_test("Vollkörper (poro=NaN) → base rate", True)
        else:
            print_test("Vollkörper (poro=NaN) → base rate", False,
                      f"Expected {base_rate:.2e}, got {rate_vollkoerper:.2e}")
            all_passed = False
        
        # Test with dry porous particle (S=0.1)
        # σ = 27000 Pa (from previous test)
        # rate = base_rate / σ = 3e-11 / 27000 ≈ 1.11e-15
        solver_dry = MockSolver(poro=0.4, sat=0.1, X0=10e-6)
        rate_dry = kernel.compute_rate(v_particle, particle_idx=0, solver=solver_dry)
        sigma_dry = 27000.0  # From Test 3
        expected_rate_dry = base_rate / sigma_dry
        
        if np.isclose(rate_dry, expected_rate_dry, rtol=1e-10):
            print_test("Dry porous (S=0.1) → corrected rate", True)
        else:
            print_test("Dry porous (S=0.1) → corrected rate", False,
                      f"Expected {expected_rate_dry:.2e}, got {rate_dry:.2e}")
            all_passed = False
        
        # Test with wet porous particle (S=0.9)
        # σ = 67356 Pa (from previous test)
        # rate = base_rate / σ = 3e-11 / 67356 ≈ 4.45e-16
        solver_wet = MockSolver(poro=0.4, sat=0.9, X0=10e-6)
        rate_wet = kernel.compute_rate(v_particle, particle_idx=0, solver=solver_wet)
        sigma_wet = 67356.0  # From Test 3
        expected_rate_wet = base_rate / sigma_wet
        
        if np.isclose(rate_wet, expected_rate_wet, rtol=1e-10):
            print_test("Wet porous (S=0.9) → corrected rate", True)
        else:
            print_test("Wet porous (S=0.9) → corrected rate", False,
                      f"Expected {expected_rate_wet:.2e}, got {rate_wet:.2e}")
            all_passed = False
        
        # Verify: Wet particles should have LOWER rate than dry (higher strength)
        if rate_wet < rate_dry:
            print_test("Wet rate < Dry rate (σ_wet > σ_dry)", True)
        else:
            print_test("Wet rate < Dry rate (σ_wet > σ_dry)", False,
                      f"Wet rate {rate_wet:.2e} should be < Dry rate {rate_dry:.2e}")
            all_passed = False
        
    except Exception as e:
        print_test("Rate calculation", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# Test 5: Sauter Diameter Computation
# =============================================================================

def test_sauter_diameter():
    """Test Sauter diameter computation from X0."""
    print_header("TEST 5: Sauter Diameter Computation")
    
    all_passed = True
    
    try:
        from wmcpbe.kernels.breakage import get_breakage_kernel
        
        kernel = get_breakage_kernel(
            'powerlaw_rumpf',
            p1=3e-2, p2=1.0, g=1000, breakrval=4, pl_v=2.0,
            k=2.5, alpha=1.15, gamma=0.072, delta=0.0,
            x_s=None  # Will be computed from X0
        )
        
        class MockSolver:
            def __init__(self, X0):
                self.X0 = np.array([X0])
        
        # Test monodisperse: x_s should equal particle diameter
        solver_mono = MockSolver(X0=10e-6)
        x_s_mono = kernel._get_x_s(solver_mono)
        if np.isclose(x_s_mono, 10e-6, rtol=1e-10):
            print_test("Monodisperse X0 → x_s = diameter", True)
        else:
            print_test("Monodisperse X0 → x_s = diameter", False,
                      f"Expected 10e-6, got {x_s_mono:.2e}")
            all_passed = False
        
        # Test polydisperse: d32 = Σ(d³) / Σ(d²)
        # Example: [5μm, 10μm, 20μm]
        # d2_sum = 25 + 100 + 400 = 525
        # d3_sum = 125 + 1000 + 8000 = 9125
        # d32 = 9125 / 525 ≈ 17.38 μm
        solver_poly = MockSolver(X0=np.array([5e-6, 10e-6, 20e-6]))
        x_s_poly = kernel._get_x_s(solver_poly)
        expected_d32 = (5**3 + 10**3 + 20**3) / (5**2 + 10**2 + 20**2) * 1e-6
        # = (125 + 1000 + 8000) / (25 + 100 + 400) * 1e-6
        # = 9125 / 525 * 1e-6 ≈ 17.38e-6
        
        if np.isclose(x_s_poly, expected_d32, rtol=1e-10):
            print_test("Polydisperse X0 → Sauter mean d32", True)
        else:
            print_test("Polydisperse X0 → Sauter mean d32", False,
                      f"Expected {expected_d32:.2e}, got {x_s_poly:.2e}")
            all_passed = False
        
        # Test explicit x_s parameter overrides X0
        kernel_explicit = get_breakage_kernel(
            'powerlaw_rumpf',
            p1=3e-2, p2=1.0, g=1000, breakrval=4, pl_v=2.0,
            k=2.5, alpha=1.15, gamma=0.072, delta=0.0,
            x_s=15e-6  # Explicit value
        )
        solver_any = MockSolver(X0=10e-6)
        x_s_explicit = kernel_explicit._get_x_s(solver_any)
        if np.isclose(x_s_explicit, 15e-6, rtol=1e-10):
            print_test("Explicit x_s overrides X0", True)
        else:
            print_test("Explicit x_s overrides X0", False,
                      f"Expected 15e-6, got {x_s_explicit:.2e}")
            all_passed = False
        
    except Exception as e:
        print_test("Sauter diameter", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# Test 6: Edge Cases and Fallbacks
# =============================================================================

def test_edge_cases():
    """Test edge cases and fallback behavior."""
    print_header("TEST 6: Edge Cases and Fallbacks")
    
    all_passed = True
    
    try:
        from wmcpbe.kernels.breakage import get_breakage_kernel
        
        kernel = get_breakage_kernel(
            'powerlaw_rumpf',
            p1=3e-2, p2=1.0, g=1000, breakrval=4, pl_v=2.0,
            k=2.5, alpha=1.15, gamma=0.072, delta=0.0, x_s=10e-6
        )
        
        v_particle = 1e-18
        
        # Test zero volume → rate = 0
        rate_zero = kernel.compute_rate(0.0)
        if rate_zero == 0.0:
            print_test("Zero volume → rate = 0", True)
        else:
            print_test("Zero volume → rate = 0", False,
                      f"Expected 0, got {rate_zero:.2e}")
            all_passed = False
        
        # Test negative volume → rate = 0
        rate_neg = kernel.compute_rate(-1e-18)
        if rate_neg == 0.0:
            print_test("Negative volume → rate = 0", True)
        else:
            print_test("Negative volume → rate = 0", False,
                      f"Expected 0, got {rate_neg:.2e}")
            all_passed = False
        
        # Test invalid porosity (poro=0) → fallback to base rate
        class MockSolverInvalidPoro:
            def __init__(self):
                self.porosity = np.array([0.0])
                self.saturation = np.array([0.5])
                self.X0 = np.array([10e-6])
        
        solver_invalid = MockSolverInvalidPoro()
        rate_invalid = kernel.compute_rate(v_particle, particle_idx=0, solver=solver_invalid)
        base_rate = kernel._compute_base_rate_powerlaw(v_particle)
        if np.isclose(rate_invalid, base_rate, rtol=1e-10):
            print_test("Invalid poro=0 → base rate", True)
        else:
            print_test("Invalid poro=0 → base rate", False,
                      f"Expected {base_rate:.2e}, got {rate_invalid:.2e}")
            all_passed = False
        
        # Test missing solver attributes → fallback to base rate
        class MockSolverNoAttr:
            pass
        
        solver_no_attr = MockSolverNoAttr()
        rate_no_attr = kernel.compute_rate(v_particle, particle_idx=0, solver=solver_no_attr)
        if np.isclose(rate_no_attr, base_rate, rtol=1e-10):
            print_test("Missing solver attributes → base rate", True)
        else:
            print_test("Missing solver attributes → base rate", False,
                      f"Expected {base_rate:.2e}, got {rate_no_attr:.2e}")
            all_passed = False
        
        # Test different BREAKRVAL variants
        for breakrval in [1, 2, 3, 4, 5]:
            kernel_br = get_breakage_kernel(
                'powerlaw_rumpf',
                p1=3e-2, p2=1.0, g=1000, breakrval=breakrval, pl_v=2.0,
                k=2.5, alpha=1.15, gamma=0.072, delta=0.0, x_s=10e-6
            )
            rate_br = kernel_br.compute_rate(v_particle)
            if rate_br >= 0 and np.isfinite(rate_br):
                print_test(f"BREAKRVAL={breakrval}", True)
            else:
                print_test(f"BREAKRVAL={breakrval}", False,
                          f"Invalid rate: {rate_br}")
                all_passed = False
        
    except Exception as e:
        print_test("Edge cases", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# Main test runner
# =============================================================================

def main():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print(" POWERLAW-RUMPF BREAKAGE KERNEL - TEST SUITE")
    print("=" * 70)
    print("\nThis script tests the powerlaw_rumpf breakage kernel.")
    print("All tests must pass before integrating into simulation.\n")
    
    results = []
    
    # Run all tests
    results.append(("Import and Creation", test_import_and_creation()))
    results.append(("Parameter Validation", test_parameter_validation()))
    results.append(("Sigma Calculation (3 Regimes)", test_sigma_calculation()))
    results.append(("Rate Calculation", test_rate_calculation()))
    results.append(("Sauter Diameter", test_sauter_diameter()))
    results.append(("Edge Cases", test_edge_cases()))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  [{status}] {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! Kernel is ready for integration.")
        print("\n  Usage example:")
        print("    solver = MCPBEBase(")
        print("        break_kernel_name='powerlaw_rumpf',")
        print("        break_kernel_params={")
        print("            'p1': 3e-2, 'p2': 1.0, 'g': 1000, 'breakrval': 4, 'pl_v': 2.0,")
        print("            'k': 2.5, 'alpha': 1.15, 'gamma': 0.072, 'delta': 0.0")
        print("        }")
        print("    )")
        return 0
    else:
        print(f"\n  ⚠ {total - passed} test(s) failed. Please fix issues before integration.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
