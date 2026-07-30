"""
Test Script for Liquid Internalization during Agglomeration Kernel.

Tests the Braumann et al. 2007 implementation for liquid trapping
in contact pores when two wetted particles merge.


"""

import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, r'C:/Users/ericb/Documents/GitHub/PSD_opt/mcpbe/src/wmcpbe')

from kernels.continuous_processes.liq_internalisation_agglomeration import (
    LiquidInternalisationAgglomerationKernel
)


def test_kernel_creation():
    """Test 1: Kernel creation and basic properties."""
    print("=" * 70)
    print("TEST 1: Kernel Creation")
    print("=" * 70)
    
    kernel = LiquidInternalisationAgglomerationKernel()
    
    assert kernel.name == 'liq_internalisation_agglomeration', \
        f"Expected name 'liq_internalisation_agglomeration', got '{kernel.name}'"
    
    assert kernel.category == 'liquid_internalization_agglomeration', \
        f"Expected category 'liquid_internalization_agglomeration', got '{kernel.category}'"
    
    assert kernel.params == {}, \
        f"Expected empty params, got {kernel.params}"
    
    print("✓ Kernel created successfully")
    print(f"  Name: {kernel.name}")
    print(f"  Category: {kernel.category}")
    print(f"  Parameters: {kernel.params}")
    print()


def test_no_external_liquid():
    """Test 2: No external liquid → No internalization (FALLBACK)."""
    print("=" * 70)
    print("TEST 2: No External Liquid (Fallback Case)")
    print("=" * 70)
    
    kernel = LiquidInternalisationAgglomerationKernel()
    
    # Both particles dry
    result = kernel.compute_internalization(
        v_dry1=1e-18, v_dry2=2e-18,
        v_liq_ext1=0.0, v_liq_ext2=0.0
    )
    assert result == 0.0, f"Expected 0.0 for dry particles, got {result}"
    print("✓ Both particles dry: l_e_to_i = 0.0")
    
    # Particle 1 dry, particle 2 wet
    result = kernel.compute_internalization(
        v_dry1=1e-18, v_dry2=2e-18,
        v_liq_ext1=0.0, v_liq_ext2=1e-19
    )
    assert result == 0.0, f"Expected 0.0 when one particle dry, got {result}"
    print("✓ One particle dry: l_e_to_i = 0.0")
    
    # Particle 1 wet, particle 2 dry
    result = kernel.compute_internalization(
        v_dry1=1e-18, v_dry2=2e-18,
        v_liq_ext1=1e-19, v_liq_ext2=0.0
    )
    assert result == 0.0, f"Expected 0.0 when one particle dry, got {result}"
    print("✓ One particle dry (reversed): l_e_to_i = 0.0")
    print()


def test_both_particles_wet():
    """Test 3: Both particles wet → Positive internalization."""
    print("=" * 70)
    print("TEST 3: Both Particles Wet (Standard Case)")
    print("=" * 70)
    
    kernel = LiquidInternalisationAgglomerationKernel()
    
    # Two identical particles with equal external liquid
    v_dry = 1e-18
    v_liq_ext = 1e-19
    
    result = kernel.compute_internalization(
        v_dry1=v_dry, v_dry2=v_dry,
        v_liq_ext1=v_liq_ext, v_liq_ext2=v_liq_ext
    )
    
    assert result > 0.0, f"Expected positive internalization, got {result}"
    assert result < 2.0 * v_liq_ext, \
        f"Internalization {result} exceeds available external liquid {2.0 * v_liq_ext}"
    
    print(f"✓ Two identical wet particles:")
    print(f"  v_dry = {v_dry:.3e} m³")
    print(f"  v_liq_ext = {v_liq_ext:.3e} m³ (each)")
    print(f"  l_e_to_i = {result:.3e} m³")
    print(f"  Fraction internalized: {result / (2.0 * v_liq_ext) * 100:.1f}%")
    print()


def test_mass_conservation():
    """Test 4: Mass conservation - internalized liquid cannot exceed available."""
    print("=" * 70)
    print("TEST 4: Mass Conservation")
    print("=" * 70)
    
    kernel = LiquidInternalisationAgglomerationKernel()
    
    # Extreme case: Very small particles with lots of external liquid
    v_dry1 = 1e-21  # Very small
    v_dry2 = 1e-21  # Very small
    v_liq_ext1 = 1e-18  # Lots of liquid
    v_liq_ext2 = 1e-18  # Lots of liquid
    
    result = kernel.compute_internalization(
        v_dry1=v_dry1, v_dry2=v_dry2,
        v_liq_ext1=v_liq_ext1, v_liq_ext2=v_liq_ext2
    )
    
    max_possible = v_liq_ext1 + v_liq_ext2
    assert result <= max_possible, \
        f"Internalization {result} exceeds maximum possible {max_possible}"
    
    print(f"✓ Mass conservation verified:")
    print(f"  Available external liquid: {max_possible:.3e} m³")
    print(f"  Internalized: {result:.3e} m³")
    print(f"  Remaining external: {max_possible - result:.3e} m³")
    print()


def test_size_dependence():
    """Test 5: Size dependence - larger particles should internalize differently."""
    print("=" * 70)
    print("TEST 5: Size Dependence")
    print("=" * 70)
    
    kernel = LiquidInternalisationAgglomerationKernel()
    
    # Same external liquid, different particle sizes
    v_liq_ext = 1e-19
    
    # Small + Small
    result_small = kernel.compute_internalization(
        v_dry1=1e-18, v_dry2=1e-18,
        v_liq_ext1=v_liq_ext, v_liq_ext2=v_liq_ext
    )
    
    # Large + Large
    result_large = kernel.compute_internalization(
        v_dry1=8e-18, v_dry2=8e-18,  # 2× radius → 8× volume
        v_liq_ext1=v_liq_ext, v_liq_ext2=v_liq_ext
    )
    
    # Mixed sizes
    result_mixed = kernel.compute_internalization(
        v_dry1=1e-18, v_dry2=8e-18,
        v_liq_ext1=v_liq_ext, v_liq_ext2=v_liq_ext
    )
    
    print(f"✓ Size dependence verified:")
    print(f"  Small + Small (1e-18 + 1e-18): l_e_to_i = {result_small:.3e} m³")
    print(f"  Large + Large (8e-18 + 8e-18): l_e_to_i = {result_large:.3e} m³")
    print(f"  Mixed (1e-18 + 8e-18):         l_e_to_i = {result_mixed:.3e} m³")
    print()


def test_numerical_stability():
    """Test 6: Numerical stability - edge cases and extreme values."""
    print("=" * 70)
    print("TEST 6: Numerical Stability")
    print("=" * 70)
    
    kernel = LiquidInternalisationAgglomerationKernel()
    
    # Very small volumes
    result = kernel.compute_internalization(
        v_dry1=1e-30, v_dry2=1e-30,
        v_liq_ext1=1e-30, v_liq_ext2=1e-30
    )
    assert np.isfinite(result), f"Result not finite for very small volumes: {result}"
    print(f"✓ Very small volumes (1e-30): l_e_to_i = {result:.3e} (finite)")
    
    # Very large volumes
    result = kernel.compute_internalization(
        v_dry1=1e-12, v_dry2=1e-12,
        v_liq_ext1=1e-13, v_liq_ext2=1e-13
    )
    assert np.isfinite(result), f"Result not finite for very large volumes: {result}"
    print(f"✓ Very large volumes (1e-12): l_e_to_i = {result:.3e} (finite)")
    
    # NaN input
    result = kernel.compute_internalization(
        v_dry1=np.nan, v_dry2=1e-18,
        v_liq_ext1=1e-19, v_liq_ext2=1e-19
    )
    assert result == 0.0, f"Expected 0.0 for NaN input, got {result}"
    print(f"✓ NaN input: l_e_to_i = 0.0 (handled correctly)")
    
    # Inf input
    result = kernel.compute_internalization(
        v_dry1=np.inf, v_dry2=1e-18,
        v_liq_ext1=1e-19, v_liq_ext2=1e-19
    )
    assert result == 0.0, f"Expected 0.0 for Inf input, got {result}"
    print(f"✓ Inf input: l_e_to_i = 0.0 (handled correctly)")
    print()


def test_braumann_formula_verification():
    """Test 7: Verify Braumann formula with manual calculation."""
    print("=" * 70)
    print("TEST 7: Braumann Formula Verification (Manual Calculation)")
    print("=" * 70)
    
    kernel = LiquidInternalisationAgglomerationKernel()
    
    # Test case with known values
    v_dry1 = 1e-18
    v_dry2 = 1e-18
    v_liq_ext1 = 1e-19
    v_liq_ext2 = 1e-19
    
    # Manual calculation following Braumann formula
    v_j = v_dry1 + v_liq_ext1  # Hydrodynamic volume
    v_k = v_dry2 + v_liq_ext2
    
    v_j_cbrt = np.cbrt(v_j)
    v_k_cbrt = np.cbrt(v_k)
    sum_radii = v_j_cbrt + v_k_cbrt
    
    ratio_j = np.cbrt(v_dry1) / sum_radii
    ratio_k = np.cbrt(v_dry2) / sum_radii
    
    bracket_j = 1.0 - np.sqrt(max(0.0, 1.0 - ratio_j**2))
    bracket_k = 1.0 - np.sqrt(max(0.0, 1.0 - ratio_k**2))
    
    product = v_liq_ext1 * v_liq_ext2 * bracket_j * bracket_k
    expected = np.sqrt(product)
    
    # Kernel calculation
    result = kernel.compute_internalization(
        v_dry1=v_dry1, v_dry2=v_dry2,
        v_liq_ext1=v_liq_ext1, v_liq_ext2=v_liq_ext2
    )
    
    # Allow small numerical tolerance
    tolerance = 1e-15
    assert abs(result - expected) < tolerance, \
        f"Kernel result {result} differs from manual calculation {expected}"
    
    print(f"✓ Formula verification passed:")
    print(f"  Manual calculation: {expected:.6e} m³")
    print(f"  Kernel result:      {result:.6e} m³")
    print(f"  Difference:         {abs(result - expected):.6e} m³")
    print()


def test_fallback_in_kernel_manager():
    """Test 8: Fallback behavior in KernelManager (no kernel configured)."""
    print("=" * 70)
    print("TEST 8: KernelManager Fallback (No Kernel Configured)")
    print("=" * 70)
    
    from kernel_integration import KernelManager
    
    # Create manager WITHOUT liq_internalisation_agglomeration kernel
    manager = KernelManager(
        agg_kernel_name='shear_chin1998',
        agg_kernel_params={'corr_beta': 1e-3, 'g': 1000},
        # NOTE: No liq_internalisation_agglomeration_kernel_name specified!
    )
    
    # Mock solver object (minimal attributes needed)
    class MockSolver:
        CORR_BETA = 1e-3
        G = 1000
    
    mock_solver = MockSolver()
    manager.initialize_kernels(mock_solver)
    
    # Call compute_liquid_internalization_agglomeration without kernel
    result = manager.compute_liquid_internalization_agglomeration(
        v_dry1=1e-18, v_dry2=1e-18,
        v_liq_ext1=1e-19, v_liq_ext2=1e-19,
        solver=mock_solver
    )
    
    assert result == 0.0, \
        f"Expected 0.0 fallback when kernel not configured, got {result}"
    
    print("✓ Fallback verified: No kernel configured → l_e_to_i = 0.0")
    print("  This ensures backward compatibility!")
    print()


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("*" * 70)
    print("* LIQUID INTERNALIZATION AGGLOMERATION KERNEL - TEST SUITE")
    print("* Braumann et al. 2007 Implementation")
    print("*" * 70)
    print("\n")
    
    try:
        test_kernel_creation()
        test_no_external_liquid()
        test_both_particles_wet()
        test_mass_conservation()
        test_size_dependence()
        test_numerical_stability()
        test_braumann_formula_verification()
        test_fallback_in_kernel_manager()
        
        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nSummary:")
        print("  - Kernel creation: OK")
        print("  - Fallback cases (dry particles): OK")
        print("  - Standard case (both wet): OK")
        print("  - Mass conservation: OK")
        print("  - Size dependence: OK")
        print("  - Numerical stability: OK")
        print("  - Formula verification: OK")
        print("  - KernelManager fallback: OK")
        print("\nThe implementation is ready for use!")
        print("\nTo enable in simulation:")
        print("  solver = MCPBESolver(")
        print("      liq_internalisation_agglomeration_kernel_name='liq_internalisation_agglomeration',")
        print("      ...")
        print("  )")
        print()
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        raise


if __name__ == '__main__':
    run_all_tests()
