"""
Test Script for WMCPBE Kernel Framework.

This script tests the new kernel architecture without modifying
the existing solver code. It verifies that:

1. All kernels can be imported and instantiated
2. Kernel methods work correctly with sample inputs
3. Factory functions return correct kernel types
4. Parameters are validated properly

Run this script to verify the kernel framework before integration.

Usage:
    python -m wmcpbe.Trials.test_kernel_framework
    
Or from the wmcpbe directory:
    python Trials/test_kernel_framework.py
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
# Test 1: Import all kernel modules
# =============================================================================

def test_imports():
    """Test that all kernel modules can be imported."""
    print_header("TEST 1: Module Imports")
    
    all_passed = True
    
    # Test base classes
    try:
        from wmcpbe.kernels.base import (
            KernelBase,
            AggregationKernel,
            BreakageKernel,
            PorosityGrowthKernel,
            CompressionKernel,
            LiquidDistributionKernel,
        )
        print_test("Import base classes", True)
    except Exception as e:
        print_test("Import base classes", False, str(e))
        all_passed = False
    
    # Test aggregation kernels
    try:
        from wmcpbe.kernels.aggregation import (
            get_aggregation_kernel,
            list_aggregation_kernels,
            ShearChinKernel,
            BrownianKernel,
            ConstantKernel,
            SumKernel,
            LiquidBridgeKernel,
        )
        print_test("Import aggregation kernels", True)
    except Exception as e:
        print_test("Import aggregation kernels", False, str(e))
        all_passed = False
    
    # Test breakage kernels
    try:
        from wmcpbe.kernels.breakage import (
            get_breakage_kernel,
            list_breakage_kernels,
            PowerLawBreakageKernel,
            StressBasedBreakageKernel,
        )
        print_test("Import breakage kernels", True)
    except Exception as e:
        print_test("Import breakage kernels", False, str(e))
        all_passed = False
    
    # Test porosity growth kernels
    try:
        from wmcpbe.kernels.porosity_growth import (
            get_porosity_growth_kernel,
            list_porosity_growth_kernels,
            VolumeMixingKernel,
            IncompleteMixingKernel,
        )
        print_test("Import porosity growth kernels", True)
    except Exception as e:
        print_test("Import porosity growth kernels", False, str(e))
        all_passed = False
    
    # Test compression kernels
    try:
        from wmcpbe.kernels.compression import (
            get_compression_kernel,
            list_compression_kernels,
            ExponentialDecayKernel,
            StressCompactionKernel,
        )
        print_test("Import compression kernels", True)
    except Exception as e:
        print_test("Import compression kernels", False, str(e))
        all_passed = False
    
    # Test liquid distribution kernels
    try:
        from wmcpbe.kernels.liquid_distribution import (
            get_liquid_distribution_kernel,
            list_liquid_distribution_kernels,
            UniformWeightedKernel,
            SurfaceWeightedKernel,
            SaturationPreferentialKernel,
        )
        print_test("Import liquid distribution kernels", True)
    except Exception as e:
        print_test("Import liquid distribution kernels", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# Test 2: Factory functions
# =============================================================================

def test_factories():
    """Test that factory functions create correct kernel instances."""
    print_header("TEST 2: Factory Functions")
    
    all_passed = True
    
    # Test aggregation factory
    try:
        from wmcpbe.kernels.aggregation import get_aggregation_kernel
        
        kernels_to_test = [
            ('shear_chin1998', {'corr_beta': 1e-3, 'g': 1000}),
            ('brownian_tsouris1995', {'corr_beta': 1.0, 'temperature': 293}),
            ('constant', {'corr_beta': 1e-15}),
            ('sum', {'corr_beta': 1e-9}),
            ('liquid_bridge', {'corr_beta': 1e-3, 'g': 1000, 'optimal_saturation': 0.5}),
        ]
        
        for name, params in kernels_to_test:
            kernel = get_aggregation_kernel(name, **params)
            assert kernel.name == name, f"Kernel name mismatch: {kernel.name} != {name}"
            assert kernel.category == 'aggregation'
        
        print_test("Aggregation factory", True)
    except Exception as e:
        print_test("Aggregation factory", False, str(e))
        all_passed = False
    
    # Test breakage factory
    try:
        from wmcpbe.kernels.breakage import get_breakage_kernel
        
        kernels_to_test = [
            ('power_law', {'p1': 3e-2, 'p2': 1.0, 'g': 1000}),
            ('stress_based', {'critical_stress': 1e6, 'weibull_modulus': 2.0}),
        ]
        
        for name, params in kernels_to_test:
            kernel = get_breakage_kernel(name, **params)
            assert kernel.name == name
            assert kernel.category == 'breakage'
        
        print_test("Breakage factory", True)
    except Exception as e:
        print_test("Breakage factory", False, str(e))
        all_passed = False
    
    # Test porosity growth factory
    try:
        from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel
        
        kernels_to_test = [
            ('volume_mixing', {}),
            ('incomplete_mixing', {'trapped_pore_fraction': 0.15}),
        ]
        
        for name, params in kernels_to_test:
            kernel = get_porosity_growth_kernel(name, **params)
            assert kernel.name == name
            assert kernel.category == 'porosity_growth'
        
        print_test("Porosity growth factory", True)
    except Exception as e:
        print_test("Porosity growth factory", False, str(e))
        all_passed = False
    
    # Test compression factory
    try:
        from wmcpbe.kernels.compression import get_compression_kernel
        
        kernels_to_test = [
            ('exponential_decay', {'rate': 0.02, 'min_porosity': 0.3}),
            ('stress_compaction', {'rate': 0.01, 'min_porosity': 0.3}),
        ]
        
        for name, params in kernels_to_test:
            kernel = get_compression_kernel(name, **params)
            assert kernel.name == name
            assert kernel.category == 'compression'
        
        print_test("Compression factory", True)
    except Exception as e:
        print_test("Compression factory", False, str(e))
        all_passed = False
    
    # Test liquid distribution factory
    try:
        from wmcpbe.kernels.liquid_distribution import get_liquid_distribution_kernel
        
        kernels_to_test = [
            ('uniform_weighted', {}),
            ('surface_weighted', {}),
            ('saturation_preferential', {'target_saturation': 0.5}),
        ]
        
        for name, params in kernels_to_test:
            kernel = get_liquid_distribution_kernel(name, **params)
            assert kernel.name == name
            assert kernel.category == 'liquid_distribution'
        
        print_test("Liquid distribution factory", True)
    except Exception as e:
        print_test("Liquid distribution factory", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# Test 3: Kernel computations
# =============================================================================

def test_computations():
    """Test that kernel computation methods work correctly."""
    print_header("TEST 3: Kernel Computations")
    
    all_passed = True
    
    # Test aggregation kernels
    try:
        from wmcpbe.kernels.aggregation import get_aggregation_kernel
        
        r1, r2 = 1e-6, 2e-6  # 1 μm and 2 μm radii
        
        # Shear kernel
        kernel = get_aggregation_kernel('shear_chin1998', corr_beta=1e-3, g=1000)
        beta = kernel.compute_beta(r1, r2)
        assert beta > 0, "Shear kernel returned non-positive beta"
        assert np.isfinite(beta), "Shear kernel returned non-finite beta"
        
        # Brownian kernel
        kernel = get_aggregation_kernel('brownian_tsouris1995')
        beta = kernel.compute_beta(r1, r2)
        assert beta > 0, "Brownian kernel returned non-positive beta"
        
        # Constant kernel
        kernel = get_aggregation_kernel('constant', corr_beta=1e-15)
        beta = kernel.compute_beta(r1, r2)
        assert abs(beta - 1e-15) < 1e-30, "Constant kernel returned wrong value"
        
        # Liquid bridge kernel (without solver - should degrade gracefully)
        kernel = get_aggregation_kernel('liquid_bridge', corr_beta=1e-3, g=1000)
        beta = kernel.compute_beta(r1, r2)
        assert beta > 0, "Liquid bridge kernel returned non-positive beta (no solver)"
        
        print_test("Aggregation computations", True)
    except Exception as e:
        print_test("Aggregation computations", False, str(e))
        all_passed = False
    
    # Test breakage kernels
    try:
        from wmcpbe.kernels.breakage import get_breakage_kernel
        
        v_particle = 1e-18  # 1 μm³
        
        # Power law kernel
        kernel = get_breakage_kernel('power_law', p1=3e-2, p2=1.0, g=1000)
        rate = kernel.compute_rate(v_particle)
        assert rate >= 0, "Power law kernel returned negative rate"
        assert np.isfinite(rate), "Power law kernel returned non-finite rate"
        
        # Stress-based kernel
        kernel = get_breakage_kernel('stress_based', critical_stress=1e6, weibull_modulus=2.0)
        rate = kernel.compute_rate(v_particle)
        assert rate >= 0, "Stress-based kernel returned negative rate"
        
        print_test("Breakage computations", True)
    except Exception as e:
        print_test("Breakage computations", False, str(e))
        all_passed = False
    
    # Test porosity growth kernels
    try:
        from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel
        
        # Volume mixing
        kernel = get_porosity_growth_kernel('volume_mixing')
        v_dry, poro = kernel.compute_merged_porosity(
            v_dry1=1e-18, poro1=0.4,
            v_dry2=2e-18, poro2=0.5
        )
        assert v_dry > 0, "Volume mixing returned non-positive dry volume"
        assert 0 <= poro <= 1, "Volume mixing returned invalid porosity"
        
        # Vollkörper case
        v_dry, poro = kernel.compute_merged_porosity(
            v_dry1=1e-18, poro1=np.nan,
            v_dry2=2e-18, poro2=np.nan
        )
        assert np.isnan(poro), "Vollkörper+Vollkörper should give Vollkörper"
        
        # Incomplete mixing
        kernel = get_porosity_growth_kernel('incomplete_mixing', trapped_pore_fraction=0.1)
        v_dry, poro = kernel.compute_merged_porosity(
            v_dry1=1e-18, poro1=0.4,
            v_dry2=2e-18, poro2=0.5
        )
        assert poro > 0.4, "Incomplete mixing should give higher porosity than volume mixing"
        
        print_test("Porosity growth computations", True)
    except Exception as e:
        print_test("Porosity growth computations", False, str(e))
        all_passed = False
    
    # Test compression kernels
    try:
        from wmcpbe.kernels.compression import get_compression_kernel
        
        # Exponential decay
        kernel = get_compression_kernel('exponential_decay', rate=0.02, min_porosity=0.3)
        poro_new = kernel.compute_porosity_decay(porosity=0.5, dt=1.0)
        assert poro_new < 0.5, "Exponential decay should reduce porosity"
        assert poro_new >= 0.3, "Porosity should not go below minimum"
        
        # No change for Vollkörper
        poro_new = kernel.compute_porosity_decay(porosity=np.nan, dt=1.0)
        assert np.isnan(poro_new), "Vollkörper should not be compressed"
        
        # Stress compaction
        kernel = get_compression_kernel('stress_compaction', rate=0.01, min_porosity=0.3)
        poro_new = kernel.compute_porosity_decay(
            porosity=0.5, dt=1.0,
            v_particle=1e-18
        )
        assert poro_new < 0.5, "Stress compaction should reduce porosity"
        
        print_test("Compression computations", True)
    except Exception as e:
        print_test("Compression computations", False, str(e))
        all_passed = False
    
    # Test liquid distribution kernels (without full solver mock)
    try:
        from wmcpbe.kernels.liquid_distribution import get_liquid_distribution_kernel
        
        # Just test instantiation and parameter validation
        kernel = get_liquid_distribution_kernel('uniform_weighted')
        assert kernel.name == 'uniform_weighted'
        
        kernel = get_liquid_distribution_kernel('surface_weighted')
        assert kernel.name == 'surface_weighted'
        
        kernel = get_liquid_distribution_kernel(
            'saturation_preferential',
            target_saturation=0.6,
            preference_strength=2.0
        )
        assert kernel.s_target == 0.6
        
        print_test("Liquid distribution initialization", True)
    except Exception as e:
        print_test("Liquid distribution initialization", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# Test 4: Parameter validation
# =============================================================================

def test_validation():
    """Test that parameter validation works correctly."""
    print_header("TEST 4: Parameter Validation")
    
    all_passed = True
    
    # Test invalid parameters
    try:
        from wmcpbe.kernels.aggregation import get_aggregation_kernel
        
        # Negative corr_beta should fail
        try:
            kernel = get_aggregation_kernel('shear_chin1998', corr_beta=-1e-3)
            print_test("Reject negative corr_beta", False, "Should have raised ValueError")
            all_passed = False
        except ValueError:
            print_test("Reject negative corr_beta", True)
        
        # Invalid optimal_saturation should fail
        try:
            kernel = get_aggregation_kernel('liquid_bridge', optimal_saturation=1.5)
            print_test("Reject invalid optimal_saturation", False, "Should have raised ValueError")
            all_passed = False
        except ValueError:
            print_test("Reject invalid optimal_saturation", True)
        
    except Exception as e:
        print_test("Parameter validation", False, str(e))
        all_passed = False
    
    # Test compression validation
    try:
        from wmcpbe.kernels.compression import get_compression_kernel
        
        # Negative rate should fail
        try:
            kernel = get_compression_kernel('exponential_decay', rate=-0.02)
            print_test("Reject negative compression rate", False, "Should have raised ValueError")
            all_passed = False
        except ValueError:
            print_test("Reject negative compression rate", True)
        
        # Invalid min_porosity should fail
        try:
            kernel = get_compression_kernel('exponential_decay', min_porosity=1.5)
            print_test("Reject invalid min_porosity", False, "Should have raised ValueError")
            all_passed = False
        except ValueError:
            print_test("Reject invalid min_porosity", True)
        
    except Exception as e:
        print_test("Compression validation", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# Test 5: List available kernels
# =============================================================================

def test_list_kernels():
    """Test that kernel listing functions work."""
    print_header("TEST 5: List Available Kernels")
    
    all_passed = True
    
    try:
        from wmcpbe.kernels.aggregation import list_aggregation_kernels
        agg_kernels = list_aggregation_kernels()
        assert len(agg_kernels) >= 4, f"Expected at least 4 agg kernels, got {len(agg_kernels)}"
        assert 'shear_chin1998' in agg_kernels
        assert 'liquid_bridge' in agg_kernels
        print(f"  Aggregation kernels: {', '.join(agg_kernels)}")
        print_test("List aggregation kernels", True)
    except Exception as e:
        print_test("List aggregation kernels", False, str(e))
        all_passed = False
    
    try:
        from wmcpbe.kernels.breakage import list_breakage_kernels
        break_kernels = list_breakage_kernels()
        assert len(break_kernels) >= 1, f"Expected at least 1 break kernel, got {len(break_kernels)}"
        print(f"  Breakage kernels: {', '.join(break_kernels)}")
        print_test("List breakage kernels", True)
    except Exception as e:
        print_test("List breakage kernels", False, str(e))
        all_passed = False
    
    try:
        from wmcpbe.kernels.porosity_growth import list_porosity_growth_kernels
        poro_kernels = list_porosity_growth_kernels()
        assert len(poro_kernels) >= 1, f"Expected at least 1 poro kernel, got {len(poro_kernels)}"
        print(f"  Porosity growth kernels: {', '.join(poro_kernels)}")
        print_test("List porosity growth kernels", True)
    except Exception as e:
        print_test("List porosity growth kernels", False, str(e))
        all_passed = False
    
    try:
        from wmcpbe.kernels.compression import list_compression_kernels
        comp_kernels = list_compression_kernels()
        assert len(comp_kernels) >= 1, f"Expected at least 1 comp kernel, got {len(comp_kernels)}"
        print(f"  Compression kernels: {', '.join(comp_kernels)}")
        print_test("List compression kernels", True)
    except Exception as e:
        print_test("List compression kernels", False, str(e))
        all_passed = False
    
    try:
        from wmcpbe.kernels.liquid_distribution import list_liquid_distribution_kernels
        liq_kernels = list_liquid_distribution_kernels()
        assert len(liq_kernels) >= 1, f"Expected at least 1 liq kernel, got {len(liq_kernels)}"
        print(f"  Liquid distribution kernels: {', '.join(liq_kernels)}")
        print_test("List liquid distribution kernels", True)
    except Exception as e:
        print_test("List liquid distribution kernels", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# Main test runner
# =============================================================================

def main():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print(" WMCPBE KERNEL FRAMEWORK - TEST SUITE")
    print("=" * 70)
    print("\nThis script tests the new kernel architecture.")
    print("All tests must pass before integrating into MCPBESolver.\n")
    
    results = []
    
    # Run all tests
    results.append(("Module Imports", test_imports()))
    results.append(("Factory Functions", test_factories()))
    results.append(("Kernel Computations", test_computations()))
    results.append(("Parameter Validation", test_validation()))
    results.append(("List Kernels", test_list_kernels()))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  [{status}] {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! Kernel framework is ready for integration.")
        print("\n  Next steps:")
        print("  1. Integrate kernels into MCPBESolver.__init__()")
        print("  2. Update _compute_beta(), _compute_break_rate_single(), etc.")
        print("  3. Run existing validation scripts to ensure backward compatibility")
        return 0
    else:
        print(f"\n  ⚠ {total - passed} test(s) failed. Please fix issues before integration.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
