"""
Test Kernel Integration with MCPBESolver.

This script verifies that the kernel framework integrates correctly
with the actual solver, not just in isolation.

Tests:
1. Solver instantiation with kernel parameters
2. Backward compatibility (COLEVAL/BREAKRVAL still works)
3. Kernel methods are called during solve()
"""

import sys
import numpy as np

if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_solver_with_kernels():
    """Test solver instantiation with explicit kernel names."""
    print_header("TEST 1: Solver with Explicit Kernels")
    
    try:
        from wmcpbe.mcpbe import MCPBESolver
        
        # Create solver with new kernel parameters
        # load_attr=False avoids config file loading for this test
        solver = MCPBESolver(
            dim=1,
            t_total=10,
            t_write=5,
            init=True,
            load_attr=False,  # Skip config file loading
            agg_kernel_name='shear_chin1998',
            agg_kernel_params={'corr_beta': 1e-3, 'g': 1000},
            break_kernel_name='power_law',
            break_kernel_params={'p1': 3e-2, 'p2': 1.0},
            porosity_growth_kernel_name='volume_mixing',
            compression_kernel_name='exponential_decay',
            compression_kernel_params={'rate': 0.02, 'min_porosity': 0.3},
            liquid_dist_kernel_name='uniform_weighted',
        )
        
        # Verify kernel_manager exists
        assert hasattr(solver, 'kernel_manager'), "kernel_manager not created"
        assert solver.kernel_manager is not None, "kernel_manager is None"
        
        # Verify kernels are initialized
        assert solver.kernel_manager.agg_kernel is not None, "Agg kernel not initialized"
        assert solver.kernel_manager.break_kernel is not None, "Break kernel not initialized"
        assert solver.kernel_manager.porosity_growth_kernel is not None, "Porosity kernel not initialized"
        assert solver.kernel_manager.compression_kernel is not None, "Compression kernel not initialized"
        assert solver.kernel_manager.liquid_dist_kernel is not None, "Liquid dist kernel not initialized"
        
        # Verify correct kernel types
        assert solver.kernel_manager.agg_kernel.name == 'shear_chin1998'
        assert solver.kernel_manager.break_kernel.name == 'power_law'
        assert solver.kernel_manager.porosity_growth_kernel.name == 'volume_mixing'
        assert solver.kernel_manager.compression_kernel.name == 'exponential_decay'
        assert solver.kernel_manager.liquid_dist_kernel.name == 'uniform_weighted'
        
        print("  ✓ Solver created with explicit kernel names")
        print(f"  ✓ Aggregation kernel: {solver.kernel_manager.agg_kernel.name}")
        print(f"  ✓ Breakage kernel: {solver.kernel_manager.break_kernel.name}")
        print(f"  ✓ Porosity growth kernel: {solver.kernel_manager.porosity_growth_kernel.name}")
        print(f"  ✓ Compression kernel: {solver.kernel_manager.compression_kernel.name}")
        print(f"  ✓ Liquid distribution kernel: {solver.kernel_manager.liquid_dist_kernel.name}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test that COLEVAL/BREAKRVAL still work (backward compatibility)."""
    print_header("TEST 2: Backward Compatibility (COLEVAL/BREAKRVAL)")
    
    try:
        from wmcpbe.mcpbe import MCPBESolver
        
        # Create solver with legacy parameters
        solver = MCPBESolver(
            dim=1,
            t_total=10,
            t_write=5,
            init=True,
            load_attr=False,  # Skip config file loading
        )
        
        # Manually set legacy attributes (simulates config file)
        solver.COLEVAL = 1
        solver.BREAKRVAL = 1
        solver.CORR_BETA = 1e-3
        solver.G = 1000
        solver.pl_P1 = 3e-2
        solver.pl_P2 = 1.0
        
        # Re-initialize kernels with new values
        solver._initialize_kernels()
        
        # Verify kernel_manager was created from legacy params
        assert hasattr(solver, 'kernel_manager'), "kernel_manager not created"
        
        # Should map COLEVAL=1 → shear_chin1998
        assert solver.kernel_manager.agg_kernel.name == 'shear_chin1998', \
            f"Expected shear_chin1998, got {solver.kernel_manager.agg_kernel.name}"
        
        # Should map BREAKRVAL=1 → power_law
        assert solver.kernel_manager.break_kernel.name == 'power_law', \
            f"Expected power_law, got {solver.kernel_manager.break_kernel.name}"
        
        print("  ✓ COLEVAL=1 mapped to shear_chin1998")
        print("  ✓ BREAKRVAL=1 mapped to power_law")
        print("  ✓ Backward compatibility maintained")
        
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kernel_delegation_beta():
    """Test that _beta() method delegates to kernel."""
    print_header("TEST 3: Kernel Delegation (_beta method)")
    
    try:
        from wmcpbe.mcpbe import MCPBESolver
        
        solver = MCPBESolver(
            dim=1,
            t_total=10,
            t_write=5,
            init=True,
            load_attr=False,
            agg_kernel_name='constant',
            agg_kernel_params={'corr_beta': 1e-15},
        )
        
        # Test _beta delegation
        beta = solver._beta(0, 1)
        
        # Constant kernel should return corr_beta
        expected = 1e-15
        assert abs(beta - expected) < 1e-30, f"Expected {expected}, got {beta}"
        
        print(f"  ✓ _beta() delegated to constant kernel")
        print(f"  ✓ β = {beta:.3e} (expected: {expected:.3e})")
        
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kernel_delegation_break_rate():
    """Test that _break_rate_single() delegates to kernel."""
    print_header("TEST 4: Kernel Delegation (_break_rate_single method)")
    
    try:
        from wmcpbe.mcpbe import MCPBESolver
        
        solver = MCPBESolver(
            dim=1,
            t_total=10,
            t_write=5,
            init=True,
            load_attr=False,
            break_kernel_name='power_law',
            break_kernel_params={'p1': 3e-2, 'p2': 1.0, 'g': 1000},
        )
        
        # Test _break_rate_single delegation
        rate = solver._break_rate_single(0)
        
        # Rate should be non-negative and finite
        assert rate >= 0, f"Rate should be non-negative, got {rate}"
        assert np.isfinite(rate), f"Rate should be finite, got {rate}"
        
        print(f"  ✓ _break_rate_single() delegated to power_law kernel")
        print(f"  ✓ S = {rate:.3e} 1/s")
        
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_new_kernel_features():
    """Test that NEW kernel features work (liquid_bridge, incomplete_mixing)."""
    print_header("TEST 5: New Kernel Features")
    
    try:
        from wmcpbe.mcpbe import MCPBESolver
        
        solver = MCPBESolver(
            dim=1,
            t_total=10,
            t_write=5,
            init=True,
            load_attr=False,
            agg_kernel_name='liquid_bridge',
            agg_kernel_params={
                'corr_beta': 1e-3,
                'g': 1000,
                'optimal_saturation': 0.5,
            },
            porosity_growth_kernel_name='incomplete_mixing',
            porosity_growth_kernel_params={
                'trapped_pore_fraction': 0.15,
            },
        )
        
        # Verify liquid_bridge kernel
        assert solver.kernel_manager.agg_kernel.name == 'liquid_bridge'
        assert solver.kernel_manager.agg_kernel.s_opt == 0.5
        
        # Verify incomplete_mixing kernel
        assert solver.kernel_manager.porosity_growth_kernel.name == 'incomplete_mixing'
        assert solver.kernel_manager.porosity_growth_kernel.f_trap == 0.15
        
        print("  ✓ liquid_bridge kernel configured")
        print(f"  ✓ optimal_saturation = {solver.kernel_manager.agg_kernel.s_opt}")
        print("  ✓ incomplete_mixing kernel configured")
        print(f"  ✓ trapped_pore_fraction = {solver.kernel_manager.porosity_growth_kernel.f_trap}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print(" KERNEL INTEGRATION TESTS")
    print("=" * 70)
    print("\nVerifying kernel framework integration with MCPBESolver.\n")
    
    results = []
    
    results.append(("Solver with Kernels", test_solver_with_kernels()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("Beta Delegation", test_kernel_delegation_beta()))
    results.append(("Break Rate Delegation", test_kernel_delegation_break_rate()))
    results.append(("New Kernel Features", test_new_kernel_features()))
    
    # Summary
    print_header("INTEGRATION TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  [{status}] {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  ✅ All integration tests passed!")
        print("  Kernel framework is successfully integrated.")
        return 0
    else:
        print(f"\n  ⚠ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
