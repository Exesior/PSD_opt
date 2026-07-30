"""
Additional Verification: Kernel Integration with Solver Context.

This script verifies that kernels work correctly in a solver-like context.
It creates mock solver objects to test kernel methods that require solver access.
"""

import sys
import numpy as np

if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockSolver:
    """Minimal mock of MCPBESolver for kernel testing."""
    
    def __init__(self):
        # Basic attributes
        self.a_tot = 10
        self.W = np.array([1.0] * 20)  # Weights
        self.X = np.array([1e-6 * (i + 1) for i in range(20)])  # Radii
        self.G = 1000  # Shear rate
        
        # Volume arrays (simplified)
        self.V_flat = np.zeros((3, 20))
        self.V_flat[0, :self.a_tot] = 1e-18 * np.arange(1, self.a_tot + 1)
        
        # Saturation (for liquid distribution tests)
        self.saturation = np.array([0.3] * 20)
        
        # RNG
        self._rng = np.random.default_rng(seed=42)
    
    def _select_particle_uniform_physical(self):
        """Mock weight-based selection."""
        W = self.W[:self.a_tot]
        probs = W / np.sum(W)
        return int(self._rng.choice(self.a_tot, p=probs))


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_aggregation_with_solver():
    """Test aggregation kernels with mock solver."""
    print_header("Aggregation Kernels with Solver Context")
    
    from wmcpbe.kernels.aggregation import get_aggregation_kernel
    
    solver = MockSolver()
    all_passed = True
    
    # Test each kernel with solver reference
    kernels_to_test = [
        ('shear_chin1998', {'corr_beta': 1e-3, 'g': 1000}),
        ('brownian_tsouris1995', {}),
        ('constant', {'corr_beta': 1e-15}),
        ('sum', {'corr_beta': 1e-9}),
        ('liquid_bridge', {'corr_beta': 1e-3, 'g': 1000, 'optimal_saturation': 0.5}),
    ]
    
    for name, params in kernels_to_test:
        try:
            kernel = get_aggregation_kernel(name, **params)
            
            # Test with solver reference
            beta = kernel.compute_beta(
                r1=1e-6, r2=2e-6,
                particle1_idx=0,
                particle2_idx=1,
                solver=solver
            )
            
            assert beta > 0, f"{name}: beta should be positive"
            assert np.isfinite(beta), f"{name}: beta should be finite"
            
            print(f"  ✓ {name}: β = {beta:.3e} m³/s")
            
        except Exception as e:
            print(f"  ✗ {name}: FAILED - {e}")
            all_passed = False
    
    return all_passed


def test_breakage_with_solver():
    """Test breakage kernels with mock solver."""
    print_header("Breakage Kernels with Solver Context")
    
    from wmcpbe.kernels.breakage import get_breakage_kernel
    
    solver = MockSolver()
    all_passed = True
    
    kernels_to_test = [
        ('power_law', {'p1': 3e-2, 'p2': 1.0, 'g': 1000}),
        ('stress_based', {'critical_stress': 1e6, 'weibull_modulus': 2.0}),
    ]
    
    for name, params in kernels_to_test:
        try:
            kernel = get_breakage_kernel(name, **params)
            
            # Test with solver reference
            rate = kernel.compute_rate(
                v_particle=1e-18,
                particle_idx=0,
                solver=solver
            )
            
            assert rate >= 0, f"{name}: rate should be non-negative"
            assert np.isfinite(rate), f"{name}: rate should be finite"
            
            print(f"  ✓ {name}: S = {rate:.3e} 1/s")
            
        except Exception as e:
            print(f"  ✗ {name}: FAILED - {e}")
            all_passed = False
    
    return all_passed


def test_porosity_with_solver():
    """Test porosity growth kernels with collision energy."""
    print_header("Porosity Growth Kernels with Collision Energy")
    
    from wmcpbe.kernels.porosity_growth import get_porosity_growth_kernel
    
    solver = MockSolver()
    all_passed = True
    
    # Estimate collision energy (simplified)
    # E_coll ~ 0.5 × m × v², v ~ G × d
    rho = 1000  # kg/m³
    d = 2e-6
    v_rel = solver.G * d
    m = rho * (4/3 * np.pi * (d/2)**3)
    E_coll = 0.5 * m * v_rel**2
    
    kernels_to_test = [
        ('volume_mixing', {}),
        ('incomplete_mixing', {'trapped_pore_fraction': 0.15, 'energy_dependent': True}),
    ]
    
    for name, params in kernels_to_test:
        try:
            kernel = get_porosity_growth_kernel(name, **params)
            
            # Test with collision energy
            v_dry, poro = kernel.compute_merged_porosity(
                v_dry1=1e-18, poro1=0.4,
                v_dry2=2e-18, poro2=0.5,
                v_liq1=1e-19, v_liq2=2e-19,
                sat1=0.3, sat2=0.4,
                collision_energy=E_coll,
                solver=solver
            )
            
            assert v_dry > 0, f"{name}: v_dry should be positive"
            assert 0 <= poro <= 1 or np.isnan(poro), f"{name}: invalid porosity"
            
            print(f"  ✓ {name}: V_dry = {v_dry:.3e} m³, ε = {poro:.3f}")
            
        except Exception as e:
            print(f"  ✗ {name}: FAILED - {e}")
            all_passed = False
    
    return all_passed


def test_compression_with_solver():
    """Test compression kernels with solver context."""
    print_header("Compression Kernels with Stress Estimation")
    
    from wmcpbe.kernels.compression import get_compression_kernel
    
    solver = MockSolver()
    all_passed = True
    
    kernels_to_test = [
        ('exponential_decay', {'rate': 0.02, 'min_porosity': 0.3}),
        ('stress_compaction', {'rate': 0.01, 'min_porosity': 0.3, 'stress_exponent': 1.5}),
    ]
    
    for name, params in kernels_to_test:
        try:
            kernel = get_compression_kernel(name, **params)
            
            # Test with particle volume and solver for stress estimation
            poro_new = kernel.compute_porosity_decay(
                porosity=0.5,
                dt=1.0,
                v_particle=1e-18,
                local_stress=None,  # Let kernel estimate from solver
                saturation=0.4,
                solver=solver
            )
            
            assert poro_new < 0.5, f"{name}: porosity should decrease"
            assert poro_new >= 0.3 or np.isnan(poro_new), f"{name}: below min_porosity"
            
            print(f"  ✓ {name}: ε(1s) = {poro_new:.3f} (from 0.5)")
            
        except Exception as e:
            print(f"  ✗ {name}: FAILED - {e}")
            all_passed = False
    
    return all_passed


def test_liquid_distribution_with_solver():
    """Test liquid distribution kernels with actual particle selection."""
    print_header("Liquid Distribution Kernels with Particle Selection")
    
    from wmcpbe.kernels.liquid_distribution import get_liquid_distribution_kernel
    
    solver = MockSolver()
    all_passed = True
    
    kernels_to_test = [
        ('uniform_weighted', {}),
        ('surface_weighted', {}),
        ('saturation_preferential', {'target_saturation': 0.5, 'preference_strength': 2.0}),
    ]
    
    for name, params in kernels_to_test:
        try:
            kernel = get_liquid_distribution_kernel(name, **params)
            
            # Test particle selection multiple times
            selections = []
            for _ in range(5):
                idx = kernel.select_target_particle(
                    solver=solver,
                    v_droplet=1e-18,
                    current_time=0.0
                )
                selections.append(idx)
                assert 0 <= idx < solver.a_tot, f"{name}: invalid index {idx}"
            
            print(f"  ✓ {name}: Selected indices = {selections}")
            
        except Exception as e:
            print(f"  ✗ {name}: FAILED - {e}")
            all_passed = False
    
    return all_passed


def test_kernel_state_independence():
    """Verify kernels don't maintain unwanted state between calls."""
    print_header("Kernel State Independence")
    
    from wmcpbe.kernels.aggregation import get_aggregation_kernel
    
    kernel = get_aggregation_kernel('constant', corr_beta=1e-15)
    
    # Multiple calls should give same result
    results = [kernel.compute_beta(1e-6, 2e-6) for _ in range(10)]
    
    if len(set(results)) == 1:
        print("  ✓ Constant kernel is stateless (same result every time)")
        return True
    else:
        print(f"  ✗ Kernel appears to have state: {results}")
        return False


def main():
    """Run integration verification tests."""
    print("\n" + "=" * 70)
    print(" KERNEL INTEGRATION VERIFICATION")
    print("=" * 70)
    print("\nVerifying kernels work correctly with solver-like objects.\n")
    
    results = []
    
    results.append(("Aggregation + Solver", test_aggregation_with_solver()))
    results.append(("Breakage + Solver", test_breakage_with_solver()))
    results.append(("Porosity + Energy", test_porosity_with_solver()))
    results.append(("Compression + Stress", test_compression_with_solver()))
    results.append(("Liquid Dist + Selection", test_liquid_distribution_with_solver()))
    results.append(("State Independence", test_kernel_state_independence()))
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  [{status}] {name}")
    
    print(f"\n  Total: {passed}/{total} verifications passed")
    
    if passed == total:
        print("\n  ✅ All integration verifications passed!")
        print("  Kernels are ready for MCPBESolver integration.")
        return 0
    else:
        print(f"\n  ⚠ {total - passed} verification(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
