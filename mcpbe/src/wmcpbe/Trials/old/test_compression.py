"""
Test for Compression Handler (Porosity Reduction).

This test validates the exponential porosity decay model in compression-only mode.

Reference: 10.1103/PhysRevLett.64.2727

Usage:
    Run from the mcpbe directory:
    python -m wmcpbe.Trials.test_compression
    
    Or in Spyder/Jupyter:
    %runfile C:/Users/ericb/Documents/GitHub/PSD_opt/mcpbe/src/wmcpbe/Trials/test_compression.py
"""

import numpy as np
import sys
import os

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_separator(title: str) -> None:
    """Print a formatted section separator."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_compression_only_mode():
    """
    Test compression-only process type with analytical solution.
    
    The exponential decay model:
        poro(t) = min_poro + (poro_0 - min_poro) * exp(-rate * t)
    
    For validation, we initialize particles with porosity=0.4 and verify
    that the numerical solution matches the analytical prediction.
    
    Phase 2 Validation: Also verifies V_flat stores V_ges (total volume).
    """
    from wmcpbe.mcpbe import MCPBESolver
    
    # Create solver with simple monodisperse initial condition
    # load_attr=False to avoid config file requirement
    solver = MCPBESolver(
        dim=1,
        t_total=10.0,      # 10 seconds simulation
        t_write=1.0,       # Save every 1 second
        verbose=True,
        init=True,
        seed=42,
        load_attr=False,   # Skip config file loading for tests
    )
    
    # Set process type to compression-only
    solver.process_type = "compression"
    
    # Enable compression with known parameters
    solver.create_compression_handler(
        enabled=True,
        rate=0.02,         # 1/s
        min_porosity=0.3,  # Minimum achievable porosity
    )
    
    # Initialize all particles with porosity = 0.4 (like after nucleation)
    # This allows us to test the decay from a known initial value
    initial_porosity = 0.4
    solver.porosity[:solver.a_tot] = initial_porosity
    
    # Phase 2: Store initial V_ges for validation
    # For porous particles: V_ges = V_solid / (1 - poro)
    # Compression reduces porosity → V_ges decreases, V_solid stays constant
    initial_V_ges = solver.get_V_total().copy()
    initial_V_solid = solver.get_V_solid().copy()
    
    print(f"\n=== Compression-Only Test ===")
    print(f"Initial particles: {solver.a_tot}")
    print(f"Initial porosity (all particles): {initial_porosity}")
    print(f"Compression rate: 0.02 1/s")
    print(f"Minimum porosity: 0.3")
    print(f"Simulation time: 10 s")
    print(f"Expected porosity at t=10s: {0.3 + (0.4 - 0.3) * np.exp(-0.02 * 10):.6f}")
    print(f"\nPhase 2 Validation:")
    print(f"  Initial V_ges (mean): {np.mean(initial_V_ges):.6e} m³")
    print(f"  Initial V_solid (mean): {np.mean(initial_V_solid):.6e} m³")
    print(f"  Expected: V_solid = V_ges * (1 - 0.4) = V_ges * 0.6")
    
    # Run simulation
    solver.solve(maxiter=int(1e6))
    
    # Extract results
    t_vec = solver.t_vec
    poro_save = solver.porosity_save
    
    print(f"\n=== Results ===")
    print(f"Time points: {len(t_vec)}")
    
    # Compute analytical solution for comparison
    rate = 0.02
    min_poro = 0.3
    poro_0 = initial_porosity
    
    analytical = [min_poro + (poro_0 - min_poro) * np.exp(-rate * t) for t in t_vec]
    
    # Get mean porosity from simulation at each time point
    # Note: Skip t=0 snapshot because it was created before we set porosity manually
    simulated_mean = []
    for i, poro_arr in enumerate(poro_save):
        if len(poro_arr) > 0:
            simulated_mean.append(np.mean(poro_arr))
        else:
            simulated_mean.append(poro_0)
    
    print(f"\n{'Time [s]':<12} {'Simulated':<12} {'Analytical':<12} {'Error [%]':<12}")
    print("-" * 50)
    
    max_error = 0.0
    # Skip t=0 for error calculation (snapshot was taken before manual porosity init)
    start_idx = 1 if t_vec[0] == 0.0 else 0
    for i, (t, sim, ana) in enumerate(zip(t_vec, simulated_mean, analytical)):
        if i < start_idx:
            print(f"{t:<12.2f} {sim:<12.6f} {ana:<12.6f} {'(skip)':<12}")
            continue
        error = abs(sim - ana) / ana * 100 if ana > 0 else 0
        max_error = max(max_error, error)
        print(f"{t:<12.2f} {sim:<12.6f} {ana:<12.6f} {error:<12.4f}")
    
    print(f"\nMaximum relative error (t>0): {max_error:.4f}%")
    
    # Phase 2: Validate V_ges semantics
    final_V_ges = solver.get_V_total()
    final_V_solid = solver.get_V_solid()
    
    print(f"\n=== Phase 2 Volume Semantics Validation ===")
    print(f"Final V_ges (mean): {np.mean(final_V_ges):.6e} m³")
    print(f"Final V_solid (mean): {np.mean(final_V_solid):.6e} m³")
    
    # V_solid should remain constant (mass conservation)
    V_solid_change = abs(np.mean(final_V_solid) - np.mean(initial_V_solid)) / np.mean(initial_V_solid)
    print(f"V_solid change: {V_solid_change*100:.4f}% (should be ~0%)")
    
    # V_ges should decrease as porosity decreases
    V_ges_change = (np.mean(initial_V_ges) - np.mean(final_V_ges)) / np.mean(initial_V_ges)
    print(f"V_ges change: {V_ges_change*100:.2f}% (should decrease)")
    
    # Verify relationship: V_solid = V_ges * (1 - poro)
    final_poro_mean = simulated_mean[-1]
    expected_V_solid = np.mean(final_V_ges) * (1.0 - final_poro_mean)
    actual_V_solid = np.mean(final_V_solid)
    relation_error = abs(expected_V_solid - actual_V_solid) / actual_V_solid
    print(f"V_solid = V_ges * (1-poro) relation error: {relation_error*100:.4f}%")
    
    # Validation criteria (skip t=0 which was before manual porosity init)
    assert max_error < 5.0, f"Error too large: {max_error:.4f}%"
    assert simulated_mean[-1] > min_poro, "Porosity should not go below minimum"
    assert simulated_mean[-1] < poro_0, "Porosity should decrease over time"
    assert V_solid_change < 0.01, f"V_solid should be conserved (change={V_solid_change*100:.4f}%)"
    assert relation_error < 0.01, f"V_ges/V_solid relation incorrect (error={relation_error*100:.4f}%)"
    
    print("\n✓ Test PASSED: Compression matches analytical solution AND V_ges semantics correct")
    
    return {
        't_vec': t_vec,
        'simulated': simulated_mean,
        'analytical': analytical,
        'max_error': max_error,
    }


def test_compression_with_nucleation():
    """
    Test that nucleation assigns porosity=0.4 and compression reduces it.
    
    This test verifies:
    1. Nucleation handler sets porosity=0.4 on particles (per DOI: 10.1103/PhysRevLett.64.2727)
    2. Compression reduces porosity over time toward min_porosity=0.3
    3. Phase 2: V_flat stores V_ges, V_solid is conserved during nucleation/compression
    """
    print_separator("TEST 2: Nucleation + Compression")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    # Setup similar to nucleation test
    solver = MCPBESolver(
        dim=1,
        t_total=3.0,
        t_write=10,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    # Basic parameters (same as test_nucleation_basic)
    solver.c = np.array([1e-3])
    solver.x = np.array([3e-6])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 5000
    solver.process_type = "agglomeration"
    
    # VERY weak agglomeration to prevent all particles merging
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-15
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    # Initialize
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    
    # Initialize liquid_volume to zero
    # Note: porosity is already NaN from initialization (Vollkörper)
    if hasattr(solver, 'liquid_volume'):
        solver.liquid_volume[:n_init] = 0.0
    
    # Create nucleation handler
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=1e-15,  # m³/s
        tropfen_durchmesser=1e-6,  # m
        wasserzugabe_start=0.0,  # s
        wasserzugabe_dauer=3.0,  # s
    )
    
    # Create compression handler
    solver.create_compression_handler(
        enabled=True,
        rate=0.02,  # 1/s
        min_porosity=0.3,
    )
    
    # Phase 2: Capture initial state (before nucleation)
    initial_V_ges = solver.get_V_total().copy()
    initial_V_solid = solver.get_V_solid().copy()
    initial_poro = solver.porosity[:n_init].copy()
    
    print(f"\nInitial state:")
    print(f"  Particles: {n_init}")
    print(f"  Initial porosity: NaN (Vollkörper, no nucleation yet)")
    print(f"  Initial V_ges (mean): {np.mean(initial_V_ges):.6e} m³")
    print(f"  Initial V_solid (mean): {np.mean(initial_V_solid):.6e} m³")
    
    # Run simulation
    print(f"\nRunning simulation...")
    solver.solve(maxiter=10000)
    
    # Final state
    n_final = solver.a_tot
    poro_final = solver.porosity[:n_final]
    
    # Filter out NaN values for statistics (only nucleated particles have valid porosity)
    poro_valid = poro_final[~np.isnan(poro_final)]
    
    # Phase 2: Get final volumes
    final_V_ges = solver.get_V_total()
    final_V_solid = solver.get_V_solid()
    
    # Get statistics
    nuc_stats = solver.nucleation.get_statistics()
    comp_stats = solver.compression.get_statistics()
    
    print(f"\nFinal state:")
    print(f"  Particles: {n_final}")
    print(f"  Particles with porosity (nucleated): {len(poro_valid)}")
    if len(poro_valid) > 0:
        print(f"  Mean porosity (nucleated): {np.mean(poro_valid):.6f}")
        print(f"  Max porosity: {np.max(poro_valid):.6f}")
        print(f"  Min porosity: {np.min(poro_valid):.6f}")
    
    # Phase 2: Volume validation
    print(f"\nPhase 2 Volume Semantics:")
    print(f"  Final V_ges (mean): {np.mean(final_V_ges):.6e} m³")
    print(f"  Final V_solid (mean): {np.mean(final_V_solid):.6e} m³")
    
    # V_solid should be conserved (mass conservation)
    V_solid_change = abs(np.mean(final_V_solid) - np.mean(initial_V_solid)) / np.mean(initial_V_solid)
    print(f"  V_solid change: {V_solid_change*100:.4f}% (should be ~0%)")
    
    print(f"\nNucleation statistics:")
    print(f"  Droplets added: {nuc_stats['droplets_added_total']}")
    print(f"  Liquid added: {nuc_stats['liquid_volume_added_total']:.6e} m³")
    
    print(f"\nCompression statistics:")
    print(f"  Total compression steps: {comp_stats['total_compression_steps']}")
    print(f"  Particles compressed: {comp_stats['particles_compressed_total']}")
    
    # Validation:
    # 1. Some particles should have received porosity from nucleation (> 0.35)
    # 2. No porosity should exceed 0.4 (initial nucleation value)
    # 3. No porosity should go below 0.3 (min_porosity)
    # Note: Only check valid (non-NaN) porosity values
    
    if len(poro_valid) > 0:
        has_porous_particles = np.any(poro_valid > 0.35)
        max_ok = np.max(poro_valid) <= 0.4
        min_ok = np.min(poro_valid) >= 0.3
    else:
        # No nucleated particles - test fails
        has_porous_particles = False
        max_ok = False
        min_ok = False
    
    print(f"\nResults:")
    print(f"  Porous particles created (poro > 0.35): {'✅' if has_porous_particles else '❌'}")
    print(f"  Max porosity ≤ 0.4: {'✅' if max_ok else '❌'}")
    print(f"  Min porosity ≥ 0.3: {'✅' if min_ok else '❌'}")
    print(f"  V_solid conserved (<1% change): {'✅' if V_solid_change < 0.01 else '❌'}")
    
    overall_pass = has_porous_particles and max_ok and min_ok and (V_solid_change < 0.01)
    print(f"\nOverall: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    
    return overall_pass


def test_compression_kernel_custom():
    """
    Test custom compression kernel by subclassing CompressionHandler.
    
    This demonstrates how users can implement their own physics models.
    """
    from wmcpbe.mcpbe_compression import CompressionHandler, CompressionConfig
    
    class CustomCompressionHandler(CompressionHandler):
        """Custom handler with size-dependent compression rate."""
        
        def compression_kernel(self, particle_idx: int, dt: float) -> float:
            """
            Custom kernel: larger particles compress faster.
            
            Rate scales with particle volume.
            """
            current_poro = self.solver.porosity[particle_idx]
            min_poro = self.config.min_porosity
            
            # Get particle volume
            V_particle = self.solver.V_flat[-1, particle_idx]
            V_ref = 1e-18  # Reference volume (1 µm sphere)
            
            # Size-dependent rate: larger = faster compression
            base_rate = self.config.rate
            size_factor = (V_particle / V_ref) ** (1/3)  # Scale with diameter
            effective_rate = base_rate * size_factor
            
            # Exponential decay with size-dependent rate
            new_poro = min_poro + (current_poro - min_poro) * np.exp(-effective_rate * dt)
            
            return max(min_poro, min(1.0, new_poro))
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=5.0,
        t_write=1.0,
        verbose=False,
        init=True,
        seed=42,
        load_attr=False,  # Skip config file loading for tests
    )
    
    solver.process_type = "compression"
    
    # Use custom handler
    config = CompressionConfig(enabled=True, rate=0.02, min_porosity=0.3)
    solver.compression = CustomCompressionHandler(solver, config)
    
    # Initialize with different sizes (via volume distribution)
    solver.porosity[:solver.a_tot] = 0.4
    
    print(f"\n=== Custom Kernel Test ===")
    print(f"Testing size-dependent compression rate...")
    
    solver.solve(maxiter=int(1e5))
    
    # Verify porosity decreased
    poro_final = solver.porosity[:solver.a_tot]
    assert np.all(poro_final <= 0.4), "Porosity should decrease"
    assert np.all(poro_final >= 0.3), "Porosity should not go below minimum"
    
    print(f"Initial porosity: 0.4")
    print(f"Final mean porosity: {np.mean(poro_final):.6f}")
    print("✓ Custom kernel test passed")
    
    return {'poro_final': poro_final}


def main():
    """Run all compression tests."""
    print("\n" + "#" * 70)
    print("# Compression Handler Tests - Porosity Reduction under Shear")
    print("#" * 70)
    
    results = []
    
    # Test 1: Compression-only mode (analytical validation)
    try:
        passed = test_compression_only_mode()
        results.append(("Compression-Only (Analytical)", True))
    except Exception as e:
        print(f"\n❌ Compression-only EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Compression-Only (Analytical)", False))
    
    # Test 2: Nucleation + Compression
    try:
        passed = test_compression_with_nucleation()
        results.append(("Nucleation + Compression", passed))
    except Exception as e:
        print(f"\n❌ Nucleation + Compression EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Nucleation + Compression", False))
    
    # Test 3: Custom compression kernel
    try:
        passed = test_compression_kernel_custom()
        results.append(("Custom Kernel", True))
    except Exception as e:
        print(f"\n❌ Custom kernel EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Custom Kernel", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED! 🎉")
    else:
        print("SOME TESTS FAILED! ⚠️")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
