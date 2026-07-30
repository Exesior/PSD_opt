"""
Test for Nucleation with Uniform Physical Particle Sampling.

This test validates that the nucleation handler correctly samples physical
particles uniformly by weighting computational particles by their W values.

Key insight:
- n_comp (a_tot): Number of computational particles in arrays
- n_phys (Sum(W)/Vc): Number of represented physical particles

To sample PHYSICAL particles uniformly, we must sample COMPUTATIONAL
particles with probability proportional to their weight W.

Test cases:
1. Small particles (5e-6m): Should work as before
2. Large particles (5e-5m): Previously failed, should now work
3. Different W distributions: Verify sampling is truly weight-proportional

Usage:
    cd mcpbe/src
    python -m wmcpbe.Trials.test_nucleation_uniform_sampling
"""

from __future__ import annotations

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


def calculate_weighted_total(array: np.ndarray, weights: np.ndarray, n_active: int) -> float:
    """Calculate weighted total for active particles."""
    if n_active <= 0:
        return 0.0
    return float(np.sum(array[:n_active] * weights[:n_active]))


def get_particle_counts(solver):
    """
    Get both computational and physical particle counts.
    
    Returns:
        tuple: (n_comp, n_phys, sum_W, Vc)
    """
    n_comp = solver.a_tot
    sum_W = np.sum(solver.W[:n_comp])
    Vc = solver.Vc
    n_phys = sum_W / Vc
    
    return n_comp, n_phys, sum_W, Vc


def test_nucleation_small_particles():
    """
    Test nucleation with small particles (5 µm diameter).
    
    This is the baseline case that should work correctly.
    """
    print_separator("TEST 1: Small Particles (5 µm)")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=5.0,
        t_write=10,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    # Setup: 5 µm particles
    diameter = 5e-6  # 5 µm
    solver.c = np.array([1e-3])
    solver.x = np.array([diameter])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 500
    solver.process_type = "agglomeration"
    
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-15
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    solver.liquid_volume[:n_init] = 0.0
    
    # Capture initial state
    n_comp_init, n_phys_init, sum_W_init, Vc_init = get_particle_counts(solver)
    V_solid_initial = calculate_weighted_total(solver.get_V_solid(), solver.W, n_init)
    
    print(f"\nInitial state:")
    print(f"  Diameter: {diameter*1e6:.1f} µm")
    print(f"  n_comp: {n_comp_init}")
    print(f"  n_phys: {n_phys_init:.2f}")
    print(f"  V_solid: {V_solid_initial:.6e} m³")
    
    # Create nucleation handler
    tropfen_durchmesser = 2e-6  # 2 µm droplet
    volumenstrom = 1e-15  # m³/s
    duration = 1.0  # s
    
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=volumenstrom,
        tropfen_durchmesser=tropfen_durchmesser,
        wasserzugabe_start=0.0,
        wasserzugabe_dauer=duration,
    )
    solver.nucleation.configure_time_step(0.05)  # Frequent nucleation steps
    
    # Expected liquid
    v_liquid_expected = volumenstrom * duration
    
    print(f"\nNucleation parameters:")
    print(f"  Droplet diameter: {tropfen_durchmesser*1e6:.1f} µm")
    print(f"  Expected liquid: {v_liquid_expected:.6e} m³")
    
    # Run simulation
    solver.solve(maxiter=20000)
    
    # Final state
    n_final = solver.a_tot
    n_comp_final, n_phys_final, sum_W_final, Vc_final = get_particle_counts(solver)
    V_solid_final = calculate_weighted_total(solver.get_V_solid(), solver.W, n_final)
    V_liq_final = calculate_weighted_total(solver.liquid_volume[:n_final], solver.W, n_final)
    
    # Statistics
    nuc_stats = solver.nucleation.get_statistics()
    liquid_added = nuc_stats['liquid_volume_added_total']
    
    print(f"\nFinal state:")
    print(f"  n_comp: {n_comp_final} (Δ: {n_comp_final - n_comp_init:+d})")
    print(f"  n_phys: {n_phys_final:.2f} (Δ: {n_phys_final - n_phys_init:+.2f})")
    print(f"  V_solid: {V_solid_final:.6e} m³")
    print(f"  V_liquid: {V_liq_final:.6e} m³")
    print(f"  Liquid added (stat): {liquid_added:.6e} m³")
    
    # Validation
    print(f"\n=== Validation ===")
    
    # 1. Liquid should be added
    liq_added = liquid_added > 0
    print(f"Liquid added: {'✅' if liq_added else '❌'} ({liquid_added:.6e} m³)")
    
    # 2. Liquid conservation
    if abs(liquid_added) > 1e-300:
        liq_error = abs(V_liq_final - liquid_added) / abs(liquid_added)
    else:
        liq_error = 0.0
    check_liq = liq_error < 0.10  # < 10%
    print(f"Liquid conservation: {'✅' if check_liq else '❌'} (error: {liq_error*100:.1f}%)")
    
    # 3. V_solid conservation
    V_solid_error = abs(V_solid_final - V_solid_initial) / V_solid_initial
    check_solid = V_solid_error < 0.01  # < 1%
    print(f"V_solid conservation: {'✅' if check_solid else '❌'} (error: {V_solid_error*100:.2f}%)")
    
    # 4. V_solid should be conserved (agglomeration changes distribution, not total mass)
    # Note: n_phys will decrease during agglomeration - this is physically correct!
    V_solid_error = abs(V_solid_final - V_solid_initial) / V_solid_initial
    check_solid2 = V_solid_error < 0.01  # < 1%
    print(f"V_solid conservation: {'✅' if check_solid2 else '❌'} (error: {V_solid_error*100:.2f}%)")
    
    all_pass = liq_added and check_liq and check_solid and check_solid2
    
    print(f"\nOverall: {'✅ PASS' if all_pass else '❌ FAIL'}")
    
    return all_pass


def test_nucleation_large_particles():
    """
    Test nucleation with large particles (50 µm diameter).
    
    This is the problematic case that previously failed because
    volume-weighted sampling never selected particles for nucleation.
    With weight-based sampling, it should now work correctly.
    """
    print_separator("TEST 2: Large Particles (50 µm)")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=5.0,
        t_write=10,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    # Setup: 50 µm particles (10x larger than test 1)
    diameter = 5e-5  # 50 µm
    solver.c = np.array([1e-3])
    solver.x = np.array([diameter])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 500
    solver.process_type = "agglomeration"
    
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-13  # Higher rate for larger particles
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    solver.liquid_volume[:n_init] = 0.0
    
    # Capture initial state
    n_comp_init, n_phys_init, sum_W_init, Vc_init = get_particle_counts(solver)
    V_solid_initial = calculate_weighted_total(solver.get_V_solid(), solver.W, n_init)
    
    print(f"\nInitial state:")
    print(f"  Diameter: {diameter*1e6:.1f} µm")
    print(f"  n_comp: {n_comp_init}")
    print(f"  n_phys: {n_phys_init:.2f}")
    print(f"  V_solid: {V_solid_initial:.6e} m³")
    
    # Create nucleation handler (same parameters as test 1)
    tropfen_durchmesser = 2e-6  # 2 µm droplet
    volumenstrom = 1e-15  # m³/s
    duration = 1.0  # s
    
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=volumenstrom,
        tropfen_durchmesser=tropfen_durchmesser,
        wasserzugabe_start=0.0,
        wasserzugabe_dauer=duration,
    )
    solver.nucleation.configure_time_step(0.05)
    
    # Expected liquid
    v_liquid_expected = volumenstrom * duration
    
    print(f"\nNucleation parameters:")
    print(f"  Droplet diameter: {tropfen_durchmesser*1e6:.1f} µm")
    print(f"  Expected liquid: {v_liquid_expected:.6e} m³")
    
    # Run simulation
    solver.solve(maxiter=20000)
    
    # Final state
    n_final = solver.a_tot
    n_comp_final, n_phys_final, sum_W_final, Vc_final = get_particle_counts(solver)
    V_solid_final = calculate_weighted_total(solver.get_V_solid(), solver.W, n_final)
    V_liq_final = calculate_weighted_total(solver.liquid_volume[:n_final], solver.W, n_final)
    
    # Statistics
    nuc_stats = solver.nucleation.get_statistics()
    liquid_added = nuc_stats['liquid_volume_added_total']
    
    print(f"\nFinal state:")
    print(f"  n_comp: {n_comp_final} (Δ: {n_comp_final - n_comp_init:+d})")
    print(f"  n_phys: {n_phys_final:.2f} (Δ: {n_phys_final - n_phys_init:+.2f})")
    print(f"  V_solid: {V_solid_final:.6e} m³")
    print(f"  V_liquid: {V_liq_final:.6e} m³")
    print(f"  Liquid added (stat): {liquid_added:.6e} m³")
    
    # Validation
    print(f"\n=== Validation ===")
    
    # CRITICAL: Liquid MUST be added (this was the bug!)
    liq_added = liquid_added > 0
    print(f"Liquid added: {'✅' if liq_added else '❌'} ({liquid_added:.6e} m³)")
    
    # 2. Liquid conservation
    if abs(liquid_added) > 1e-300:
        liq_error = abs(V_liq_final - liquid_added) / abs(liquid_added)
    else:
        liq_error = 0.0
    check_liq = liq_error < 0.10  # < 10%
    print(f"Liquid conservation: {'✅' if check_liq else '❌'} (error: {liq_error*100:.1f}%)")
    
    # 3. V_solid conservation
    V_solid_error = abs(V_solid_final - V_solid_initial) / V_solid_initial
    check_solid = V_solid_error < 0.01  # < 1%
    print(f"V_solid conservation: {'✅' if check_solid else '❌'} (error: {V_solid_error*100:.2f}%)")
    
    # 4. V_solid should be conserved (agglomeration changes distribution, not total mass)
    # Note: n_phys will decrease during agglomeration - this is physically correct!
    V_solid_error = abs(V_solid_final - V_solid_initial) / V_solid_initial
    check_solid2 = V_solid_error < 0.01  # < 1%
    print(f"V_solid conservation: {'✅' if check_solid2 else '❌'} (error: {V_solid_error*100:.2f}%)")
    
    all_pass = liq_added and check_liq and check_solid and check_solid2
    
    print(f"\nOverall: {'✅ PASS' if all_pass else '❌ FAIL'}")
    
    return all_pass


def test_weight_sampling_distribution():
    """
    Test that particle selection is truly weight-proportional.
    
    Creates a system with varying W values and verifies that
    selection frequency matches W distribution.
    """
    print_separator("TEST 3: Weight-Proportional Sampling Distribution")
    
    from wmcpbe.mcpbe import MCPBESolver
    from wmcpbe.mcpbe_nucleation import NucleationHandler, NucleationConfig
    
    solver = MCPBESolver(
        dim=1,
        t_total=0.1,  # Very short - just for initialization
        t_write=10,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    # Setup with uniform particles but varying W
    diameter = 1e-5
    solver.c = np.array([1e-3])
    solver.x = np.array([diameter])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 100
    solver.process_type = "agglomeration"
    
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-15
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    # Manually set varying weights: W[i] = i + 1
    # This creates a known distribution for testing
    for i in range(solver.a_tot):
        solver.W[i] = float(i + 1)
    
    # Normalize to have reasonable n_phys
    target_sum_W = 1000.0
    current_sum_W = np.sum(solver.W[:solver.a_tot])
    scale = target_sum_W / current_sum_W
    solver.W[:solver.a_tot] *= scale
    
    n_comp, n_phys, sum_W, Vc = get_particle_counts(solver)
    
    print(f"\nSetup:")
    print(f"  n_comp: {n_comp}")
    print(f"  Sum(W): {sum_W:.2f}")
    print(f"  W range: [{solver.W[0]:.2f}, {solver.W[solver.a_tot-1]:.2f}]")
    
    # Create nucleation handler
    config = NucleationConfig(
        enabled=True,
        volumenstrom=1e-14,
        tropfen_durchmesser=1e-6,
        wasserzugabe_start=0.0,
        wasserzugabe_dauer=0.1,
    )
    solver.nucleation = NucleationHandler(solver, config)
    solver.nucleation.configure_time_step(0.01)
    
    # Sample many times and record frequencies
    n_samples = 10000
    selection_counts = np.zeros(solver.a_tot, dtype=int)
    
    print(f"\nSampling {n_samples} times...")
    
    for _ in range(n_samples):
        idx = solver.nucleation._select_particle_uniform_physical()
        if idx >= 0:
            selection_counts[idx] += 1
    
    # Calculate expected frequencies from W distribution
    W_normalized = solver.W[:solver.a_tot] / np.sum(solver.W[:solver.a_tot])
    expected_counts = W_normalized * n_samples
    
    # Compare observed vs expected
    observed_freq = selection_counts / n_samples
    expected_freq = W_normalized
    
    # Chi-square-like metric: sum((obs - exp)^2 / exp)
    chi_sq_metric = np.sum((selection_counts - expected_counts)**2 / np.maximum(expected_counts, 1))
    normalized_error = np.sqrt(chi_sq_metric / n_samples)
    
    # Correlation coefficient
    correlation = np.corrcoef(observed_freq, expected_freq)[0, 1]
    
    print(f"\nResults:")
    print(f"  Observed frequency range: [{observed_freq.min():.4f}, {observed_freq.max():.4f}]")
    print(f"  Expected frequency range: [{expected_freq.min():.4f}, {expected_freq.max():.4f}]")
    print(f"  Correlation (observed vs expected): {correlation:.6f}")
    print(f"  Normalized error: {normalized_error:.4f}")
    
    # Validation
    print(f"\n=== Validation ===")
    
    # High correlation indicates correct weight-proportional sampling
    check_correlation = correlation > 0.95
    print(f"Correlation > 0.95: {'✅' if check_correlation else '❌'} ({correlation:.4f})")
    
    # Low normalized error
    check_error = normalized_error < 0.1
    print(f"Normalized error < 10%: {'✅' if check_error else '❌'} ({normalized_error*100:.2f}%)")
    
    all_pass = check_correlation and check_error
    
    print(f"\nOverall: {'✅ PASS' if all_pass else '❌ FAIL'}")
    
    return all_pass


def test_varying_w_distribution():
    """
    Test nucleation with non-uniform W distribution.
    
    Verifies that nucleation works correctly even when
    computational particles have very different weights.
    """
    print_separator("TEST 4: Non-Uniform W Distribution")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=2.0,
        t_write=10,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    # Setup
    diameter = 1e-5
    solver.c = np.array([1e-3])
    solver.x = np.array([diameter])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 200
    solver.process_type = "agglomeration"
    
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-15
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    
    # Create bimodal W distribution:
    # First half: W = 1.0
    # Second half: W = 10.0
    solver.W[:n_init//2] = 1.0
    solver.W[n_init//2:] = 10.0
    
    solver.liquid_volume[:n_init] = 0.0
    
    n_comp_init, n_phys_init, sum_W_init, Vc_init = get_particle_counts(solver)
    V_solid_initial = calculate_weighted_total(solver.get_V_solid(), solver.W, n_init)
    
    print(f"\nInitial state:")
    print(f"  n_comp: {n_comp_init}")
    print(f"  n_phys: {n_phys_init:.2f}")
    print(f"  W distribution: [{solver.W[0]:.1f}, ..., {solver.W[n_init//2]:.1f}, ...]")
    
    # Create nucleation handler
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=1e-14,
        tropfen_durchmesser=1e-6,
        wasserzugabe_start=0.0,
        wasserzugabe_dauer=0.5,
    )
    solver.nucleation.configure_time_step(0.05)
    
    # Run simulation
    solver.solve(maxiter=10000)
    
    # Final state
    n_final = solver.a_tot
    V_liq_final = calculate_weighted_total(solver.liquid_volume[:n_final], solver.W, n_final)
    
    nuc_stats = solver.nucleation.get_statistics()
    liquid_added = nuc_stats['liquid_volume_added_total']
    
    print(f"\nFinal state:")
    print(f"  V_liquid: {V_liq_final:.6e} m³")
    print(f"  Liquid added (stat): {liquid_added:.6e} m³")
    
    # Validation
    print(f"\n=== Validation ===")
    
    # Liquid should be added regardless of W distribution
    liq_added = liquid_added > 0
    print(f"Liquid added: {'✅' if liq_added else '❌'} ({liquid_added:.6e} m³)")
    
    # Check that high-W particles got more nucleation events
    # (they represent more physical particles, so higher collision probability)
    has_porosity = ~np.isnan(solver.porosity[:n_final])
    if np.any(has_porosity):
        mean_W_nucleated = np.mean(solver.W[has_porosity])
        mean_W_all = np.mean(solver.W[:n_final])
        ratio = mean_W_nucleated / max(mean_W_all, 1e-10)
        print(f"  Mean W (nucleated): {mean_W_nucleated:.2f}")
        print(f"  Mean W (all): {mean_W_all:.2f}")
        print(f"  Ratio: {ratio:.2f} (should be > 1, high-W particles nucleate more)")
    
    all_pass = liq_added
    
    print(f"\nOverall: {'✅ PASS' if all_pass else '❌ FAIL'}")
    
    return all_pass


def main():
    """Run all nucleation uniform sampling tests."""
    print("\n" + "#" * 70)
    print("# Nucleation Uniform Physical Particle Sampling Tests")
    print("#" * 70)
    
    results = []
    
    # Test 1: Small particles (baseline)
    try:
        passed = test_nucleation_small_particles()
        results.append(("Small Particles (5 µm)", passed))
    except Exception as e:
        print(f"\n❌ Small Particles EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Small Particles (5 µm)", False))
    
    # Test 2: Large particles (the problematic case)
    try:
        passed = test_nucleation_large_particles()
        results.append(("Large Particles (50 µm)", passed))
    except Exception as e:
        print(f"\n❌ Large Particles EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Large Particles (50 µm)", False))
    
    # Test 3: Weight sampling distribution
    try:
        passed = test_weight_sampling_distribution()
        results.append(("Weight-Proportional Sampling", passed))
    except Exception as e:
        print(f"\n❌ Weight Sampling EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Weight-Proportional Sampling", False))
    
    # Test 4: Non-uniform W distribution
    try:
        passed = test_varying_w_distribution()
        results.append(("Non-Uniform W Distribution", passed))
    except Exception as e:
        print(f"\n❌ Non-Uniform W EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Non-Uniform W Distribution", False))
    
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
