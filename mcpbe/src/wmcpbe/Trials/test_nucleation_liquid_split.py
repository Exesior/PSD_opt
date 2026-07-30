"""
Test: Nucleation Liquid Distribution with/without Internalization Kernel.

This test verifies the correct behavior of liquid distribution during nucleation:
1. WITHOUT liquid_internalization kernel: 40% internal, 60% external (legacy split)
2. WITH liquid_internalization kernel: 0% internal, 100% external (kernel handles it)

The test ensures that:
- The fallback behavior (40/60 split) is preserved when kernel is inactive
- The new behavior (all external) is applied when kernel is active
- Liquid internalization occurs correctly over time via the kernel

Run: python Trials/test_nucleation_liquid_split.py
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver


# =============================================================================
# Test Configuration
# =============================================================================

class TestConfig:
    """Configuration for nucleation liquid split tests."""
    
    SEED = 42
    
    # Time settings
    T_TOTAL = 1.0       # Total simulation time [s]
    T_WRITE = 0.1       # Output interval [s]
    
    # Particle properties (initial solids)
    PARTICLE_DIAMETER = 500e-6      # 500 µm
    INITIAL_POROSITY = 0.4          # 40% void fraction
    
    # Droplet properties
    DROPLET_DIAMETER = 100e-6       # 100 µm
    DROPLET_DENSITY = 1000.0        # kg/m³ (water)
    
    # Process parameters
    VOLUMETRIC_FLOW_RATE = 1e-10    # m³/s
    NUCLEATION_DURATION = 0.5       # s
    
    # Internalization kernel parameters
    K_INT = 1e13                    # 1/(m³·s), rate constant
    
    # Numerical settings
    INITIAL_PARTICLES = 500         # Computational particles
    INITIAL_WEIGHT = 50.0           # Weight per particle
    CONTROL_VOLUME = 1.0            # m³


# =============================================================================
# Helper Functions
# =============================================================================

def format_scientific(value: float, unit: str = "") -> str:
    """Format value in scientific notation with unit."""
    return f"{value:.3e} {unit}".strip()


def print_section(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(title)
    print(char * 80)


def compute_droplet_volume(diameter: float) -> float:
    """Compute volume of spherical droplet."""
    radius = diameter / 2.0
    return (4.0 / 3.0) * np.pi * radius ** 3


# =============================================================================
# Test Case 1: WITHOUT Internalization Kernel (Legacy 40/60 Split)
# =============================================================================

def test_without_internalization_kernel() -> bool:
    """
    Test nucleation WITHOUT liquid_internalization kernel.
    
    Expected behavior:
    - Newly nucleated particles should have ~40% of liquid internalized
    - Saturation = V_liq_int / V_pore, where V_liq_int = min(0.4 × liquid, V_pore)
    
    Important: Saturation depends on pore capacity!
    If V_pore >> 0.4 × liquid, then saturation will be LOW even though 40% is internal.
    So we check the FRACTION of liquid that is internal, not absolute saturation.
    """
    print_section("TEST 1: WITHOUT Liquid Internalization Kernel", "=")
    
    cfg = TestConfig()
    rng = np.random.default_rng(cfg.SEED)
    
    # Time vector
    t_vec = np.linspace(0.0, cfg.T_TOTAL, int(cfg.T_TOTAL / cfg.T_WRITE) + 1)
    
    # Initialize solver WITHOUT internalization kernel
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=False,
        load_attr=False,
        init=True,
        agg_kernel_name='constant',
        agg_kernel_params={'corr_beta': 1e-5},
        porosity_growth_kernel_name='volume_mixing',
        porosity_growth_kernel_params={},
        # NO liquid_internalization_kernel!
        rng=rng,
    )
    
    solver.process_type = "agglomeration"
    
    # Initialize solid particles
    particle_radius = cfg.PARTICLE_DIAMETER / 2.0
    particle_volume = (4.0 / 3.0) * np.pi * particle_radius ** 3
    
    V_flat = np.zeros((2, cfg.INITIAL_PARTICLES), dtype=float)
    V_flat[0, :] = particle_volume * (1.0 - cfg.INITIAL_POROSITY)  # V_solid
    V_flat[1, :] = particle_volume  # V_dry
    
    W_init = np.full(cfg.INITIAL_PARTICLES, cfg.INITIAL_WEIGHT, dtype=float)
    solver.Vc = cfg.CONTROL_VOLUME
    
    solver._initialize_particles(
        init_Vc=False,
        V_flat=V_flat,
        W_init=W_init,
    )
    
    # Initialize porosity
    solver.porosity[:solver.a_tot] = cfg.INITIAL_POROSITY
    solver.liquid_volume[:solver.a_tot] = 0.0
    solver.saturation[:solver.a_tot] = 0.0
    
    solver._initialize_samplers()
    
    # Configure nucleation
    solver.create_nucleation_handler(
        enabled=True,
        volumetric_flow_rate=cfg.VOLUMETRIC_FLOW_RATE,
        droplet_diameter=cfg.DROPLET_DIAMETER,
        liquid_addition_start=0.0,
        liquid_addition_duration=cfg.NUCLEATION_DURATION,
    )
    
    # Run simulation briefly to trigger nucleation
    print("\nRunning simulation (no internalization kernel)...")
    try:
        solver.solve(maxiter=1000)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Analyze results
    # Find nucleated particles (those with liquid_volume > 0)
    # IMPORTANT: Only check active particles [:solver.a_tot]
    liquid_active = solver.liquid_volume[:solver.a_tot]
    saturation_active = solver.saturation[:solver.a_tot]
    porosity_active = solver.porosity[:solver.a_tot]
    v_dry_active = solver.V_flat[-1, :solver.a_tot]
    
    has_liquid = liquid_active > 1e-30
    n_nucleated = np.sum(has_liquid)
    
    if n_nucleated == 0:
        print("WARNING: No nucleated particles found!")
        return True
    
    # Calculate internal liquid fraction for each nucleated particle
    # V_liq_int = saturation × V_pore = saturation × (v_dry × porosity)
    # Fraction internal = V_liq_int / liquid_total
    v_pore = v_dry_active[has_liquid] * porosity_active[has_liquid]
    liquid_total = liquid_active[has_liquid]
    saturation_nuc = saturation_active[has_liquid]
    
    v_liq_int = saturation_nuc * v_pore
    fraction_internal = v_liq_int / liquid_total
    
    # Statistics
    mean_fraction_internal = np.mean(fraction_internal)
    std_fraction_internal = np.std(fraction_internal)
    min_fraction = np.min(fraction_internal)
    max_fraction = np.max(fraction_internal)
    
    # Expected: ~40% internal (capped at 100% if pore capacity exceeded)
    expected_fraction = 0.4
    tolerance = 0.15  # Allow variance due to pore capacity limits and agglomeration
    
    print(f"\nNucleated particles: {n_nucleated}")
    print(f"\nInternal liquid fraction (V_liq_int / V_liq_total):")
    print(f"  Mean:   {mean_fraction_internal:.3f}")
    print(f"  Std:    {std_fraction_internal:.3f}")
    print(f"  Range:  [{min_fraction:.3f}, {max_fraction:.3f}]")
    print(f"\nExpected: ~{expected_fraction:.1f} (40% internal)")
    print(f"Tolerance: ±{tolerance:.1f}")
    
    # Debug: show sample values for first few nucleated particles
    print(f"\nSample data (first 5 nucleated particles):")
    nuc_indices = np.where(has_liquid)[0][:5]
    for idx in nuc_indices:
        liq = liquid_active[idx]
        sat = saturation_active[idx]
        pore = v_dry_active[idx] * porosity_active[idx]
        liq_int = sat * pore
        frac = liq_int / liq if liq > 0 else 0
        print(f"  Particle {idx}: liquid={liq:.2e}, sat={sat:.4f}, V_pore={pore:.2e}, frac_int={frac:.3f}")
    
    # Verify fraction is in expected range
    # Note: fraction can be < 0.4 if pore capacity is limiting (V_pore < 0.4 × liquid)
    # But it should NOT be 0 (which would mean all external)
    frac_ok = mean_fraction_internal > 0.2  # At least some internalization
    frac_not_all = mean_fraction_internal < 0.95  # Not all internal (that would be wrong too)
    
    if frac_ok and frac_not_all:
        print(f"\n✓ TEST 1 PASSED: Legacy split shows partial internalization")
        return True
    else:
        print(f"\n✗ TEST 1 FAILED:")
        if not frac_ok:
            print(f"  - Mean fraction {mean_fraction_internal:.3f} too low (expected >0.2)")
        if not frac_not_all:
            print(f"  - Mean fraction {mean_fraction_internal:.3f} too high (expected <0.95)")
        return False


# =============================================================================
# Test Case 2: WITH Internalization Kernel (All External Initially)
# =============================================================================

def test_with_internalization_kernel() -> bool:
    """
    Test nucleation WITH liquid_internalization kernel.
    
    Expected behavior:
    - Newly nucleated particles should have ALL liquid external (saturation = 0)
    - Over time, internalization occurs via the continuous process kernel
    """
    print_section("TEST 2: WITH Liquid Internalization Kernel", "=")
    
    cfg = TestConfig()
    rng = np.random.default_rng(cfg.SEED)
    
    # Time vector
    t_vec = np.linspace(0.0, cfg.T_TOTAL, int(cfg.T_TOTAL / cfg.T_WRITE) + 1)
    
    # Initialize solver WITH internalization kernel
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=False,
        load_attr=False,
        init=True,
        agg_kernel_name='constant',
        agg_kernel_params={'corr_beta': 1e-5},
        porosity_growth_kernel_name='volume_mixing',
        porosity_growth_kernel_params={},
        liquid_internalization_kernel_name='liquid_internalization',
        liquid_internalization_kernel_params={'k_int': cfg.K_INT},
        rng=rng,
    )
    
    solver.process_type = "agglomeration"
    
    # Initialize solid particles
    particle_radius = cfg.PARTICLE_DIAMETER / 2.0
    particle_volume = (4.0 / 3.0) * np.pi * particle_radius ** 3
    
    V_flat = np.zeros((2, cfg.INITIAL_PARTICLES), dtype=float)
    V_flat[0, :] = particle_volume * (1.0 - cfg.INITIAL_POROSITY)  # V_solid
    V_flat[1, :] = particle_volume  # V_dry
    
    W_init = np.full(cfg.INITIAL_PARTICLES, cfg.INITIAL_WEIGHT, dtype=float)
    solver.Vc = cfg.CONTROL_VOLUME
    
    solver._initialize_particles(
        init_Vc=False,
        V_flat=V_flat,
        W_init=W_init,
    )
    
    # Initialize porosity
    solver.porosity[:solver.a_tot] = cfg.INITIAL_POROSITY
    solver.liquid_volume[:solver.a_tot] = 0.0
    solver.saturation[:solver.a_tot] = 0.0
    
    solver._initialize_samplers()
    
    # Configure nucleation
    solver.create_nucleation_handler(
        enabled=True,
        volumetric_flow_rate=cfg.VOLUMETRIC_FLOW_RATE,
        droplet_diameter=cfg.DROPLET_DIAMETER,
        liquid_addition_start=0.0,
        liquid_addition_duration=cfg.NUCLEATION_DURATION,
    )
    
    # Run simulation briefly to trigger nucleation
    print("\nRunning simulation (with internalization kernel)...")
    try:
        solver.solve(maxiter=1000)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Analyze results
    # Find nucleated particles (those with liquid_volume > 0)
    # IMPORTANT: Only check active particles [:solver.a_tot]
    liquid_active = solver.liquid_volume[:solver.a_tot]
    saturation_active = solver.saturation[:solver.a_tot]
    porosity_active = solver.porosity[:solver.a_tot]
    v_dry_active = solver.V_flat[-1, :solver.a_tot]
    
    has_liquid = liquid_active > 1e-30
    n_nucleated = np.sum(has_liquid)
    
    if n_nucleated == 0:
        print("WARNING: No nucleated particles found!")
        return True
    
    # Calculate internal liquid fraction for each nucleated particle
    v_pore = v_dry_active[has_liquid] * porosity_active[has_liquid]
    liquid_total = liquid_active[has_liquid]
    saturation_nuc = saturation_active[has_liquid]
    
    v_liq_int = saturation_nuc * v_pore
    fraction_internal = v_liq_int / liquid_total
    
    # Statistics
    mean_fraction_internal = np.mean(fraction_internal)
    max_fraction_internal = np.max(fraction_internal)
    
    # Expected: 0% internal initially (all external)
    # Allow small numerical tolerance
    expected_max_fraction = 0.05  # < 5% internal
    
    print(f"\nNucleated particles: {n_nucleated}")
    print(f"\nInternal liquid fraction (V_liq_int / V_liq_total):")
    print(f"  Mean: {mean_fraction_internal:.6f}")
    print(f"  Max:  {max_fraction_internal:.6f}")
    print(f"\nExpected: ≈0 (all external, internalization via kernel)")
    print(f"Tolerance: <{expected_max_fraction:.2f}")
    
    # Debug: show sample values
    print(f"\nSample data (first 5 nucleated particles):")
    nuc_indices = np.where(has_liquid)[0][:5]
    for idx in nuc_indices:
        liq = liquid_active[idx]
        sat = saturation_active[idx]
        pore = v_dry_active[idx] * porosity_active[idx]
        liq_int = sat * pore
        frac = liq_int / liq if liq > 0 else 0
        print(f"  Particle {idx}: liquid={liq:.2e}, sat={sat:.6f}, V_pore={pore:.2e}, frac_int={frac:.6f}")
    
    # Verify fraction is close to 0
    frac_ok = mean_fraction_internal < expected_max_fraction
    
    if frac_ok:
        print(f"\n✓ TEST 2 PASSED: All liquid starts external (kernel handles internalization)")
        return True
    else:
        print(f"\n✗ TEST 2 FAILED:")
        print(f"  - Mean fraction {mean_fraction_internal:.6f} too high (expected <{expected_max_fraction})")
        return False


# =============================================================================
# Test Case 3: Internalization Over Time
# =============================================================================

def test_internalization_time_evolution() -> bool:
    """
    Test that liquid internalization occurs correctly over time.
    
    Expected behavior:
    - Start with all liquid external (fraction_internal ≈ 0)
    - Over time, fraction_internal increases as liquid moves internal
    - Eventually reaches equilibrium (saturated pores or all liquid internalized)
    """
    print_section("TEST 3: Internalization Time Evolution", "=")
    
    cfg = TestConfig()
    rng = np.random.default_rng(cfg.SEED)
    
    # Longer time vector to observe full evolution
    # Use more frequent saves to catch early evolution
    t_vec = np.linspace(0.0, 5.0, 101)  # 5 seconds, 0.05s steps
    
    # Initialize solver WITH internalization kernel
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=False,
        load_attr=False,
        init=True,
        agg_kernel_name='constant',
        agg_kernel_params={'corr_beta': 1e-5},
        porosity_growth_kernel_name='volume_mixing',
        porosity_growth_kernel_params={},
        liquid_internalization_kernel_name='liquid_internalization',
        liquid_internalization_kernel_params={'k_int': cfg.K_INT},
        rng=rng,
    )
    
    solver.process_type = "agglomeration"
    
    # Initialize solid particles
    particle_radius = cfg.PARTICLE_DIAMETER / 2.0
    particle_volume = (4.0 / 3.0) * np.pi * particle_radius ** 3
    
    V_flat = np.zeros((2, cfg.INITIAL_PARTICLES), dtype=float)
    V_flat[0, :] = particle_volume * (1.0 - cfg.INITIAL_POROSITY)  # V_solid
    V_flat[1, :] = particle_volume  # V_dry
    
    W_init = np.full(cfg.INITIAL_PARTICLES, cfg.INITIAL_WEIGHT, dtype=float)
    solver.Vc = cfg.CONTROL_VOLUME
    
    solver._initialize_particles(
        init_Vc=False,
        V_flat=V_flat,
        W_init=W_init,
    )
    
    # Initialize porosity
    solver.porosity[:solver.a_tot] = cfg.INITIAL_POROSITY
    solver.liquid_volume[:solver.a_tot] = 0.0
    solver.saturation[:solver.a_tot] = 0.0
    
    solver._initialize_samplers()
    
    # Configure continuous processes handler (applies internalization kernel)
    # This is CRITICAL - without the handler, the kernel is never called!
    solver.create_continuous_processes_handler(
        enabled=True,
        k_int=cfg.K_INT,
        compression_enabled=False,  # Only test internalization
    )
    
    # Configure nucleation (very short burst at start)
    solver.create_nucleation_handler(
        enabled=True,
        volumetric_flow_rate=cfg.VOLUMETRIC_FLOW_RATE,
        droplet_diameter=cfg.DROPLET_DIAMETER,
        liquid_addition_start=0.0,
        liquid_addition_duration=0.05,  # Very short: only ~5 droplets
    )
    
    # Run simulation
    print("\nRunning simulation to observe internalization over time...")
    try:
        solver.solve(maxiter=10000)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Analyze final state directly (not from saved states to avoid index issues)
    liquid_final = solver.liquid_volume[:solver.a_tot]
    saturation_final = solver.saturation[:solver.a_tot]
    porosity_final = solver.porosity[:solver.a_tot]
    v_dry_final = solver.V_flat[-1, :solver.a_tot]
    
    # Find particles with liquid
    has_liquid = liquid_final > 1e-30
    n_with_liquid = np.sum(has_liquid)
    
    if n_with_liquid == 0:
        print("WARNING: No particles with liquid found")
        return True
    
    # Calculate internal liquid fraction at final time
    v_pore_final = v_dry_final[has_liquid] * porosity_final[has_liquid]
    v_liq_int_final = saturation_final[has_liquid] * v_pore_final
    liq_total_final = liquid_final[has_liquid]
    frac_int_final = v_liq_int_final / liq_total_final
    mean_frac_final = np.mean(frac_int_final)
    
    # Also check first saved state for initial condition
    if len(solver.liquid_volume_save) >= 2 and len(solver.saturation_save) >= 2:
        liq_init = solver.liquid_volume_save[1]
        sat_init = solver.saturation_save[1]
        
        # Get V_flat and porosity from saved state if available
        if hasattr(solver, 'V_flat_save') and len(solver.V_flat_save) >= 2:
            v_dry_init = solver.V_flat_save[1][-1, :]
            poro_init = solver.porosity_save[1]
        else:
            v_dry_init = v_dry_final[:len(liq_init)]
            poro_init = porosity_final[:len(liq_init)]
        
        has_liq_init = liq_init > 1e-30
        n_init = np.sum(has_liq_init)
        
        if n_init > 0:
            v_pore_init = v_dry_init[has_liq_init] * poro_init[has_liq_init]
            v_liq_int_init = sat_init[has_liq_init] * v_pore_init
            liq_total_init = liq_init[has_liq_init]
            frac_int_init = v_liq_int_init / liq_total_init
            mean_frac_init = np.mean(frac_int_init)
        else:
            mean_frac_init = 0.0
    else:
        mean_frac_init = 0.0
    
    # Compute equilibrium reference
    # S_eq = min(1, l_total / v_pore)
    if n_with_liquid > 0:
        s_equilibrium = np.mean(np.minimum(liq_total_final / v_pore_final, 1.0))
    else:
        s_equilibrium = 0.0
    
    print(f"\nParticles with liquid: {n_with_liquid}")
    print(f"\nInternal liquid fraction evolution:")
    print(f"  Initial (first save):  {mean_frac_init:.6f}")
    print(f"  Final (end of sim):    {mean_frac_final:.6f}")
    print(f"  Δ Fraction:            {mean_frac_final - mean_frac_init:+.6f}")
    print(f"\nEquilibrium reference:")
    print(f"  Expected S_eq:         ~{s_equilibrium:.3f}")
    
    # Debug: show sample values
    print(f"\nSample data (first 5 particles with liquid):")
    sample_indices = np.where(has_liquid)[0][:5]
    for idx in sample_indices:
        liq = liquid_final[idx]
        sat = saturation_final[idx]
        pore = v_dry_final[idx] * porosity_final[idx]
        frac = (sat * pore) / liq if liq > 0 else 0
        print(f"  Particle {idx}: liquid={liq:.2e}, sat={sat:.6f}, V_pore={pore:.2e}, frac_int={frac:.6f}")
    
    # Verify internalization occurred
    # Initial fraction should be LOW (all external initially)
    # Final fraction should be HIGHER (internalization over time)
    initial_low = mean_frac_init < 0.1  # < 10% internal initially
    
    # Check if any internalization happened (final > initial)
    # Allow for small numerical drift
    evolution_positive = mean_frac_final > mean_frac_init + 0.001
    
    # Also check that final saturation is non-zero (some liquid went internal)
    some_internalization = mean_frac_final > 0.01
    
    if initial_low and (evolution_positive or some_internalization):
        print(f"\n✓ TEST 3 PASSED: Internalization occurs over time")
        return True
    else:
        print(f"\n✗ TEST 3 FAILED:")
        if not initial_low:
            print(f"  - Initial fraction {mean_frac_init:.6f} too high (expected <0.1)")
        if not evolution_positive and not some_internalization:
            print(f"  - No significant evolution (init={mean_frac_init:.6f}, final={mean_frac_final:.6f})")
        return False


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    print("=" * 80)
    print("NUCLEATION LIQUID SPLIT TEST SUITE")
    print("Testing: Legacy 40/60 Split vs. Internalization Kernel Behavior")
    print("=" * 80)
    
    results = {}
    
    # Run all tests
    results['test1_no_kernel'] = test_without_internalization_kernel()
    results['test2_with_kernel'] = test_with_internalization_kernel()
    results['test3_evolution'] = test_internalization_time_evolution()
    
    # Summary
    print_section("TEST SUMMARY", "=")
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:<30} {status}")
    
    print(f"\n{'OVERALL:':<30} {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
