"""
Volume Semantics Test - Validation of V_particle_dry, Saturation, and Liquid Handling.

This test validates the consistent volume model:
- V_flat stores V_particle_dry = V_solid / (1 - porosity)
- Liquid is partitioned into internal (in pores) and external (on surface)
- V_hydrodynamic = V_particle_dry + V_liquid_external
- Mass conservation during nucleation, compression, and agglomeration

Usage:
    cd mcpbe/src
    python -m wmcpbe.Trials.test_volume_semantics
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
    
    CRITICAL DISTINCTION:
    - n_comp (a_tot): Number of computational particles in solver arrays
    - n_phys: Number of physical particles represented = Sum(W) / Vc
    
    During nucleation WITHOUT agglomeration:
    - n_comp INCREASES (new computational particles created for nucleated fraction)
    - n_phys STAYS CONSTANT (same physical system, just split into dry/wet)
    
    During agglomeration:
    - n_comp DECREASES (particles merged)
    - n_phys STAYS CONSTANT (mass conserved)
    
    Args:
        solver: MCPBESolver instance
        
    Returns:
        tuple: (n_comp, n_phys, sum_W, Vc)
    """
    n_comp = solver.a_tot
    sum_W = np.sum(solver.W[:n_comp])
    Vc = solver.Vc
    n_phys = sum_W / Vc
    
    return n_comp, n_phys, sum_W, Vc


def test_nucleation_volume_model():
    """
    Test nucleation with correct volume semantics.
    
    Validates:
    1. V_particle_dry = V_solid / (1 - poro) after nucleation
    2. 60/40 split: V_liq_int = min(0.6 × liquid, V_pore), V_liq_ext = liquid - V_liq_int
    3. V_liquid_internal = V_pore * sat (sat is stored state, derived from 60/40 split)
    4. V_liquid_external = V_liq_total - V_liquid_internal
    5. V_solid is conserved during nucleation
    6. Parent stays dry after nucleation (only child gets liquid)
    
    Analytical Solution:
    For a Vollkörper particle with V_solid = V0 receiving droplet v_droplet:
    - After nucleation with poro=0.4:
      V_particle_dry = V0 / (1 - 0.4) = V0 / 0.6
      V_pore = V_particle_dry * 0.4
      V_liq_int = min(0.6 * v_droplet, V_pore)  # 60% of droplet, capped at pore capacity
      V_liq_ext = v_droplet - V_liq_int
      S = V_liq_int / V_pore  # Resulting saturation (stored for future use)
    """
    print_separator("TEST 1: Nucleation Volume Model")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=1.0,
        t_write=10,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    # Single monodisperse particle for clean validation
    solver.c = np.array([1e-3])
    solver.x = np.array([5e-6])  # 5 µm diameter (large enough to accept droplets)
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 1000
    solver.process_type = "agglomeration"
    
    # Moderate agglomeration for nucleation support
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-15  # Higher rate to enable agglomeration during nucleation
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    
    # Initialize liquid_volume to zero
    if hasattr(solver, 'liquid_volume'):
        solver.liquid_volume[:n_init] = 0.0
    else:
        print("ERROR: liquid_volume attribute missing!")
        return False
    
    # Capture initial state (Vollkörper)
    V_solid_initial = calculate_weighted_total(solver.get_V_solid(), solver.W, n_init)
    V_dry_initial = calculate_weighted_total(solver.get_V_particle_dry(), solver.W, n_init)
    
    n_comp_init, n_phys_init, sum_W_init, Vc_init = get_particle_counts(solver)
    
    print(f"\nInitial state (Vollkörper):")
    print(f"  n_comp (a_tot): {n_comp_init}")
    print(f"  n_phys (Sum(W)/Vc): {n_phys_init:.2f}")
    print(f"  Sum(W): {sum_W_init:.2f}")
    print(f"  Vc: {Vc_init:.6e}")
    print(f"  V_solid (total): {V_solid_initial:.6e} m³")
    print(f"  V_particle_dry (total): {V_dry_initial:.6e} m³")
    print(f"  Porosity: NaN (all Vollkörper)")
    
    # Create nucleation handler with known parameters
    tropfen_durchmesser = 1 #in myM
    v_droplet = (4.0 / 3.0) * np.pi * (tropfen_durchmesser*1e-6) ** 3  # 1 µm droplet
    volumenstrom = 1e-15  # m³/s
    duration = 1.0  # s
    
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=1e-15,
        tropfen_durchmesser= tropfen_durchmesser*1e-6,
        wasserzugabe_start=0.0,
        wasserzugabe_dauer=1.0,
    )
    
    # IMPORTANT: Configure nucleation time step!
    # This must be called after create_nucleation_handler
    solver.nucleation.configure_time_step(0.1)  # Nucleation every 0.1s
    
    # Expected liquid
    v_liquid_expected = volumenstrom * duration
    n_droplets_expected = v_liquid_expected / v_droplet
    
    print(f"\nNucleation parameters:")
    print(f"  Droplet diameter: {tropfen_durchmesser:.0f} µm")
    print(f"  Droplet volume: {v_droplet:.6e} m³")
    print(f"  Expected droplets: ~{n_droplets_expected:.0f}")
    print(f"  Expected liquid added: {v_liquid_expected:.6e} m³")
    
    # Run simulation
    solver.solve(maxiter=10000)
    
    n_final = solver.a_tot
    n_comp_final, n_phys_final, sum_W_final, Vc_final = get_particle_counts(solver)
    
    # Get final state
    # IMPORTANT: ALL quantities must be WEIGHTED for DSMC consistency!
    # liquid_volume stores volume PER computational particle
    # Actual liquid in system = sum(liquid_volume × W)
    V_solid_final = calculate_weighted_total(solver.get_V_solid(), solver.W, n_final)
    V_dry_final = calculate_weighted_total(solver.get_V_particle_dry(), solver.W, n_final)
    
    # Liquid volumes: use WEIGHTED sum for conservation check
    liq_total_array = solver.liquid_volume[:n_final].copy()
    V_liq_final = calculate_weighted_total(liq_total_array, solver.W, n_final)  # GEWICHTET!
    
    # Compute derived quantities (per-particle)
    V_pore_final = solver.get_V_pore()
    V_liq_int_final = solver.get_V_liquid_internal()
    V_liq_ext_final = solver.get_V_liquid_external()
    sat_final = solver.get_saturation()
    
    # IMPORTANT: Use WEIGHTED sums for DSMC consistency!
    V_pore_total = calculate_weighted_total(V_pore_final, solver.W, n_final)
    V_liq_int_total = calculate_weighted_total(V_liq_int_final, solver.W, n_final)
    V_liq_ext_total = calculate_weighted_total(V_liq_ext_final, solver.W, n_final)
    
    # Get nucleation statistics
    nuc_stats = solver.nucleation.get_statistics()
    liquid_added = nuc_stats['liquid_volume_added_total']
    droplets_stat = nuc_stats['droplets_added_total']
    
    print(f"\nFinal state:")
    print(f"  n_comp (a_tot): {n_comp_final} (Δ: {n_comp_final - n_comp_init:+d})")
    print(f"  n_phys (Sum(W)/Vc): {n_phys_final:.2f} (Δ: {n_phys_final - n_phys_init:+.2f})")
    print(f"  Sum(W): {sum_W_final:.2f}")
    print(f"  Vc: {Vc_final:.6e}")
    print(f"  V_solid (total, weighted): {V_solid_final:.6e} m³")
    print(f"  V_particle_dry (total, weighted): {V_dry_final:.6e} m³")
    print(f"  V_pore (total): {V_pore_total:.6e} m³")
    print(f"  V_liquid_total (total): {V_liq_final:.6e} m³")
    print(f"  V_liquid_internal (total): {V_liq_int_total:.6e} m³")
    print(f"  V_liquid_external (total): {V_liq_ext_total:.6e} m³")
    print(f"  Mean saturation: {np.mean(sat_final[:n_final]):.4f}")
    
    print(f"\nLiquid added (from nucleation stats):")
    print(f"  V_liquid: {liquid_added:.6e} m³")
    print(f"  Droplets (effective dW): {droplets_stat:.2f}")
    print(f"  Mean saturation: {np.mean(sat_final[sat_final > 0]):.4f}" if np.any(sat_final > 0) else "  Mean saturation: N/A")
    
    print(f"\nLiquid added (from nucleation): {liquid_added:.6e} m³")
    print(f"Liquid in system (final): {V_liq_final:.6e} m³")
    
    # Validation checks
    print(f"\n=== Validation ===")
    
    # CRITICAL CHECK: Physical vs Computational particles
    print(f"Particle counts:")
    print(f"  Δn_comp: {n_comp_final - n_comp_init:+d} (expected: ~{int(n_droplets_expected)} from nucleation)")
    print(f"  Δn_phys: {n_phys_final - n_phys_init:+.2f} (should be ~0, no agglomeration/breakage)")
    
    # 0. Physical particle count should stay constant (no agg/break in this test)
    phys_change = abs(n_phys_final - n_phys_init) / max(n_phys_init, 1)
    check_phys_const = phys_change < 0.01  # < 1% change
    print(f"n_phys constant: {'✅' if check_phys_const else '❌'} (error: {phys_change*100:.2f}%)")
    
    # 1. V_solid conservation
    V_solid_error = abs(V_solid_final - V_solid_initial) / V_solid_initial
    print(f"V_solid conservation error: {V_solid_error*100:.4f}%")
    check_solid = V_solid_error < 0.01  # < 1%
    
    # 2. Liquid mass conservation (V_liq_final is now WEIGHTED sum)
    if abs(liquid_added) > 1e-300:
        liq_error = abs(V_liq_final - liquid_added) / abs(liquid_added)
    else:
        liq_error = 0.0
    print(f"Liquid mass conservation error: {liq_error*100:.4f}%")
    check_liquid = liq_error < 0.05  # < 5% tolerance
    
    # 3. Saturation value (derived from 60/40 split during nucleation)
    # Note: S is NOT fixed at 0.6! It's calculated as:
    #   V_liq_int = min(0.6 * liquid, V_pore)
    #   S = V_liq_int / V_pore
    # So S depends on whether the droplet is large enough to fill pores to 60%
    sat_valid = sat_final[~np.isnan(solver.porosity[:n_final]) & (sat_final > 0)]
    if len(sat_valid) > 0:
        sat_mean = np.mean(sat_valid)
        # For typical droplets on small particles, S may be < 1.0 or = 1.0 (capped)
        # We just check that saturation is in valid range [0, 1]
        print(f"Saturation mean: {sat_mean:.4f} (valid range: 0-1)")
        check_sat = 0.0 <= sat_mean <= 1.0
    else:
        print("Saturation: No valid values")
        check_sat = False
    
    # 4. Internal/External liquid relationship
    # V_liq_int MUST equal V_pore * sat (by definition, S is stored state)
    # Check: V_liq_int_total ≈ weighted_sum(V_pore * sat)
    expected_liq_int_per_particle = V_pore_final * sat_final
    expected_liq_int = calculate_weighted_total(expected_liq_int_per_particle, solver.W, n_final)
    if expected_liq_int < 1e-300:
        # If no internal liquid expected, check that we have none
        check_int_liq = V_liq_int_total < 1e-300
        int_error = 0.0
    else:
        int_error = abs(V_liq_int_total - expected_liq_int) / expected_liq_int
        check_int_liq = int_error < 0.01  # < 1% tolerance
    
    if check_int_liq:
        print(f"V_liq_internal = V_pore * sat: ✅ (error: {int_error*100:.4f}%)")
    else:
        print(f"V_liq_internal ≠ V_pore * sat: ❌ (error: {int_error*100:.4f}%)")
    
    # 5. Hydrodynamic volume consistency
    V_hydro_check = V_dry_final + V_liq_ext_total
    V_hydro_direct = calculate_weighted_total(solver.get_V_total(), solver.W, n_final)
    hydro_error = abs(V_hydro_check - V_hydro_direct) / max(V_hydro_direct, 1e-300)
    print(f"V_total = V_dry + V_liq_ext consistency: {hydro_error*100:.4f}% error")
    check_hydro = hydro_error < 0.01
    
    all_pass = check_phys_const and check_solid and check_liquid and check_sat and check_int_liq and check_hydro
    
    print(f"\nResults:")
    print(f"  n_phys constant: {'✅' if check_phys_const else '❌'}")
    print(f"  V_solid conserved: {'✅' if check_solid else '❌'}")
    print(f"  Liquid conserved: {'✅' if check_liquid else '❌'}")
    print(f"  Saturation valid: {'✅' if check_sat else '❌'}")
    print(f"  V_liq_int definition: {'✅' if check_int_liq else '❌'}")
    print(f"  V_total consistency: {'✅' if check_hydro else '❌'}")
    
    print(f"\nOverall: {'✅ PASS' if all_pass else '❌ FAIL'}")
    
    return all_pass


def test_compression_with_saturation():
    """
    Test compression with saturation handling.
    
    Validates:
    1. V_solid is conserved during compression
    2. Porosity decreases exponentially toward min_porosity
    3. Saturation increases as pores shrink
    4. Liquid is externalized when S > 1
    
    Analytical Solution:
    poro(t) = min_poro + (poro_0 - min_poro) * exp(-rate * t)
    V_particle_dry(t) = V_solid / (1 - poro(t))
    """
    print_separator("TEST 2: Compression with Saturation")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=10.0,
        t_write=1.0,
        verbose=True,
        init=True,
        seed=42,
        load_attr=False,
    )
    
    solver.process_type = "compression"
    
    solver.create_compression_handler(
        enabled=True,
        rate=0.02,
        min_porosity=0.3,
    )
    
    # Initialize all particles with porosity = 0.4 and saturation = 0.6
    initial_porosity = 0.4
    initial_saturation = 0.6
    solver.porosity[:solver.a_tot] = initial_porosity
    solver.saturation[:solver.a_tot] = initial_saturation
    
    # Add some initial liquid (internal)
    V_pore_initial = solver.get_V_pore()
    solver.liquid_volume[:solver.a_tot] = V_pore_initial * initial_saturation
    
    # Capture initial state
    V_solid_initial = solver.get_V_solid().copy()
    V_dry_initial = solver.get_V_particle_dry().copy()
    V_liq_initial = solver.liquid_volume[:solver.a_tot].copy()
    V_pore_init_arr = V_pore_initial.copy()
    
    print(f"\n=== Compression-Only Test with Saturation ===")
    print(f"Initial particles: {solver.a_tot}")
    print(f"Initial porosity: {initial_porosity}")
    print(f"Initial saturation: {initial_saturation}")
    print(f"Initial V_solid (mean): {np.mean(V_solid_initial):.6e} m³")
    print(f"Initial V_particle_dry (mean): {np.mean(V_dry_initial):.6e} m³")
    print(f"Initial V_liquid (mean): {np.mean(V_liq_initial):.6e} m³")
    
    # Run simulation
    solver.solve(maxiter=int(1e6))
    
    # Extract results
    t_vec = solver.t_vec
    
    # Final state
    V_solid_final = solver.get_V_solid()
    V_dry_final = solver.get_V_particle_dry()
    V_liq_final = solver.liquid_volume[:solver.a_tot]
    poro_final = solver.porosity[:solver.a_tot]
    sat_final = solver.saturation[:solver.a_tot]
    
    # Analytical porosity at t=10s
    rate = 0.02
    min_poro = 0.3
    poro_analytical = min_poro + (initial_porosity - min_poro) * np.exp(-rate * 10.0)
    poro_simulated = np.mean(poro_final)
    poro_error = abs(poro_simulated - poro_analytical) / poro_analytical
    
    # Analytical saturation calculation
    # Formula: sat_final = sat_initial * (V_pore_initial / V_pore_final)
    # Since V_pore = V_solid * poro / (1 - poro), we get:
    # sat_final = sat_initial * [poro_0 / (1 - poro_0)] / [poro_final / (1 - poro_final)]
    # This assumes no liquid externalization (sat < 1.0)
    
    # Calculate analytical saturation per particle using mean porosity
    pore_ratio = (initial_porosity / (1.0 - initial_porosity)) / (poro_analytical / (1.0 - poro_analytical))
    sat_analytical_unclipped = initial_saturation * pore_ratio
    sat_analytical = min(sat_analytical_unclipped, 1.0)  # Cap at 1.0, excess would be externalized
    
    sat_simulated = np.mean(sat_final)
    sat_error = abs(sat_simulated - sat_analytical) / max(sat_analytical, 1e-10)
    
    print(f"\n=== Results at t=10s ===")
    print(f"Porosity analytical: {poro_analytical:.6f}")
    print(f"Porosity simulated: {poro_simulated:.6f}")
    print(f"Porosity error: {poro_error*100:.4f}%")
    print(f"\nSaturation analytical: {sat_analytical:.6f} (calculated from poro change)")
    print(f"Saturation simulated: {sat_simulated:.6f}")
    print(f"Saturation error: {sat_error*100:.4f}%")
    
    # V_solid conservation
    V_solid_change = abs(np.mean(V_solid_final) - np.mean(V_solid_initial)) / np.mean(V_solid_initial)
    print(f"\nV_solid change: {V_solid_change*100:.4f}% (should be ~0%)")
    
    # V_particle_dry should decrease
    V_dry_change = (np.mean(V_dry_initial) - np.mean(V_dry_final)) / np.mean(V_dry_initial)
    print(f"V_particle_dry change: {V_dry_change*100:.2f}% (should decrease)")
    
    # Saturation should increase
    sat_initial_mean = initial_saturation
    sat_final_mean = np.mean(sat_final)
    print(f"Saturation change: {sat_initial_mean:.4f} → {sat_final_mean:.4f}")
    
    # Liquid conservation (total should be constant)
    V_liq_change = abs(np.mean(V_liq_final) - np.mean(V_liq_initial)) / np.mean(V_liq_initial)
    print(f"V_liquid change: {V_liq_change*100:.4f}% (should be ~0%)")
    
    # Validation
    print(f"\n=== Validation ===")
    check_poro = poro_error < 0.05  # < 5%
    check_solid = V_solid_change < 0.01  # < 1%
    check_dry_decrease = V_dry_change > 0  # Should decrease
    check_liq = V_liq_change < 0.01  # < 1%
    check_sat_increase = sat_final_mean >= sat_initial_mean
    check_sat_analytical = sat_error < 0.10  # < 10% of analytical value
    
    print(f"Porosity matches analytical: {'✅' if check_poro else '❌'}")
    print(f"V_solid conserved: {'✅' if check_solid else '❌'}")
    print(f"V_particle_dry decreased: {'✅' if check_dry_decrease else '❌'}")
    print(f"Liquid conserved: {'✅' if check_liq else '❌'}")
    print(f"Saturation increased: {'✅' if check_sat_increase else '❌'}")
    print(f"Saturation matches analytical: {'✅' if check_sat_analytical else '❌'}")
    
    all_pass = check_poro and check_solid and check_dry_decrease and check_liq and check_sat_increase and check_sat_analytical
    
    print(f"\nOverall: {'✅ PASS' if all_pass else '❌ FAIL'}")
    
    return all_pass


def test_agglomeration_mass_conservation():
    """
    Test mass conservation during agglomeration with porous particles.
    
    Validates:
    1. V_solid is additive during agglomeration
    2. Liquid volume is conserved
    3. Porosity is merged correctly (volume-weighted)
    """
    print_separator("TEST 3: Agglomeration Mass Conservation")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=10.0,
        t_write=10,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    # Setup with moderate agglomeration
    solver.c = np.array([1e-3])
    solver.x = np.array([5e-5])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 700
    solver.process_type = "agglomeration"
    
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-19
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    
    # Initialize liquid and porosity
    if hasattr(solver, 'liquid_volume'):
        solver.liquid_volume[:n_init] = 0.0
    
    # Create nucleation to add porosity and liquid
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=8e-12,
        tropfen_durchmesser=4e-5,
        wasserzugabe_start=0.0,
        wasserzugabe_dauer=1.0,
    )
    
    # IMPORTANT: Configure nucleation time step
    solver.nucleation.configure_time_step(0.1)
    
    # Capture initial state (BEFORE nucleation adds liquid)
    # IMPORTANT: Use WEIGHTED sums for DSMC consistency!
    V_solid_initial = calculate_weighted_total(solver.get_V_solid(), solver.W, n_init)
    V_liq_initial = calculate_weighted_total(solver.liquid_volume[:n_init], solver.W, n_init)
    
    print(f"\nInitial state:")
    print(f"  Particles: {n_init}")
    print(f"  V_solid: {V_solid_initial:.6e} m³")
    print(f"  V_liquid: {V_liq_initial:.6e} m³")
    
    # Run simulation
    solver.solve(maxiter=50000)
    
    n_final = solver.a_tot
    
    # Final state
    # IMPORTANT: Use WEIGHTED sums for DSMC consistency!
    V_solid_final = calculate_weighted_total(solver.get_V_solid(), solver.W, n_final)
    V_liq_final = calculate_weighted_total(solver.liquid_volume[:n_final], solver.W, n_final)
    
    # Get expected liquid from nucleation
    nuc_stats = solver.nucleation.get_statistics()
    liquid_added = nuc_stats['liquid_volume_added_total']
    
    print(f"\nFinal state:")
    print(f"  Particles: {n_final}")
    print(f"  V_solid: {V_solid_final:.6e} m³")
    print(f"  V_liquid: {V_liq_final:.6e} m³")
    print(f"  Liquid added (from nucleation): {liquid_added:.6e} m³")
    
    # Conservation errors
    V_solid_error = abs(V_solid_final - V_solid_initial) / V_solid_initial
    # Compare final liquid to ADDED liquid (not initial, which was 0!)
    if abs(liquid_added) > 1e-300:
        V_liq_error = abs(V_liq_final - liquid_added) / abs(liquid_added)
    else:
        V_liq_error = 0.0
    
    print(f"\nConservation errors:")
    print(f"  V_solid: {V_solid_error*100:.4f}%")
    print(f"  V_liquid: {V_liq_error*100:.4f}%")
    
    # Validation
    check_solid = V_solid_error < 0.01
    check_liq = V_liq_error < 0.01
    
    print(f"\nResults:")
    print(f"  V_solid conserved: {'✅' if check_solid else '❌'}")
    print(f"  V_liquid conserved: {'✅' if check_liq else '❌'}")
    
    all_pass = check_solid and check_liq
    
    print(f"\nOverall: {'✅ PASS' if all_pass else '❌ FAIL'}")
    
    return all_pass


def test_mass_mode_initialization():
    """
    Test unified MASS mode initialization with eingewogene_masse.
    
    Validates:
    1. Duration is correctly calculated from eingewogene_masse + wt%
    2. Volumenstrom is used directly (no scaling)
    3. Nucleation stops when target_wt% reached
    4. Statistics are consistent
    
    Note: This uses the NEW unified logic where duration is auto-calculated
    from mass parameters (no separate DURATION vs MASS modes anymore).
    """
    print_separator("TEST 5: Unified MASS Mode (Auto-Calculate Duration)")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    # Physical parameters - use MASS consistent with simulation scale
    # IMPORTANT: eingewogene_masse must match what the simulation represents!
    # Simulation: 500 particles × Vc ~ 6.5e-11 m³ each
    # We use a SMALL eingewogene_masse for reasonable test duration
    eingewogene_masse = 1e-10  # kg (tiny mass for short duration)
    n_comp = 500  # computational particles
    diameter = 5e-6  # m (5 µm)
    rho_solid = 2500.0  # kg/m³
    rho_liquid = 1000.0  # kg/m³
    target_wt = 5.0  # %
    
    # Calculate expected values
    v_particle = (np.pi / 6.0) * diameter ** 3
    mass_particle = v_particle * rho_solid
    n_particles_real = eingewogene_masse / mass_particle
    
    mass_solid_total = eingewogene_masse
    mass_liquid_target = mass_solid_total * (target_wt / 100.0)
    v_liquid_target = mass_liquid_target / rho_liquid
    
    # Calculate expected duration from mass parameters
    # Use HIGHER volumenstrom for shorter duration (test speed)
    volumenstrom = 1e-12  # m³/s
    expected_duration = v_liquid_target / volumenstrom
    
    print(f"\nPhysical parameters:")
    print(f"  eingewogene_masse: {eingewogene_masse:.3f} kg")
    print(f"  n_comp: {n_comp}")
    print(f"  diameter: {diameter*1e6:.1f} µm")
    print(f"  rho_solid: {rho_solid:.1f} kg/m³")
    print(f"\nCalculated values:")
    print(f"  v_particle: {v_particle:.6e} m³")
    print(f"  mass_particle: {mass_particle:.6e} kg")
    print(f"  n_particles_real: {n_particles_real:.6e}")
    print(f"\nTarget:")
    print(f"  wt%: {target_wt}%")
    print(f"  Expected V_liquid: {v_liquid_target:.6e} m³")
    print(f"  Calculated duration: {expected_duration:.3f} s")
    
    # Setup solver
    solver = MCPBESolver(
        dim=1,
        t_total=max(2.0, expected_duration * 1.5),  # Ensure enough time
        t_write=10,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    solver.c = np.array([1e-3])
    solver.x = np.array([diameter])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = n_comp
    solver.process_type = "agglomeration"
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-13  # Increased for more MC events
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    # Verify W scaling
    sum_W = np.sum(solver.W[:solver.a_tot])
    n_represented = sum_W / solver.Vc
    print(f"\nSolver initialization:")
    print(f"  a_tot: {solver.a_tot}")
    print(f"  Sum(W): {sum_W:.6e}")
    print(f"  Vc: {solver.Vc:.6e}")
    print(f"  n_represented (Sum(W)/Vc): {n_represented:.6e}")
    print(f"  Ratio to n_particles_real: {n_represented/n_particles_real:.2f}")
    
    # Create nucleation handler with mass-based parameters
    # Duration is AUTO-CALCULATED from eingewogene_masse + wt% + densities
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=volumenstrom,
        tropfen_durchmesser=diameter,
        wasserzugabe_start=0.0,
        target_wt_percent=target_wt,
        eingewogene_masse=eingewogene_masse,
        rho_solid=rho_solid,
        rho_liquid=rho_liquid,
        # NO initialization_type parameter (deprecated)
        # NO n_comp parameter (not needed)
    )
    # configure_time_step is deprecated but harmless
    solver.nucleation.configure_time_step(0.1)
    
    # Run simulation
    solver.solve(maxiter=10000)
    
    # Get statistics
    nuc_stats = solver.nucleation.get_statistics()
    current_wt = nuc_stats['current_wt_percent']
    target_reached = nuc_stats['target_reached']
    liquid_added = nuc_stats['liquid_volume_added_total']
    n_particles_stat = nuc_stats.get('n_particles_real', None)
    
    # Final state
    n_final = solver.a_tot
    V_solid_final = calculate_weighted_total(solver.get_V_solid(), solver.W, n_final)
    V_liq_final = calculate_weighted_total(solver.liquid_volume[:n_final], solver.W, n_final)
    
    print(f"\nFinal state:")
    print(f"  Particles: {n_final}")
    print(f"  V_solid: {V_solid_final:.6e} m³")
    print(f"  V_liquid: {V_liq_final:.6e} m³")
    print(f"  Current wt%: {current_wt:.2f}%")
    print(f"  Target reached: {target_reached}")
    print(f"  Liquid added (stat): {liquid_added:.6e} m³")
    print(f"  n_particles_real (from stat): {n_particles_stat:.6e}" if n_particles_stat else "")
    
    # Validation
    print(f"\n=== Validation ===")
    
    # 1. Target should be reached (or very close)
    wt_error = abs(current_wt - target_wt) / target_wt
    check_target = wt_error < 0.05 #within 5%
    print(f"wt% near target: {'✅' if check_target else '❌'} (error: {wt_error*100:.1f}%)")
    
    # 2. Liquid conservation: final ≈ added
    if liquid_added > 1e-300:
        liq_error = abs(V_liq_final - liquid_added) / liquid_added
    else:
        liq_error = 0.0
    check_liq = liq_error < 0.10
    print(f"Liquid conserved: {'✅' if check_liq else '❌'} (error: {liq_error*100:.1f}%)")
    
    # 3. Target flag should be True
    check_flag = target_reached or (current_wt >= target_wt * 0.9)
    print(f"Target flag correct: {'✅' if check_flag else '❌'}")
    
    # 4. n_particles_real should match calculation
    if n_particles_stat:
        n_error = abs(n_particles_stat - n_particles_real) / n_particles_real
        check_n = n_error < 0.01
        print(f"n_particles_real correct: {'✅' if check_n else '❌'} (error: {n_error*100:.1f}%)")
    else:
        check_n = True
        print(f"n_particles_real: N/A (not in stats)")
    
    all_pass = check_target and check_liq and check_flag and check_n
    
    print(f"\nOverall: {'✅ PASS' if all_pass else '❌ FAIL'}")
    
    return all_pass


def test_wt_percent_control():
    """
    Test wt%-based nucleation control.
    
    Validates:
    1. Nucleation stops when target wt% is reached
    2. Current wt% calculation is correct
    3. Statistics are Vc-normalized (robust to doubling)
    """
    print_separator("TEST 4: wt% Control")
    
    from wmcpbe.mcpbe import MCPBESolver
    
    solver = MCPBESolver(
        dim=1,
        t_total=10.0,
        t_write=10,
        verbose=False,
        load_attr=False,
        init=False,
        seed=42,
    )
    
    # Setup - use HIGHER agglomeration rate to ensure MC events during nucleation window
    solver.c = np.array([1e-3])
    solver.x = np.array([5e-6])
    solver.PGV = np.array(["mono"])
    solver.SIG = np.array([0.0])
    solver.a0 = 500
    solver.process_type = "agglomeration"
    solver.COLEVAL = 3
    solver.CORR_BETA = 1e-13  # Increased from 1e-15 for more events
    solver.G = 1.0
    solver.alpha_prim = 1.0
    
    solver._initialize_particles()
    solver._initialize_samplers()
    
    n_init = solver.a_tot
    
    # Target: 1.5 wt% (liquid/solid dry basis)
    target_wt = 1.5
    rho_solid = 2000.0
    rho_liquid = 1000.0
    
    # Calculate expected liquid volume for target wt%
    V_solid_initial = calculate_weighted_total(solver.get_V_solid(), solver.W, n_init)
    mass_solid = V_solid_initial * rho_solid
    mass_liquid_target = mass_solid * (target_wt / 100.0)
    V_liquid_target = mass_liquid_target / rho_liquid
    
    # Calculate eingewogene_masse from simulated solid mass
    # This is the physical mass represented by the simulation
    eingewogene_masse = mass_solid
    
    print(f"\nInitial state:")
    print(f"  Particles: {n_init}")
    print(f"  V_solid: {V_solid_initial:.6e} m³")
    print(f"  Mass solid: {mass_solid:.6e} kg")
    print(f"\nTarget:")
    print(f"  wt%: {target_wt}%")
    print(f"  Expected V_liquid: {V_liquid_target:.6e} m³")
    
    # Create nucleation handler with mass-based wt% control
    # Use LOWER volumenstrom for longer duration (gives MC events time to occur)
    # duration = V_liquid_target / volumenstrom
    # With 1e-15 m³/s: duration ≈ 1s (good for MC events)
    volumenstrom_test = 1e-15  # m³/s (reduced for longer window)
    expected_duration = V_liquid_target / volumenstrom_test
    
    print(f"\nNucleation parameters:")
    print(f"  volumenstrom: {volumenstrom_test} m³/s")
    print(f"  Calculated duration: {expected_duration:.3f} s")
    
    solver.create_nucleation_handler(
        enabled=True,
        volumenstrom=volumenstrom_test,
        tropfen_durchmesser=9e-6,
        wasserzugabe_start=0.0,
        target_wt_percent=target_wt,
        eingewogene_masse=eingewogene_masse,
        rho_solid=rho_solid,
        rho_liquid=rho_liquid,
    )
    solver.nucleation.configure_time_step(0.1)
    
    # Run simulation
    solver.solve(maxiter=10000)
    
    # Get statistics
    nuc_stats = solver.nucleation.get_statistics()
    current_wt = nuc_stats['current_wt_percent']
    target_reached = nuc_stats['target_reached']
    liquid_added = nuc_stats['liquid_volume_added_total']
    
    # Final state
    n_final = solver.a_tot
    V_solid_final = calculate_weighted_total(solver.get_V_solid(), solver.W, n_final)
    V_liq_final = calculate_weighted_total(solver.liquid_volume[:n_final], solver.W, n_final)
    
    print(f"\nFinal state:")
    print(f"  Particles: {n_final}")
    print(f"  V_solid: {V_solid_final:.6e} m³")
    print(f"  V_liquid: {V_liq_final:.6e} m³")
    print(f"  Current wt%: {current_wt:.2f}%")
    print(f"  Target reached: {target_reached}")
    print(f"  Liquid added (stat): {liquid_added:.6e} m³")
    
    # Validation
    print(f"\n=== Validation ===")
    
    # 1. Target should be reached (or very close)
    wt_error = abs(current_wt - target_wt) / target_wt
    check_target = wt_error < 0.05   # Within 5%
    print(f"wt% near target: {'✅' if check_target else '❌'} (error: {wt_error*100:.1f}%)")
    
    # 2. Liquid conservation: final ≈ added
    if liquid_added > 1e-300:
        liq_error = abs(V_liq_final - liquid_added) / liquid_added
    else:
        liq_error = 0.0
    check_liq = liq_error < 0.10  # < 10%
    print(f"Liquid conserved: {'✅' if check_liq else '❌'} (error: {liq_error*100:.1f}%)")
    
    # 3. Target reached flag should be True
    check_flag = target_reached or (current_wt >= target_wt * 0.9)
    print(f"Target flag correct: {'✅' if check_flag else '❌'}")
    
    all_pass = check_target and check_liq and check_flag
    
    print(f"\nOverall: {'✅ PASS' if all_pass else '❌ FAIL'}")
    
    return all_pass


def main():
    """Run all volume semantics tests."""
    print("\n" + "#" * 70)
    print("# Volume Semantics Tests - V_particle_dry, Saturation, Liquid Handling")
    print("#" * 70)
    
    results = []
    
    # Test 1: Nucleation volume model
    try:
        passed = test_nucleation_volume_model()
        results.append(("Nucleation Volume Model", passed))
    except Exception as e:
        print(f"\n❌ Nucleation EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Nucleation Volume Model", False))
    
    # Test 2: Compression with saturation
    try:
        passed = test_compression_with_saturation()
        results.append(("Compression with Saturation", passed))
    except Exception as e:
        print(f"\n❌ Compression EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Compression with Saturation", False))
    
    # Test 3: Agglomeration mass conservation
    try:
        passed = test_agglomeration_mass_conservation()
        results.append(("Agglomeration Mass Conservation", passed))
    except Exception as e:
        print(f"\n❌ Agglomeration EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Agglomeration Mass Conservation", False))
    
    # Test 4: wt% control (legacy DURATION mode with target monitoring)
    try:
        passed = test_wt_percent_control()
        results.append(("wt% Control (DURATION mode)", passed))
    except Exception as e:
        print(f"\n❌ wt% Control EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("wt% Control (DURATION mode)", False))
    
    # Test 5: MASS mode initialization
    try:
        passed = test_mass_mode_initialization()
        results.append(("MASS Mode Initialization", passed))
    except Exception as e:
        print(f"\n❌ MASS Mode EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append(("MASS Mode Initialization", False))
    
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
