"""
Comprehensive Test: Nucleation + Agglomeration with Stokes Acceptance + Breakage + Compression.

This test verifies the MCPBE solver with the Stokes agglomeration acceptance criterion
(Braumann et al. 2007) integrated into the agglomeration process:

- Nucleation: Liquid droplet addition during defined time window
- Agglomeration: Particle collisions with Stokes acceptance criterion
  - St < St_crit: Particles agglomerate (viscous forces dominate)
  - St > St_crit: Particles bounce (inertia dominates)
- Breakage: Particle fragmentation under shear stress
- Compression: Porosity reduction due to mechanical stress

Test Parameters (validated working configuration):
    - Particle diameter: 500 µm (initial)
    - Droplet diameter: 100 µm
    - Flow rate: 3e-10 m³/s
    - Agglomeration coefficient: 1e-5
    - Stokes parameters: U_coll=1.0 m/s, binder_viscosity=0.1 Pa·s, h_a=500 nm
    - Simulation duration: 4 seconds

Reference:
    Braumann, A., Kraft, M., & Wagner, L. (2007). Modelling and validation of granulation
    with heterogeneous binder dispersion and chemical reaction. Chemical Engineering Science,
    62(17), 4709-4720.
"""

import numpy as np
import sys
import os
import time
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver


# =============================================================================
# Test Configuration
# =============================================================================

class TestConfig:
    """Physical and numerical parameters for Stokes acceptance test."""
    
    # Seed for reproducibility
    SEED = 42
    
    # Time settings
    T_TOTAL = 2.0       # Total simulation time [s]
    T_WRITE = 0.1       # Output interval [s]
    
    # Particle properties (initial)
    PARTICLE_DIAMETER = 500e-6      # 500 µm (typical granule size)
    PARTICLE_DENSITY = 2500.0       # kg/m³ (solid material)
    INITIAL_POROSITY = 0.0          # 20% void fraction
    
    # Droplet properties
    DROPLET_DIAMETER = 500e-6       # 100 µm (typical spray droplet size)
    DROPLET_DENSITY = 1000.0        # kg/m³ (water)
    
    # Process parameters
    VOLUMETRIC_FLOW_RATE = 3e-9     # m³/s (moderate flow rate)
    NUCLEATION_DURATION = 1.0       # s
    AGG_COEFFICIENT = 5e-5          # Constant kernel coefficient
    
    # Stokes acceptance criterion parameters (Braumann et al. 2007)
    STOKES_ENABLED = True
    U_COLL = 1.0                    # m/s - collision velocity (typical for high-shear)
    BINDER_VISCOSITY = 0.1          # Pa·s - binder viscosity (aqueous binder)
    RHO_SOLID = 2500.0              # kg/m³ - solid density
    RHO_LIQUID = 1000.0             # kg/m³ - liquid density
    H_A = 50e-9                     # m - surface roughness (50 nm, reduced for better agglomeration)
    STOKES_DEBUG = False            # Enable detailed debug output for each collision
    
    # Breakage parameters (DISABLED for Stokes validation)
    BREAKAGE_ENABLED = True
    BREAKAGE_RATE = 0.001            # Fragmentation rate [1/s]
    BREAKAGE_DAUGHTERS = 2          # Number of daughter particles
    
    # Compression parameters
    COMPRESSION_ENABLED = True
    COMPRESSION_RATE = 0.05         # Porosity decay rate [1/s]
    MIN_POROSITY = 0.1              # Minimum achievable porosity
    
    # Numerical settings
    A0_INITIAL = 500               # Initial particle count (computational particles)
    INITIAL_WEIGHT = 100.0          # Weight per particle
    CONTROL_VOLUME = 1.0            


# =============================================================================
# Helper Functions
# =============================================================================

def format_scientific(value: float, unit: str = "") -> str:
    """Format value in scientific notation with unit."""
    return f"{value:.3e} {unit}".strip()


def format_percentage(value: float) -> str:
    """Format value as percentage with sign."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value*100:+.4f}%"


def print_section(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(title)
    print(char * 80)


# =============================================================================
# Main Test Function
# =============================================================================

def run_stokes_acceptance_test() -> Dict[str, Any]:
    """
    Run comprehensive MCPBE simulation with Stokes acceptance criterion.
    
    Returns:
        Dictionary containing test results and statistics
    """
    
    # -------------------------------------------------------------------------
    # Setup & Initialization
    # -------------------------------------------------------------------------
    
    print_section("STOKES ACCEPTANCE CRITERION TEST", "=")
    print("Physics Modules: Nucleation + Agglomeration (Stokes) + Breakage + Compression")
    print("\nReference: Braumann et al. (2007), Chem. Eng. Sci. 62(17), 4709-4720")
    
    cfg = TestConfig()
    
    # Derived quantities
    particle_radius = cfg.PARTICLE_DIAMETER / 2.0
    particle_volume = (4.0 / 3.0) * np.pi * particle_radius ** 3
    
    droplet_radius = cfg.DROPLET_DIAMETER / 2.0
    droplet_volume = (4.0 / 3.0) * np.pi * droplet_radius ** 3
    
    expected_liquid = cfg.VOLUMETRIC_FLOW_RATE * cfg.NUCLEATION_DURATION
    expected_droplets = expected_liquid / droplet_volume
    
    # -------------------------------------------------------------------------
    # Print Configuration
    # -------------------------------------------------------------------------
    
    print_section("PARTICLE PROPERTIES")
    print(f"  Diameter:           {cfg.PARTICLE_DIAMETER*1e6:.1f} µm")
    print(f"  Volume:             {format_scientific(particle_volume, 'm³')}")
    print(f"  Density:            {cfg.PARTICLE_DENSITY:.1f} kg/m³")
    print(f"  Initial porosity:   {cfg.INITIAL_POROSITY*100:.1f} %")
    
    print_section("DROPLET PROPERTIES")
    print(f"  Diameter:           {cfg.DROPLET_DIAMETER*1e6:.1f} µm")
    print(f"  Volume:             {format_scientific(droplet_volume, 'm³')}")
    print(f"  Density:            {cfg.DROPLET_DENSITY:.1f} kg/m³")
    
    print_section("PROCESS PARAMETERS")
    print(f"  Volumetric flow:    {format_scientific(cfg.VOLUMETRIC_FLOW_RATE, 'm³/s')}")
    print(f"  Nucleation window:  [0.0 s, {cfg.NUCLEATION_DURATION:.1f} s]")
    print(f"  Agg. coefficient:   {format_scientific(cfg.AGG_COEFFICIENT)}")
    
    print_section("STOKES ACCEPTANCE CRITERION (Braumann et al. 2007)")
    print(f"  Enabled:            {cfg.STOKES_ENABLED}")
    print(f"  Collision velocity: {cfg.U_COLL:.2f} m/s")
    print(f"  Binder viscosity:   {cfg.BINDER_VISCOSITY:.2f} Pa·s")
    print(f"  Solid density:      {cfg.RHO_SOLID:.1f} kg/m³")
    print(f"  Liquid density:     {cfg.RHO_LIQUID:.1f} kg/m³")
    print(f"  Surface roughness:  {cfg.H_A*1e9:.1f} nm")
    print(f"\n  Criterion: St < St_crit → accept, St > St_crit → reject")
    print(f"  where:")
    print(f"    St = (m_harm × U) / (3 × π × η × R_harm²)")
    print(f"    St_crit = (1 + 1/e_coag) × ln(h / h_a)")
    
    print_section("BREAKAGE & COMPRESSION")
    breakage_status = "(enabled)" if cfg.BREAKAGE_ENABLED and cfg.BREAKAGE_RATE > 0 else "(disabled)"
    compression_status = "(enabled)" if cfg.COMPRESSION_ENABLED else "(disabled)"
    print(f"  Breakage kernel:    {'power_law' if cfg.BREAKAGE_ENABLED else 'None'} {breakage_status}")
    print(f"  Breakage rate:      {cfg.BREAKAGE_RATE:.2e} 1/s")
    print(f"  Compression kernel: {'exponential_decay' if cfg.COMPRESSION_ENABLED else 'None'} {compression_status}")
    print(f"  Compression rate:   {cfg.COMPRESSION_RATE:.2f} 1/s")
    
    print_section("EXPECTED RESULTS")
    print(f"  Total liquid:       {format_scientific(expected_liquid, 'm³')}")
    print(f"  Total droplets:     {expected_droplets:.2f}")
    print(f"  Simulation time:    {cfg.T_TOTAL:.1f} s")
    
    # -------------------------------------------------------------------------
    # Solver Initialization
    # -------------------------------------------------------------------------
    
    print_section("INITIALIZING SOLVER", "-")
    
    # Time vector
    t_vec = np.linspace(0.0, cfg.T_TOTAL, int(cfg.T_TOTAL / cfg.T_WRITE) + 1)
    
    # Create RNG with fixed seed
    rng = np.random.default_rng(cfg.SEED)
    
    print(f"\nCreating solver (seed={cfg.SEED})...")
    
    # Configure solver with Stokes acceptance kernel
    print("\nUsing Stokes agglomeration acceptance criterion:")
    print("  - Kernel: stokes_krit (physics-based, Braumann et al. 2007)")
    print("  - Integration: After partner selection, before agglomeration")
    print("  - Effect: Only collisions with St < St_crit lead to agglomeration")
    
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=True,
        load_attr=False,
        init=True,
        agg_kernel_name='constant',
        agg_kernel_params={'corr_beta': cfg.AGG_COEFFICIENT},
        # NEW: Stokes acceptance kernel
        agg_acceptance_kernel_name='stokes_krit',
        agg_acceptance_kernel_params={
            'U_coll': cfg.U_COLL,
            'binder_viscosity': cfg.BINDER_VISCOSITY,
            'rho_solid': cfg.RHO_SOLID,
            'rho_liquid': cfg.RHO_LIQUID,
            'h_a': cfg.H_A,
            'debug': cfg.STOKES_DEBUG,  # Enable debug output
        },
        # Breakage: None disables breakage completely
        break_kernel_name=None if not cfg.BREAKAGE_ENABLED else 'power_law',
        break_kernel_params={
            'p1': cfg.BREAKAGE_RATE,
            'p2': 0.0,
            'g': 1000,
            'breakrval': 1,  # Constant rate
        } if cfg.BREAKAGE_ENABLED else None,
        compression_kernel_name='exponential_decay' if cfg.COMPRESSION_ENABLED else None,
        compression_kernel_params={
            'rate': cfg.COMPRESSION_RATE,
            'min_porosity': cfg.MIN_POROSITY,
        } if cfg.COMPRESSION_ENABLED else None,
        porosity_growth_kernel_name='cone_model',
        porosity_growth_kernel_params={},  # Uses default: 0.0 (Vollkörper), grows via ΔV
        rng=rng,
    )
    
    # Configure process type
    solver.process_type = "mix"  # Agglomeration + Breakage
    solver.recon_enable = False
    
    # Initialize particles with custom properties
    # IMPORTANT: V_flat structure for dim=1:
    #   V_flat has shape (dim+1, n_particles) = (2, n_particles)
    #   V_flat[0, :] = V_solid (solid volume only, conserved during compression!)
    #   V_flat[1, :] = V_dry (total dry volume = V_solid + V_pore)
    # Relation: V_solid = V_dry × (1 - porosity)
    V_flat = np.zeros((2, cfg.A0_INITIAL), dtype=float)  # dim+1 = 2 rows
    V_flat[0, :] = particle_volume * (1.0 - cfg.INITIAL_POROSITY)  # V_solid
    V_flat[1, :] = particle_volume  # V_dry (geometric volume of particle)
    
    W_init = np.full(cfg.A0_INITIAL, cfg.INITIAL_WEIGHT, dtype=float)
    
    solver.Vc = cfg.CONTROL_VOLUME
    
    print("\nInitializing particles...")
    solver._initialize_particles(
        init_Vc=False,
        V_flat=V_flat,
        W_init=W_init,
    )
    
    # Initialize porosity for all particles (NO initial liquid)
    solver.porosity[:solver.a_tot] = cfg.INITIAL_POROSITY
    solver.liquid_volume[:solver.a_tot] = 0.0
    solver.saturation[:solver.a_tot] = 0.0
    
    print(f"\nInitial liquid: 0.0 m³ (dry particles)")
    print(f"Initial saturation: 0.0%")
    
    # Calculate initial solid mass (before any physics)
    v_dry_init = solver.V_flat[-1, :solver.a_tot]
    porosity_init = solver.porosity[:solver.a_tot]
    weights_init = solver.W[:solver.a_tot]
    
    valid_poro_init = ~np.isnan(porosity_init)
    v_solid_init = np.zeros_like(v_dry_init)
    v_solid_init[valid_poro_init] = v_dry_init[valid_poro_init] * (1.0 - porosity_init[valid_poro_init])
    v_solid_init[~valid_poro_init] = v_dry_init[~valid_poro_init]  # Vollkörper
    
    initial_solid_volume = np.sum(v_solid_init * weights_init)
    initial_solid_mass = initial_solid_volume * cfg.PARTICLE_DENSITY
    
    print(f"  a_tot:      {solver.a_tot}")
    print(f"  Vc:         {format_scientific(solver.Vc, 'm³')}")
    print(f"  W[0:5]:     {solver.W[:5]}")
    print(f"  porosity:   {solver.porosity[0]:.3f}")
    
    # Initialize samplers
    print("\nInitializing samplers...")
    solver._initialize_samplers()
    
    # Configure nucleation
    print("\nConfiguring nucleation...")
    solver.create_nucleation_handler(
        enabled=True,
        volumetric_flow_rate=cfg.VOLUMETRIC_FLOW_RATE,
        droplet_diameter=cfg.DROPLET_DIAMETER,
        liquid_addition_start=0.0,
        liquid_addition_duration=cfg.NUCLEATION_DURATION,
    )
    
    # Configure compression
    if cfg.COMPRESSION_ENABLED:
        print("Configuring compression...")
        solver.create_compression_handler(
            enabled=True,
            rate=cfg.COMPRESSION_RATE,
            min_porosity=cfg.MIN_POROSITY,
        )
    
    # -------------------------------------------------------------------------
    # Run Simulation
    # -------------------------------------------------------------------------
    
    print_section("RUNNING SIMULATION", "=")
    
    start_time = time.time()
    
    try:
        solver.solve(maxiter=int(1e9))
    except Exception as e:
        print(f"\n❌ ERROR during solve: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    
    elapsed_real = time.time() - start_time
    
    # -------------------------------------------------------------------------
    # Collect Results
    # -------------------------------------------------------------------------
    
    print_section("RESULTS", "=")
    
    # Nucleation statistics
    nuc_stats = solver.nucleation.get_statistics()
    actual_liquid = nuc_stats['liquid_volume_added_total']
    actual_droplets = nuc_stats['droplets_added_total']
    
    # Event counts
    agg_events = solver.real_agg_events
    break_events = solver.real_break_events
    total_events = agg_events + break_events
    
    # Liquid in system (sum over all particles)
    liquid_in_system = np.sum(solver.liquid_volume[:solver.a_tot] * solver.W[:solver.a_tot])
    
    # Solid mass conservation check
    v_dry = solver.V_flat[-1, :solver.a_tot]
    porosity = solver.porosity[:solver.a_tot]
    weights = solver.W[:solver.a_tot]
    
    valid_poro = ~np.isnan(porosity)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid_poro] = v_dry[valid_poro] * (1.0 - porosity[valid_poro])
    v_solid[~valid_poro] = v_dry[~valid_poro]  # Vollkörper
    
    final_solid_volume = np.sum(v_solid * weights)
    final_solid_mass = final_solid_volume * cfg.PARTICLE_DENSITY
    solid_mass_error = (final_solid_mass - initial_solid_mass) / initial_solid_mass
    
    print(f"\n[SOLID MASS DEBUG]")
    print(f"  Initial solid mass: {initial_solid_mass:.6e} kg")
    print(f"  Final solid mass:   {final_solid_mass:.6e} kg")
    print(f"  Δ Mass:             {final_solid_mass - initial_solid_mass:.6e} kg")
    print(f"  Error:              {solid_mass_error*100:+.4f}%")
    print(f"  n_comp:             {solver.a_tot}")
    print(f"  n_Vollkörper:       {np.sum(~valid_poro)}")
    print(f"  n_porous:           {np.sum(valid_poro)}")
    
    # Porosity statistics
    porosity_active = solver.porosity[:solver.a_tot]
    porosity_mean = np.mean(porosity_active[~np.isnan(porosity_active)])
    porosity_min = np.nanmin(porosity_active)
    porosity_max = np.nanmax(porosity_active)
    
    # Errors
    error_liquid = (actual_liquid - expected_liquid) / expected_liquid
    error_system = (liquid_in_system - expected_liquid) / expected_liquid
    
    # -------------------------------------------------------------------------
    # Print Results
    # -------------------------------------------------------------------------
    
    print(f"\n{'Liquid Addition:':<25}")
    print(f"  {'Expected:':<20} {format_scientific(expected_liquid, 'm³')}")
    print(f"  {'Added (stats):':<20} {format_scientific(actual_liquid, 'm³')}")
    print(f"  {'In system:':<20} {format_scientific(liquid_in_system, 'm³')}")
    print(f"  {'Error (added):':<20} {format_percentage(error_liquid)}")
    print(f"  {'Error (system):':<20} {format_percentage(error_system)}")
    
    print(f"\n{'Solid Mass Conservation:' :<25}")
    print(f"  {'Initial:':<20} {format_scientific(initial_solid_mass, 'kg')}")
    print(f"  {'Final:':<20} {format_scientific(final_solid_mass, 'kg')}")
    print(f"  {'Error:':<20} {format_percentage(solid_mass_error)}")
    
    print(f"\n{'Event Statistics:':<25}")
    print(f"  {'Agglomeration:':<20} {agg_events:,.0f} events")
    print(f"  {'Breakage:':<20} {break_events:,.0f} events")
    print(f"  {'Total:':<20} {total_events:,.0f} events")
    if break_events > 0:
        print(f"  {'Agg/Break ratio:':<20} {agg_events/break_events:.2f}")
    else:
        print(f"  {'Agg/Break ratio:':<20} N/A (no breakage)")
    
    print(f"\n{'Droplet Count:':<25}")
    print(f"  {'Expected:':<20} {expected_droplets:.2f}")
    print(f"  {'Actual:':<20} {actual_droplets:.2f}")
    print(f"  {'Accuracy:':<20} {actual_droplets/expected_droplets*100:.2f}%")
    
    print(f"\n{'Porosity Evolution:':<25}")
    print(f"  {'Initial:':<20} {cfg.INITIAL_POROSITY:.3f}")
    print(f"  {'Final (mean):':<20} {porosity_mean:.3f}")
    print(f"  {'Final (min/max):':<20} {porosity_min:.3f} / {porosity_max:.3f}")
    
    print(f"\n{'Particle Evolution:':<25}")
    print(f"  {'Initial n_comp (a0):':<20} {cfg.A0_INITIAL:,}")
    print(f"  {'Final n_comp:':<20} {solver.a_tot:,}")
    print(f"  {'Final n_phys:':<20} {np.sum(solver.W[:solver.a_tot])/solver.Vc:,.0f}")
    
    print(f"\n{'Performance:':<25}")
    print(f"  {'Real time:':<20} {elapsed_real:.2f} s")
    print(f"  {'Simulated time:':<20} {cfg.T_TOTAL:.1f} s")
    print(f"  {'Speedup:':<20} {cfg.T_TOTAL/elapsed_real:.1f}x")
    print(f"  {'Events/sec:':<20} {total_events/elapsed_real:,.0f}")
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    
    print_section("VALIDATION", "=")
    
    tolerance_liquid = 0.01  # 1%
    tolerance_mass = 0.001   # 0.1%
    
    checks = []
    
    # Check 1: Liquid addition accuracy
    if abs(error_liquid) < tolerance_liquid:
        print(f"✓ Liquid addition: PASS (error={format_percentage(error_liquid)})")
        checks.append(True)
    else:
        print(f"✗ Liquid addition: FAIL (error={format_percentage(error_liquid)}, tolerance={tolerance_liquid*100:.1f}%)")
        checks.append(False)
    
    # Check 2: Solid mass conservation
    if abs(solid_mass_error) < tolerance_mass:
        print(f"✓ Solid mass conservation: PASS (error={format_percentage(solid_mass_error)})")
        checks.append(True)
    else:
        print(f"✗ Solid mass conservation: FAIL (error={format_percentage(solid_mass_error)}, tolerance={tolerance_mass*100:.2f}%)")
        checks.append(False)
    
    # Check 3: Droplet count accuracy
    droplet_accuracy = actual_droplets / expected_droplets
    if 0.95 < droplet_accuracy < 1.05:
        print(f"✓ Droplet count: PASS (accuracy={droplet_accuracy*100:.2f}%)")
        checks.append(True)
    else:
        print(f"✗ Droplet count: FAIL (accuracy={droplet_accuracy*100:.2f}%, expected 95-105%)")
        checks.append(False)
    
    # Check 4: Stokes kernel active
    has_stokes = (
        hasattr(solver, 'kernel_manager') and
        solver.kernel_manager.agglomeration_acceptance_kernel is not None and
        solver.kernel_manager.agglomeration_acceptance_kernel.name == 'stokes_krit'
    )
    if has_stokes:
        print(f"✓ Stokes acceptance kernel: ACTIVE")
        print(f"  - U_coll: {solver.kernel_manager.agglomeration_acceptance_kernel.U_coll:.2f} m/s")
        print(f"  - binder_viscosity: {solver.kernel_manager.agglomeration_acceptance_kernel.binder_viscosity:.2f} Pa·s")
        print(f"  - h_a: {solver.kernel_manager.agglomeration_acceptance_kernel.h_a*1e9:.1f} nm")
        checks.append(True)
    else:
        print(f"✗ Stokes acceptance kernel: NOT FOUND")
        checks.append(False)
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(checks)
    total = len(checks)
    print(f"Checks passed: {passed}/{total}")
    
    if all(checks):
        print("\n🎉 ALL VALIDATION CHECKS PASSED!")
        return {
            'success': True,
            'checks_passed': passed,
            'checks_total': total,
            'liquid_error': error_liquid,
            'mass_error': solid_mass_error,
            'agg_events': agg_events,
            'break_events': break_events,
            'final_particles': solver.a_tot,
            'elapsed_time': elapsed_real,
        }
    else:
        print(f"\n⚠️  {total - passed} CHECK(S) FAILED!")
        return {
            'success': False,
            'checks_passed': passed,
            'checks_total': total,
            'failed_checks': [i for i, c in enumerate(checks) if not c],
        }


# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == '__main__':
    results = run_stokes_acceptance_test()
    
    if results['success']:
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("TEST COMPLETED WITH ERRORS")
        print("="*80)
        sys.exit(1)
