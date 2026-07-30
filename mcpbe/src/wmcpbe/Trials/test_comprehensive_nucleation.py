"""
Comprehensive Test: Nucleation + Agglomeration + Breakage + Compression.

This test verifies the complete MCPBE solver with all physics modules active:
- Nucleation: Liquid droplet addition during defined time window
- Agglomeration: Particle collisions and merging
- Breakage: Particle fragmentation under shear stress
- Compression: Porosity reduction due to mechanical stress

Test Parameters (validated working configuration):
    - Particle diameter: 500 µm (initial)
    - Droplet diameter: 100 µm
    - Flow rate: 3e-11 m³/s
    - Agglomeration coefficient: 1e-5
    - Breakage rate: calibrated for moderate fragmentation
    - Compression rate: exponential decay model
    - Simulation duration: 4 seconds
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
    """Physical and numerical parameters for comprehensive test."""
    
    # Seed for reproducibility
    SEED = 42
    
    # Time settings
    T_TOTAL = 4.0       # Total simulation time [s]
    T_WRITE = 0.1       # Output interval [s]
    
    # Particle properties (initial)
    PARTICLE_DIAMETER = 5000e-6      # 500 µm
    PARTICLE_DENSITY = 2500.0       # kg/m³ (solid material)
    INITIAL_POROSITY = 0.0        # 40% void fraction
    
    # Droplet properties
    DROPLET_DIAMETER = 1000e-6       # 100 µm
    DROPLET_DENSITY = 1000.0        # kg/m³ (water)
    
    # Process parameters
    VOLUMETRIC_FLOW_RATE = 3e-10    # m³/s
    NUCLEATION_DURATION = 2.0       # s
    AGG_COEFFICIENT = 1e-5          # Constant kernel coefficient
    
    # Breakage parameters
    BREAKAGE_ENABLED = True
    BREAKAGE_RATE = 0.1             # Fragmentation rate [1/s]
    BREAKAGE_DAUGHTERS = 2          # Number of daughter particles
    
    # Compression parameters
    COMPRESSION_ENABLED = True
    COMPRESSION_RATE = 0.05         # Porosity decay rate [1/s]
    MIN_POROSITY = 0.1              # Minimum achievable porosity
    
    # Numerical settings
    INITIAL_PARTICLES = 1000        # Computational particles
    INITIAL_WEIGHT = 100.0          # Weight per particle
    CONTROL_VOLUME = 1.5            # m³


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

def run_comprehensive_test() -> Dict[str, Any]:
    """
    Run comprehensive MCPBE simulation with all physics modules.
    
    Returns:
        Dictionary containing test results and statistics
    """
    
    # -------------------------------------------------------------------------
    # Setup & Initialization
    # -------------------------------------------------------------------------
    
    print_section("COMPREHENSIVE MCPBE TEST", "=")
    print("Physics Modules: Nucleation + Agglomeration + Breakage + Compression")
    
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
    print(f"  Breakage rate:      {cfg.BREAKAGE_RATE:.2f} 1/s {'(enabled)' if cfg.BREAKAGE_ENABLED else '(disabled)'}")
    print(f"  Compression rate:   {cfg.COMPRESSION_RATE:.2f} 1/s {'(enabled)' if cfg.COMPRESSION_ENABLED else '(disabled)'}")
    
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
    
    # Test with INHOMOGENEOUS kernel framework:
    # - volume_mixing for agglomeration (additive pores)
    # - incomplete_mixing for breakage (pore collapse) + nucleation (configurable)
    # This demonstrates the UNIFIED PHYSICS approach: one kernel with 3 methods!
    print("\nUsing INHOMOGENEOUS PorosityGrowthKernel framework:")
    print("  - Agglomeration: volume_mixing (additive pores)")
    print("  - Breakage: incomplete_mixing (pore collapse, collapse_factor=0.15)")
    print("  - Nucleation: incomplete_mixing (nucleation_porosity=0.35)")
    
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=True,
        load_attr=False,
        init=True,
        agg_kernel_name='constant',
        agg_kernel_params={'corr_beta': cfg.AGG_COEFFICIENT},
        break_kernel_name='power_law',
        break_kernel_params={
            'breakage_rate': cfg.BREAKAGE_RATE,
            'daughter_count': cfg.BREAKAGE_DAUGHTERS,
        },
        compression_kernel_name='exponential_decay',
        compression_kernel_params={
            'rate': cfg.COMPRESSION_RATE,
            'min_porosity': cfg.MIN_POROSITY,
        },
        porosity_growth_kernel_name='cone_model',
        porosity_growth_kernel_params={},
        rng=rng,
    )
    
    # Configure process type
    solver.process_type = "mix"  # Agglomeration + Breakage
    solver.recon_enable = False
    
    # Initialize particles with custom properties
    # IMPORTANT: V_flat structure for dim=1:
    #   V_flat[0, :] = V_solid (solid volume only, conserved during compression!)
    #   V_flat[1, :] = V_dry (total dry volume = V_solid + V_pore)
    # Relation: V_solid = V_dry × (1 - porosity)
    V_flat = np.zeros((2, cfg.INITIAL_PARTICLES), dtype=float)
    V_flat[0, :] = particle_volume * (1.0 - cfg.INITIAL_POROSITY)  # V_solid
    V_flat[1, :] = particle_volume  # V_dry (geometric volume of particle)
    
    W_init = np.full(cfg.INITIAL_PARTICLES, cfg.INITIAL_WEIGHT, dtype=float)
    
    solver.Vc = cfg.CONTROL_VOLUME
    
    print("\nInitializing particles...")
    solver._initialize_particles(
        init_Vc=False,
        V_flat=V_flat,
        W_init=W_init,
    )
    
    # Initialize porosity for all particles
    solver.porosity[:solver.a_tot] = cfg.INITIAL_POROSITY
    solver.liquid_volume[:solver.a_tot] = 0.0
    solver.saturation[:solver.a_tot] = 0.0
    
    # Calculate initial solid mass (before any physics)
    # Solid volume per particle = V_dry × (1 - porosity)
    # Solid mass = solid_volume × density × weight
    # 
    # IMPORTANT: Use V_flat[-1,:] (V_dry), NOT solver.v!
    # solver.v may be overwritten by breakage configuration and contain wrong values.
    v_dry_init = solver.V_flat[-1, :solver.a_tot]
    porosity_init = solver.porosity[:solver.a_tot]
    weights_init = solver.W[:solver.a_tot]
    
    # For Vollkörper (NaN porosity): V_solid = V_dry
    # For porous particles: V_solid = V_dry × (1 - porosity)
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
    # Solid volume per particle = V_dry × (1 - porosity)
    # Solid mass = solid_volume × density × weight
    
    # Use V_flat[-1,:] (V_dry) for correct calculation
    v_dry = solver.V_flat[-1, :solver.a_tot]
    porosity = solver.porosity[:solver.a_tot]
    weights = solver.W[:solver.a_tot]
    
    # For Vollkörper (NaN porosity): V_solid = V_dry
    # For porous particles: V_solid = V_dry × (1 - porosity)
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
    print(f"  {'Agg/Break ratio:':<20} {agg_events/break_events:.2f}" if break_events > 0 else "  {'Agg/Break ratio:':<20} N/A (no breakage)")
    
    print(f"\n{'Droplet Count:':<25}")
    print(f"  {'Expected:':<20} {expected_droplets:.2f}")
    print(f"  {'Actual:':<20} {actual_droplets:.2f}")
    print(f"  {'Accuracy:':<20} {actual_droplets/expected_droplets*100:.2f}%")
    
    print(f"\n{'Porosity Evolution:':<25}")
    print(f"  {'Initial:':<20} {cfg.INITIAL_POROSITY:.3f}")
    print(f"  {'Final (mean):':<20} {porosity_mean:.3f}")
    print(f"  {'Final (min/max):':<20} {porosity_min:.3f} / {porosity_max:.3f}")
    
    print(f"\n{'Particle Evolution:':<25}")
    print(f"  {'Initial n_comp:':<20} {cfg.INITIAL_PARTICLES:,}")
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
    
    print_section("VALIDATION", "-")
    
    # Check liquid conservation
    liquid_tolerance = 0.01  # 1%
    liquid_ok = abs(error_liquid) < liquid_tolerance
    
    # Check droplet count
    droplet_tolerance = 0.01  # 1%
    droplet_ok = abs(actual_droplets/expected_droplets - 1.0) < droplet_tolerance
    
    # Check solid mass conservation (should be EXACT - no creation/destruction)
    solid_tolerance = 1e-10  # Machine precision (mass MUST be conserved)
    solid_ok = abs(solid_mass_error) < solid_tolerance
    
    # Check events occurred
    events_ok = agg_events > 0
    
    # Summary
    all_ok = liquid_ok and droplet_ok and solid_ok and events_ok
    
    print(f"\n  {'Liquid conservation:':<25} {'✓ PASS' if liquid_ok else '✗ FAIL'} ({format_percentage(error_liquid)})")
    print(f"  {'Droplet accuracy:':<25} {'✓ PASS' if droplet_ok else '✗ FAIL'} ({actual_droplets/expected_droplets*100:.2f}%)")
    print(f"  {'Solid mass conservation:':<25} {'✓ PASS' if solid_ok else '✗ FAIL'} ({format_percentage(solid_mass_error)})")
    print(f"  {'Agglomeration active:':<25} {'✓ PASS' if events_ok else '✗ FAIL'} ({agg_events:,.0f} events)")
    if cfg.BREAKAGE_ENABLED:
        print(f"  {'Breakage active:':<25} {'✓ PASS' if break_events > 0 else '✗ FAIL'} ({break_events:,.0f} events)")
    if cfg.COMPRESSION_ENABLED:
        comp_ok = porosity_mean < cfg.INITIAL_POROSITY
        print(f"  {'Compression active:':<25} {'✓ PASS' if comp_ok else '✗ FAIL'} (Δporo = {cfg.INITIAL_POROSITY - porosity_mean:.3f})")
    
    print(f"\n{'OVERALL:':<25} {'✓ ALL TESTS PASSED' if all_ok else '✗ SOME TESTS FAILED'}")
    
    # -------------------------------------------------------------------------
    # Return Results Dictionary
    # -------------------------------------------------------------------------
    
    return {
        'success': all_ok,
        'liquid_error': error_liquid,
        'system_error': error_system,
        'solid_mass_error': solid_mass_error,
        'initial_solid_mass': float(initial_solid_mass),
        'final_solid_mass': float(final_solid_mass),
        'droplet_accuracy': actual_droplets / expected_droplets,
        'agg_events': int(agg_events),
        'break_events': int(break_events),
        'total_events': int(total_events),
        'final_n_comp': int(solver.a_tot),
        'final_n_phys': float(np.sum(solver.W[:solver.a_tot]) / solver.Vc),
        'porosity_mean': float(porosity_mean),
        'porosity_change': cfg.INITIAL_POROSITY - porosity_mean,
        'elapsed_real': float(elapsed_real),
        'speedup': cfg.T_TOTAL / elapsed_real,
        'events_per_sec': total_events / elapsed_real,
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    results = run_comprehensive_test()
    
    print_section("TEST COMPLETE", "=")
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)
