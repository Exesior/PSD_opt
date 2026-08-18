"""
Comprehensive Test: PowerLaw-Rumpf Breakage + Full Physics Suite.

This test verifies the complete MCPBE solver with all advanced physics modules:
- Nucleation: Liquid droplet addition during defined time window
- Agglomeration: Particle collisions with constant kernel
- Agglomeration Acceptance: Stokes criterion (Braumann et al. 2007)
- Breakage: PowerLaw-Rumpf with porosity/saturation-dependent strength
- Porosity Growth: Cone model for pore formation
- Compression: Porosity reduction over time
- Liquid Internalization: Capillary-driven pore filling (continuous)
- Liquid Internalization during Agglomeration: Contact pore trapping (event-based)

Test Configuration:
    - Initial particles: 1000 computational particles
    - Particle diameter: 500 µm (monodisperse)
    - Droplet diameter: 100 µm
    - Simulation duration: 10 seconds (extended run)
    - All physics modules active

Usage:
    python -m wmcpbe.Trials.test_powerlaw_rumpf_full
    
Or from the wmcpbe directory:
    python Trials/test_powerlaw_rumpf_full.py
"""

import numpy as np
import sys
import os
import time
from typing import Dict, Any

# Add parent directory to path for imports
# Robust approach that works with %runfile, direct execution, and module execution
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from wmcpbe import MCPBESolver


# =============================================================================
# Test Configuration
# =============================================================================

class TestConfig:
    """Physical and numerical parameters for comprehensive test."""
    
    # Seed for reproducibility
    SEED = 205
    
    # Time settings
    T_TOTAL = 100.0      # Total simulation time [s] - EXTENDED RUN
    T_WRITE = 0.2       # Output interval [s]
    
    # Particle properties (initial)
    PARTICLE_DIAMETER = 34e-6            # 700 µm
    PARTICLE_DENSITY = 500.0             # kg/m³ (solid material)
    INITIAL_POROSITY = 0.0                # 80% void fraction
    
    # Droplet properties
    DROPLET_DIAMETER = 20e-6             # 200 µm
    DROPLET_DENSITY = 1000.0              # kg/m³ (water)
    
    # Process parameters
    VOLUMETRIC_FLOW_RATE = 3e-10       # m³/s
    NUCLEATION_DURATION = 10.0         # s
    AGG_COEFFICIENT = 4            # Constant kernel coefficient [m³/s] - HIGH for testing
    BATCH_SIZE = 20                   # Wie viele Tropfen werden identisch verteilt?
    
    # Breakage parameters (PowerLaw-Rumpf)
    BREAKAGE_ENABLED = True
    PL_P1 = 4e13                         # Pre-factor [1/s·Pa·m^(-3*alpha)]
    PL_P2 = 1.0                       # Volume exponent in S = P1*G*V^P2
    G = 1000.0                        # Shear rate [1/s]
    BREAKRVAL = 4                     # Volume-based power law
    # Rumpf strength parameters
    RUMPF_K = 2.5                     # Fitting parameter dry [2.2-2.8] - MIN VALUE
    RUMPF_ALPHA = 1.0                 # Fitting parameter wet [1.0-1.33] - MIN VALUE
    RUMPF_GAMMA = 0.072               # Surface tension [N/m] - REDUCED (surfactant)
    RUMPF_DELTA = 0.0                 # Contact angle [rad] (perfect wetting)
    
    # Compression parameters
    COMPRESSION_ENABLED = True
    COMPRESSION_RATE = 0.02             # Porosity decay rate [1/s]
    MIN_POROSITY = 0.2                # Minimum achievable porosity
    
    # Liquid internalization parameters
    LIQ_INTERN_ENABLED = True
    LIQ_INTERN_RATE = 1e12             # Rate constant [1/(m³·s)]
    
    # Liquid internalization during agglomeration
    LIQ_INTERN_AGG_ENABLED = True
    
    # Agglomeration acceptance (Stokes criterion)
    STOKES_ENABLED = True
    BINDER_VISCOSITY = 0.1            # Pa·s
    COLLISION_VELOCITY = 0.5          # m/s
    H_A = 500e-9                      # m (half-distance of closest approach)
    
    # Numerical settings
    INITIAL_PARTICLES = 2000          # Computational particles
    INITIAL_WEIGHT = 600             # Weight per particle
    CONTROL_VOLUME = 1               # m³ 
    
    # Merger configuration
    MERGER_TOLERANCE = 1e-4               # Relative tolerance for matching (0.0001%)
    USE_HASH_INDEX = True    
    
    #Solver parameters
    AGG_DW_MIN = 1.0
    AGG_DW_MAX = 20.0
    BREAK_DW_MAX = 50.0
    SIZEEVAL = 0


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
    Run comprehensive MCPBE simulation with PowerLaw-Rumpf breakage and full physics.
    
    Returns:
        Dictionary containing test results and statistics
    """
    
    # -------------------------------------------------------------------------
    # Setup & Initialization
    # -------------------------------------------------------------------------
    
    print_section("COMPREHENSIVE MCPBE TEST: POWERLAW-RUMPF + FULL PHYSICS", "=")
    print("Physics Modules:")
    print("  - Nucleation: Liquid droplet addition")
    print("  - Agglomeration: Constant kernel")
    print("  - Agglomeration Acceptance: Stokes criterion")
    print("  - Breakage: PowerLaw-Rumpf (porosity/saturation-dependent)")
    print("  - Porosity Growth: Cone model")
    print("  - Compression: Exponential decay")
    print("  - Liquid Internalization: Continuous (capillary-driven)")
    print("  - Liquid Internalization (Agg): Event-based (contact pores)")
    
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
    print(f"  Agg. coefficient:   {format_scientific(cfg.AGG_COEFFICIENT, 'm³/s')}")
    
    print_section("BREAKAGE (POWERLAW-RUMPF)")
    print(f"  Enabled:            {cfg.BREAKAGE_ENABLED}")
    print(f"  P1:                 {format_scientific(cfg.PL_P1)}")
    print(f"  P2:                 {cfg.PL_P2}")
    print(f"  G:                  {cfg.G:.1f} 1/s")
    print(f"  BREAKRVAL:          {cfg.BREAKRVAL}")
    print(f"  Rumpf k:            {cfg.RUMPF_K}")
    print(f"  Rumpf alpha:        {cfg.RUMPF_ALPHA}")
    print(f"  Gamma:              {cfg.RUMPF_GAMMA:.3f} N/m")
    print(f"  Delta:              {cfg.RUMPF_DELTA:.2f} rad")
    
    print_section("COMPRESSION")
    print(f"  Enabled:            {cfg.COMPRESSION_ENABLED}")
    print(f"  Rate:               {cfg.COMPRESSION_RATE:.2f} 1/s")
    print(f"  Min porosity:       {cfg.MIN_POROSITY:.2f}")
    
    print_section("LIQUID INTERNALIZATION")
    print(f"  Continuous:         {cfg.LIQ_INTERN_ENABLED} (rate={cfg.LIQ_INTERN_RATE:.1e})")
    print(f"  During Agg:         {cfg.LIQ_INTERN_AGG_ENABLED}")
    
    print_section("AGGLOMERATION ACCEPTANCE (STOKES)")
    print(f"  Enabled:            {cfg.STOKES_ENABLED}")
    print(f"  Binder viscosity:   {cfg.BINDER_VISCOSITY:.2f} Pa·s")
    print(f"  Collision velocity: {cfg.COLLISION_VELOCITY:.2f} m/s")
    print(f"  h_a:                {cfg.H_A*1e9:.1f} nm")
    
    print_section("EXPECTED RESULTS")
    print(f"  Total liquid:       {format_scientific(expected_liquid, 'm³')}")
    print(f"  Total droplets:     {expected_droplets:.2f}")
    print(f"  Simulation time:    {cfg.T_TOTAL:.1f} s")
    print(f"  Initial particles:  {cfg.INITIAL_PARTICLES:,}")
    
    # -------------------------------------------------------------------------
    # Solver Initialization
    # -------------------------------------------------------------------------
    
    print_section("INITIALIZING SOLVER", "-")
    
    # Time vector
    t_vec = np.linspace(0.0, cfg.T_TOTAL, int(cfg.T_TOTAL / cfg.T_WRITE) + 1)
    
    # Create RNG with fixed seed
    rng = np.random.default_rng(cfg.SEED)
    
    print(f"\nCreating solver (seed={cfg.SEED})...")
    
    print("\nKernel configuration:")
    print("  - Aggregation: constant kernel")
    print("  - Agg. Acceptance: stokes_krit")
    print("  - Breakage: powerlaw_rumpf (porosity/saturation-dependent strength)")
    print("  - Porosity Growth: cone_model")
    print("  - Porosity Compression: porosity_compression kernel")
    print("  - Liq. Internalization: continuous")
    print("  - Liq. Internalization (Agg): braumann_2007")
    print("  - Propensity mode: moment (O(n) acceleration)")
    
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=True,
        load_attr=False,
        init=True,
        rng=rng,
        # Aggregation kernel
        agg_kernel_name='shear_chin1998',
        agg_kernel_params={'corr_beta': cfg.AGG_COEFFICIENT,'g':cfg.G},
        # Agglomeration acceptance kernel
        agg_acceptance_kernel_name='stokes_krit',
        agg_acceptance_kernel_params={
            'U_coll': cfg.COLLISION_VELOCITY,
            'binder_viscosity': cfg.BINDER_VISCOSITY,
            'rho_solid': cfg.PARTICLE_DENSITY,
            'rho_liquid': cfg.DROPLET_DENSITY,
            'h_a': cfg.H_A,
        },
        # Breakage kernel (PowerLaw-Rumpf)
        break_kernel_name='powerlaw_rumpf',
        break_kernel_params={
            'p1': cfg.PL_P1,
            'p2': cfg.PL_P2,
            'g': cfg.G,
            'breakrval': cfg.BREAKRVAL,
            'k': cfg.RUMPF_K,
            'alpha': cfg.RUMPF_ALPHA,
            'gamma': cfg.RUMPF_GAMMA,
            'delta': cfg.RUMPF_DELTA,
            'x_s': None,  # Compute from X0
        },
        # Porosity growth kernel
        porosity_growth_kernel_name='cone_model',
        porosity_growth_kernel_params={},
        # Continuous processes kernels (compression + liquid internalization)
        porosity_compression_kernel_name='porosity_compression',
        porosity_compression_kernel_params={
            'rate': cfg.COMPRESSION_RATE,
            'min_porosity': cfg.MIN_POROSITY,
        },
        liquid_internalization_kernel_name='liquid_internalization',
        liquid_internalization_kernel_params={
            'k_int': cfg.LIQ_INTERN_RATE,
        },
        # Liquid internalization during agglomeration (event-based)
        liq_internalisation_agglomeration_kernel_name='liq_internalisation_agglomeration',
        liq_internalisation_agglomeration_kernel_params={},
    )
    solver.mcpbe_debug = False 
    # Enable moment-based propensity calculation for O(n) acceleration
    # (instead of default O(n²) pairwise evaluation)
    solver.agg_propensity_mode = "moment"
    
    # Configure process type
    solver.process_type = "mix"  # Agglomeration + Breakage
    solver.recon_enable = False

    # MC packet sizes -- must be set before _initialize_samplers() reads them.
    # See TestConfig.AGG_DW_MIN/_MAX/BREAK_DW_MAX above (B-06).
    solver.agg_dW_min = cfg.AGG_DW_MIN
    solver.agg_dW_max = cfg.AGG_DW_MAX
    solver.break_dW_max = cfg.BREAK_DW_MAX
    solver.SIZEEVAL = cfg.SIZEEVAL
    
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
    v_dry = solver.V_flat[-1, :solver.a_tot]
    porosity = solver.porosity[:solver.a_tot]
    weights = solver.W[:solver.a_tot]
    
    valid_poro = ~np.isnan(porosity)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid_poro] = v_dry[valid_poro] * (1.0 - porosity[valid_poro])
    v_solid[~valid_poro] = v_dry[~valid_poro]  # Vollkörper
    
    initial_solid_volume = np.sum(v_solid * weights)
    initial_solid_mass = initial_solid_volume * cfg.PARTICLE_DENSITY
    
    print(f"  a_tot:      {solver.a_tot}")
    print(f"  Vc:         {format_scientific(solver.Vc, 'm³')}")
    print(f"  W[0:5]:     {solver.W[:5]}")
    print(f"  porosity:   {solver.porosity[0]:.3f}")
    print(f"  saturation: {solver.saturation[0]:.3f}")
    
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
        batch_size = cfg.BATCH_SIZE,
    )
    
    # Configure continuous processes (liquid internalization + compression)
    # Note: Compression is now handled here, NOT via create_compression_handler()
    if cfg.LIQ_INTERN_ENABLED or cfg.COMPRESSION_ENABLED:
        print("Configuring continuous processes...")
        solver.create_continuous_processes_handler(
            enabled=True,
            k_int=cfg.LIQ_INTERN_RATE if cfg.LIQ_INTERN_ENABLED else 0.0,
            compression_enabled=cfg.COMPRESSION_ENABLED,
            compression_rate=cfg.COMPRESSION_RATE,
            min_porosity=cfg.MIN_POROSITY,
        )
    
    # -------------------------------------------------------------------------
    # Run Simulation
    # -------------------------------------------------------------------------
    
    print_section("RUNNING SIMULATION", "=")
    print(f"Simulating {cfg.T_TOTAL:.1f} s with all physics modules active...\n")
    
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
    
    # Accepted/rejected agglomeration events (if Stokes criterion active)
    if hasattr(solver, '_agg_accepted_count'):
        agg_accepted = solver._agg_accepted_count
        agg_rejected = solver._agg_rejected_count if hasattr(solver, '_agg_rejected_count') else 0
    else:
        agg_accepted = agg_events
        agg_rejected = 0
    
    # Liquid in system (sum over all particles)
    liquid_in_system = np.sum(solver.liquid_volume[:solver.a_tot] * solver.W[:solver.a_tot])
    
    # Internal vs external liquid
    liquid_internal = np.sum(solver.liquid_volume[:solver.a_tot] * solver.W[:solver.a_tot])
    # External liquid would need separate tracking
    
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
    print(f"  Error:              {format_percentage(solid_mass_error)}")
    print(f"  n_comp:             {solver.a_tot}")
    print(f"  n_Vollkörper:       {np.sum(~valid_poro)}")
    print(f"  n_porous:           {np.sum(valid_poro)}")
    
    # Porosity statistics
    porosity_active = solver.porosity[:solver.a_tot]
    porosity_mean = np.mean(porosity_active[~np.isnan(porosity_active)])
    porosity_min = np.nanmin(porosity_active)
    porosity_max = np.nanmax(porosity_active)
    
    # Saturation statistics
    saturation_active = solver.saturation[:solver.a_tot]
    sat_valid = ~np.isnan(saturation_active)
    saturation_mean = np.mean(saturation_active[sat_valid]) if np.any(sat_valid) else 0.0
    saturation_min = np.nanmin(saturation_active)
    saturation_max = np.nanmax(saturation_active)
    
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
    print(f"  {'Accepted:':<20} {agg_accepted:,.0f} events")
    if agg_rejected > 0:
        print(f"  {'Rejected (Stokes):':<20} {agg_rejected:,.0f} events")
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
    print(f"  {'Δ Porosity:':<20} {cfg.INITIAL_POROSITY - porosity_mean:.3f}")
    
    print(f"\n{'Saturation Evolution:':<25}")
    print(f"  {'Initial:':<20} 0.000")
    print(f"  {'Final (mean):':<20} {saturation_mean:.3f}")
    print(f"  {'Final (min/max):':<20} {saturation_min:.3f} / {saturation_max:.3f}")
    
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
    liquid_tolerance = 0.05  # 5% (more lenient for long runs)
    liquid_ok = abs(error_liquid) < liquid_tolerance
    
    # Check droplet count
    droplet_tolerance = 0.05  # 5%
    droplet_ok = abs(actual_droplets/expected_droplets - 1.0) < droplet_tolerance
    
    # Check solid mass conservation (should be EXACT - no creation/destruction)
    solid_tolerance = 1e-8  # Slightly relaxed for long runs
    solid_ok = abs(solid_mass_error) < solid_tolerance
    
    # Check events occurred
    events_ok = agg_events > 0
    
    # Check breakage occurred (if enabled)
    breakage_ok = break_events > 0 if cfg.BREAKAGE_ENABLED else True
    
    # Check compression occurred: porosity must have CHANGED, not necessarily
    # decreased. Der alte Check verlangte porosity_mean < INITIAL_POROSITY. Das
    # war nie ein Kompressions-Test: er war nur erfuellt, solange die Nucleation
    # die Porositaet auf 0 setzte und den Mittelwert nach unten riss. Seit die
    # Porositaet erhalten bleibt, konkurrieren zwei echte Effekte -- Kompression
    # zieht eps mit `rate` Richtung MIN_POROSITY, cone_model schiebt bei jeder
    # Agglomeration Porenvolumen nach -- und der Mittelwert darf in beide
    # Richtungen laufen.
    poro_delta = abs(porosity_mean - cfg.INITIAL_POROSITY)
    comp_ok = poro_delta > 1e-6 if cfg.COMPRESSION_ENABLED else True
    
    # Summary
    all_ok = liquid_ok and droplet_ok and solid_ok and events_ok and breakage_ok and comp_ok
    
    print(f"\n  {'Liquid conservation:':<25} {'✓ PASS' if liquid_ok else '✗ FAIL'} ({format_percentage(error_liquid)})")
    print(f"  {'Droplet accuracy:':<25} {'✓ PASS' if droplet_ok else '✗ FAIL'} ({actual_droplets/expected_droplets*100:.2f}%)")
    print(f"  {'Solid mass conservation:':<25} {'✓ PASS' if solid_ok else '✗ FAIL'} ({format_percentage(solid_mass_error)})")
    print(f"  {'Agglomeration active:':<25} {'✓ PASS' if events_ok else '✗ FAIL'} ({agg_events:,.0f} events)")
    if agg_rejected > 0:
        print(f"  {'Stokes rejection active:':<25} {'✓ PASS' if agg_rejected > 0 else '✗ FAIL'} ({agg_rejected:,.0f} rejected)")
    if cfg.BREAKAGE_ENABLED:
        print(f"  {'Breakage active:':<25} {'✓ PASS' if breakage_ok else '✗ FAIL'} ({break_events:,.0f} events)")
    if cfg.COMPRESSION_ENABLED:
        print(f"  {'Porosity evolved:':<25} {'✓ PASS' if comp_ok else '✗ FAIL'} (|Δporo| = {poro_delta:.3f}, {cfg.INITIAL_POROSITY:.3f} → {porosity_mean:.3f})")
    
    print(f"\n{'OVERALL:':<25} {'✓ ALL TESTS PASSED' if all_ok else '✗ SOME TESTS FAILED'}")
    
    # -------------------------------------------------------------------------
    # Return Results Dictionary
    # -------------------------------------------------------------------------
    
    return {
        'success': all_ok,
        'liquid_error': float(error_liquid),
        'system_error': float(error_system),
        'solid_mass_error': float(solid_mass_error),
        'initial_solid_mass': float(initial_solid_mass),
        'final_solid_mass': float(final_solid_mass),
        'droplet_accuracy': float(actual_droplets / expected_droplets),
        'agg_events': int(agg_events),
        'agg_accepted': int(agg_accepted),
        'agg_rejected': int(agg_rejected),
        'break_events': int(break_events),
        'total_events': int(total_events),
        'final_n_comp': int(solver.a_tot),
        'final_n_phys': float(np.sum(solver.W[:solver.a_tot]) / solver.Vc),
        'porosity_mean': float(porosity_mean),
        'porosity_change': float(cfg.INITIAL_POROSITY - porosity_mean),
        'saturation_mean': float(saturation_mean),
        'elapsed_real': float(elapsed_real),
        'speedup': float(cfg.T_TOTAL / elapsed_real),
        'events_per_sec': float(total_events / elapsed_real),
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    results = run_comprehensive_test()
    
    print_section("TEST COMPLETE", "=")
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)
