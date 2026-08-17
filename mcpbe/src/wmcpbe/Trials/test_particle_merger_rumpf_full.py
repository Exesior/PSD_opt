"""
A/B Test: Particle Merger Performance Comparison - Full Physics Suite.

This test compares solver performance WITH and WITHOUT the particle merging
optimization under FULL PHYSICS conditions:
    - Breakage: PowerLaw-Rumpf (porosity/saturation-dependent strength)
    - Agglomeration: Constant kernel with Stokes acceptance criterion
    - Nucleation: Liquid droplet addition during defined time window
    - Porosity Growth: Cone model (geometric pore formation)
    - Compression: Exponential porosity decay
    - Liquid Internalization: Continuous (capillary-driven) + Event-based (during agg)

This is the REAL-WORLD scenario where the merger should shine:
    - Diverse particle properties (V_dry, poro, liquid, sat)
    - High breakage rates → many fragments to merge
    - Agglomeration → property mixing → more merge opportunities
    
Usage:
    python -m wmcpbe.Trials.test_particle_merger_rumpf_full
    
Or from the wmcpbe directory:
    python Trials/test_particle_merger_rumpf_full.py
"""

import numpy as np
import sys
import os
import time
from typing import Dict, Any, Tuple

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from wmcpbe import MCPBESolver


# =============================================================================
# Test Configuration (MIRRORED FROM test_powerlaw_rumpf_full.py)
# =============================================================================

class TestConfig:
    """Physical and numerical parameters for comprehensive test."""
    
    # Seed for reproducibility
    SEED = 42
    
    # Time settings
    T_TOTAL = 10.0      # Total simulation time [s] - EXTENDED RUN
    T_WRITE = 0.2       # Output interval [s]
    
    # Particle properties (initial)
    PARTICLE_DIAMETER = 34e-6            # 700 µm
    PARTICLE_DENSITY = 2500.0             # kg/m³ (solid material)
    INITIAL_POROSITY = 0.8                # 80% void fraction
    
    # Droplet properties
    DROPLET_DIAMETER = 10e-6             # 200 µm
    DROPLET_DENSITY = 1000.0              # kg/m³ (water)
    
    # Process parameters
    VOLUMETRIC_FLOW_RATE = 1e-9       # m³/s
    NUCLEATION_DURATION = 4.0         # s
    AGG_COEFFICIENT = 8000            # Constant kernel coefficient [m³/s] - HIGH for testing
    BATCH_SIZE = 100                   # Wie viele Tropfen werden identisch verteilt?   
    
    # Breakage parameters (PowerLaw-Rumpf)
    BREAKAGE_ENABLED = True
    PL_P1 = 50                         # Pre-factor [1/s·Pa·m^(-3*alpha)]
    PL_P2 = 1.0                       # Volume exponent in S = P1*G*V^P2
    G = 1000.0                        # Shear rate [1/s]
    BREAKRVAL = 4                     # Volume-based power law
    # Rumpf strength parameters
    RUMPF_K = 2.5                     # Fitting parameter dry [2.2-2.8] - MIN VALUE
    RUMPF_ALPHA = 1.0                 # Fitting parameter wet [1.0-1.33] - MIN VALUE
    RUMPF_GAMMA = 0.036               # Surface tension [N/m] - REDUCED (surfactant)
    RUMPF_DELTA = 0.0                 # Contact angle [rad] (perfect wetting)
    
    # Compression parameters
    COMPRESSION_ENABLED = True
    COMPRESSION_RATE = 0.2             # Porosity decay rate [1/s]
    MIN_POROSITY = 0.2                # Minimum achievable porosity
    
    # Liquid internalization parameters
    LIQ_INTERN_ENABLED = True
    LIQ_INTERN_RATE = 1e-8             # Rate constant [1/(m³·s)]
    
    # Liquid internalization during agglomeration
    LIQ_INTERN_AGG_ENABLED = True
    
    # Agglomeration acceptance (Stokes criterion)
    STOKES_ENABLED = True
    BINDER_VISCOSITY = 0.1            # Pa·s
    COLLISION_VELOCITY = 0.5          # m/s
    H_A = 500e-9                      # m (half-distance of closest approach)
    STOKES_DEBUG = False              # Enable ONLY for debugging (floods terminal!)
    
    # Numerical settings
    INITIAL_PARTICLES = 2000          # Computational particles
    INITIAL_WEIGHT = 700             # Weight per particle
    CONTROL_VOLUME = 1               # m³ 
    
    # Merger configuration
    MERGER_TOLERANCE = 1e-2               # Relative tolerance for matching (0.0001%)
    USE_HASH_INDEX = True                 # Enable hash-based O(1) lookup



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


def create_solver(
    t_vec: np.ndarray,
    rng: np.random.Generator,
    enable_merger: bool,
    merger_tol: float = 1e-6,
    use_hash: bool = True
) -> MCPBESolver:
    """
    Create and configure solver with optional particle merger.
    
    Parameters
    ----------
    t_vec : np.ndarray
        Time vector for output
    rng : np.random.Generator
        RNG instance (shared between runs for identical randomness)
    enable_merger : bool
        Whether to enable particle merging optimization
    merger_tol : float
        Relative tolerance for property matching
    use_hash : bool
        Whether to use hash-based index for O(1) lookup
    
    Returns
    -------
    MCPBESolver
        Configured solver instance (NOT yet initialized)
    """
    cfg = TestConfig()
    
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=False,  # Disable verbose for cleaner test output
        load_attr=False,
        init=True,
        rng=rng,
        # Aggregation kernel (constant, high rate for testing)
        agg_kernel_name='shear_chin1998',
        agg_kernel_params={'corr_beta': cfg.AGG_COEFFICIENT,'g':cfg.G},
        # Agglomeration acceptance kernel (Stokes criterion)
        agg_acceptance_kernel_name='stokes_krit',
        agg_acceptance_kernel_params={
            'U_coll': cfg.COLLISION_VELOCITY,
            'binder_viscosity': cfg.BINDER_VISCOSITY,
            'rho_solid': cfg.PARTICLE_DENSITY,
            'rho_liquid': cfg.DROPLET_DENSITY,
            'h_a': cfg.H_A,
            'debug': cfg.STOKES_DEBUG,  # Enable per-collision debug output
        },
        # Breakage kernel (PowerLaw-Rumpf with porosity/saturation dependence)
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
        # Porosity growth (Cone model - geometric pore formation)
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
    solver.agg_propensity_mode = "moment"  # O(n) acceleration
    solver.process_type = "mix"  # Agglomeration + Breakage
    solver.recon_enable = False  # Disable reconstruction for isolated test
    
    # Configure particle merger BEFORE _initialize_samplers()
    # IMPORTANT: Do NOT call _initialize_samplers() here!
    # It will be called in initialize_particles() AFTER particles are created.
    solver._enable_particle_merging = enable_merger
    if enable_merger:
        solver._fragment_merge_tol = merger_tol
        solver._use_merger_hash_index = use_hash
    
    return solver


def initialize_particles(solver: MCPBESolver) -> Tuple[float, float]:
    """
    Initialize particles with uniform properties.
    
    Returns
    -------
    (initial_solid_volume, initial_solid_mass) : Tuple[float, float]
    """
    cfg = TestConfig()
    
    # Calculate particle volume
    particle_radius = cfg.PARTICLE_DIAMETER / 2.0
    particle_volume = (4.0 / 3.0) * np.pi * particle_radius ** 3
    
    # V_flat structure for dim=1:
    #   V_flat[0, :] = V_solid
    #   V_flat[1, :] = V_dry
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
    
    # Initialize porosity for all particles
    solver.porosity[:solver.a_tot] = cfg.INITIAL_POROSITY
    solver.liquid_volume[:solver.a_tot] = 0.0
    solver.saturation[:solver.a_tot] = 0.0
    
    # Calculate initial solid mass
    v_dry = solver.V_flat[-1, :solver.a_tot]
    porosity = solver.porosity[:solver.a_tot]
    weights = solver.W[:solver.a_tot]
    
    valid_poro = ~np.isnan(porosity)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid_poro] = v_dry[valid_poro] * (1.0 - porosity[valid_poro])
    v_solid[~valid_poro] = v_dry[~valid_poro]
    
    initial_solid_volume = np.sum(v_solid * weights)
    initial_solid_mass = initial_solid_volume * cfg.PARTICLE_DENSITY
    
    # Initialize samplers (this calls _prepare_break_config which creates _particle_merger)
    solver._initialize_samplers()
    
    # DEBUG: Check if merger was properly initialized AFTER particles exist
    if hasattr(solver, '_enable_particle_merging') and solver._enable_particle_merging:
        print(f"\n[DEBUG post-init] Merger initialization check:")
        print(f"  _enable_particle_merging: {solver._enable_particle_merging}")
        print(f"  Has _particle_merger: {hasattr(solver, '_particle_merger')}")
        print(f"  _particle_merger object: {solver._particle_merger}")
        if solver._particle_merger is not None:
            print(f"  Merger tol_rel: {solver._particle_merger.tol_rel}")
            print(f"  Merger use_hash_index: {solver._particle_merger.use_hash_index}")
            print(f"  Hash index size: {len(solver._particle_merger._hash_index)}")
            print(f"  Particles in hash: {sum(len(v) for v in solver._particle_merger._hash_index.values())}")
            print(f"  a_tot (active particles): {solver.a_tot}")
            if len(solver._particle_merger._hash_index) == 0:
                print(f"  ⚠️  WARNING: Hash index is EMPTY! Merger won't find any matches.")
                print(f"     This is expected - hash is populated on-demand during breakage/agg events.")
        else:
            print(f"  ❌ ERROR: Merger should be enabled but is None!")
    
    # Configure nucleation
    solver.create_nucleation_handler(
        enabled=True,
        volumetric_flow_rate=cfg.VOLUMETRIC_FLOW_RATE,
        droplet_diameter=cfg.DROPLET_DIAMETER,
        liquid_addition_start=0.0,
        liquid_addition_duration=cfg.NUCLEATION_DURATION,
        batch_size=cfg.BATCH_SIZE,
    )
    
    # Configure continuous processes (liquid internalization + compression)
    if cfg.LIQ_INTERN_ENABLED or cfg.COMPRESSION_ENABLED:
        solver.create_continuous_processes_handler(
            enabled=True,
            k_int=cfg.LIQ_INTERN_RATE if cfg.LIQ_INTERN_ENABLED else 0.0,
            compression_enabled=cfg.COMPRESSION_ENABLED,
            compression_rate=cfg.COMPRESSION_RATE,
            min_porosity=cfg.MIN_POROSITY,
        )
    
    return initial_solid_volume, initial_solid_mass


def run_simulation(solver: MCPBESolver, label: str) -> Dict[str, Any]:
    """
    Run simulation and collect statistics.
    
    Parameters
    ----------
    solver : MCPBESolver
        Initialized solver instance
    label : str
        Label for output ("WITH MERGER" or "WITHOUT MERGER")
    
    Returns
    -------
    dict
        Dictionary with performance metrics and statistics
    """
    cfg = TestConfig()
    
    print(f"\n{'='*80}")
    print(f"RUNNING: {label}")
    print(f"{'='*80}")
    
    # Pre-solve state
    print(f"\nInitial state:")
    print(f"  n_comp:     {solver.a_tot:,}")
    print(f"  n_phys:     {np.sum(solver.W[:solver.a_tot])/solver.Vc:,.0f}")
    print(f"  Vc:         {format_scientific(solver.Vc, 'm³')}")
    print(f"  porosity:   {solver.porosity[0]:.3f}")
    print(f"  saturation: {solver.saturation[0]:.3f}")
    
    # Pre-solve merger status check
    if hasattr(solver, '_particle_merger') and solver._particle_merger is not None:
        print(f"\n[DEBUG pre-solve] Merger ready:")
        print(f"  Hash index size: {len(solver._particle_merger._hash_index)}")
        print(f"  Initial particles in index: {sum(len(v) for v in solver._particle_merger._hash_index.values())}")
    else:
        print(f"\n[DEBUG pre-solve] No merger active (expected for WITHOUT MERGER run)")
    
    # Run simulation
    start_time = time.time()
    
    try:
        solver.solve(maxiter=int(1e9))
    except Exception as e:
        print(f"\n❌ ERROR during solve: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    
    elapsed_real = time.time() - start_time
    
    # Post-solve state
    print(f"\nFinal state:")
    print(f"  n_comp:     {solver.a_tot:,}")
    print(f"  n_phys:     {np.sum(solver.W[:solver.a_tot])/solver.Vc:,.0f}")
    
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
    
    print(f"\nEvent statistics:")
    print(f"  Agglomeration:      {agg_events:,.0f} events")
    print(f"  Accepted:           {agg_accepted:,.0f} events")
    if agg_rejected > 0:
        print(f"  Rejected (Stokes):  {agg_rejected:,.0f} events")
    print(f"  Breakage:           {break_events:,.0f} events")
    print(f"  Total:              {total_events:,.0f} events")
    if break_events > 0:
        print(f"  Agg/Break ratio:    {agg_events/break_events:.2f}")
    
    # Solid mass conservation check
    v_dry = solver.V_flat[-1, :solver.a_tot]
    porosity = solver.porosity[:solver.a_tot]
    weights = solver.W[:solver.a_tot]
    
    valid_poro = ~np.isnan(porosity)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid_poro] = v_dry[valid_poro] * (1.0 - porosity[valid_poro])
    v_solid[~valid_poro] = v_dry[~valid_poro]
    
    final_solid_volume = np.sum(v_solid * weights)
    final_solid_mass = final_solid_volume * cfg.PARTICLE_DENSITY
    
    # Nucleation statistics
    nuc_stats = solver.nucleation.get_statistics()
    actual_liquid = nuc_stats['liquid_volume_added_total']
    actual_droplets = nuc_stats['droplets_added_total']
    
    # Porosity statistics
    porosity_active = solver.porosity[:solver.a_tot]
    porosity_mean = np.mean(porosity_active[~np.isnan(porosity_active)])
    porosity_min = np.nanmin(porosity_active)
    porosity_max = np.nanmax(porosity_active)
    
    # Saturation statistics
    saturation_active = solver.saturation[:solver.a_tot]
    sat_valid = ~np.isnan(saturation_active)
    saturation_mean = np.mean(saturation_active[sat_valid]) if np.any(sat_valid) else 0.0
    
    # Merger statistics (if enabled)
    merger_stats = None
    if hasattr(solver, '_particle_merger') and solver._particle_merger is not None:
        merger_stats = solver._particle_merger.get_statistics()
        print(f"\n🔍 Merger statistics:")
        print(f"  Lookups:      {merger_stats['lookups']:,}")
        print(f"  Merges:       {merger_stats['merges']:,}")
        print(f"  Creates:      {merger_stats['creates']:,}")
        print(f"  Merge rate:   {merger_stats['merge_rate']*100:.1f}%")
        
        # Debug: Compare expected vs actual lookups
        break_evts = solver.real_break_events
        avg_frags_per_event = 4  # Typical for PowerLaw-Rumpf
        expected_lookups = break_evts * avg_frags_per_event
        lookup_ratio = merger_stats['lookups'] / expected_lookups if expected_lookups > 0 else 0
        print(f"\n[DEBUG] Expected lookups: ~{expected_lookups:,} ({break_evts:,} breaks × {avg_frags_per_event} frags)")
        print(f"[DEBUG] Actual lookups:   {merger_stats['lookups']:,}")
        print(f"[DEBUG] Lookup ratio:     {lookup_ratio*100:.1f}%")
        
        # DEBUG: Print breakage processing stats
        if hasattr(solver, '_break_debug_stats'):
            stats = solver._break_debug_stats
            
            # NOTE: real_break_events counts "packet weight" (dW), not Python calls!
            # One _do_one_break() call can represent dW=25 real events.
            avg_dw = solver.real_break_events / max(1, stats['attempted'])
            
            print(f"\n[BREAKAGE DEBUG] Fragment processing pipeline:")
            print(f"  Real break events (dW sum):   {solver.real_break_events:,.1f}")
            print(f"  Python _do_one_break calls:   {stats['attempted']:,}")
            print(f"  Avg dW per call:              {avg_dw:.1f}")
            print(f"")
            print(f"  [Per Python Call]:")
            print(f"    Passed weight check:        {stats['passed_weight_check']:,} ({stats['passed_weight_check']/max(1,stats['attempted'])*100:.1f}%)")
            print(f"    Fragments produced:         {stats['produced_fragments']:,} ({stats['produced_fragments']/max(1,stats['passed_weight_check']):.2f} per call)")
            print(f"    Passed volume filter:       {stats['passed_volume_filter']:,} ({stats['passed_volume_filter']/max(1,stats['produced_fragments'])*100:.1f}%)")
            print(f"    Filtered out (vol<=0):      {stats.get('filtered_out', 0):,}")
            print(f"    Reached merger:             {stats['reached_merger']:,} ({stats['reached_merger']/max(1,stats['passed_volume_filter'])*100:.1f}%)")
            print(f"      → Merged:                 {stats['merged']:,} ({stats['merged']/max(1,stats['reached_merger'])*100:.1f}%)")
            print(f"      → Created new:            {stats['created_new']:,} ({stats['created_new']/max(1,stats['reached_merger'])*100:.1f}%)")
            print(f"")
            print(f"  [Scaled to Real Events (×{avg_dw:.1f})]:")
            print(f"    Total fragments:            ~{int(stats['produced_fragments']*avg_dw):,}")
            print(f"    Merger lookups:             ~{int(stats['reached_merger']*avg_dw):,}")
            print(f"    Merges:                     ~{int(stats['merged']*avg_dw):,}")
        
        if lookup_ratio < 0.5:
            print(f"\n⚠️  WARNING: Less than 50% of fragments are being looked up!")
            print(f"   Possible causes:")
            print(f"   - Merger not called in _break_apply_and_maintain()")
            print(f"   - Fragments filtered out before merger")
            print(f"   - Merger disabled mid-simulation")
    
    # Performance metrics
    print(f"\nPerformance:")
    print(f"  Real time:    {elapsed_real:.2f} s")
    print(f"  Simulated:    {cfg.T_TOTAL:.1f} s")
    print(f"  Speedup:      {cfg.T_TOTAL/elapsed_real:.1f}x")
    print(f"  Events/sec:   {total_events/elapsed_real:,.0f}")
    
    # Additional physics stats
    print(f"\nPhysics statistics:")
    print(f"  Liquid added:     {format_scientific(actual_liquid, 'm³')} ({actual_droplets:.0f} droplets)")
    print(f"  Porosity (mean):  {porosity_mean:.3f} (init: {cfg.INITIAL_POROSITY:.3f})")
    print(f"  Saturation (mean): {saturation_mean:.3f}")
    
    return {
        'success': True,
        'final_n_comp': int(solver.a_tot),
        'final_n_phys': float(np.sum(solver.W[:solver.a_tot]) / solver.Vc),
        'final_solid_mass': float(final_solid_mass),
        'agg_events': int(agg_events),
        'agg_accepted': int(agg_accepted),
        'agg_rejected': int(agg_rejected),
        'break_events': int(break_events),
        'total_events': int(total_events),
        'elapsed_real': float(elapsed_real),
        'speedup': float(cfg.T_TOTAL / elapsed_real),
        'events_per_sec': float(total_events / elapsed_real),
        'merger_stats': merger_stats,
        'actual_liquid': float(actual_liquid),
        'actual_droplets': float(actual_droplets),
        'porosity_mean': float(porosity_mean),
        'saturation_mean': float(saturation_mean),
    }


# =============================================================================
# Main Test Function
# =============================================================================

def run_merger_comparison_test() -> Dict[str, Any]:
    """
    Run A/B comparison test: WITH vs WITHOUT particle merger.
    
    Returns
    -------
    dict
        Combined results dictionary
    """
    
    # -------------------------------------------------------------------------
    # Setup & Configuration
    # -------------------------------------------------------------------------
    
    print_section("PARTICLE MERGER A/B COMPARISON TEST - FULL PHYSICS", "=")
    print("\nPurpose:")
    print("  Compare solver performance with and without particle merging")
    print("  under FULL PHYSICS conditions (PowerLaw-Rumpf + Stokes + Nucleation).")
    print("\nPhysics Modules:")
    print("  - Breakage: PowerLaw-Rumpf (porosity/saturation-dependent)")
    print("  - Agglomeration: Constant kernel")
    print("  - Agg. Acceptance: Stokes criterion")
    print("  - Nucleation: Liquid droplet addition")
    print("  - Porosity Growth: Cone model")
    print("  - Compression: Exponential decay")
    print("  - Liq. Internalization: Continuous + Event-based")
    
    cfg = TestConfig()
    
    print_section("CONFIGURATION")
    print(f"  Seed:               {cfg.SEED}")
    print(f"  Simulation time:    {cfg.T_TOTAL:.1f} s")
    print(f"  Initial particles:  {cfg.INITIAL_PARTICLES:,}")
    print(f"  Particle diameter:  {cfg.PARTICLE_DIAMETER*1e6:.0f} µm")
    print(f"  Initial porosity:   {cfg.INITIAL_POROSITY:.2f}")
    print(f"  Agg coefficient:    {format_scientific(cfg.AGG_COEFFICIENT, 'm³/s')}")
    print(f"  Breakage P1:        {cfg.PL_P1:.2f} 1/s")
    print(f"  Rumpf k/alpha:      {cfg.RUMPF_K}/{cfg.RUMPF_ALPHA}")
    print(f"  Flow rate:          {format_scientific(cfg.VOLUMETRIC_FLOW_RATE, 'm³/s')}")
    print(f"  Merger tolerance:   {cfg.MERGER_TOLERANCE*100:.4f}%")
    print(f"  Hash index:         {cfg.USE_HASH_INDEX}")
    
    # -------------------------------------------------------------------------
    # Create time vector and shared RNG
    # -------------------------------------------------------------------------
    
    t_vec = np.linspace(0.0, cfg.T_TOTAL, int(cfg.T_TOTAL / cfg.T_WRITE) + 1)
    
    # Create TWO independent RNGs with same seed for identical randomness
    rng_with_merger = np.random.default_rng(cfg.SEED)
    rng_without_merger = np.random.default_rng(cfg.SEED)
    
    # -------------------------------------------------------------------------
    # Expected liquid calculation
    # -------------------------------------------------------------------------
    
    droplet_radius = cfg.DROPLET_DIAMETER / 2.0
    droplet_volume = (4.0 / 3.0) * np.pi * droplet_radius ** 3
    expected_liquid = cfg.VOLUMETRIC_FLOW_RATE * cfg.NUCLEATION_DURATION
    expected_droplets = expected_liquid / droplet_volume
    
    print_section("EXPECTED RESULTS")
    print(f"  Total liquid:       {format_scientific(expected_liquid, 'm³')}")
    print(f"  Total droplets:     {expected_droplets:.2f}")
    
    # -------------------------------------------------------------------------
    # RUN A: WITH PARTICLE MERGER
    # -------------------------------------------------------------------------
    
    print_section("RUN A: WITH PARTICLE MERGER", "-")
    
    solver_a = create_solver(
        t_vec=t_vec,
        rng=rng_with_merger,
        enable_merger=True,
        merger_tol=cfg.MERGER_TOLERANCE,
        use_hash=cfg.USE_HASH_INDEX,
    )
    
    initial_vol_a, initial_mass_a = initialize_particles(solver_a)
    
    results_a = run_simulation(solver_a, "WITH PARTICLE MERGER")
    
    if not results_a['success']:
        print("\n❌ Run A failed!")
        return {'success': False, 'error_a': results_a.get('error', 'Unknown')}
    
    # -------------------------------------------------------------------------
    # RUN B: WITHOUT PARTICLE MERGER
    # -------------------------------------------------------------------------
    
    print_section("RUN B: WITHOUT PARTICLE MERGER", "-")
    
    solver_b = create_solver(
        t_vec=t_vec,
        rng=rng_without_merger,
        enable_merger=False,
    )
    
    initial_vol_b, initial_mass_b = initialize_particles(solver_b)
    
    results_b = run_simulation(solver_b, "WITHOUT PARTICLE MERGER")
    
    if not results_b['success']:
        print("\n❌ Run B failed!")
        return {'success': False, 'error_b': results_b.get('error', 'Unknown')}
    
    # -------------------------------------------------------------------------
    # Compare Results
    # -------------------------------------------------------------------------
    
    print_section("COMPARISON RESULTS", "=")
    
    # Extract key metrics
    n_comp_a = results_a['final_n_comp']
    n_comp_b = results_b['final_n_comp']
    
    time_a = results_a['elapsed_real']
    time_b = results_b['elapsed_real']
    
    events_a = results_a['total_events']
    events_b = results_b['total_events']
    
    break_a = results_a['break_events']
    break_b = results_b['break_events']
    
    agg_a = results_a['agg_events']
    agg_b = results_b['agg_events']
    
    # Calculate relative differences
    n_comp_reduction = (1 - n_comp_a / n_comp_b) * 100 if n_comp_b > 0 else 0
    time_change = (time_a - time_b) / time_b * 100 if time_b > 0 else 0
    throughput_a = events_a / time_a if time_a > 0 else 0
    throughput_b = events_b / time_b if time_b > 0 else 0
    throughput_change = (throughput_a - throughput_b) / throughput_b * 100 if throughput_b > 0 else 0
    
    # Mass conservation check (should be identical)
    mass_error_a = (results_a['final_solid_mass'] - initial_mass_a) / initial_mass_a
    mass_error_b = (results_b['final_solid_mass'] - initial_mass_b) / initial_mass_b
    
    # Liquid accuracy
    liquid_error_a = (results_a['actual_liquid'] - expected_liquid) / expected_liquid
    liquid_error_b = (results_b['actual_liquid'] - expected_liquid) / expected_liquid
    
    print(f"\n{'Metric':<30} {'With Merger':<20} {'Without Merger':<20} {'Change':<15}")
    print(f"{'-'*85}")
    print(f"{'Final n_comp:':<30} {n_comp_a:<20,} {n_comp_b:<20,} {format_percentage(-n_comp_reduction/100):<15}")
    print(f"{'Final n_phys:':<30} {results_a['final_n_phys']:<20,.0f} {results_b['final_n_phys']:<20,.0f} {'-':<15}")
    print(f"{'Real time:':<30} {time_a:<20.2f} s {time_b:<20.2f} s {format_percentage(time_change/100):<15}")
    print(f"{'Total events:':<30} {events_a:<20,} {events_b:<20,} {'-':<15}")
    print(f"{'  Breakage:':<30} {break_a:<20,} {break_b:<20,} {'-':<15}")
    print(f"{'  Agglomeration:':<30} {agg_a:<20,} {agg_b:<20,} {'-':<15}")
    print(f"{'Throughput:':<30} {throughput_a:<20,.0f} evt/s {throughput_b:<20,.0f} evt/s {format_percentage(throughput_change/100):<15}")
    print(f"{'Mass error:':<30} {format_percentage(mass_error_a):<20} {format_percentage(mass_error_b):<20} {'-':<15}")
    print(f"{'Liquid error:':<30} {format_percentage(liquid_error_a):<20} {format_percentage(liquid_error_b):<20} {'-':<15}")
    print(f"{'Porosity (final):':<30} {results_a['porosity_mean']:<20.3f} {results_b['porosity_mean']:<20.3f} {'-':<15}")
    print(f"{'Saturation (final):':<30} {results_a['saturation_mean']:<20.3f} {results_b['saturation_mean']:<20.3f} {'-':<15}")
    
    # Merger-specific stats
    if results_a['merger_stats']:
        stats = results_a['merger_stats']
        print(f"\n{'Merger Statistics (Run A only):':<30}")
        print(f"  Lookups:      {stats['lookups']:,}")
        print(f"  Merges:       {stats['merges']:,}")
        print(f"  Creates:      {stats['creates']:,}")
        print(f"  Merge rate:   {stats['merge_rate']*100:.1f}%")
        
        # Expected vs actual creates
        expected_creates = stats['lookups']  # Without merger, all would be creates
        avoided_creates = stats['merges']
        if expected_creates > 0:
            print(f"\n  Avoided creates: {avoided_creates:,} ({avoided_creates/expected_creates*100:.1f}% reduction)")
        else:
            print(f"\n  No merge opportunities (0 lookups - no breakage events occurred)")
        
        # Estimate long-term benefit
        if n_comp_reduction > 0:
            print(f"\n  Long-term benefit: {n_comp_reduction:.1f}% fewer particles")
            print(f"    → O(n²) operations: {(1 - (n_comp_a/n_comp_b)**2)*100:.1f}% faster")
            print(f"    → Memory usage: {n_comp_reduction:.1f}% lower")
    
    # Validation
    print_section("VALIDATION", "-")
    
    # Check that merger actually reduced n_comp
    n_comp_ok = n_comp_a <= n_comp_b  # Should be equal or lower with merger
    
    # Check mass conservation (both should be good)
    mass_tolerance = 1e-6
    mass_ok_a = abs(mass_error_a) < mass_tolerance
    mass_ok_b = abs(mass_error_b) < mass_tolerance
    
    # Check liquid conservation (both should be good)
    liquid_tolerance = 0.05  # 5%
    liquid_ok_a = abs(liquid_error_a) < liquid_tolerance
    liquid_ok_b = abs(liquid_error_b) < liquid_tolerance
    
    # Check events occurred in both runs
    events_ok_a = events_a > 0
    events_ok_b = events_b > 0
    
    # Check breakage occurred
    breakage_ok_a = break_a > 0
    breakage_ok_b = break_b > 0
    
    # Check agglomeration occurred
    agg_ok_a = agg_a > 0
    agg_ok_b = agg_b > 0
    
    # Check event counts similar (same RNG seed → similar events)
    events_similar = abs(events_a - events_b) / max(events_a, events_b) < 0.10 if events_a > 0 else True
    
    all_ok = (n_comp_ok and mass_ok_a and mass_ok_b and 
              liquid_ok_a and liquid_ok_b and
              events_ok_a and events_ok_b and 
              breakage_ok_a and breakage_ok_b)
    
    print(f"\n  {'n_comp reduction:':<25} {'✓ PASS' if n_comp_ok else '✗ FAIL'} ({n_comp_reduction:.1f}% reduction)")
    print(f"  {'Mass conservation (A):':<25} {'✓ PASS' if mass_ok_a else '✗ FAIL'} ({format_percentage(mass_error_a)})")
    print(f"  {'Mass conservation (B):':<25} {'✓ PASS' if mass_ok_b else '✗ FAIL'} ({format_percentage(mass_error_b)})")
    print(f"  {'Liquid conservation (A):':<25} {'✓ PASS' if liquid_ok_a else '✗ FAIL'} ({format_percentage(liquid_error_a)})")
    print(f"  {'Liquid conservation (B):':<25} {'✓ PASS' if liquid_ok_b else '✗ FAIL'} ({format_percentage(liquid_error_b)})")
    print(f"  {'Events occurred (A):':<25} {'✓ PASS' if events_ok_a else '✗ FAIL'} ({events_a:,} events)")
    print(f"  {'Events occurred (B):':<25} {'✓ PASS' if events_ok_b else '✗ FAIL'} ({events_b:,} events)")
    print(f"  {'Breakage active (A):':<25} {'✓ PASS' if breakage_ok_a else '✗ FAIL'} ({break_a:,} events)")
    print(f"  {'Breakage active (B):':<25} {'✓ PASS' if breakage_ok_b else '✗ FAIL'} ({break_b:,} events)")
    print(f"  {'Agglomeration active (A):':<25} {'✓ PASS' if agg_ok_a else '✗ FAIL'} ({agg_a:,} events)")
    print(f"  {'Agglomeration active (B):':<25} {'✓ PASS' if agg_ok_b else '✗ FAIL'} ({agg_b:,} events)")
    print(f"  {'Event counts similar:':<25} {'✓ PASS' if events_similar else '✗ FAIL'} (Δ={abs(events_a-events_b):,})")
    
    print(f"\n{'OVERALL:':<25} {'✓ ALL TESTS PASSED' if all_ok else '✗ SOME TESTS FAILED'}")
    
    # Performance recommendation
    if time_a < time_b:
        print(f"\n📈 RECOMMENDATION: Particle merger is FASTER for this scenario!")
        print(f"   Speedup: {time_b/time_a:.2f}x ({time_change:.1f}% time reduction)")
        if n_comp_reduction > 10:
            print(f"   Combined benefit: {n_comp_reduction:.1f}% fewer particles + {abs(time_change):.1f}% faster!")
    elif time_a > time_b:
        print(f"\n📉 RECOMMENDATION: Particle merger adds overhead for this scenario.")
        print(f"   Overhead: {time_a/time_b:.2f}x ({time_change:.1f}% slower)")
        if n_comp_reduction > 20:
            print(f"   BUT: n_comp reduced by {n_comp_reduction:.1f}% - may benefit LONG runs!")
            print(f"   Consider enabling for simulations > {cfg.T_TOTAL*2:.0f} s")
        elif n_comp_reduction > 10:
            print(f"   Trade-off: {n_comp_reduction:.1f}% fewer particles vs {time_change:.1f}% slower")
    else:
        print(f"\n➡️  RECOMMENDATION: Neutral performance impact.")
        if n_comp_reduction > 5:
            print(f"   Benefit: {n_comp_reduction:.1f}% fewer particles at no time cost!")
    
    # -------------------------------------------------------------------------
    # Return Results Dictionary
    # -------------------------------------------------------------------------
    
    return {
        'success': all_ok,
        'run_a': results_a,
        'run_b': results_b,
        'comparison': {
            'n_comp_reduction': float(n_comp_reduction),
            'time_change': float(time_change),
            'throughput_change': float(throughput_change),
            'mass_error_a': float(mass_error_a),
            'mass_error_b': float(mass_error_b),
            'liquid_error_a': float(liquid_error_a),
            'liquid_error_b': float(liquid_error_b),
            'events_similar': events_similar,
        },
        'initial_mass_a': float(initial_mass_a),
        'initial_mass_b': float(initial_mass_b),
        'expected_liquid': float(expected_liquid),
        'expected_droplets': float(expected_droplets),
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    results = run_merger_comparison_test()
    
    print_section("TEST COMPLETE", "=")
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)
