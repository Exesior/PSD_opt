"""
A/B Test: Particle Merger Performance Comparison.

This test compares solver performance WITH and WITHOUT the particle merging
optimization. The particle merger reduces n_comp by finding existing particles
with similar properties (V_dry, liquid, porosity, saturation) and adding weight
to them instead of creating new computational particles.

Test Configuration:
    - Breakage-dominated scenario (PowerLaw kernel)
    - Agglomeration active (constant kernel)
    - No nucleation/compression (isolated breakage/agg physics)
    - Identical seeds for both runs
    
Expected Results:
    - WITH merger: Lower n_comp, potentially slower per-event (lookup overhead)
    - WITHOUT merger: Higher n_comp, faster per-event but more total work
    
Usage:
    python -m wmcpbe.Trials.test_particle_merger_comparison
    
Or from the wmcpbe directory:
    python Trials/test_particle_merger_comparison.py
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
# Test Configuration
# =============================================================================

class TestConfig:
    """Physical and numerical parameters for merger comparison test."""
    
    # Seed for reproducibility (SAME for both runs!)
    SEED = 42
    
    # Time settings
    T_TOTAL = 4.0       # Total simulation time [s]
    T_WRITE = 0.1       # Output interval [s]
    
    # Particle properties (initial)
    PARTICLE_DIAMETER = 500e-6            # 500 µm
    PARTICLE_DENSITY = 2500.0             # kg/m³ (solid material)
    INITIAL_POROSITY = 0.0                # Non-porous (Vollkörper)
    
    # Process parameters
    AGG_COEFFICIENT = 1e-9                # Constant kernel coefficient [m³/s]
    BREAKAGE_P1 = 0.1                     # Breakage rate pre-factor [1/s]
    
    # Numerical settings
    INITIAL_PARTICLES = 2000               # Computational particles (lower for faster test)
    INITIAL_WEIGHT = 100                  # Weight per particle
    CONTROL_VOLUME = 1                 # m³
    
    # Breakage configuration
    BREAK_DW_CONST = 25.0                 # Packet size for breakage
    FRAG_NUM = 4                          # Number of fragments per break (deterministic)
    
    # Merger configuration
    MERGER_TOLERANCE = 1e-6               # Relative tolerance for matching (0.0001%)
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
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=False,  # Disable verbose for cleaner test output
        load_attr=False,
        init=True,
        rng=rng,
        # Aggregation kernel (constant, low rate)
        agg_kernel_name='constant',
        agg_kernel_params={'corr_beta': TestConfig.AGG_COEFFICIENT},
        # Breakage kernel (PowerLaw, moderate rate)
        break_kernel_name='power_law',
        break_kernel_params={
            'p1': TestConfig.BREAKAGE_P1,
        },
        # Porosity growth (volume_mixing for simplicity)
        porosity_growth_kernel_name='volume_mixing',
        porosity_growth_kernel_params={},
    )
    
    solver.mcpbe_debug = False
    solver.agg_propensity_mode = "moment"  # O(n) acceleration
    solver.process_type = "mix"  # Agglomeration + Breakage
    solver.recon_enable = False  # Disable reconstruction for isolated test
    
    # Configure particle merger BEFORE _initialize_samplers()
    solver._enable_particle_merging = enable_merger
    if enable_merger:
        solver._fragment_merge_tol = merger_tol
        solver.merger_lookup = "hash" if use_hash else "scan"
    
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
    
    # Initialize porosity (all non-porous)
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
    
    # Initialize samplers
    solver._initialize_samplers()
    
    # Configure breakage packet size
    solver._break_dW_const = cfg.BREAK_DW_CONST
    solver.frag_num = cfg.FRAG_NUM
    
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
    
    print(f"\nEvent statistics:")
    print(f"  Agglomeration:  {agg_events:,.0f} events")
    print(f"  Breakage:       {break_events:,.0f} events")
    print(f"  Total:          {total_events:,.0f} events")
    
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
    
    # Merger statistics (if enabled)
    merger_stats = None
    if hasattr(solver, '_particle_merger') and solver._particle_merger is not None:
        merger_stats = solver._particle_merger.get_statistics()
        print(f"\nMerger statistics:")
        print(f"  Lookups:      {merger_stats['lookups']:,}")
        print(f"  Merges:       {merger_stats['merges']:,}")
        print(f"  Creates:      {merger_stats['creates']:,}")
        print(f"  Merge rate:   {merger_stats['merge_rate']*100:.1f}%")
    
    # Performance metrics
    print(f"\nPerformance:")
    print(f"  Real time:    {elapsed_real:.2f} s")
    print(f"  Simulated:    {cfg.T_TOTAL:.1f} s")
    print(f"  Speedup:      {cfg.T_TOTAL/elapsed_real:.1f}x")
    print(f"  Events/sec:   {total_events/elapsed_real:,.0f}")
    
    return {
        'success': True,
        'final_n_comp': int(solver.a_tot),
        'final_n_phys': float(np.sum(solver.W[:solver.a_tot]) / solver.Vc),
        'final_solid_mass': float(final_solid_mass),
        'agg_events': int(agg_events),
        'break_events': int(break_events),
        'total_events': int(total_events),
        'elapsed_real': float(elapsed_real),
        'speedup': float(cfg.T_TOTAL / elapsed_real),
        'events_per_sec': float(total_events / elapsed_real),
        'merger_stats': merger_stats,
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
    
    print_section("PARTICLE MERGER A/B COMPARISON TEST", "=")
    print("\nPurpose:")
    print("  Compare solver performance with and without particle merging.")
    print("  The merger reduces n_comp by reusing existing particles with")
    print("  similar properties instead of creating new ones.")
    print("\nPhysics:")
    print("  - Agglomeration: constant kernel (low rate)")
    print("  - Breakage: PowerLaw kernel (moderate rate)")
    print("  - No nucleation/compression (isolated breakage/agg)")
    
    cfg = TestConfig()
    
    print_section("CONFIGURATION")
    print(f"  Seed:               {cfg.SEED}")
    print(f"  Simulation time:    {cfg.T_TOTAL:.1f} s")
    print(f"  Initial particles:  {cfg.INITIAL_PARTICLES:,}")
    print(f"  Particle diameter:  {cfg.PARTICLE_DIAMETER*1e6:.0f} µm")
    print(f"  Agg coefficient:    {format_scientific(cfg.AGG_COEFFICIENT, 'm³/s')}")
    print(f"  Breakage P1:        {cfg.BREAKAGE_P1:.2f} 1/s")
    print(f"  Fragments/break:    {cfg.FRAG_NUM}")
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
    
    # Calculate relative differences
    n_comp_reduction = (1 - n_comp_a / n_comp_b) * 100 if n_comp_b > 0 else 0
    time_change = (time_a - time_b) / time_b * 100 if time_b > 0 else 0
    throughput_a = events_a / time_a if time_a > 0 else 0
    throughput_b = events_b / time_b if time_b > 0 else 0
    throughput_change = (throughput_a - throughput_b) / throughput_b * 100 if throughput_b > 0 else 0
    
    # Mass conservation check (should be identical)
    mass_error_a = (results_a['final_solid_mass'] - initial_mass_a) / initial_mass_a
    mass_error_b = (results_b['final_solid_mass'] - initial_mass_b) / initial_mass_b
    
    print(f"\n{'Metric':<30} {'With Merger':<20} {'Without Merger':<20} {'Change':<15}")
    print(f"{'-'*85}")
    print(f"{'Final n_comp:':<30} {n_comp_a:<20,} {n_comp_b:<20,} {format_percentage(-n_comp_reduction/100):<15}")
    print(f"{'Final n_phys:':<30} {results_a['final_n_phys']:<20,.0f} {results_b['final_n_phys']:<20,.0f} {'-':<15}")
    print(f"{'Real time:':<30} {time_a:<20.2f} s {time_b:<20.2f} s {format_percentage(time_change/100):<15}")
    print(f"{'Total events:':<30} {events_a:<20,} {events_b:<20,} {'-':<15}")
    print(f"{'Throughput:':<30} {throughput_a:<20,.0f} evt/s {throughput_b:<20,.0f} evt/s {format_percentage(throughput_change/100):<15}")
    print(f"{'Mass error:':<30} {format_percentage(mass_error_a):<20} {format_percentage(mass_error_b):<20} {'-':<15}")
    
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
        print(f"\n  Avoided creates: {avoided_creates:,} ({avoided_creates/expected_creates*100:.1f}% reduction)")
    
    # Validation
    print_section("VALIDATION", "-")
    
    # Check that merger actually reduced n_comp
    n_comp_ok = n_comp_a <= n_comp_b  # Should be equal or lower with merger
    
    # Check mass conservation (both should be good)
    mass_tolerance = 1e-6
    mass_ok_a = abs(mass_error_a) < mass_tolerance
    mass_ok_b = abs(mass_error_b) < mass_tolerance
    
    # Check events occurred in both runs
    events_ok_a = events_a > 0
    events_ok_b = events_b > 0
    
    # Check identical event counts (same RNG seed → same events)
    # Note: May differ slightly due to floating-point order in merger
    events_similar = abs(events_a - events_b) / max(events_a, events_b) < 0.01 if events_a > 0 else True
    
    all_ok = n_comp_ok and mass_ok_a and mass_ok_b and events_ok_a and events_ok_b
    
    print(f"\n  {'n_comp reduction:':<25} {'✓ PASS' if n_comp_ok else '✗ FAIL'} ({n_comp_reduction:.1f}% reduction)")
    print(f"  {'Mass conservation (A):':<25} {'✓ PASS' if mass_ok_a else '✗ FAIL'} ({format_percentage(mass_error_a)})")
    print(f"  {'Mass conservation (B):':<25} {'✓ PASS' if mass_ok_b else '✗ FAIL'} ({format_percentage(mass_error_b)})")
    print(f"  {'Events occurred (A):':<25} {'✓ PASS' if events_ok_a else '✗ FAIL'} ({events_a:,} events)")
    print(f"  {'Events occurred (B):':<25} {'✓ PASS' if events_ok_b else '✗ FAIL'} ({events_b:,} events)")
    print(f"  {'Event counts similar:':<25} {'✓ PASS' if events_similar else '✗ FAIL'} (Δ={abs(events_a-events_b):,})")
    
    print(f"\n{'OVERALL:':<25} {'✓ ALL TESTS PASSED' if all_ok else '✗ SOME TESTS FAILED'}")
    
    # Performance recommendation
    if time_a < time_b:
        print(f"\n📈 RECOMMENDATION: Particle merger is FASTER for this scenario!")
        print(f"   Speedup: {time_b/time_a:.2f}x ({time_change:.1f}% time reduction)")
    elif time_a > time_b:
        print(f"\n📉 RECOMMENDATION: Particle merger adds overhead for this scenario.")
        print(f"   Overhead: {time_a/time_b:.2f}x ({time_change:.1f}% slower)")
        if n_comp_reduction > 20:
            print(f"   BUT: n_comp reduced by {n_comp_reduction:.1f}% - may benefit long runs!")
    else:
        print(f"\n➡️  RECOMMENDATION: Neutral performance impact.")
    
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
            'events_similar': events_similar,
        },
        'initial_mass_a': float(initial_mass_a),
        'initial_mass_b': float(initial_mass_b),
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    results = run_merger_comparison_test()
    
    print_section("TEST COMPLETE", "=")
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)
