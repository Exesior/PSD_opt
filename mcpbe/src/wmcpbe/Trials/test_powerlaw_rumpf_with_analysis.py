"""
Comprehensive Test: PowerLaw-Rumpf Breakage + Full Physics Suite WITH ANALYSIS.

This test extends test_powerlaw_rumpf_full.py with detailed post-processing
and coupled multi-property analysis using ParticlePropertyContainer.

NEW FEATURES:
- Coupled property analysis (porosity vs. volume, saturation vs. size, etc.)
- Time-resolved correlation plots
- Particle subset tracking over time
- Statistical summaries of property distributions

Usage:
    python -m wmcpbe.Trials.test_powerlaw_rumpf_with_analysis
    
Or from the wmcpbe directory:
    python Trials/test_powerlaw_rumpf_with_analysis.py
"""

import numpy as np
import sys
import os
import time
from typing import Dict, Any

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
    """Physical and numerical parameters for comprehensive test."""
    
    # Seed for reproducibility
    SEED = 42
    
    # Time settings
    T_TOTAL = 20.0      # Total simulation time [s] - EXTENDED RUN
    T_WRITE = 0.2       # Output interval [s]
    
    # Particle properties (initial)
    PARTICLE_DIAMETER = 700e-6        # 500 µm
    PARTICLE_DENSITY = 2500.0         # kg/m³ (solid material)
    INITIAL_POROSITY = 0.8           # 40% void fraction
    
    # Droplet properties
    DROPLET_DIAMETER = 200e-6         # 100 µm
    DROPLET_DENSITY = 1000.0          # kg/m³ (water)
    
    # Process parameters
    VOLUMETRIC_FLOW_RATE = 7e-7       # m³/s
    NUCLEATION_DURATION = 10.0         # s
    AGG_COEFFICIENT = 3e-6            # Constant kernel coefficient [m³/s] - HIGH for testing
    BATCH_SIZE = 50                   # How many droplets are distributed identically per event   
    
    # Breakage parameters (PowerLaw-Rumpf)
    BREAKAGE_ENABLED = True
    PL_P1 = 5                       # Pre-factor [1/s·Pa·m^(-3*alpha)]
    PL_P2 = 1.0                       # Volume exponent in S = P1*G*V^P2
    G = 10000.0                        # Shear rate [1/s]
    BREAKRVAL = 4                     # Volume-based power law
    # Rumpf strength parameters - REDUCED for higher breakage rates
    RUMPF_K = 2.5                     # Fitting parameter dry [2.2-2.8] - MIN VALUE
    RUMPF_ALPHA = 1.0                 # Fitting parameter wet [1.0-1.33] - MIN VALUE
    RUMPF_GAMMA = 0.072               # Surface tension [N/m] - REDUCED (surfactant)
    RUMPF_DELTA = 0.0                 # Contact angle [rad] (perfect wetting)
    
    # Compression parameters
    COMPRESSION_ENABLED = True
    COMPRESSION_RATE = 0.02           # Porosity decay rate [1/s]
    MIN_POROSITY = 0.15               # Minimum achievable porosity
    
    # Liquid internalization parameters
    LIQ_INTERN_ENABLED = True
    LIQ_INTERN_RATE = 1e8             # Rate constant [1/(m³·s)]
    
    # Liquid internalization during agglomeration
    LIQ_INTERN_AGG_ENABLED = True
    
    # Agglomeration acceptance (Stokes criterion)
    STOKES_ENABLED = True
    BINDER_VISCOSITY = 0.1            # Pa·s
    COLLISION_VELOCITY = 0.5          # m/s
    H_A = 500e-9                      # m (half-distance of closest approach)
    
    # Numerical settings
    INITIAL_PARTICLES = 800          # Computational particles
    INITIAL_WEIGHT = 500.0             # Weight per particle
    CONTROL_VOLUME = 1              # m³ # Seed for reproducibility
    

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
# Analysis Functions (NEW)
# =============================================================================

def analyze_property_correlations(props, t_indices=None):
    """
    Analyze correlations between particle properties at selected times.
    
    Parameters
    ----------
    props : ParticlePropertyContainer
        Container with all particle properties
    t_indices : list of int, optional
        Time indices to analyze. If None, analyzes first, middle, last.
    """
    if t_indices is None:
        t_indices = [0, props.T // 2, props.T - 1]
    
    print_section("PROPERTY CORRELATION ANALYSIS")
    
    results = {}
    
    for t_idx in t_indices:
        if t_idx >= props.T:
            continue
        
        t = props.t_vec[t_idx]
        n_active = props.n_active(t_idx)
        
        print(f"\n--- Time t = {t:.2f} s ({n_active} active particles) ---")
        
        # Extract data for this time
        V_dry = props.V_dry[t_idx, :n_active]
        X = props.X[t_idx, :n_active]
        W = props.W[t_idx, :n_active]
        
        # Correlation: Porosity vs. Volume
        if props.porosity is not None:
            poro = props.porosity[t_idx, :n_active]
            
            # Statistics by size bin
            d50 = np.percentile(X, 50)
            small_mask = X < d50
            large_mask = X >= d50
            
            if np.any(small_mask):
                poro_small = np.nanmean(poro[small_mask])
                print(f"  Mean porosity (small particles, d<{d50*1e6:.0f}µm): {poro_small:.3f}")
            
            if np.any(large_mask):
                poro_large = np.nanmean(poro[large_mask])
                print(f"  Mean porosity (large particles, d>{d50*1e6:.0f}µm): {poro_large:.3f}")
            
            # Overall statistics
            valid_poro = ~np.isnan(poro)
            if np.any(valid_poro):
                print(f"  Porosity stats: mean={np.mean(poro[valid_poro]):.3f}, "
                      f"std={np.std(poro[valid_poro]):.3f}, "
                      f"min={np.min(poro[valid_poro]):.3f}, "
                      f"max={np.max(poro[valid_poro]):.3f}")
            
            results[f't{t_idx}_poro_mean'] = np.mean(poro[valid_poro]) if np.any(valid_poro) else np.nan
        
        # Correlation: Saturation vs. Porosity
        if props.saturation is not None and props.porosity is not None:
            sat = props.saturation[t_idx, :n_active]
            poro = props.porosity[t_idx, :n_active]
            
            valid = ~np.isnan(sat) & ~np.isnan(poro)
            if np.any(valid):
                # Correlation coefficient
                corr = np.corrcoef(sat[valid], poro[valid])[0, 1]
                print(f"  Saturation-Porosity correlation: {corr:+.3f}")
                
                # Saturation bins
                low_sat = sat[valid] < 0.3
                high_sat = sat[valid] >= 0.7
                
                if np.any(low_sat):
                    print(f"  Mean porosity (low sat <0.3): {np.mean(poro[valid][low_sat]):.3f}")
                if np.any(high_sat):
                    print(f"  Mean porosity (high sat >0.7): {np.mean(poro[valid][high_sat]):.3f}")
                
                results[f't{t_idx}_sat_poro_corr'] = corr
        
        # Correlation: Liquid Volume vs. Size
        if props.liquid_volume is not None:
            liq_vol = props.liquid_volume[t_idx, :n_active]
            
            valid = ~np.isnan(liq_vol) & (liq_vol > 0)
            if np.any(valid):
                # Correlation with diameter
                corr = np.corrcoef(X[valid], liq_vol[valid])[0, 1]
                print(f"  Liquid Volume - Diameter correlation: {corr:+.3f}")
                
                # Total liquid in different size ranges
                total_liq = np.sum(liq_vol[valid] * W[valid])
                liq_in_small = np.sum(liq_vol[valid][small_mask[valid]] * W[valid][small_mask[valid]]) if np.any(small_mask[valid]) else 0
                liq_fraction_small = liq_in_small / total_liq if total_liq > 0 else 0
                print(f"  Liquid fraction in small particles: {liq_fraction_small:.1%}")
                
                results[f't{t_idx}_liq_X_corr'] = corr
    
    return results


def analyze_time_evolution(props):
    """
    Track property evolution for particle subsets over time.
    
    Parameters
    ----------
    props : ParticlePropertyContainer
        Container with all particle properties
    """
    print_section("TIME EVOLUTION ANALYSIS")
    
    results = {}
    
    # Define initial subsets by size
    X_initial = props.X[0, :]
    n_initial = props.n_active(0)
    
    d33 = np.percentile(X_initial[:n_initial], 33)
    d66 = np.percentile(X_initial[:n_initial], 66)
    
    small_mask = X_initial[:n_initial] < d33
    medium_mask = (X_initial[:n_initial] >= d33) & (X_initial[:n_initial] < d66)
    large_mask = X_initial[:n_initial] >= d66
    
    print(f"\nInitial particle classification:")
    print(f"  Small:   d < {d33*1e6:.0f} µm  → {np.sum(small_mask)} particles")
    print(f"  Medium:  {d33*1e6:.0f} ≤ d < {d66*1e6:.0f} µm  → {np.sum(medium_mask)} particles")
    print(f"  Large:   d ≥ {d66*1e6:.0f} µm  → {np.sum(large_mask)} particles")
    
    # Track porosity evolution for each subset
    if props.porosity is not None:
        print(f"\nPorosity evolution by initial size class:")
        print(f"  {'Time [s]':<12} {'Small':<12} {'Medium':<12} {'Large':<12}")
        print(f"  {'-'*10:<12} {'-'*10:<12} {'-'*10:<12} {'-'*10:<12}")
        
        for t_idx in range(0, props.T, max(1, props.T // 10)):
            t = props.t_vec[t_idx]
            
            # Extend masks to current N_max
            current_n = props.n_active(t_idx)
            small_ext = np.zeros(props.N_max, dtype=bool)
            medium_ext = np.zeros(props.N_max, dtype=bool)
            large_ext = np.zeros(props.N_max, dtype=bool)
            small_ext[:n_initial] = small_mask
            medium_ext[:n_initial] = medium_mask
            large_ext[:n_initial] = large_mask
            
            # Get porosity time series
            poro_small, _ = props.time_series(small_ext, "porosity", "mean")
            poro_medium, _ = props.time_series(medium_ext, "porosity", "mean")
            poro_large, _ = props.time_series(large_ext, "porosity", "mean")
            
            print(f"  {t:<12.2f} {poro_small[t_idx]:<12.3f} {poro_medium[t_idx]:<12.3f} {poro_large[t_idx]:<12.3f}")
        
        results['porosity_evolution'] = {
            'small': poro_small,
            'medium': poro_medium,
            'large': poro_large,
            'time': props.t_vec,
        }
    
    # Track saturation evolution
    if props.saturation is not None:
        print(f"\nSaturation evolution by initial size class:")
        print(f"  {'Time [s]':<12} {'Small':<12} {'Medium':<12} {'Large':<12}")
        print(f"  {'-'*10:<12} {'-'*10:<12} {'-'*10:<12} {'-'*10:<12}")
        
        for t_idx in range(0, props.T, max(1, props.T // 10)):
            t = props.t_vec[t_idx]
            
            current_n = props.n_active(t_idx)
            small_ext = np.zeros(props.N_max, dtype=bool)
            medium_ext = np.zeros(props.N_max, dtype=bool)
            large_ext = np.zeros(props.N_max, dtype=bool)
            small_ext[:n_initial] = small_mask
            medium_ext[:n_initial] = medium_mask
            large_ext[:n_initial] = large_mask
            
            sat_small, _ = props.time_series(small_ext, "saturation", "mean")
            sat_medium, _ = props.time_series(medium_ext, "saturation", "mean")
            sat_large, _ = props.time_series(large_ext, "saturation", "mean")
            
            print(f"  {t:<12.2f} {sat_small[t_idx]:<12.3f} {sat_medium[t_idx]:<12.3f} {sat_large[t_idx]:<12.3f}")
        
        results['saturation_evolution'] = {
            'small': sat_small,
            'medium': sat_medium,
            'large': sat_large,
            'time': props.t_vec,
        }
    
    return results


def generate_scatter_data_for_export(props, t_indices=None):
    """
    Generate scatter data suitable for external plotting (e.g., matplotlib, pandas).
    
    Returns a dictionary of DataFrames (if pandas available) or structured arrays.
    """
    if t_indices is None:
        t_indices = list(range(props.T))
    
    all_data = []
    
    for t_idx in t_indices:
        if t_idx >= props.T:
            continue
        
        n_active = props.n_active(t_idx)
        t = props.t_vec[t_idx]
        
        # Build record array
        records = {
            't': np.full(n_active, t),
            't_idx': np.full(n_active, t_idx),
            'particle_idx': np.arange(n_active),
            'V_dry': props.V_dry[t_idx, :n_active],
            'V_solid': props.V_solid[t_idx, :n_active],
            'X': props.X[t_idx, :n_active],
            'W': props.W[t_idx, :n_active],
        }
        
        if props.liquid_volume is not None:
            records['liquid_volume'] = props.liquid_volume[t_idx, :n_active]
        
        if props.porosity is not None:
            records['porosity'] = props.porosity[t_idx, :n_active]
        
        if props.saturation is not None:
            records['saturation'] = props.saturation[t_idx, :n_active]
        
        all_data.append(records)
    
    # Combine all times
    combined = {}
    for key in all_data[0].keys():
        combined[key] = np.concatenate([d[key] for d in all_data])
    
    return combined


# =============================================================================
# Main Test Function
# =============================================================================

def run_comprehensive_test() -> Dict[str, Any]:
    """
    Run comprehensive MCPBE simulation with PowerLaw-Rumpf breakage and full physics,
    including detailed post-processing analysis.
    """
    
    # -------------------------------------------------------------------------
    # Setup & Initialization
    # -------------------------------------------------------------------------
    
    print_section("COMPREHENSIVE MCPBE TEST: POWERLAW-RUMPF + FULL PHYSICS + ANALYSIS", "=")
    
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
    
    print("Physics Modules Active:")
    print("  ✓ Nucleation: Liquid droplet addition")
    print("  ✓ Agglomeration: Constant kernel")
    print("  ✓ Agglomeration Acceptance: Stokes criterion")
    print("  ✓ Breakage: PowerLaw-Rumpf (porosity/saturation-dependent)")
    print("  ✓ Porosity Growth: Cone model")
    print("  ✓ Compression: Exponential decay")
    print("  ✓ Liquid Internalization: Continuous + Event-based")
    
    print_section("KEY PARAMETERS")
    print(f"  Initial particles:  {cfg.INITIAL_PARTICLES:,}")
    print(f"  Initial diameter:   {cfg.PARTICLE_DIAMETER*1e6:.0f} µm")
    print(f"  Initial porosity:   {cfg.INITIAL_POROSITY:.1%}")
    print(f"  Simulation time:    {cfg.T_TOTAL:.1f} s")
    print(f"  Output interval:    {cfg.T_WRITE:.1f} s → {int(cfg.T_TOTAL/cfg.T_WRITE)+1} snapshots")
    
    # -------------------------------------------------------------------------
    # Solver Initialization
    # -------------------------------------------------------------------------
    
    print_section("INITIALIZING SOLVER", "-")
    
    t_vec = np.linspace(0.0, cfg.T_TOTAL, int(cfg.T_TOTAL / cfg.T_WRITE) + 1)
    rng = np.random.default_rng(cfg.SEED)
    
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=True,
        load_attr=False,
        init=True,
        rng=rng,
        agg_kernel_name='constant',
        agg_kernel_params={'corr_beta': cfg.AGG_COEFFICIENT},
        agg_acceptance_kernel_name='stokes_krit',
        agg_acceptance_kernel_params={
            'U_coll': cfg.COLLISION_VELOCITY,
            'binder_viscosity': cfg.BINDER_VISCOSITY,
            'rho_solid': cfg.PARTICLE_DENSITY,
            'rho_liquid': cfg.DROPLET_DENSITY,
            'h_a': cfg.H_A,
        },
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
            'x_s': None,
        },
        porosity_growth_kernel_name='cone_model',
        porosity_growth_kernel_params={},
        porosity_compression_kernel_name='porosity_compression',
        porosity_compression_kernel_params={
            'rate': cfg.COMPRESSION_RATE,
            'min_porosity': cfg.MIN_POROSITY,
        },
        liquid_internalization_kernel_name='liquid_internalization',
        liquid_internalization_kernel_params={
            'k_int': cfg.LIQ_INTERN_RATE,
        },
        liq_internalisation_agglomeration_kernel_name='braumann_2007',
        liq_internalisation_agglomeration_kernel_params={},
    )
    
    solver.mcpbe_debug = False
    solver.agg_propensity_mode = "moment"
    solver.process_type = "mix"
    solver.recon_enable = False
    
    # Initialize particles
    V_flat = np.zeros((2, cfg.INITIAL_PARTICLES), dtype=float)
    V_flat[0, :] = particle_volume * (1.0 - cfg.INITIAL_POROSITY)  # V_solid
    V_flat[1, :] = particle_volume  # V_dry
    
    W_init = np.full(cfg.INITIAL_PARTICLES, cfg.INITIAL_WEIGHT, dtype=float)
    solver.Vc = cfg.CONTROL_VOLUME
    
    solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
    solver.porosity[:solver.a_tot] = cfg.INITIAL_POROSITY
    solver.liquid_volume[:solver.a_tot] = 0.0
    solver.saturation[:solver.a_tot] = 0.0
    
    solver._initialize_samplers()
    
    # Configure handlers
    solver.create_nucleation_handler(
        enabled=True,
        volumetric_flow_rate=cfg.VOLUMETRIC_FLOW_RATE,
        droplet_diameter=cfg.DROPLET_DIAMETER,
        liquid_addition_start=0.0,
        liquid_addition_duration=cfg.NUCLEATION_DURATION,
        batch_size=cfg.BATCH_SIZE,
    )
    
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
    # Basic Results
    # -------------------------------------------------------------------------
    
    print_section("BASIC RESULTS", "=")
    
    nuc_stats = solver.nucleation.get_statistics()
    actual_liquid = nuc_stats['liquid_volume_added_total']
    actual_droplets = nuc_stats['droplets_added_total']
    
    print(f"\nLiquid Addition:")
    print(f"  Expected: {format_scientific(expected_liquid, 'm³')}")
    print(f"  Added:    {format_scientific(actual_liquid, 'm³')}")
    print(f"  Error:    {format_percentage((actual_liquid - expected_liquid) / expected_liquid)}")
    
    print(f"\nEvent Statistics:")
    print(f"  Agglomeration: {solver.real_agg_events:,} events")
    print(f"  Breakage:      {solver.real_break_events:,} events")
    
    print(f"\nPerformance:")
    print(f"  Real time:     {elapsed_real:.2f} s")
    print(f"  Speedup:       {cfg.T_TOTAL/elapsed_real:.1f}x")
    
    # -------------------------------------------------------------------------
    # COUPLED MULTI-PROPERTY ANALYSIS (NEW!)
    # -------------------------------------------------------------------------
    
    print_section("COUPLED MULTI-PROPERTY ANALYSIS (NEW!)", "=")
    
    # Extract all properties into container
    print("\nExtracting particle properties...")
    props = solver.get_all_properties()
    
    print(f"\nParticlePropertyContainer initialized:")
    print(f"  Time points:     {props.T}")
    print(f"  Max particles:   {props.N_max:,}")
    print(f"  Available props: V_dry, V_solid, X, W", end="")
    if props.porosity is not None:
        print(", porosity", end="")
    if props.saturation is not None:
        print(", saturation", end="")
    if props.liquid_volume is not None:
        print(", liquid_volume", end="")
    print()
    
    # Analysis 1: Property correlations at selected times
    corr_results = analyze_property_correlations(props, t_indices=[0, props.T//2, props.T-1])
    
    # Analysis 2: Time evolution of subsets
    evol_results = analyze_time_evolution(props)
    
    # Analysis 3: Export-ready data structure
    print_section("DATA EXPORT PREPARATION", "-")
    export_data = generate_scatter_data_for_export(props, t_indices=None)
    total_records = len(export_data['t'])
    print(f"\nExport data prepared:")
    print(f"  Total records:   {total_records:,} (time × particles)")
    print(f"  Fields:          {list(export_data.keys())}")
    
    # Example: Show first few records
    print(f"\nSample records (first 5):")
    for i in range(min(5, total_records)):
        line = f"  t={export_data['t'][i]:.2f}s, idx={export_data['particle_idx'][i]}, "
        line += f"d={export_data['X'][i]*1e6:.0f}µm, "
        if 'porosity' in export_data:
            line += f"poro={export_data['porosity'][i]:.3f}, "
        if 'saturation' in export_data:
            line += f"sat={export_data['saturation'][i]:.3f}"
        print(line)
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    
    print_section("VALIDATION", "-")
    
    liquid_tolerance = 0.05
    liquid_ok = abs((actual_liquid - expected_liquid) / expected_liquid) < liquid_tolerance
    
    solid_mass_initial = np.sum(solver.V0[0, :] * solver.W0) * cfg.PARTICLE_DENSITY
    v_dry_final = solver.V_flat[-1, :solver.a_tot]
    porosity_final = solver.porosity[:solver.a_tot]
    valid_poro = ~np.isnan(porosity_final)
    v_solid_final = np.zeros_like(v_dry_final)
    v_solid_final[valid_poro] = v_dry_final[valid_poro] * (1.0 - porosity_final[valid_poro])
    v_solid_final[~valid_poro] = v_dry_final[~valid_poro]
    solid_mass_final = np.sum(v_solid_final * solver.W[:solver.a_tot]) * cfg.PARTICLE_DENSITY
    solid_error = (solid_mass_final - solid_mass_initial) / solid_mass_initial
    solid_ok = abs(solid_error) < 1e-8
    
    events_ok = solver.real_agg_events > 0
    breakage_ok = solver.real_break_events > 0 if cfg.BREAKAGE_ENABLED else True
    
    all_ok = liquid_ok and solid_ok and events_ok and breakage_ok
    
    print(f"\n  Liquid conservation:     {'✓ PASS' if liquid_ok else '✗ FAIL'}")
    print(f"  Solid mass conservation: {'✓ PASS' if solid_ok else '✗ FAIL'} (error={format_percentage(solid_error)})")
    print(f"  Agglomeration active:    {'✓ PASS' if events_ok else '✗ FAIL'}")
    if cfg.BREAKAGE_ENABLED:
        print(f"  Breakage active:         {'✓ PASS' if breakage_ok else '✗ FAIL'}")
    
    print(f"\n{'OVERALL:':<25} {'✓ ALL TESTS PASSED' if all_ok else '✗ SOME TESTS FAILED'}")
    
    # -------------------------------------------------------------------------
    # Return Results
    # -------------------------------------------------------------------------
    
    return {
        'success': all_ok,
        'basic_results': {
            'liquid_error': (actual_liquid - expected_liquid) / expected_liquid,
            'solid_mass_error': solid_error,
            'agg_events': solver.real_agg_events,
            'break_events': solver.real_break_events,
            'elapsed_real': elapsed_real,
        },
        'correlation_results': corr_results,
        'evolution_results': evol_results,
        'export_data': export_data,
        'props_container': props,  # For interactive exploration
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    results = run_comprehensive_test()
    
    print_section("TEST COMPLETE", "=")
    print("\nThe 'props' container is available for interactive exploration:")
    print("  >>> props = results['props_container']")
    print("  >>> x, y, w = props.scatter_data(t_idx=5, x='X', y='porosity')")
    print("  >>> mask = props.filter(t_idx=5, condition='saturation > 0.5')")
    print("  >>> series, t = props.time_series(mask, 'porosity')")
    
    sys.exit(0 if results['success'] else 1)
