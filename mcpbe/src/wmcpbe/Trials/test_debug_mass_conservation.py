"""
Debug Test: Mass Conservation Tracking.

This test tracks V_flat, porosity, and solid mass at critical points:
1. After initialization
2. After each agglomeration event (sampled)
3. After each breakage event (sampled)
4. After compression steps
5. Final mass balance

Goal: Identify WHERE mass is lost/created.
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver


# =============================================================================
# Test Configuration
# =============================================================================

class TestConfig:
    """Physical and numerical parameters for debug test."""
    
    SEED = 42
    T_TOTAL = 1.0        # Short duration for debugging
    T_WRITE = 0.05       # Frequent output
    
    PARTICLE_DIAMETER = 500e-6
    PARTICLE_DENSITY = 2500.0
    INITIAL_POROSITY = 0.4  # 40% porosity
    
    DROPLET_DIAMETER = 100e-6
    DROPLET_DENSITY = 1000.0
    
    VOLUMETRIC_FLOW_RATE = 3e-11
    NUCLEATION_DURATION = 0.3  # Short nucleation window
    
    AGG_COEFFICIENT = 1e-5
    BREAKAGE_ENABLED = True
    BREAKAGE_RATE = 0.1
    BREAKAGE_DAUGHTERS = 2
    
    COMPRESSION_ENABLED = True
    COMPRESSION_RATE = 0.05
    MIN_POROSITY = 0.1
    
    INITIAL_PARTICLES = 100
    INITIAL_WEIGHT = 100.0
    CONTROL_VOLUME = 1.0


# =============================================================================
# Debug Helpers
# =============================================================================

def print_separator(title=""):
    print(f"\n{'='*80}")
    if title:
        print(title)
        print('='*80)


def debug_particle_state(solver, label="", sample_size=5):
    """Print detailed particle state for first few particles."""
    a_tot = solver.a_tot
    if a_tot == 0:
        print(f"[{label}] No particles!")
        return
    
    sample_size = min(sample_size, a_tot)
    
    print(f"\n[{label}] a_tot={a_tot}")
    print(f"  V_flat[-1, 0:{sample_size}] (V_dry):     {solver.V_flat[-1, :sample_size]}")
    print(f"  V_flat[0, 0:{sample_size}] (V_comp0):    {solver.V_flat[0, :sample_size]}")
    if solver.dim > 1:
        print(f"  V_flat[1, 0:{sample_size}] (V_comp1):    {solver.V_flat[1, :sample_size]}")
    print(f"  porosity[0:{sample_size}]:             {solver.porosity[:sample_size]}")
    print(f"  W[0:{sample_size}]:                    {solver.W[:sample_size]}")
    
    # Check consistency: V_flat[0,:] should be V_solid = V_dry * (1 - poro)
    v_dry_sample = solver.V_flat[-1, :sample_size]
    v_comp0_sample = solver.V_flat[0, :sample_size]
    poro_sample = solver.porosity[:sample_size]
    
    v_solid_expected = v_dry_sample * (1.0 - poro_sample)
    v_solid_expected[~np.isfinite(poro_sample)] = v_dry_sample[~np.isfinite(poro_sample)]  # Vollkörper
    
    print(f"  V_solid_expected (V_dry*(1-p)):      {v_solid_expected}")
    print(f"  V_flat[0,:] - V_solid_expected:      {v_comp0_sample - v_solid_expected}")
    
    # Ratio check
    ratio = np.divide(v_comp0_sample, v_solid_expected, 
                      out=np.ones_like(v_solid_expected), 
                      where=v_solid_expected > 0)
    print(f"  V_flat[0,:] / V_solid_expected:      {ratio}")


def compute_solid_mass_detailed(solver, density, label=""):
    """
    Compute solid mass with detailed breakdown.
    
    Returns dict with all intermediate values.
    """
    a_tot = solver.a_tot
    if a_tot == 0:
        return {'mass': 0.0, 'details': {}}
    
    v_dry = solver.V_flat[-1, :a_tot].copy()
    v_comp0 = solver.V_flat[0, :a_tot].copy()
    porosity = solver.porosity[:a_tot].copy()
    weights = solver.W[:a_tot].copy()
    
    valid_poro = ~np.isnan(porosity)
    n_vollkoerper = np.sum(~valid_poro)
    n_porous = np.sum(valid_poro)
    
    # Method 1: Calculate from V_dry and porosity (what test uses)
    v_solid_from_dry = np.zeros_like(v_dry)
    v_solid_from_dry[valid_poro] = v_dry[valid_poro] * (1.0 - porosity[valid_poro])
    v_solid_from_dry[~valid_poro] = v_dry[~valid_poro]  # Vollkörper: V_solid = V_dry
    
    mass_from_dry = np.sum(v_solid_from_dry * weights) * density
    
    # Method 2: Use V_flat[0,:] directly (assuming it's V_solid)
    v_solid_from_comp0 = v_comp0.copy()
    # For Vollkörper, V_flat[0,:] should also be V_solid
    mass_from_comp0 = np.sum(v_solid_from_comp0 * weights) * density
    
    # Method 3: Total V_dry minus pore volume
    v_pore = np.zeros_like(v_dry)
    v_pore[valid_poro] = v_dry[valid_poro] * porosity[valid_poro]
    v_solid_method3 = v_dry - v_pore
    mass_method3 = np.sum(v_solid_method3 * weights) * density
    
    return {
        'mass_from_dry': mass_from_dry,
        'mass_from_comp0': mass_from_comp0,
        'mass_method3': mass_method3,
        'n_comp': a_tot,
        'n_vollkoerper': n_vollkoerper,
        'n_porous': n_porous,
        'sum_W': np.sum(weights),
        'sum_v_dry': np.sum(v_dry * weights),
        'sum_v_comp0': np.sum(v_comp0 * weights),
        'mean_poro': np.nanmean(porosity),
        'details': {
            'v_dry_sample': v_dry[:5],
            'v_comp0_sample': v_comp0[:5],
            'poro_sample': porosity[:5],
            'W_sample': weights[:5],
            'v_solid_from_dry_sample': v_solid_from_dry[:5],
        }
    }


# =============================================================================
# Main Test Function
# =============================================================================

def run_debug_test():
    """Run debug test with detailed mass tracking."""
    
    print_separator("DEBUG MASS CONSERVATION TEST")
    print("Tracking V_flat, porosity, and solid mass at critical points")
    
    cfg = TestConfig()
    
    # Derived quantities
    particle_radius = cfg.PARTICLE_DIAMETER / 2.0
    particle_volume = (4.0 / 3.0) * np.pi * particle_radius ** 3
    
    droplet_radius = cfg.DROPLET_DIAMETER / 2.0
    droplet_volume = (4.0 / 3.0) * np.pi * droplet_radius ** 3
    
    expected_liquid = cfg.VOLUMETRIC_FLOW_RATE * cfg.NUCLEATION_DURATION
    
    # -------------------------------------------------------------------------
    # Print Configuration
    # -------------------------------------------------------------------------
    
    print(f"\nParticle diameter:  {cfg.PARTICLE_DIAMETER*1e6:.1f} µm")
    print(f"Particle volume:    {particle_volume:.3e} m³")
    print(f"Initial porosity:   {cfg.INITIAL_POROSITY*100:.1f} %")
    print(f"Initial particles:  {cfg.INITIAL_PARTICLES}")
    print(f"Simulation time:    {cfg.T_TOTAL:.2f} s")
    print(f"Compression:        {'ON' if cfg.COMPRESSION_ENABLED else 'OFF'} (rate={cfg.COMPRESSION_RATE})")
    
    # -------------------------------------------------------------------------
    # Solver Initialization
    # -------------------------------------------------------------------------
    
    print_separator("INITIALIZATION")
    
    t_vec = np.linspace(0.0, cfg.T_TOTAL, int(cfg.T_TOTAL / cfg.T_WRITE) + 1)
    rng = np.random.default_rng(cfg.SEED)
    
    solver = MCPBESolver(
        dim=1,
        t_vec=t_vec,
        verbose=False,
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
        rng=rng,
    )
    
    solver.process_type = "mix"
    solver.recon_enable = False
    
    # Initialize particles
    # IMPORTANT: V_flat structure for dim=1:
    #   V_flat[0, :] = V_solid (solid volume only)
    #   V_flat[1, :] = V_dry (total dry volume = V_solid + V_pore)
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
    
    # Set porosity for all particles
    solver.porosity[:solver.a_tot] = cfg.INITIAL_POROSITY
    solver.liquid_volume[:solver.a_tot] = 0.0
    solver.saturation[:solver.a_tot] = 0.0
    
    # -------------------------------------------------------------------------
    # DEBUG: After Initialization
    # -------------------------------------------------------------------------
    
    print_separator("DEBUG: AFTER INITIALIZATION")
    
    debug_particle_state(solver, "AFTER_INIT")
    
    mass_info_init = compute_solid_mass_detailed(solver, cfg.PARTICLE_DENSITY, "INIT")
    print(f"\n  Mass from V_dry*(1-p):  {mass_info_init['mass_from_dry']:.6e} kg")
    print(f"  Mass from V_flat[0,:]:  {mass_info_init['mass_from_comp0']:.6e} kg")
    print(f"  Mass method 3:          {mass_info_init['mass_method3']:.6e} kg")
    print(f"  sum(V_dry * W):         {mass_info_init['sum_v_dry']:.6e} m³")
    print(f"  sum(V_comp0 * W):       {mass_info_init['sum_v_comp0']:.6e} m³")
    
    initial_mass = mass_info_init['mass_from_dry']
    
    # Initialize samplers
    solver._initialize_samplers()
    
    # Configure nucleation
    solver.create_nucleation_handler(
        enabled=True,
        volumetric_flow_rate=cfg.VOLUMETRIC_FLOW_RATE,
        droplet_diameter=cfg.DROPLET_DIAMETER,
        liquid_addition_start=0.0,
        liquid_addition_duration=cfg.NUCLEATION_DURATION,
    )
    
    # Configure compression
    if cfg.COMPRESSION_ENABLED:
        solver.create_compression_handler(
            enabled=True,
            rate=cfg.COMPRESSION_RATE,
            min_porosity=cfg.MIN_POROSITY,
        )
    
    # -------------------------------------------------------------------------
    # Run Simulation with Debug Hooks
    # -------------------------------------------------------------------------
    
    print_separator("RUNNING SIMULATION WITH DEBUG HOOKS")
    
    # Monkey-patch solver methods to add debug output
    original_do_one_agg = solver._do_one_agg
    original_do_one_break = solver._do_one_break
    original_apply_compression = solver.compression._apply_compression if hasattr(solver, 'compression') else None
    
    event_counter = {'agg': 0, 'break': 0, 'comp': 0}
    state = {'last_mass': initial_mass}  # Mutable container for closure
    
    def debug_agg_hook():
        result = original_do_one_agg()
        event_counter['agg'] += 1
        if event_counter['agg'] <= 5 or event_counter['agg'] % 50 == 0:
            mass_info = compute_solid_mass_detailed(solver, cfg.PARTICLE_DENSITY, f"AGG_{event_counter['agg']}")
            dm = mass_info['mass_from_dry'] - state['last_mass']
            if abs(dm) > 1e-15:
                print(f"\n[AGG #{event_counter['agg']}] MASS CHANGE: {dm:+.3e} kg ({dm/state['last_mass']*100:+.4f}%)")
                debug_particle_state(solver, f"AGG_{event_counter['agg']}", sample_size=3)
            state['last_mass'] = mass_info['mass_from_dry']
        return result
    
    def debug_break_hook():
        result = original_do_one_break()
        event_counter['break'] += 1
        if event_counter['break'] <= 5 or event_counter['break'] % 50 == 0:
            mass_info = compute_solid_mass_detailed(solver, cfg.PARTICLE_DENSITY, f"BREAK_{event_counter['break']}")
            dm = mass_info['mass_from_dry'] - state['last_mass']
            if abs(dm) > 1e-15:
                print(f"\n[BREAK #{event_counter['break']}] MASS CHANGE: {dm:+.3e} kg ({dm/state['last_mass']*100:+.4f}%)")
                debug_particle_state(solver, f"BREAK_{event_counter['break']}", sample_size=3)
            state['last_mass'] = mass_info['mass_from_dry']
        return result
    
    def debug_compression_hook(dt):
        if original_apply_compression is None:
            return
        mass_before = compute_solid_mass_detailed(solver, cfg.PARTICLE_DENSITY, 'COMP_BEFORE')['mass_from_dry']
        result = original_apply_compression(dt)
        event_counter['comp'] += 1
        mass_after = compute_solid_mass_detailed(solver, cfg.PARTICLE_DENSITY, 'COMP_AFTER')['mass_from_dry']
        dm = mass_after - mass_before
        if abs(dm) > 1e-15 or event_counter['comp'] <= 3:
            print(f"\n[COMP #{event_counter['comp']}] dt={dt:.4f}s | MASS CHANGE: {dm:+.3e} kg ({dm/mass_before*100:+.4f}%)")
            debug_particle_state(solver, f"COMP_{event_counter['comp']}_AFTER", sample_size=3)
        return result
    
    # Apply hooks
    solver._do_one_agg = debug_agg_hook
    solver._do_one_break = debug_break_hook
    if hasattr(solver, 'compression') and original_apply_compression is not None:
        solver.compression._apply_compression = debug_compression_hook
    
    # Run simulation
    start_time = time.time()
    
    try:
        solver.solve(maxiter=int(1e7))
    except Exception as e:
        print(f"\nERROR during solve: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    elapsed = time.time() - start_time
    
    # -------------------------------------------------------------------------
    # Final Analysis
    # -------------------------------------------------------------------------
    
    print_separator("FINAL ANALYSIS")
    
    debug_particle_state(solver, "FINAL")
    
    mass_info_final = compute_solid_mass_detailed(solver, cfg.PARTICLE_DENSITY, "FINAL")
    
    print(f"\n{'Mass Calculation Method:':<30}")
    print(f"  Initial (from V_dry):     {initial_mass:.6e} kg")
    print(f"  Final (from V_dry):       {mass_info_final['mass_from_dry']:.6e} kg")
    print(f"  Final (from V_flat[0]):   {mass_info_final['mass_from_comp0']:.6e} kg")
    print(f"  Final (method 3):         {mass_info_final['mass_method3']:.6e} kg")
    
    error_dry = (mass_info_final['mass_from_dry'] - initial_mass) / initial_mass
    error_comp0 = (mass_info_final['mass_from_comp0'] - initial_mass) / initial_mass
    
    print(f"\n{'Mass Error:':<30}")
    print(f"  Using V_dry*(1-p):        {error_dry*100:+.4f}%")
    print(f"  Using V_flat[0,:] direct: {error_comp0*100:+.4f}%")
    
    print(f"\n{'Volume Sums (weighted):':<30}")
    print(f"  sum(V_dry * W):           {mass_info_final['sum_v_dry']:.6e} m³")
    print(f"  sum(V_comp0 * W):         {mass_info_final['sum_v_comp0']:.6e} m³")
    print(f"  Ratio V_comp0/V_dry:      {mass_info_final['sum_v_comp0']/mass_info_final['sum_v_dry']:.4f}")
    print(f"  Expected (1-poro_mean):   {1.0 - mass_info_final['mean_poro']:.4f}")
    
    print(f"\n{'Event Counts:':<30}")
    print(f"  Agglomeration:  {event_counter['agg']}")
    print(f"  Breakage:       {event_counter['break']}")
    print(f"  Compression:    {event_counter['comp']}")
    
    print(f"\n{'Performance:':<30}")
    print(f"  Real time:      {elapsed:.2f} s")
    print(f"  Events/sec:     {(event_counter['agg']+event_counter['break'])/elapsed:.0f}")
    
    # -------------------------------------------------------------------------
    # Key Insight
    # -------------------------------------------------------------------------
    
    print_separator("KEY INSIGHT")
    
    if abs(error_dry) > 1e-6:
        print(f"\n⚠ MASS ERROR DETECTED: {error_dry*100:+.4f}%")
        
        # Check if V_flat[0,:] is consistent with V_solid
        v_dry = solver.V_flat[-1, :solver.a_tot]
        v_comp0 = solver.V_flat[0, :solver.a_tot]
        poro = solver.porosity[:solver.a_tot]
        
        v_solid_expected = v_dry * (1.0 - poro)
        v_solid_expected[~np.isfinite(poro)] = v_dry[~np.isfinite(poro)]
        
        diff = v_comp0 - v_solid_expected
        max_diff = np.max(np.abs(diff))
        mean_ratio = np.mean(v_comp0 / v_solid_expected)
        
        print(f"\n  V_flat[0,:] vs V_solid_expected:")
        print(f"    Max absolute difference:  {max_diff:.3e} m³")
        print(f"    Mean ratio:               {mean_ratio:.4f} (expected: 1.0)")
        
        if max_diff > 1e-20:
            print(f"\n  → V_flat[0,:] is NOT equal to V_solid!")
            print(f"  → This explains the mass conservation error!")
        else:
            print(f"\n  → V_flat[0,:] equals V_solid, error must be elsewhere.")
    else:
        print(f"\n✓ Mass conserved within tolerance: {error_dry*100:+.4f}%")
    
    return {
        'initial_mass': initial_mass,
        'final_mass_dry': mass_info_final['mass_from_dry'],
        'final_mass_comp0': mass_info_final['mass_from_comp0'],
        'error_dry': error_dry,
        'error_comp0': error_comp0,
        'events': event_counter,
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    results = run_debug_test()
    
    print_separator("TEST COMPLETE")
    
    if results:
        print(f"\nResults summary:")
        print(f"  Mass error (V_dry method):  {results['error_dry']*100:+.4f}%")
        print(f"  Mass error (V_flat[0] meth):{results['error_comp0']*100:+.4f}%")
