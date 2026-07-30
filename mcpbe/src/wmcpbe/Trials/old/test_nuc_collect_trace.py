"""
Trace particle collection during nucleation.
Shows every time particles are agglomerated to collect enough V_solid.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver
import numpy as np
import time

print("=" * 80)
print("NUCLEATION COLLECTION TRACE")
print("=" * 80)

# Configuration
SEED = 42
T_TOTAL = 4.0
PARTICLE_DIAMETER = 500e-6
PARTICLE_DENSITY = 2500.0
INITIAL_PARTICLES = 1000
INITIAL_WEIGHT = 100.0
CONTROL_VOLUME = 1.0
AGG_COEFFICIENT = 1e-5
FLOW_RATE = 3e-11
NUC_DURATION = 2.0
DROPLET_DIAMETER = 100e-6

particle_volume = (np.pi / 6.0) * PARTICLE_DIAMETER ** 3
droplet_volume = (np.pi / 6.0) * DROPLET_DIAMETER ** 3

def calc_solid_mass(solver):
    v_dry = solver.V_flat[-1, :solver.a_tot]
    poro = solver.porosity[:solver.a_tot]
    w = solver.W[:solver.a_tot]
    
    valid_poro = ~np.isnan(poro)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid_poro] = v_dry[valid_poro] * (1.0 - poro[valid_poro])
    v_solid[~valid_poro] = v_dry[~valid_poro]
    
    return np.sum(v_solid * w) * PARTICLE_DENSITY

# Create solver
rng = np.random.default_rng(SEED)
t_vec = np.linspace(0.0, T_TOTAL, 41)

print("\nCreating solver...")
solver = MCPBESolver(
    dim=1,
    t_vec=t_vec,
    verbose=False,
    load_attr=False,
    init=True,
    agg_kernel_name='constant',
    agg_kernel_params={'corr_beta': AGG_COEFFICIENT},
    rng=rng,
)

solver.process_type = "agglomeration"
solver.recon_enable = False

V_flat = np.zeros((2, INITIAL_PARTICLES), dtype=float)
V_flat[0, :] = particle_volume
V_flat[1, :] = particle_volume
W_init = np.full(INITIAL_PARTICLES, INITIAL_WEIGHT, dtype=float)
solver.Vc = CONTROL_VOLUME

print("Initializing particles...")
solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
solver.porosity[:solver.a_tot] = np.nan
solver.liquid_volume[:solver.a_tot] = 0.0

print("Initializing samplers...")
solver._initialize_samplers()

print("Configuring nucleation...")
solver.create_nucleation_handler(
    enabled=True,
    volumetric_flow_rate=FLOW_RATE,
    droplet_diameter=DROPLET_DIAMETER,
    liquid_addition_start=0.0,
    liquid_addition_duration=NUC_DURATION,
)

initial_mass = calc_solid_mass(solver)
print(f"\nInitial solid mass: {initial_mass:.10e} kg")

# Monkey-patch _perform_nucleation_agglomeration
original_perform = solver.nucleation._perform_nucleation_agglomeration
collect_count = [0]
collection_events = []

def traced_perform(i, j):
    """Trace every collection agglomeration."""
    # Get state BEFORE
    V_dry_i = float(solver.V_flat[-1, i]) if i < solver.a_tot else 0
    V_dry_j = float(solver.V_flat[-1, j]) if j < solver.a_tot else 0
    poro_i = solver.porosity[i] if i < solver.a_tot else np.nan
    poro_j = solver.porosity[j] if j < solver.a_tot else np.nan
    W_i = float(solver.W[i]) if i < solver.a_tot else 0
    W_j = float(solver.W[j]) if j < solver.a_tot else 0
    
    V_solid_i = V_dry_i * (1.0 - poro_i) if not np.isnan(poro_i) else V_dry_i
    V_solid_j = V_dry_j * (1.0 - poro_j) if not np.isnan(poro_j) else V_dry_j
    
    mass_before = calc_solid_mass(solver)
    
    # Call original
    result = original_perform(i, j)
    
    # Get state AFTER
    mass_after = calc_solid_mass(solver)
    delta_mass = mass_after - mass_before
    
    collect_count[0] += 1
    
    # Get child properties
    if result >= 0 and result < solver.a_tot:
        V_dry_child = solver.V_flat[-1, result]
        poro_child = solver.porosity[result]
        V_solid_child = V_dry_child * (1.0 - poro_child) if not np.isnan(poro_child) else V_dry_child
        
        collection_events.append({
            'collect_num': collect_count[0],
            'i': i, 'j': j,
            'child_idx': result,
            'V_dry_i': V_dry_i,
            'V_dry_j': V_dry_j,
            'poro_i': poro_i,
            'poro_j': poro_j,
            'V_solid_i': V_solid_i,
            'V_solid_j': V_solid_j,
            'V_solid_expected': V_solid_i + V_solid_j,
            'V_dry_child': V_dry_child,
            'poro_child': poro_child,
            'V_solid_child': V_solid_child,
            'delta_mass': delta_mass,
        })
    
    return result

solver.nucleation._perform_nucleation_agglomeration = traced_perform

# Run simulation
print(f"\nRunning simulation for {T_TOTAL}s...")
print("Tracing every collection agglomeration...\n")

start = time.time()
solver.solve(maxiter=int(1e7))
elapsed = time.time() - start

final_mass = calc_solid_mass(solver)
total_error = (final_mass - initial_mass) / initial_mass * 100

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nCollection agglomerations: {collect_count[0]}")
print(f"Total MC agg events:       {int(solver.real_agg_events)}")
print(f"\nInitial mass: {initial_mass:.10e} kg")
print(f"Final mass:   {final_mass:.10e} kg")
print(f"Total error:  {total_error:+.6f}%")

if len(collection_events) > 0:
    print(f"\n{'Coll#':>5} | {'Δ Mass [kg]':>14} | {'V_s_exp':>10} | {'V_s_ch':>10} | {'Ratio':>7} | {'Poro':>6}")
    print("-" * 65)
    
    for e in collection_events[:50]:  # Show first 50
        ratio = e['V_solid_child'] / e['V_solid_expected'] if e['V_solid_expected'] > 0 else 0
        poro_str = f"{e['poro_child']:.3f}" if not np.isnan(e['poro_child']) else "NaN"
        print(f"{e['collect_num']:5d} | {e['delta_mass']:>+14.6e} | {e['V_solid_expected']:>10.3e} | {e['V_solid_child']:>10.3e} | {ratio:>7.4f} | {poro_str:>6}")
    
    if len(collection_events) > 50:
        print(f"... and {len(collection_events) - 50} more collections")
    
    # Analyze pattern
    ratios = [e['V_solid_child'] / e['V_solid_expected'] for e in collection_events if e['V_solid_expected'] > 0]
    print(f"\nPattern Analysis:")
    print(f"  Mean V_solid ratio: {np.mean(ratios):.4f} (should be 1.0)")
    print(f"  Min/Max ratio:      {np.min(ratios):.4f} / {np.max(ratios):.4f}")
    
    mass_from_collections = sum(e['delta_mass'] for e in collection_events)
    print(f"\nCumulative mass change from collections: {mass_from_collections:+.6e} kg")
    print(f"Actual total mass loss:                   {final_mass - initial_mass:+.6e} kg")

print(f"\nPerformance:")
print(f"  Time:       {elapsed:.2f}s")

print("\n" + "=" * 80)
