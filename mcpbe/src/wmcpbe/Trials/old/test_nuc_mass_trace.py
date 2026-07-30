"""
Trace solid mass after EVERY nucleation event to find exact loss point.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver
import numpy as np
import time

print("=" * 80)
print("NUCLEATION MASS TRACE - Event-by-Event Analysis")
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

# Derived
particle_volume = (np.pi / 6.0) * PARTICLE_DIAMETER ** 3
droplet_volume = (np.pi / 6.0) * DROPLET_DIAMETER ** 3
expected_liquid = FLOW_RATE * NUC_DURATION

def calc_solid_mass(solver):
    """Calculate total solid mass from current state."""
    v_dry = solver.V_flat[-1, :solver.a_tot]
    poro = solver.porosity[:solver.a_tot]
    w = solver.W[:solver.a_tot]
    
    valid_poro = ~np.isnan(poro)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid_poro] = v_dry[valid_poro] * (1.0 - poro[valid_poro])
    v_solid[~valid_poro] = v_dry[~valid_poro]
    
    return np.sum(v_solid * w) * PARTICLE_DENSITY

# Create RNG
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

# Initialize particles
V_flat = np.zeros((2, INITIAL_PARTICLES), dtype=float)
V_flat[0, :] = particle_volume
V_flat[1, :] = particle_volume
W_init = np.full(INITIAL_PARTICLES, INITIAL_WEIGHT, dtype=float)
solver.Vc = CONTROL_VOLUME

print("Initializing particles...")
solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
solver.porosity[:solver.a_tot] = np.nan
solver.liquid_volume[:solver.a_tot] = 0.0
solver.saturation[:solver.a_tot] = 0.0

print("Initializing samplers...")
solver._initialize_samplers()

# Configure nucleation with custom handler for tracing
print("Configuring nucleation with mass tracing...")
solver.create_nucleation_handler(
    enabled=True,
    volumetric_flow_rate=FLOW_RATE,
    droplet_diameter=DROPLET_DIAMETER,
    liquid_addition_start=0.0,
    liquid_addition_duration=NUC_DURATION,
)

# Initial mass
initial_mass = calc_solid_mass(solver)
print(f"\nInitial solid mass: {initial_mass:.10e} kg")

# Monkey-patch nucleation step() to trace mass
original_step = solver.nucleation.step
event_count = [0]
mass_errors = []

def traced_step(current_time, solver_last_dt):
    """Wrapper around nucleation.step() that traces mass."""
    mass_before = calc_solid_mass(solver)
    
    result = original_step(current_time, solver_last_dt)
    
    mass_after = calc_solid_mass(solver)
    delta_mass = mass_after - mass_before
    rel_error = delta_mass / initial_mass * 100 if initial_mass > 0 else 0
    
    event_count[0] += 1
    
    # Log significant events
    if abs(rel_error) > 1e-8:  # More than numerical noise
        mass_errors.append({
            'event': event_count[0],
            'time': current_time,
            'mass_before': mass_before,
            'mass_after': mass_after,
            'delta': delta_mass,
            'rel_error_pct': rel_error,
        })
        
        print(f"\n[EVENT {event_count[0]:4d}] t={current_time:.4f}s")
        print(f"  Mass before: {mass_before:.10e} kg")
        print(f"  Mass after:  {mass_after:.10e} kg")
        print(f"  Δ Mass:      {delta_mass:+.6e} kg ({rel_error:+.6f}%)")
        print(f"  Cumulative:  {mass_after - initial_mass:+.6e} kg ({(mass_after-initial_mass)/initial_mass*100:+.6f}%)")
    
    return result

solver.nucleation.step = traced_step

# Run simulation
print(f"\nRunning simulation for {T_TOTAL}s...")
print("Tracing mass after every nucleation event...\n")

start = time.time()
solver.solve(maxiter=int(1e7))
elapsed = time.time() - start

# Final analysis
final_mass = calc_solid_mass(solver)
total_error = (final_mass - initial_mass) / initial_mass * 100

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nTotal nucleation events: {event_count[0]}")
print(f"Events with mass change: {len(mass_errors)}")
print(f"\nInitial mass: {initial_mass:.10e} kg")
print(f"Final mass:   {final_mass:.10e} kg")
print(f"Total error:  {total_error:+.6f}%")

if len(mass_errors) > 0:
    print(f"\n{'Event':>6} | {'Time [s]':>10} | {'Δ Mass [kg]':>14} | {'Rel Error [%]':>12} | {'Cumulative [%]':>13}")
    print("-" * 65)
    
    cumulative = 0.0
    for err in mass_errors[:20]:  # Show first 20
        cumulative += err['rel_error_pct']
        print(f"{err['event']:6d} | {err['time']:>10.4f} | {err['delta']:>14.6e} | {err['rel_error_pct']:>+12.6f} | {cumulative:>+13.6f}")
    
    if len(mass_errors) > 20:
        print(f"... and {len(mass_errors) - 20} more events")
        
        # Show last 5
        print("\nLast 5 events:")
        for err in mass_errors[-5:]:
            print(f"{err['event']:6d} | {err['time']:>10.4f} | {err['delta']:>14.6e} | {err['rel_error_pct']:>+12.6f}")

# Analyze porous particles
poro_final = solver.porosity[:solver.a_tot]
valid_poro = ~np.isnan(poro_final)
v_dry_final = solver.V_flat[-1, :solver.a_tot]
w_final = solver.W[:solver.a_tot]

print(f"\n{'Particle Analysis:':<25}")
print(f"  n_comp total:     {solver.a_tot:,}")
print(f"  n_Vollkörper:     {np.sum(~valid_poro):,}")
print(f"  n_porous:         {np.sum(valid_poro):,}")

if np.sum(valid_poro) > 0:
    print(f"\n  Porous particle statistics:")
    print(f"    Mean porosity:    {np.mean(poro_final[valid_poro]):.4f}")
    print(f"    Mean V_dry:       {np.mean(v_dry_final[valid_poro]):.6e} m³")
    print(f"    Mean W:           {np.mean(w_final[valid_poro]):.1f}")
    
    # Check V_dry consistency
    print(f"\n  V_dry consistency check:")
    print(f"    Expected V_dry (Vollkörper): {particle_volume:.6e} m³")
    print(f"    Expected V_dry (poro=0.4):   {particle_volume/0.6:.6e} m³")
    print(f"    Actual mean V_dry (porous):  {np.mean(v_dry_final[valid_poro]):.6e} m³")

print(f"\nPerformance:")
print(f"  Time:       {elapsed:.2f}s")
print(f"  Agg events: {int(solver.real_agg_events):,}")

print("\n" + "=" * 80)
