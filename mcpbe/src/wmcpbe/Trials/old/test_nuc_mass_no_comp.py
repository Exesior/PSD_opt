"""
Nucleation + Agglomeration ONLY (NO compression, NO breakage).
This isolates the mass conservation issue.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver
import numpy as np
import time

print("=" * 80)
print("NUCLEATION + AGGLOMERATION MASS TEST (NO COMPRESSION)")
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
expected_liquid = FLOW_RATE * NUC_DURATION

def calc_solid_mass(solver):
    v_dry = solver.V_flat[-1, :solver.a_tot]
    poro = solver.porosity[:solver.a_tot]
    w = solver.W[:solver.a_tot]
    
    valid_poro = ~np.isnan(poro)
    v_solid = np.zeros_like(v_dry)
    v_solid[valid_poro] = v_dry[valid_poro] * (1.0 - poro[valid_poro])
    v_solid[~valid_poro] = v_dry[~valid_poro]
    
    return np.sum(v_solid * w) * PARTICLE_DENSITY

# Create solver WITHOUT compression
rng = np.random.default_rng(SEED)
t_vec = np.linspace(0.0, T_TOTAL, 41)

print("\nCreating solver (agg only, NO compression)...")
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
solver.porosity[:solver.a_tot] = np.nan  # All Vollkörper initially
solver.liquid_volume[:solver.a_tot] = 0.0
solver.saturation[:solver.a_tot] = 0.0

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
print(f"Expected V_dry (Vollkörper): {particle_volume:.6e} m³")
print(f"Expected V_dry (poro=0.4):   {particle_volume/0.6:.6e} m³")

# Run simulation
print(f"\nRunning simulation for {T_TOTAL}s...")
start = time.time()
solver.solve(maxiter=int(1e7))
elapsed = time.time() - start

final_mass = calc_solid_mass(solver)
total_error = (final_mass - initial_mass) / initial_mass * 100

# Analyze final state
poro_final = solver.porosity[:solver.a_tot]
v_dry_final = solver.V_flat[-1, :solver.a_tot]
w_final = solver.W[:solver.a_tot]

valid_poro = ~np.isnan(poro_final)
n_voll = np.sum(~valid_poro)
n_porous = np.sum(valid_poro)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nMass Conservation:")
print(f"  Initial mass: {initial_mass:.10e} kg")
print(f"  Final mass:   {final_mass:.10e} kg")
print(f"  Δ Mass:       {final_mass - initial_mass:+.6e} kg")
print(f"  Error:        {total_error:+.6f}%")

print(f"\nParticle Statistics:")
print(f"  n_comp total:     {solver.a_tot:,}")
print(f"  n_Vollkörper:     {n_voll:,}")
print(f"  n_porous:         {n_porous:,}")

if n_porous > 0:
    print(f"\n  Porous particles:")
    print(f"    Mean porosity:    {np.mean(poro_final[valid_poro]):.4f} (should be 0.4)")
    print(f"    Mean V_dry:       {np.mean(v_dry_final[valid_poro]):.6e} m³")
    print(f"    Expected V_dry:   {particle_volume/0.6:.6e} m³")
    print(f"    Ratio:            {np.mean(v_dry_final[valid_poro]) / (particle_volume/0.6):.4f}")
    
    # Check if V_dry is consistent with V_solid and poro
    v_solid_porous = v_dry_final[valid_poro] * (1.0 - poro_final[valid_poro])
    print(f"\n  V_solid consistency:")
    print(f"    Mean V_solid (porous): {np.mean(v_solid_porous):.6e} m³")
    print(f"    Expected V_solid:      {particle_volume:.6e} m³")
    print(f"    Ratio:                 {np.mean(v_solid_porous) / particle_volume:.4f}")

print(f"\nPerformance:")
print(f"  Time:       {elapsed:.2f}s")
print(f"  Agg events: {int(solver.real_agg_events):,}")

if abs(total_error) > 1e-4:
    print(f"\n❌ MASS ERROR DETECTED: {total_error:+.6f}%")
else:
    print(f"\n✓ MASS CONSERVED within tolerance")

print("\n" + "=" * 80)
