"""
Debug test for mass conservation with BREAKAGE only.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver
import numpy as np
import time

print("=" * 80)
print("MASS CONSERVATION DEBUG TEST - BREAKAGE ONLY")
print("=" * 80)

# Configuration
SEED = 42
T_TOTAL = 2.0
PARTICLE_DIAMETER = 500e-6
PARTICLE_DENSITY = 2500.0
INITIAL_PARTICLES = 1000
INITIAL_WEIGHT = 100.0
CONTROL_VOLUME = 1.0
BREAKAGE_RATE = 0.1

# Derived
particle_radius = PARTICLE_DIAMETER / 2.0
particle_volume = (4.0 / 3.0) * np.pi * particle_radius ** 3

print(f"\nParticle diameter: {PARTICLE_DIAMETER*1e6:.1f} µm")
print(f"Particle volume:   {particle_volume:.6e} m³")
print(f"Breakage rate:     {BREAKAGE_RATE:.2f} 1/s")

# Create RNG
rng = np.random.default_rng(SEED)
t_vec = np.linspace(0.0, T_TOTAL, 21)

print("\nCreating solver (breakage only)...")
solver = MCPBESolver(
    dim=1,
    t_vec=t_vec,
    verbose=False,
    load_attr=False,
    init=True,
    break_kernel_name='power_law',
    break_kernel_params={
        'breakage_rate': BREAKAGE_RATE,
        'daughter_count': 2,
    },
    rng=rng,
)

solver.process_type = "breakage"
solver.recon_enable = False

# Initialize particles
V_flat = np.zeros((2, INITIAL_PARTICLES), dtype=float)
V_flat[0, :] = particle_volume
V_flat[1, :] = particle_volume
W_init = np.full(INITIAL_PARTICLES, INITIAL_WEIGHT, dtype=float)
solver.Vc = CONTROL_VOLUME

print("Initializing particles...")
solver._initialize_particles(init_Vc=False, V_flat=V_flat, W_init=W_init)
solver.porosity[:solver.a_tot] = np.nan  # Vollkörper
solver.liquid_volume[:solver.a_tot] = 0.0

print("Initializing samplers...")
solver._initialize_samplers()

# Calculate initial solid mass
v_dry_init = solver.V_flat[-1, :solver.a_tot].copy()
w_init = solver.W[:solver.a_tot].copy()
initial_solid_volume = np.sum(v_dry_init * w_init)  # Vollkörper: V_solid = V_dry
initial_solid_mass = initial_solid_volume * PARTICLE_DENSITY

print(f"\nInitial state:")
print(f"  n_comp:     {solver.a_tot}")
print(f"  V_dry:      {v_dry_init[0]:.6e} m³")
print(f"  W[0]:       {w_init[0]:.1f}")
print(f"  Solid vol:  {initial_solid_volume:.6e} m³")
print(f"  Solid mass: {initial_solid_mass:.6e} kg")

# Run simulation
print(f"\nRunning simulation for {T_TOTAL}s...")
start = time.time()
solver.solve(maxiter=int(1e6))
elapsed = time.time() - start

# Calculate final solid mass
v_dry_final = solver.V_flat[-1, :solver.a_tot]
w_final = solver.W[:solver.a_tot]
poro_final = solver.porosity[:solver.a_tot]

# For Vollkörper (NaN): V_solid = V_dry
valid_poro = ~np.isnan(poro_final)
v_solid_final = np.zeros_like(v_dry_final)
v_solid_final[valid_poro] = v_dry_final[valid_poro] * (1.0 - poro_final[valid_poro])
v_solid_final[~valid_poro] = v_dry_final[~valid_poro]

final_solid_volume = np.sum(v_solid_final * w_final)
final_solid_mass = final_solid_volume * PARTICLE_DENSITY
mass_error = (final_solid_mass - initial_solid_mass) / initial_solid_mass

print(f"\nFinal state:")
print(f"  n_comp:       {solver.a_tot}")
print(f"  n_Vollkörper: {np.sum(~valid_poro)}")
print(f"  n_porous:     {np.sum(valid_poro)}")
print(f"  V_dry[0]:     {v_dry_final[0]:.6e} m³")
print(f"  W[0]:         {w_final[0]:.1f}")
print(f"  poro[0]:      {poro_final[0]}")
print(f"  Solid vol:    {final_solid_volume:.6e} m³")
print(f"  Solid mass:   {final_solid_mass:.6e} kg")

print(f"\nMass Conservation:")
print(f"  Δ Mass:  {final_solid_mass - initial_solid_mass:.6e} kg")
print(f"  Error:   {mass_error*100:+.4f}%")

print(f"\nPerformance:")
print(f"  Time:     {elapsed:.2f}s")
print(f"  Events:   {int(solver.real_break_events)}")

if abs(mass_error) > 1e-6:
    print("\n❌ MASS NOT CONSERVED!")
    
    # Detailed analysis
    print("\n=== DETAILED ANALYSIS ===")
    
    total_weight_init = np.sum(w_init)
    total_weight_final = np.sum(w_final)
    print(f"\nTotal weight:")
    print(f"  Initial: {total_weight_init:.1f}")
    print(f"  Final:   {total_weight_final:.1f}")
    print(f"  Δ:       {total_weight_final - total_weight_init:+.1f}")
    
    total_vdry_w_init = np.sum(v_dry_init * w_init)
    total_vdry_w_final = np.sum(v_dry_final * w_final)
    print(f"\nTotal V_dry × W:")
    print(f"  Initial: {total_vdry_w_init:.6e}")
    print(f"  Final:   {total_vdry_w_final:.6e}")
    print(f"  Δ:       {(total_vdry_w_final - total_vdry_w_init)/total_vdry_w_init*100:+.4f}%")
else:
    print("\n✓ MASS CONSERVED!")

print("\n" + "=" * 80)
