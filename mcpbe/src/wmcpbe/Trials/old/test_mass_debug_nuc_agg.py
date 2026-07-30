"""
Debug test: Nucleation + Agglomeration (no breakage/compression).
This isolates the mass conservation issue in nucleation+agg interaction.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmcpbe import MCPBESolver
import numpy as np
import time

print("=" * 80)
print("MASS CONSERVATION TEST - NUCLEATION + AGGLOMERATION")
print("=" * 80)

# Configuration (same as comprehensive test)
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
particle_radius = PARTICLE_DIAMETER / 2.0
particle_volume = (4.0 / 3.0) * np.pi * particle_radius ** 3
droplet_volume = (4.0 / 3.0) * np.pi * (DROPLET_DIAMETER/2)**3
expected_liquid = FLOW_RATE * NUC_DURATION

print(f"\nParticle diameter: {PARTICLE_DIAMETER*1e6:.1f} µm")
print(f"Particle volume:   {particle_volume:.6e} m³")
print(f"Droplet diameter:  {DROPLET_DIAMETER*1e6:.1f} µm")
print(f"Droplet volume:    {droplet_volume:.6e} m³")
print(f"Flow rate:         {FLOW_RATE:.2e} m³/s")
print(f"Nuc. duration:     {NUC_DURATION:.1f} s")
print(f"Expected liquid:   {expected_liquid:.6e} m³ ({expected_liquid/droplet_volume:.1f} droplets)")

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
solver.porosity[:solver.a_tot] = np.nan  # Vollkörper initially
solver.liquid_volume[:solver.a_tot] = 0.0
solver.saturation[:solver.a_tot] = 0.0

print("Initializing samplers...")
solver._initialize_samplers()

# Configure nucleation
print("Configuring nucleation...")
solver.create_nucleation_handler(
    enabled=True,
    volumetric_flow_rate=FLOW_RATE,
    droplet_diameter=DROPLET_DIAMETER,
    liquid_addition_start=0.0,
    liquid_addition_duration=NUC_DURATION,
)

# Calculate initial solid mass
v_dry_init = solver.V_flat[-1, :solver.a_tot].copy()
w_init = solver.W[:solver.a_tot].copy()
poro_init = solver.porosity[:solver.a_tot].copy()

# For Vollkörper (NaN): V_solid = V_dry
valid_poro_init = ~np.isnan(poro_init)
v_solid_init = np.zeros_like(v_dry_init)
v_solid_init[valid_poro_init] = v_dry_init[valid_poro_init] * (1.0 - poro_init[valid_poro_init])
v_solid_init[~valid_poro_init] = v_dry_init[~valid_poro_init]

initial_solid_volume = np.sum(v_solid_init * w_init)
initial_solid_mass = initial_solid_volume * PARTICLE_DENSITY

print(f"\nInitial state:")
print(f"  n_comp:        {solver.a_tot}")
print(f"  n_Vollkörper:  {np.sum(~valid_poro_init)}")
print(f"  Solid vol:     {initial_solid_volume:.6e} m³")
print(f"  Solid mass:    {initial_solid_mass:.6e} kg")

# Run simulation
print(f"\nRunning simulation for {T_TOTAL}s...")
start = time.time()
solver.solve(maxiter=int(1e7))
elapsed = time.time() - start

# Calculate final solid mass
v_dry_final = solver.V_flat[-1, :solver.a_tot]
w_final = solver.W[:solver.a_tot]
poro_final = solver.porosity[:solver.a_tot]

valid_poro_final = ~np.isnan(poro_final)
v_solid_final = np.zeros_like(v_dry_final)
v_solid_final[valid_poro_final] = v_dry_final[valid_poro_final] * (1.0 - poro_final[valid_poro_final])
v_solid_final[~valid_poro_final] = v_dry_final[~valid_poro_final]

final_solid_volume = np.sum(v_solid_final * w_final)
final_solid_mass = final_solid_volume * PARTICLE_DENSITY
mass_error = (final_solid_mass - initial_solid_mass) / initial_solid_mass

print(f"\nFinal state:")
print(f"  n_comp:        {solver.a_tot}")
print(f"  n_Vollkörper:  {np.sum(~valid_poro_final)}")
print(f"  n_porous:      {np.sum(valid_poro_final)}")
print(f"  Solid vol:     {final_solid_volume:.6e} m³")
print(f"  Solid mass:    {final_solid_mass:.6e} kg")

print(f"\nMass Conservation:")
print(f"  Δ Mass:  {final_solid_mass - initial_solid_mass:.6e} kg")
print(f"  Error:   {mass_error*100:+.4f}%")

print(f"\nNucleation Statistics:")
nuc_stats = solver.nucleation.get_statistics()
print(f"  Liquid added:    {nuc_stats['liquid_volume_added_total']:.6e} m³")
print(f"  Droplets added:  {nuc_stats['droplets_added_total']:.2f}")

print(f"\nPerformance:")
print(f"  Time:       {elapsed:.2f}s")
print(f"  Agg events: {int(solver.real_agg_events):,}")

if abs(mass_error) > 1e-6:
    print("\n❌ MASS NOT CONSERVED!")
    
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
    
    # Check porous particles
    if np.sum(valid_poro_final) > 0:
        print(f"\nPorous particles created: {np.sum(valid_poro_final)}")
        print(f"  Mean porosity: {np.mean(poro_final[valid_poro_final]):.4f}")
        
        # Sample some porous particles
        porous_idx = np.where(valid_poro_final)[0][:10]
        print(f"\n  First {len(porous_idx)} porous particles:")
        for idx in porous_idx:
            print(f"    [{idx:4d}] V_dry={v_dry_final[idx]:.6e}, poro={poro_final[idx]:.4f}, W={w_final[idx]:.1f}, liq={solver.liquid_volume[idx]:.6e}")
else:
    print("\n✓ MASS CONSERVED!")

print("\n" + "=" * 80)
